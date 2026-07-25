from __future__ import annotations

import argparse
import os

import torch
import triton

from .common import add_student_pkg, atomic_json_dump, result_path
from .cuda_phase_profile import _median_event_seconds
from .cuda_scaling import make_problem


def profile(
    rank: int,
    tile_rows: int,
    tile_cols: int,
    setup_num_warps: int,
    sweep_num_warps: int,
    repeats: int,
    seed: int,
    solve_width: int = 2,
    solve_right: bool = False,
) -> dict:
    add_student_pkg()
    import doublesided_cholesky as dc

    _, L, _, Aw, z, w = make_problem(1, rank, seed)
    L_work = L[0].clone()
    workspace = dc.TritonTiledWorkspace(Aw[0], solve_width=solve_width)
    block = triton.next_power_of_2(rank)

    def pack():
        dc._pack_solve_rhs_kernel[(1,)](
            z[0],
            w[0],
            workspace.rhs,
            r=rank,
            BLOCK=block,
            RHS_WIDTH=solve_width,
            num_warps=setup_num_warps,
        )

    def solve():
        if solve_right:
            torch.linalg.solve_triangular(
                L_work.mT,
                workspace.rhs.mT,
                upper=True,
                left=False,
                out=workspace.solve.mT,
            )
        else:
            torch.linalg.solve_triangular(
                L_work,
                workspace.rhs,
                upper=False,
                out=workspace.solve,
            )

    def params():
        dc._parallel_update_params_kernel[(1,)](
            workspace.solve,
            workspace.vz,
            workspace.vw,
            workspace.cs,
            workspace.ss,
            workspace.prefix_c,
            r=rank,
            BLOCK=block,
            RHS_WIDTH=solve_width,
            num_warps=setup_num_warps,
        )

    def cholesky_update():
        dc._parallel_cholesky_update_kernel[
            (triton.cdiv(rank, tile_rows), 1)
        ](
            L_work,
            workspace.solve,
            workspace.vz,
            workspace.cs,
            r=rank,
            BLOCK_R=block,
            TILE_ROWS=tile_rows,
            RHS_WIDTH=solve_width,
            num_warps=sweep_num_warps,
        )

    def left():
        dc._left_sweep_cumsum_kernel[
            (triton.cdiv(rank, tile_cols), 1)
        ](
            Aw[0],
            workspace.aw_mid,
            workspace.cs,
            workspace.ss,
            workspace.prefix_c,
            r=rank,
            BLOCK_R=block,
            TILE_COLS=tile_cols,
            num_warps=sweep_num_warps,
        )

    def right():
        dc._right_sweep_cumsum_rank1_kernel[
            (triton.cdiv(rank, tile_rows), 1)
        ](
            workspace.aw_mid,
            workspace.aw_out,
            workspace.cs,
            workspace.ss,
            workspace.prefix_c,
            workspace.vz,
            workspace.vw,
            r=rank,
            BLOCK_R=block,
            TILE_ROWS=tile_rows,
            num_warps=sweep_num_warps,
        )

    def total():
        dc.incremental_update_triton_parallel(
            L_work,
            Aw[0],
            z[0],
            w[0],
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            setup_num_warps=setup_num_warps,
            sweep_num_warps=sweep_num_warps,
            solve_width=solve_width,
            solve_right=solve_right,
            workspace=workspace,
        )

    # Compile every specialization and establish its inputs.
    pack()
    solve()
    params()
    cholesky_update()
    left()
    right()
    torch.cuda.synchronize()

    pack_seconds = _median_event_seconds(pack, repeats)
    pack()
    solve_seconds = _median_event_seconds(solve, repeats)
    solve()
    params_seconds = _median_event_seconds(params, repeats)
    params()
    cholesky_seconds = _median_event_seconds(
        cholesky_update,
        repeats,
        prepare=lambda: L_work.copy_(L[0]),
    )
    left_seconds = _median_event_seconds(left, repeats)
    left()
    right_seconds = _median_event_seconds(right, repeats)
    total_seconds = _median_event_seconds(
        total,
        repeats,
        prepare=lambda: L_work.copy_(L[0]),
    )
    phases = {
        "pack_seconds": pack_seconds,
        "solve_seconds": solve_seconds,
        "params_seconds": params_seconds,
        "cholesky_update_seconds": cholesky_seconds,
        "left_sweep_seconds": left_seconds,
        "right_sweep_and_rank1_seconds": right_seconds,
    }
    payload = {
        "experiment": "cuda_parallel_profile",
        "rank": rank,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "setup_num_warps": setup_num_warps,
        "sweep_num_warps": sweep_num_warps,
        "solve_width": solve_width,
        "solve_right": solve_right,
        "repeats": repeats,
        **phases,
        "sum_phase_seconds": sum(phases.values()),
        "end_to_end_seconds": total_seconds,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    side = "right" if solve_right else "left"
    stem = (
        f"cuda_parallel_profile_r{rank}_rhs{solve_width}_{side}_"
        f"{os.environ.get('SLURM_JOB_ID', 'local')}"
    )
    output = result_path("cuda_tuning", stem)
    atomic_json_dump(payload, output)
    print(payload, flush=True)
    print(f"wrote {output}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--tile-rows", type=int, required=True)
    parser.add_argument("--tile-cols", type=int, required=True)
    parser.add_argument("--setup-num-warps", type=int, default=4)
    parser.add_argument("--sweep-num-warps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--solve-width", type=int, default=2)
    parser.add_argument("--solve-right", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    profile(
        rank=args.rank,
        tile_rows=args.tile_rows,
        tile_cols=args.tile_cols,
        setup_num_warps=args.setup_num_warps,
        sweep_num_warps=args.sweep_num_warps,
        repeats=args.repeats,
        seed=args.seed,
        solve_width=args.solve_width,
        solve_right=args.solve_right,
    )
