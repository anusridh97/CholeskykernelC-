from __future__ import annotations

import statistics

import torch

from .common import add_student_pkg
from .cuda_scaling import make_problem


def _median_event_seconds(function, repeats: int, prepare=None) -> float:
    samples = []
    for _ in range(repeats):
        if prepare is not None:
            prepare()
            torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / 1000.0)
    return statistics.median(samples)


def profile_optimized_phases(
    rank: int,
    impl: str,
    setup_num_warps: int,
    sweep_num_warps: int,
    tile_rows: int,
    tile_cols: int,
    repeats: int = 20,
    seed: int = 0,
) -> dict | None:
    """Measure the three launches in an optimized large-rank update."""
    if impl not in ("tiled_v2", "scan", "cumsum"):
        return None

    import triton

    add_student_pkg()
    import doublesided_cholesky as dc

    _, L, _, Aw, z, w = make_problem(1, rank, seed)
    L_work = L[0].clone()
    workspace = dc.TritonTiledWorkspace(Aw[0])
    block = triton.next_power_of_2(rank)

    def setup():
        dc._setup_update_solve_kernel[(1,)](
            L_work,
            z[0],
            w[0],
            workspace.vz,
            workspace.vw,
            workspace.cs,
            workspace.ss,
            workspace.prefix_c,
            r=rank,
            BLOCK=block,
            STORE_PREFIX=impl == "cumsum",
            num_warps=setup_num_warps,
        )

    if impl == "tiled_v2":
        def left():
            dc._left_sweep_cols_batched_kernel[
                (triton.cdiv(rank, tile_cols), 1)
            ](
                Aw[0],
                workspace.aw_mid,
                workspace.cs,
                workspace.ss,
                r=rank,
                TILE_COLS=tile_cols,
                num_warps=sweep_num_warps,
            )

        def right():
            dc._right_sweep_rows_rank1_kernel[
                (triton.cdiv(rank, tile_rows), 1)
            ](
                workspace.aw_mid,
                workspace.aw_out,
                workspace.cs,
                workspace.ss,
                workspace.vz,
                workspace.vw,
                r=rank,
                TILE_ROWS=tile_rows,
                num_warps=sweep_num_warps,
            )

        def total():
            dc.incremental_update_triton_tiled_v2(
                L_work,
                Aw[0],
                z[0],
                w[0],
                tile_rows=tile_rows,
                tile_cols=tile_cols,
                setup_num_warps=setup_num_warps,
                sweep_num_warps=sweep_num_warps,
                workspace=workspace,
            )
    elif impl == "scan":
        def left():
            dc._left_sweep_scan_kernel[
                (triton.cdiv(rank, tile_cols), 1)
            ](
                Aw[0],
                workspace.aw_mid,
                workspace.cs,
                workspace.ss,
                r=rank,
                BLOCK_R=block,
                TILE_COLS=tile_cols,
                num_warps=sweep_num_warps,
            )

        def right():
            dc._right_sweep_scan_rank1_kernel[
                (triton.cdiv(rank, tile_rows), 1)
            ](
                workspace.aw_mid,
                workspace.aw_out,
                workspace.cs,
                workspace.ss,
                workspace.vz,
                workspace.vw,
                r=rank,
                BLOCK_R=block,
                TILE_ROWS=tile_rows,
                num_warps=sweep_num_warps,
            )

        def total():
            dc.incremental_update_triton_scan(
                L_work,
                Aw[0],
                z[0],
                w[0],
                tile_rows=tile_rows,
                tile_cols=tile_cols,
                setup_num_warps=setup_num_warps,
                sweep_num_warps=sweep_num_warps,
                workspace=workspace,
            )
    else:
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
            dc.incremental_update_triton_cumsum(
                L_work,
                Aw[0],
                z[0],
                w[0],
                tile_rows=tile_rows,
                tile_cols=tile_cols,
                setup_num_warps=setup_num_warps,
                sweep_num_warps=sweep_num_warps,
                workspace=workspace,
            )

    # Compile and establish all phase inputs before collecting samples.
    L_work.copy_(L[0])
    setup()
    left()
    right()
    torch.cuda.synchronize()

    setup_seconds = _median_event_seconds(
        setup,
        repeats,
        prepare=lambda: L_work.copy_(L[0]),
    )
    setup()
    torch.cuda.synchronize()
    left_seconds = _median_event_seconds(left, repeats)
    left()
    torch.cuda.synchronize()
    right_seconds = _median_event_seconds(right, repeats)
    total_seconds = _median_event_seconds(
        total,
        repeats,
        prepare=lambda: L_work.copy_(L[0]),
    )
    return {
        "rank": rank,
        "impl": impl,
        "setup_num_warps": setup_num_warps,
        "sweep_num_warps": sweep_num_warps,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "repeats": repeats,
        "setup_seconds": setup_seconds,
        "left_sweep_seconds": left_seconds,
        "right_sweep_and_rank1_seconds": right_seconds,
        "sum_phase_seconds": setup_seconds + left_seconds + right_seconds,
        "end_to_end_seconds": total_seconds,
        "workspace_bytes": workspace.nbytes,
    }
