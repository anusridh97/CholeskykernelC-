from __future__ import annotations

import argparse
import os

import torch

from .common import (
    add_student_pkg,
    atomic_json_dump,
    result_path,
    tensor_relative_error,
)
from .cuda_scaling import batched_whiten, make_problem


def run(
    rank: int,
    steps: int,
    tile_rows: int,
    tile_cols: int,
    setup_num_warps: int,
    sweep_num_warps: int,
    seed: int,
) -> dict:
    add_student_pkg()
    from doublesided_cholesky import (
        TritonTiledWorkspace,
        incremental_update_triton_parallel,
    )

    G, L, M, Aw, _, _ = make_problem(1, rank, seed)
    L_work = L.clone()
    Aw_work = Aw
    workspace = TritonTiledWorkspace(Aw, solve_width=2)
    generator = torch.Generator(device="cuda").manual_seed(seed + 1000)

    for _ in range(steps):
        z = (
            torch.randn(1, rank, device="cuda", generator=generator)
            * 0.05
        )
        w = (
            torch.randn(1, rank, device="cuda", generator=generator)
            * 0.05
        )
        G.add_(z.unsqueeze(2) @ z.unsqueeze(1))
        M.add_(z.unsqueeze(2) @ w.unsqueeze(1))
        Aw_work, _ = incremental_update_triton_parallel(
            L_work,
            Aw_work,
            z,
            w,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            setup_num_warps=setup_num_warps,
            sweep_num_warps=sweep_num_warps,
            workspace=workspace,
        )

    torch.cuda.synchronize()
    L_exact = torch.linalg.cholesky(G)
    Aw_exact = batched_whiten(L_exact, M)
    payload = {
        "experiment": "cuda_parallel_stability",
        "rank": rank,
        "steps": steps,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "setup_num_warps": setup_num_warps,
        "sweep_num_warps": sweep_num_warps,
        "L_relative_error": tensor_relative_error(L_work, L_exact),
        "Aw_relative_error": tensor_relative_error(Aw_work, Aw_exact),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    output = result_path(
        "cuda_tuning",
        f"cuda_parallel_stability_r{rank}_s{steps}_"
        f"{os.environ.get('SLURM_JOB_ID', 'local')}",
    )
    atomic_json_dump(payload, output)
    print(payload, flush=True)
    print(f"wrote {output}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--tile-rows", type=int, required=True)
    parser.add_argument("--tile-cols", type=int, required=True)
    parser.add_argument("--setup-num-warps", type=int, required=True)
    parser.add_argument("--sweep-num-warps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        rank=args.rank,
        steps=args.steps,
        tile_rows=args.tile_rows,
        tile_cols=args.tile_cols,
        setup_num_warps=args.setup_num_warps,
        sweep_num_warps=args.sweep_num_warps,
        seed=args.seed,
    )
