from __future__ import annotations

import argparse
import math
import os

import torch

from .common import (
    add_student_pkg,
    atomic_json_dump,
    result_path,
    tensor_relative_error,
)
from .cuda_scaling import batched_whiten, make_problem


def make_conditioned_problem(rank: int, condition: float, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    raw = torch.randn(
        rank,
        rank,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    q, _ = torch.linalg.qr(raw)
    eigenvalues = torch.logspace(
        0.0,
        math.log10(condition),
        rank,
        device="cuda",
        dtype=torch.float32,
    )
    G = (q * eigenvalues.unsqueeze(0)) @ q.mT
    G = ((G + G.mT) * 0.5).unsqueeze(0).contiguous()
    L = torch.linalg.cholesky(G).contiguous()
    M = (
        torch.randn(
            1,
            rank,
            rank,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        / math.sqrt(rank)
    )
    Aw = batched_whiten(L, M)
    return G, L, M, Aw


def run(
    rank: int,
    steps: int,
    tile_rows: int,
    tile_cols: int,
    setup_num_warps: int,
    sweep_num_warps: int,
    seed: int,
    large: bool = False,
    inplace: bool = False,
    condition: float | None = None,
) -> dict:
    add_student_pkg()
    from doublesided_cholesky import (
        TritonTiledWorkspace,
        TritonInverseInplaceWorkspace,
        incremental_update_triton_inverse_factor,
        incremental_update_triton_inverse_factor_large,
        incremental_update_triton_inverse_factor_inplace,
    )

    if condition is None:
        G, L, M, Aw, _, _ = make_problem(1, rank, seed)
    else:
        G, L, M, Aw = make_conditioned_problem(rank, condition, seed)
    initial_condition = float(torch.linalg.cond(G[0]).item())
    Linv = torch.linalg.inv(L).tril().contiguous()
    workspace = (
        TritonInverseInplaceWorkspace(Aw)
        if inplace
        else TritonTiledWorkspace(Aw, solve_width=2)
    )
    generator = torch.Generator(device="cuda").manual_seed(seed + 1000)
    checkpoints = sorted(
        {value for value in (1, 10, 50, 100, 200, steps) if value <= steps}
    )
    checkpoint_rows = []

    Aw_work = Aw
    for step in range(1, steps + 1):
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
        if inplace:
            Aw_work, _ = incremental_update_triton_inverse_factor_inplace(
                Linv,
                Aw_work,
                z,
                w,
                tile_rows=tile_rows,
                tile_cols=tile_cols,
                setup_num_warps=setup_num_warps,
                sweep_num_warps=sweep_num_warps,
                workspace=workspace,
            )
        elif large:
            Aw_work, _ = incremental_update_triton_inverse_factor_large(
                Linv,
                Aw_work,
                z,
                w,
                workspace=workspace,
            )
        else:
            Aw_work, _ = incremental_update_triton_inverse_factor(
                Linv,
                Aw_work,
                z,
                w,
                tile_rows=tile_rows,
                tile_cols=tile_cols,
                setup_num_warps=setup_num_warps,
                sweep_num_warps=sweep_num_warps,
                workspace=workspace,
            )
        if step in checkpoints:
            torch.cuda.synchronize()
            L_checkpoint = torch.linalg.cholesky(G)
            Linv_checkpoint = torch.linalg.inv(L_checkpoint)
            Aw_checkpoint = batched_whiten(L_checkpoint, M)
            identity = torch.eye(
                rank,
                device=Linv.device,
                dtype=Linv.dtype,
            ).unsqueeze(0)
            inverse_residual = tensor_relative_error(
                Linv @ L_checkpoint,
                identity,
            )
            checkpoint_rows.append(
                {
                    "step": step,
                    "Linv_relative_error": tensor_relative_error(
                        Linv,
                        Linv_checkpoint,
                    ),
                    "Aw_relative_error": tensor_relative_error(
                        Aw_work,
                        Aw_checkpoint,
                    ),
                    "inverse_residual": inverse_residual,
                }
            )

    torch.cuda.synchronize()
    L_exact = torch.linalg.cholesky(G)
    Linv_exact = torch.linalg.inv(L_exact)
    Aw_exact = batched_whiten(L_exact, M)
    payload = {
        "experiment": "cuda_inverse_stability",
        "rank": rank,
        "steps": steps,
        "seed": seed,
        "target_condition": condition,
        "initial_measured_condition": initial_condition,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "setup_num_warps": setup_num_warps,
        "sweep_num_warps": sweep_num_warps,
        "implementation": (
            "inplace" if inplace else "large" if large else "single_block"
        ),
        "Linv_relative_error": tensor_relative_error(Linv, Linv_exact),
        "Aw_relative_error": tensor_relative_error(Aw_work, Aw_exact),
        "checkpoints": checkpoint_rows,
        "max_Linv_relative_error": max(
            row["Linv_relative_error"] for row in checkpoint_rows
        ),
        "max_Aw_relative_error": max(
            row["Aw_relative_error"] for row in checkpoint_rows
        ),
        "max_inverse_residual": max(
            row["inverse_residual"] for row in checkpoint_rows
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    implementation = payload["implementation"].replace("_", "-")
    condition_suffix = (
        ""
        if condition is None
        else f"_cond{condition:g}"
    )
    output = result_path(
        "cuda_tuning",
        f"cuda_inverse_stability_{implementation}_"
        f"r{rank}_s{steps}_seed{seed}{condition_suffix}_"
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
    parser.add_argument("--setup-num-warps", type=int, default=4)
    parser.add_argument("--sweep-num-warps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--condition", type=float)
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
        large=args.large,
        inplace=args.inplace,
        condition=args.condition,
    )
