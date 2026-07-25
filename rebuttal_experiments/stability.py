from __future__ import annotations

import argparse
import os
import platform
import socket
import time

import torch

from .common import (
    add_student_pkg,
    atomic_json_dump,
    configure_threads,
    result_path,
    set_reproducible_seed,
    tensor_relative_error,
)


def whiten(L: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    left = torch.linalg.solve_triangular(L, M, upper=False)
    return torch.linalg.solve_triangular(L, left.mT, upper=False).mT.contiguous()


def make_spd(rank: int, condition: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(
        torch.randn(rank, rank, dtype=torch.float64, generator=generator)
    )
    eigenvalues = torch.logspace(
        0.0, torch.log10(torch.tensor(condition)).item(), rank, dtype=torch.float64
    )
    matrix = (q * eigenvalues.unsqueeze(0)) @ q.mT
    return ((matrix + matrix.mT) * 0.5).contiguous()


def diagnostics(
    step: int,
    G_exact: torch.Tensor,
    M_exact: torch.Tensor,
    L_computed: torch.Tensor,
    Aw_computed: torch.Tensor,
) -> dict:
    L_exact = torch.linalg.cholesky(G_exact)
    Aw_exact = whiten(L_exact, M_exact)
    L64 = L_computed.double()
    Aw64 = Aw_computed.double()
    reconstructed_g = L64 @ L64.mT
    reconstructed_m = L64 @ Aw64 @ L64.mT
    return {
        "step": step,
        "measured_condition": float(torch.linalg.cond(G_exact).item()),
        "Aw_spectral_norm": float(torch.linalg.matrix_norm(Aw_exact, ord=2).item()),
        "L_relative_error": tensor_relative_error(L64, L_exact),
        "Aw_relative_error": tensor_relative_error(Aw64, Aw_exact),
        "factor_backward_error": tensor_relative_error(reconstructed_g, G_exact),
        "operator_backward_error": tensor_relative_error(reconstructed_m, M_exact),
    }


def run(args: argparse.Namespace) -> dict:
    configure_threads(args.threads)
    set_reproducible_seed(args.seed)
    add_student_pkg()
    import fast_cholesky

    dtype = torch.float32 if args.precision == "float32" else torch.float64
    G_exact = make_spd(args.rank, args.condition, args.seed)
    lambda_floor = float(torch.linalg.eigvalsh(G_exact)[0].item())
    M_exact = G_exact - lambda_floor * torch.eye(args.rank, dtype=torch.float64)

    status = "ok"
    failure = None
    checkpoints = sorted(
        set(step for step in (1, 10, 50, 100, 200, args.steps) if step <= args.steps)
    )
    records = []
    reset_count = 0
    started = time.perf_counter()

    try:
        L = torch.linalg.cholesky(G_exact.to(dtype)).contiguous()
        Aw = whiten(L, M_exact.to(dtype))
        records.append(diagnostics(0, G_exact, M_exact, L, Aw))
        generator = torch.Generator().manual_seed(args.seed + 991)

        for step in range(1, args.steps + 1):
            scale = args.update_scale * (torch.trace(G_exact) / args.rank).sqrt()
            z64 = (
                torch.randn(args.rank, dtype=torch.float64, generator=generator)
                * scale
                / args.rank**0.5
            )
            z = z64.to(dtype)
            Aw, _ = fast_cholesky.fast_incremental_update(
                L, Aw, z.clone(), z.clone()
            )
            G_exact = G_exact + torch.outer(z64, z64)
            M_exact = M_exact + torch.outer(z64, z64)

            if args.reset_interval > 0 and step % args.reset_interval == 0:
                L = torch.linalg.cholesky(G_exact.to(dtype)).contiguous()
                Aw = whiten(L, M_exact.to(dtype))
                reset_count += 1

            if step in checkpoints:
                records.append(diagnostics(step, G_exact, M_exact, L, Aw))
    except Exception as error:
        status = "failed"
        failure = f"{type(error).__name__}: {error}"

    payload = {
        "experiment": "positive_update_stability",
        "rank": args.rank,
        "target_condition": args.condition,
        "precision": args.precision,
        "reset_interval": args.reset_interval,
        "steps": args.steps,
        "seed": args.seed,
        "update_scale": args.update_scale,
        "lambda_floor": lambda_floor,
        "status": status,
        "failure": failure,
        "reset_count": reset_count,
        "wall_seconds": time.perf_counter() - started,
        "records": records,
        "environment": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    }
    stem = (
        f"stability_r{args.rank}_cond{args.condition:g}_{args.precision}"
        f"_reset{args.reset_interval}_seed{args.seed}"
    )
    output = result_path("stability", stem)
    atomic_json_dump(payload, output)
    print(f"status={status} wrote {output}", flush=True)
    if status != "ok" and args.fail_on_numerical_failure:
        raise RuntimeError(failure)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--condition", type=float, required=True)
    parser.add_argument("--precision", choices=("float32", "float64"), required=True)
    parser.add_argument("--reset-interval", type=int, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--update-scale", type=float, default=0.01)
    parser.add_argument("--fail-on-numerical-failure", action="store_true")
    parser.add_argument(
        "--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
