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


def rbf(left: torch.Tensor, right: torch.Tensor, lengthscale: float) -> torch.Tensor:
    distance = torch.cdist(left / lengthscale, right / lengthscale)
    return torch.exp(-0.5 * distance.square())


def make_near_duplicate_kernel(
    rank: int, delta: float, jitter: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    points = torch.rand(rank, 3, dtype=torch.float64, generator=generator)
    pairs = max(1, rank // 8)
    noise = torch.randn(pairs, 3, dtype=torch.float64, generator=generator)
    noise /= torch.linalg.norm(noise, dim=1, keepdim=True)
    points[1 : 2 * pairs : 2] = (
        points[0 : 2 * pairs : 2] + delta * noise
    ).clamp(0.0, 1.0)
    G = rbf(points, points, lengthscale=0.4)
    G.diagonal().add_(jitter)
    return ((G + G.mT) * 0.5).contiguous(), points


def one_configuration(args, delta: float, seed: int, fast_cholesky) -> dict:
    dtype = torch.float32 if args.precision == "float32" else torch.float64
    G_exact, points = make_near_duplicate_kernel(
        args.rank, delta, args.jitter, seed
    )
    generator = torch.Generator().manual_seed(seed + 7001)
    if args.operator == "cumulative":
        M_exact = G_exact - args.jitter * torch.eye(
            args.rank, dtype=torch.float64
        )
    else:
        raw = torch.randn(
            args.rank, args.rank, dtype=torch.float64, generator=generator
        )
        M_exact = raw / torch.linalg.matrix_norm(raw, ord=2)

    status = "ok"
    failure = None
    records = []
    started = time.perf_counter()
    checkpoints = sorted(
        {step for step in (0, 1, 10, 50, args.steps) if step <= args.steps}
    )
    try:
        L = torch.linalg.cholesky(G_exact.to(dtype)).contiguous()
        Aw = whiten(L, M_exact.to(dtype))
        for step in range(args.steps + 1):
            if step in checkpoints:
                L_exact = torch.linalg.cholesky(G_exact)
                Aw_exact = whiten(L_exact, M_exact)
                L64 = L.double()
                Aw64 = Aw.double()
                records.append(
                    {
                        "step": step,
                        "condition": float(torch.linalg.cond(G_exact).item()),
                        "Aw_spectral_norm": float(
                            torch.linalg.matrix_norm(Aw_exact, ord=2).item()
                        ),
                        "L_relative_error": tensor_relative_error(L64, L_exact),
                        "Aw_relative_error": tensor_relative_error(Aw64, Aw_exact),
                        "factor_backward_error": tensor_relative_error(
                            L64 @ L64.mT, G_exact
                        ),
                        "operator_backward_error": tensor_relative_error(
                            L64 @ Aw64 @ L64.mT, M_exact
                        ),
                    }
                )
            if step == args.steps:
                break
            x_new = torch.rand(1, 3, dtype=torch.float64, generator=generator)
            z64 = (
                args.update_scale
                * rbf(points, x_new, lengthscale=0.4).squeeze(1)
            )
            z = z64.to(dtype)
            Aw, _ = fast_cholesky.fast_incremental_update(
                L, Aw, z.clone(), z.clone()
            )
            G_exact.add_(torch.outer(z64, z64))
            M_exact.add_(torch.outer(z64, z64))
            if args.reset_interval > 0 and (step + 1) % args.reset_interval == 0:
                L = torch.linalg.cholesky(G_exact.to(dtype)).contiguous()
                Aw = whiten(L, M_exact.to(dtype))
    except Exception as error:
        status = "failed"
        failure = f"{type(error).__name__}: {error}"
    return {
        "delta": delta,
        "jitter": args.jitter,
        "seed": seed,
        "status": status,
        "failure": failure,
        "wall_seconds": time.perf_counter() - started,
        "records": records,
    }


def run(args: argparse.Namespace) -> dict:
    configure_threads(args.threads)
    set_reproducible_seed(0)
    add_student_pkg()
    import fast_cholesky

    rows = []
    for delta in args.deltas:
        for seed in args.seeds:
            row = one_configuration(args, delta, seed, fast_cholesky)
            rows.append(row)
            final = row["records"][-1] if row["records"] else {}
            print(
                f"precision={args.precision} operator={args.operator} "
                f"delta={delta:g} seed={seed} status={row['status']} "
                f"cond={final.get('condition')} "
                f"Aw_error={final.get('Aw_relative_error')}",
                flush=True,
            )

    payload = {
        "experiment": "near_duplicate_kernel_stability",
        "rank": args.rank,
        "precision": args.precision,
        "operator": args.operator,
        "jitter": args.jitter,
        "steps": args.steps,
        "reset_interval": args.reset_interval,
        "update_scale": args.update_scale,
        "rows": rows,
        "environment": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    output = result_path(
        "stability",
        f"kernel_r{args.rank}_{args.precision}_{args.operator}",
    )
    atomic_json_dump(payload, output)
    print(f"wrote {output}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--precision", choices=("float32", "float64"), required=True)
    parser.add_argument("--operator", choices=("cumulative", "random"), required=True)
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--reset-interval", type=int, default=50)
    parser.add_argument("--update-scale", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
