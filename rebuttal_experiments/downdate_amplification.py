from __future__ import annotations

import argparse
import os
import platform
import socket

import torch

from .common import (
    atomic_json_dump,
    configure_threads,
    result_path,
    set_reproducible_seed,
)


def run(args: argparse.Namespace) -> dict:
    configure_threads(args.threads)
    set_reproducible_seed(args.seed)
    r = args.rank
    raw = torch.randn(r, r, dtype=torch.float64)
    G = raw @ raw.mT + r * torch.eye(r, dtype=torch.float64)
    L = torch.linalg.cholesky(G)
    direction = torch.randn(r, dtype=torch.float64)
    direction /= torch.linalg.norm(direction)
    A = torch.randn(r, r, dtype=torch.float64)
    rows = []

    for leverage in args.leverages:
        z = leverage**0.5 * (L @ direction)
        measured = float((z @ torch.cholesky_solve(z[:, None], L).squeeze()).item())
        G_minus = G - torch.outer(z, z)
        L_minus = torch.linalg.cholesky(G_minus)
        P = torch.linalg.solve_triangular(L_minus, L, upper=False)
        _, singular_values, vh = torch.linalg.svd(P)
        top_right = vh[0]
        worst_perturbation = torch.outer(top_right, top_right)
        transported = P @ worst_perturbation @ P.mT
        amplification = float(
            (
                torch.linalg.matrix_norm(transported, ord=2)
                / torch.linalg.matrix_norm(worst_perturbation, ord=2)
            ).item()
        )
        bound = 1.0 / (1.0 - leverage)

        A32 = A.float().double()
        roundoff_perturbation = A32 - A
        random_amplification = float(
            (
                torch.linalg.matrix_norm(
                    P @ roundoff_perturbation @ P.mT, ord=2
                )
                / torch.linalg.matrix_norm(roundoff_perturbation, ord=2)
            ).item()
        )
        row = {
            "rank": r,
            "target_leverage": leverage,
            "measured_leverage": measured,
            "theoretical_bound": bound,
            "worst_direction_amplification": amplification,
            "float32_roundoff_amplification": random_amplification,
            "P_spectral_norm_squared": float(singular_values[0].item() ** 2),
            "guard_accept_tau_half": leverage <= 0.5,
            "damping_for_tau_half": min(1.0, 0.5 / leverage),
        }
        rows.append(row)
        print(
            f"r={r} leverage={leverage:.6f} amplification={amplification:.3f} "
            f"bound={bound:.3f}",
            flush=True,
        )

    payload = {
        "experiment": "downdate_amplification",
        "rank": r,
        "seed": args.seed,
        "rows": rows,
        "environment": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    output = result_path("stability", f"downdate_r{r}_seed{args.seed}")
    atomic_json_dump(payload, output)
    print(f"wrote {output}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument(
        "--leverages", type=float, nargs="+", default=[0.1, 0.5, 0.9, 0.99, 0.999]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
