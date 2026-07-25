from __future__ import annotations

import argparse
import os
import platform
import socket
import statistics
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


def median_time(function, repeats: int) -> float:
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        values.append(time.perf_counter() - start)
    return statistics.median(values)


def median_prepared_time(prepare, function, repeats: int) -> float:
    values = []
    for _ in range(repeats):
        arguments = prepare()
        start = time.perf_counter()
        function(*arguments)
        values.append(time.perf_counter() - start)
    return statistics.median(values)


def run(args: argparse.Namespace) -> dict:
    configure_threads(args.threads)
    set_reproducible_seed(args.seed)
    add_student_pkg()
    import fast_cholesky

    r = args.rank
    raw = torch.randn(r, r, dtype=torch.float64)
    G = raw @ raw.mT + r * torch.eye(r, dtype=torch.float64)
    L = torch.linalg.cholesky(G).contiguous()
    M = torch.randn(r, r, dtype=torch.float64)
    Aw = whiten(L, M)
    rows = []

    for k in args.ks:
        Z = torch.randn(r, k, dtype=torch.float64) * 0.05
        W = torch.randn(r, k, dtype=torch.float64) * 0.05

        def sequential(L_work, Aw_work):
            for column in range(k):
                Aw_work, _ = fast_cholesky.fast_incremental_update(
                    L_work,
                    Aw_work,
                    Z[:, column].clone(),
                    W[:, column],
                )
            return L_work, Aw_work

        def refactor():
            G_new = G + Z @ Z.mT
            M_new = M + Z @ W.mT
            L_new = torch.linalg.cholesky(G_new).contiguous()
            return L_new, whiten(L_new, M_new)

        for _ in range(2):
            sequential(L.clone(), Aw.clone())
            refactor()
        L_seq, Aw_seq = sequential(L.clone(), Aw.clone())
        L_exact, Aw_exact = refactor()
        l_error = tensor_relative_error(L_seq, L_exact)
        aw_error = tensor_relative_error(Aw_seq, Aw_exact)
        if l_error > 1e-9 or aw_error > 1e-9:
            raise RuntimeError(
                f"rank-k mismatch r={r} k={k}: L={l_error:.3e}, Aw={aw_error:.3e}"
            )
        repeats = args.repeats if k <= 8 else max(3, args.repeats // 2)
        sequential_seconds = median_prepared_time(
            lambda: (L.clone(), Aw.clone()), sequential, repeats
        )
        refactor_seconds = median_time(refactor, repeats)
        row = {
            "rank": r,
            "k": k,
            "repeats": repeats,
            "sequential_seconds": sequential_seconds,
            "refactor_seconds": refactor_seconds,
            "speedup": refactor_seconds / sequential_seconds,
            "L_relative_error": l_error,
            "Aw_relative_error": aw_error,
        }
        rows.append(row)
        print(
            f"r={r} k={k} sequential={1e3*sequential_seconds:.3f}ms "
            f"refact={1e3*refactor_seconds:.3f}ms "
            f"speedup={row['speedup']:.3f}x",
            flush=True,
        )

    payload = {
        "experiment": "rank_k",
        "rank": r,
        "seed": args.seed,
        "threads": args.threads,
        "predicted_flop_break_even_k": (7.0 / 54.0) * r,
        "rows": rows,
        "environment": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    output = result_path("rank_k", f"rank_k_r{r}_seed{args.seed}")
    atomic_json_dump(payload, output)
    print(f"wrote {output}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
