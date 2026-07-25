from __future__ import annotations

import argparse
import os
import platform
import socket
import statistics
import sys
import time

import torch

from .common import (
    ROOT,
    add_student_pkg,
    atomic_json_dump,
    configure_threads,
    result_path,
    set_reproducible_seed,
    tensor_relative_error,
)


def median_time(function, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def median_prepared_time(prepare, function, repeats: int) -> float:
    """Time only the operation; construct a fresh mutable state before timing."""
    samples = []
    for _ in range(repeats):
        arguments = prepare()
        started = time.perf_counter()
        function(*arguments)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def run(args: argparse.Namespace) -> dict:
    configure_threads(args.threads)
    set_reproducible_seed(args.seed)
    add_student_pkg()
    sys.path.insert(0, str(ROOT / "rebuttal_experiments"))
    import fast_cholesky
    import one_sided_cholesky

    r = args.rank
    generator = torch.Generator().manual_seed(args.seed)
    raw = torch.randn(r, r, generator=generator)
    G = raw @ raw.mT + r * torch.eye(r)
    L = torch.linalg.cholesky(G).contiguous()
    z = torch.randn(r, generator=generator) * 0.1
    zero = torch.zeros(r)
    rows = []

    for p in args.widths:
        B = torch.randn(r, p, generator=generator) / max(1, p) ** 0.5
        C = torch.randn(r, p, generator=generator) / max(1, p) ** 0.5
        M = B @ C.mT
        Vb = torch.linalg.solve_triangular(L, B, upper=False).contiguous()
        Vc = torch.linalg.solve_triangular(L, C, upper=False).contiguous()
        Aw = (Vb @ Vc.mT).contiguous()

        def two_sided(L_work, z_work):
            return fast_cholesky.fast_incremental_update(
                L_work, Aw, z_work, zero
            )

        def drag_along(L_work, left, right, z_work):
            return one_sided_cholesky.one_sided_update(
                L_work, left, right, z_work
            )

        def drag_and_materialize(L_work, left, right, z_work):
            left, right = drag_along(L_work, left, right, z_work)
            return left @ right.mT

        def fresh_solve():
            L_new = torch.linalg.cholesky(G + torch.outer(z, z)).contiguous()
            return (
                torch.linalg.solve_triangular(L_new, B, upper=False),
                torch.linalg.solve_triangular(L_new, C, upper=False),
            )

        def fresh_and_materialize():
            left, right = fresh_solve()
            return left @ right.mT

        for _ in range(3):
            two_sided(L.clone(), z.clone())
            drag_along(L.clone(), Vb.clone(), Vc.clone(), z.clone())
            drag_and_materialize(
                L.clone(), Vb.clone(), Vc.clone(), z.clone()
            )
            fresh_solve()
            fresh_and_materialize()

        two_aw, _ = two_sided(L.clone(), z.clone())
        drag_aw = drag_and_materialize(
            L.clone(), Vb.clone(), Vc.clone(), z.clone()
        )
        fresh_aw = fresh_and_materialize()
        error_drag = tensor_relative_error(drag_aw, two_aw)
        error_fresh = tensor_relative_error(fresh_aw, two_aw)
        if error_drag > 2e-4 or error_fresh > 2e-4:
            raise RuntimeError(
                f"representation mismatch r={r} p={p}: "
                f"drag={error_drag:.3e}, fresh={error_fresh:.3e}"
            )

        repeats = args.repeats if p <= r else max(5, args.repeats // 2)
        timings = {
            "two_sided_seconds": median_prepared_time(
                lambda: (L.clone(), z.clone()), two_sided, repeats
            ),
            "drag_seconds": median_prepared_time(
                lambda: (L.clone(), Vb.clone(), Vc.clone(), z.clone()),
                drag_along,
                repeats,
            ),
            "drag_materialize_seconds": median_prepared_time(
                lambda: (L.clone(), Vb.clone(), Vc.clone(), z.clone()),
                drag_and_materialize,
                repeats,
            ),
            "fresh_solve_seconds": median_time(fresh_solve, repeats),
            "fresh_materialize_seconds": median_time(
                fresh_and_materialize, repeats
            ),
        }
        row = {
            "rank": r,
            "factor_width": p,
            "repeats": repeats,
            "drag_relative_error": error_drag,
            "fresh_relative_error": error_fresh,
            "dense_state_bytes": int((r * (r + 1) // 2 + r * r) * 4),
            "factor_state_bytes": int((r * (r + 1) // 2 + 2 * r * p) * 4),
            **timings,
        }
        rows.append(row)
        print(
            f"r={r} p={p} drag={1e3*timings['drag_seconds']:.3f}ms "
            f"drag+mat={1e3*timings['drag_materialize_seconds']:.3f}ms "
            f"two={1e3*timings['two_sided_seconds']:.3f}ms",
            flush=True,
        )

    payload = {
        "experiment": "representation_crossover",
        "rank": r,
        "seed": args.seed,
        "threads": args.threads,
        "rows": rows,
        "analytic_memory_crossover_p": r / 2,
        "analytic_drag_time_crossover_p": 1.5 * r,
        "environment": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    output = result_path("representation", f"crossover_r{r}_seed{args.seed}")
    atomic_json_dump(payload, output)
    print(f"wrote {output}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--widths", type=int, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
