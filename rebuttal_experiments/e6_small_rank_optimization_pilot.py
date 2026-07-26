"""Development-only E6 pilot for the r=256, k=32 crossover.

This compares the already-confirmed compact implementation with a direct
right-application recurrence that avoids both dense transpose copies.  Pilot
seeds are implementation-selection data and must never enter confirmation
statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import time
from typing import Any, Callable

import torch

from .common import (
    ROOT,
    add_student_pkg,
    configure_threads,
    set_reproducible_seed,
    tensor_relative_error,
)
from .rank_k import apply_compact_inverse_left, whiten


RANK = 256
UPDATE_COUNT = 32
THREADS = 1
BLOCK_SIZE = 32
PILOT_SEEDS = tuple(range(87001, 87006))
METHODS = ("confirmed_compact", "direct_right_compact", "fresh")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _balanced_samples(
    methods: dict[str, tuple[Callable[..., Any], bool]],
    *,
    L: torch.Tensor,
    Aw: torch.Tensor,
    repeats: int,
) -> tuple[dict[str, list[float]], list[list[str]]]:
    orders = list(itertools.permutations(METHODS))
    if repeats % len(orders):
        raise ValueError("repeats must be divisible by six")
    samples = {name: [] for name in METHODS}
    schedule: list[list[str]] = []
    for repeat in range(repeats):
        order = orders[repeat % len(orders)]
        schedule.append(list(order))
        for name in order:
            function, mutable = methods[name]
            arguments = (L.clone(), Aw.clone()) if mutable else ()
            started = time.perf_counter()
            function(*arguments)
            samples[name].append(time.perf_counter() - started)
    return samples, schedule


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed not in PILOT_SEEDS:
        raise ValueError(f"seed must be one of the development seeds {PILOT_SEEDS}")
    if args.repeats <= 0 or args.repeats % 6:
        raise ValueError("--repeats must be a positive multiple of six")
    configure_threads(THREADS)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("pilot requires a one-thread PyTorch runtime")
    set_reproducible_seed(args.seed)
    add_student_pkg()
    import fast_cholesky

    raw = torch.randn(RANK, RANK, dtype=torch.float64)
    G = raw @ raw.mT + RANK * torch.eye(RANK, dtype=torch.float64)
    L = torch.linalg.cholesky(G).contiguous()
    M = torch.randn(RANK, RANK, dtype=torch.float64)
    Aw = whiten(L, M)
    Z = torch.randn(RANK, UPDATE_COUNT, dtype=torch.float64) * 0.05
    W = torch.randn(RANK, UPDATE_COUNT, dtype=torch.float64) * 0.05

    U = torch.empty_like(Z)
    Y = torch.empty_like(W)
    d = torch.empty(RANK, dtype=torch.float64)
    Q = torch.empty_like(Z)
    factor_workspace = torch.empty(
        UPDATE_COUNT * UPDATE_COUNT + 2 * UPDATE_COUNT,
        dtype=torch.float64,
    )
    buffer = torch.empty_like(Aw)
    scratch = torch.empty_like(Aw)
    prefix = torch.empty(UPDATE_COUNT, RANK, dtype=torch.float64)
    coefficient = torch.empty(BLOCK_SIZE, BLOCK_SIZE, dtype=torch.float64)
    apply_workspace = torch.empty(UPDATE_COUNT * RANK, dtype=torch.float64)

    def prepare_factor(L_work: torch.Tensor, Aw_work: torch.Tensor) -> None:
        torch.linalg.solve_triangular(
            L_work, Z, upper=False, out=U
        )
        torch.linalg.solve_triangular(
            L_work, W, upper=False, out=Y
        )
        torch.mm(U, Y.mT, out=buffer)
        buffer.add_(Aw_work)
        fast_cholesky.fast_compact_factor_out(
            L_work, U, d, Q, factor_workspace
        )

    def confirmed_compact(
        L_work: torch.Tensor, Aw_work: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prepare_factor(L_work, Aw_work)
        apply_compact_inverse_left(
            buffer,
            scratch,
            U,
            Q,
            d,
            block_size=BLOCK_SIZE,
            prefix=prefix,
            coefficient=coefficient,
        )
        buffer.copy_(scratch.mT)
        apply_compact_inverse_left(
            buffer,
            scratch,
            U,
            Q,
            d,
            block_size=BLOCK_SIZE,
            prefix=prefix,
            coefficient=coefficient,
        )
        Aw_work.copy_(scratch.mT)
        return L_work, Aw_work

    def direct_right_compact(
        L_work: torch.Tensor, Aw_work: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prepare_factor(L_work, Aw_work)
        fast_cholesky.fast_compact_apply_left_out(
            buffer, U, Q, d, scratch, apply_workspace
        )
        fast_cholesky.fast_compact_apply_right_out(
            scratch, U, Q, d, Aw_work, apply_workspace
        )
        return L_work, Aw_work

    def fresh() -> tuple[torch.Tensor, torch.Tensor]:
        G_new = G + Z @ Z.mT
        M_new = M + Z @ W.mT
        L_new = torch.linalg.cholesky(G_new).contiguous()
        return L_new, whiten(L_new, M_new)

    L_fresh, Aw_fresh = fresh()
    correctness: dict[str, dict[str, float]] = {}
    for name, function in (
        ("confirmed_compact", confirmed_compact),
        ("direct_right_compact", direct_right_compact),
    ):
        L_actual, Aw_actual = function(L.clone(), Aw.clone())
        correctness[name] = {
            "L_relative_error": tensor_relative_error(L_actual, L_fresh),
            "Aw_relative_error": tensor_relative_error(Aw_actual, Aw_fresh),
        }
        if max(correctness[name].values()) > 1e-9:
            raise RuntimeError(f"{name} failed correctness: {correctness[name]}")

    methods: dict[str, tuple[Callable[..., Any], bool]] = {
        "confirmed_compact": (confirmed_compact, True),
        "direct_right_compact": (direct_right_compact, True),
        "fresh": (fresh, False),
    }
    for function, mutable in methods.values():
        function(*(L.clone(), Aw.clone()) if mutable else ())
    samples, schedule = _balanced_samples(
        methods,
        L=L,
        Aw=Aw,
        repeats=args.repeats,
    )
    medians = {
        name: statistics.median(values) for name, values in samples.items()
    }
    result = {
        "schema_version": 1,
        "experiment": "e6_small_rank_optimization_pilot",
        "role": "development_only_excluded_from_confirmation",
        "seed": args.seed,
        "rank": RANK,
        "k": UPDATE_COUNT,
        "dtype": "float64",
        "threads": THREADS,
        "repeats_per_method": args.repeats,
        "methods": list(METHODS),
        "timing_samples_seconds": samples,
        "timing_order_schedule": schedule,
        "median_seconds": medians,
        "fresh_over_confirmed_compact": (
            medians["fresh"] / medians["confirmed_compact"]
        ),
        "fresh_over_direct_right_compact": (
            medians["fresh"] / medians["direct_right_compact"]
        ),
        "confirmed_over_direct_right_speedup": (
            medians["confirmed_compact"]
            / medians["direct_right_compact"]
        ),
        "correctness": correctness,
        "environment": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "process_affinity": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_account": os.environ.get("SLURM_JOB_ACCOUNT"),
            "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "git_commit": _git("rev-parse", "HEAD"),
            "driver_sha256": _sha256(Path(__file__).resolve()),
            "kernel_sha256": _sha256(
                ROOT / "student_pkg" / "fast_update.cpp"
            ),
            "extension_sha256": _sha256(
                Path(fast_cholesky.__file__).resolve()
            ),
        },
    }
    _write_once(Path(args.output), result)
    print(
        f"seed={args.seed} confirmed={1e3*medians['confirmed_compact']:.3f}ms "
        f"direct={1e3*medians['direct_right_compact']:.3f}ms "
        f"fresh={1e3*medians['fresh']:.3f}ms "
        f"fresh/direct={result['fresh_over_direct_right_compact']:.3f}x",
        flush=True,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
