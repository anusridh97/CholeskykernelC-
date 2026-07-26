"""Frozen E3 pilot for a dense metric-only (w=0) specialization.

This driver is deliberately separate from the canonical E3 and E6 drivers.
It reads a pre-frozen protocol, keeps thin-output and dense-output contracts
distinct, retains every timing sample, and refuses to overwrite an existing
pilot result.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

import torch

from .common import ROOT, configure_threads, tensor_relative_error


EXPERIMENTS = ROOT / "rebuttal_experiments"
DEFAULT_PROTOCOL = EXPERIMENTS / "e3_metric_only_pilot_protocol.json"
REQUIRED_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
DENSE_PATHS = (
    "metric_only_specialized",
    "generic_w0_ablation",
    "drag_materialize",
    "fresh_materialize",
)
THIN_PATHS = ("drag_factor_only", "fresh_factor_only")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen pilot output: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def descriptive(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty timing sample")
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError(f"invalid timing sample: {values}")
    ordered = sorted(float(value) for value in values)
    return {
        "median": statistics.median(ordered),
        "mean": statistics.mean(ordered),
        "sample_sd": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "n": len(ordered),
        "values": [float(value) for value in values],
    }


def scalar_stats(values: list[float | int]) -> dict[str, Any]:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("cannot summarize an empty scalar sample")
    return {
        "mean": statistics.mean(materialized),
        "median": statistics.median(materialized),
        "sample_sd": (
            statistics.stdev(materialized) if len(materialized) > 1 else 0.0
        ),
        "minimum": min(materialized),
        "maximum": max(materialized),
        "n": len(materialized),
        "values": materialized,
    }


def whiten(L: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    left = torch.linalg.solve_triangular(L, M, upper=False)
    return torch.linalg.solve_triangular(
        L, left.mT, upper=False
    ).mT.contiguous()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def first_cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return None
    for line in cpuinfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def find_unique_binary(pattern: str) -> Path:
    matches = sorted(EXPERIMENTS.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one binary matching {pattern}, found {matches}"
        )
    return matches[0]


def validate_protocol(protocol: dict[str, Any], path: Path) -> None:
    if protocol.get("status") != "frozen_before_first_metric_only_timing":
        raise ValueError("pilot protocol is not marked frozen")
    if protocol.get("dtype") != "float64":
        raise ValueError("pilot must use FP64")
    if int(protocol.get("threads", 0)) != 1:
        raise ValueError("pilot must use one thread")
    if list(protocol.get("seeds", [])) != [73001, 73002, 73003]:
        raise ValueError("unexpected pilot seeds")
    if set(protocol.get("paths", {})) != set(DENSE_PATHS + THIN_PATHS):
        raise ValueError("pilot path set differs from the frozen contract")
    for name in DENSE_PATHS:
        if protocol["paths"][name]["output_contract"] != "dense_same_output":
            raise ValueError(f"{name} is not declared dense same-output")
    for name in THIN_PATHS:
        if protocol["paths"][name]["output_contract"] != "thin_factor_only":
            raise ValueError(f"{name} is not declared thin factor-only")
    if not protocol.get("retain_all_rows"):
        raise ValueError("protocol must retain unfavorable rows")
    if protocol.get("report_edit_authorized"):
        raise ValueError("pilot protocol must not authorize report edits")
    if protocol.get("canonical_result_edit_authorized"):
        raise ValueError("pilot protocol must not authorize canonical edits")
    if not path.is_file():
        raise FileNotFoundError(path)


def configure_execution(protocol: dict[str, Any]) -> dict[str, Any]:
    invalid_environment = {
        name: os.environ.get(name)
        for name in REQUIRED_THREAD_ENV
        if os.environ.get(name) != "1"
    }
    if invalid_environment:
        raise RuntimeError(
            "launch with one-thread BLAS/OpenMP variables; invalid values: "
            f"{invalid_environment}"
        )
    affinity = {int(cpu) for cpu in protocol["cpu_affinity"]}
    available = os.sched_getaffinity(0)
    if not affinity <= available:
        raise RuntimeError(
            f"requested affinity {sorted(affinity)} not in {sorted(available)}"
        )
    os.sched_setaffinity(0, affinity)
    configure_threads(int(protocol["threads"]))
    if torch.get_num_threads() != 1:
        raise RuntimeError(f"torch intra-op threads={torch.get_num_threads()}")
    if torch.get_num_interop_threads() != 1:
        raise RuntimeError(f"torch inter-op threads={torch.get_num_interop_threads()}")
    return {
        "thread_environment": {
            name: os.environ.get(name) for name in REQUIRED_THREAD_ENV
        },
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }


def load_extensions():
    path = str(EXPERIMENTS)
    if path not in sys.path:
        sys.path.insert(0, path)
    import e3_metric_only
    import one_sided_cholesky

    return e3_metric_only, one_sided_cholesky


def make_cases(
    *,
    L: torch.Tensor,
    Aw: torch.Tensor,
    z: torch.Tensor,
    zero: torch.Tensor,
    G: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    Vb: torch.Tensor,
    Vc: torch.Tensor,
    e3_metric_only,
    one_sided_cholesky,
) -> dict[str, tuple[Callable[[], tuple[Any, ...]], Callable[..., Any]]]:
    rank = L.shape[0]

    def prepare_specialized() -> tuple[torch.Tensor, ...]:
        return (
            L.clone(),
            Aw.clone(),
            z.clone(),
            torch.empty(5 * rank, dtype=L.dtype),
        )

    def execute_specialized(
        L_work: torch.Tensor,
        Aw_work: torch.Tensor,
        z_work: torch.Tensor,
        workspace: torch.Tensor,
    ):
        return e3_metric_only.metric_only_update_(
            L_work, Aw_work, z_work, workspace
        )

    def prepare_ablation() -> tuple[torch.Tensor, ...]:
        return (
            L.clone(),
            Aw.clone(),
            z.clone(),
            zero,
            torch.empty(5 * rank, dtype=L.dtype),
        )

    def execute_ablation(
        L_work: torch.Tensor,
        Aw_work: torch.Tensor,
        z_work: torch.Tensor,
        w_work: torch.Tensor,
        workspace: torch.Tensor,
    ):
        return e3_metric_only.generic_update_(
            L_work, Aw_work, z_work, w_work, workspace
        )

    def prepare_drag() -> tuple[torch.Tensor, ...]:
        return L.clone(), Vb.clone(), Vc.clone(), z.clone()

    def execute_drag_factor(
        L_work: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        z_work: torch.Tensor,
    ):
        left_out, right_out = one_sided_cholesky.one_sided_update(
            L_work, left, right, z_work
        )
        return L_work, left_out, right_out

    def execute_drag_materialize(
        L_work: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        z_work: torch.Tensor,
    ):
        left_out, right_out = one_sided_cholesky.one_sided_update(
            L_work, left, right, z_work
        )
        return L_work, left_out @ right_out.mT

    def prepare_fresh() -> tuple[()]:
        return ()

    def execute_fresh_factor():
        L_new = torch.linalg.cholesky(G + torch.outer(z, z)).contiguous()
        left = torch.linalg.solve_triangular(L_new, B, upper=False)
        right = torch.linalg.solve_triangular(L_new, C, upper=False)
        return L_new, left, right

    def execute_fresh_materialize():
        L_new, left, right = execute_fresh_factor()
        return L_new, left @ right.mT

    return {
        "metric_only_specialized": (
            prepare_specialized,
            execute_specialized,
        ),
        "generic_w0_ablation": (prepare_ablation, execute_ablation),
        "drag_materialize": (prepare_drag, execute_drag_materialize),
        "fresh_materialize": (prepare_fresh, execute_fresh_materialize),
        "drag_factor_only": (prepare_drag, execute_drag_factor),
        "fresh_factor_only": (prepare_fresh, execute_fresh_factor),
    }


def validate_outputs(
    cases: dict[str, tuple[Callable[[], tuple[Any, ...]], Callable[..., Any]]],
    *,
    L_fresh: torch.Tensor,
    Aw_fresh: torch.Tensor,
    left_fresh: torch.Tensor,
    right_fresh: torch.Tensor,
    tolerance: float,
) -> dict[str, dict[str, Any]]:
    output = {}
    for name, (prepare, execute) in cases.items():
        result = execute(*prepare())
        if name in DENSE_PATHS:
            L_actual, Aw_actual = result
            metrics = {
                "L_relative_error": tensor_relative_error(L_actual, L_fresh),
                "Aw_relative_error": tensor_relative_error(Aw_actual, Aw_fresh),
                "maximum_relative_error": max(
                    tensor_relative_error(L_actual, L_fresh),
                    tensor_relative_error(Aw_actual, Aw_fresh),
                ),
            }
        else:
            L_actual, left_actual, right_actual = result
            materialized = left_actual @ right_actual.mT
            metrics = {
                "L_relative_error": tensor_relative_error(L_actual, L_fresh),
                "left_relative_error": tensor_relative_error(
                    left_actual, left_fresh
                ),
                "right_relative_error": tensor_relative_error(
                    right_actual, right_fresh
                ),
                "materialized_Aw_relative_error": tensor_relative_error(
                    materialized, Aw_fresh
                ),
            }
            metrics["maximum_relative_error"] = max(metrics.values())
        if metrics["maximum_relative_error"] > tolerance:
            raise RuntimeError(
                f"correctness failure for {name}: {metrics}, tolerance={tolerance}"
            )
        output[name] = metrics
    return output


def time_cases(
    cases: dict[str, tuple[Callable[[], tuple[Any, ...]], Callable[..., Any]]],
    *,
    names: list[str],
    seed: int,
    width_index: int,
    warmups: int,
    repeats: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    for name in names:
        prepare, execute = cases[name]
        for _ in range(warmups):
            result = execute(*prepare())
            del result

    samples: dict[str, list[float]] = {name: [] for name in names}
    schedule = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for repetition in range(repeats):
            offset = (seed + width_index + repetition) % len(names)
            order = names[offset:] + names[:offset]
            schedule.append({"repetition": repetition, "order": order})
            for name in order:
                prepare, execute = cases[name]
                arguments = prepare()
                started = time.perf_counter_ns()
                result = execute(*arguments)
                elapsed = (time.perf_counter_ns() - started) * 1e-9
                samples[name].append(elapsed)
                del result
                del arguments
    finally:
        if gc_was_enabled:
            gc.enable()
    return {name: descriptive(values) for name, values in samples.items()}, schedule


def run_cell(
    *,
    rank: int,
    width: int,
    width_index: int,
    seed: int,
    warmups: int,
    repeats: int,
    tolerance: float,
    names: list[str],
    e3_metric_only,
    one_sided_cholesky,
) -> dict[str, Any]:
    dtype = torch.float64
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(rank, rank, dtype=dtype, generator=generator)
    G = raw @ raw.mT + rank * torch.eye(rank, dtype=dtype)
    L = torch.linalg.cholesky(G).contiguous()
    z = 0.1 * torch.randn(rank, dtype=dtype, generator=generator)

    # Recreate the frozen source experiment's sequential width generation so
    # every width is a deterministic prefix of one seed-specific stream.
    B = C = None
    protocol_widths = [
        1, 4, 8, 16, 32, 64, 128, 256, 512
    ] if rank == 256 else [
        1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
    ]
    for current_width in protocol_widths:
        B_candidate = (
            torch.randn(
                rank, current_width, dtype=dtype, generator=generator
            )
            / max(1, current_width) ** 0.5
        )
        C_candidate = (
            torch.randn(
                rank, current_width, dtype=dtype, generator=generator
            )
            / max(1, current_width) ** 0.5
        )
        if current_width == width:
            B, C = B_candidate, C_candidate
            break
    if B is None or C is None:
        raise ValueError(f"width {width} is not frozen for rank {rank}")

    M = B @ C.mT
    Vb = torch.linalg.solve_triangular(L, B, upper=False).contiguous()
    Vc = torch.linalg.solve_triangular(L, C, upper=False).contiguous()
    Aw = (Vb @ Vc.mT).contiguous()
    zero = torch.zeros(rank, dtype=dtype)

    L_fresh = torch.linalg.cholesky(G + torch.outer(z, z)).contiguous()
    left_fresh = torch.linalg.solve_triangular(L_fresh, B, upper=False)
    right_fresh = torch.linalg.solve_triangular(L_fresh, C, upper=False)
    Aw_fresh = whiten(L_fresh, M)
    cases = make_cases(
        L=L,
        Aw=Aw,
        z=z,
        zero=zero,
        G=G,
        B=B,
        C=C,
        Vb=Vb,
        Vc=Vc,
        e3_metric_only=e3_metric_only,
        one_sided_cholesky=one_sided_cholesky,
    )
    accuracy = validate_outputs(
        cases,
        L_fresh=L_fresh,
        Aw_fresh=Aw_fresh,
        left_fresh=left_fresh,
        right_fresh=right_fresh,
        tolerance=tolerance,
    )
    timings, schedule = time_cases(
        cases,
        names=names,
        seed=seed,
        width_index=width_index,
        warmups=warmups,
        repeats=repeats,
    )
    speedups = {}
    denominator = timings["metric_only_specialized"]["values"]
    for competitor in (
        "generic_w0_ablation",
        "drag_materialize",
        "fresh_materialize",
    ):
        numerator = timings[competitor]["values"]
        ratios = [left / right for left, right in zip(numerator, denominator)]
        speedups[f"metric_only_vs_{competitor}"] = scalar_stats(ratios)

    return {
        "rank": rank,
        "factor_width": width,
        "seed": seed,
        "dtype": "float64",
        "threads": 1,
        "warmups_per_path": warmups,
        "timed_repeats_per_path": repeats,
        "accuracy": accuracy,
        "timings_seconds": timings,
        "dense_same_output_speedups": speedups,
        "timing_schedule": schedule,
        "output_contracts": {
            name: (
                "dense_same_output" if name in DENSE_PATHS else "thin_factor_only"
            )
            for name in names
        },
    }


def aggregate_rows(rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["rank"], row["factor_width"])].append(row)
    aggregate = []
    for (rank, width), items in sorted(groups.items()):
        if len(items) != 3:
            raise AssertionError(
                f"expected three seeds for r={rank}, p={width}, found {len(items)}"
            )
        path_stats = {
            name: scalar_stats(
                [
                    item["timings_seconds"][name]["median"]
                    for item in items
                ]
            )
            for name in names
        }
        speedup_stats = {}
        for competitor in (
            "generic_w0_ablation",
            "drag_materialize",
            "fresh_materialize",
        ):
            ratios = [
                item["timings_seconds"][competitor]["median"]
                / item["timings_seconds"]["metric_only_specialized"]["median"]
                for item in items
            ]
            speedup_stats[f"metric_only_vs_{competitor}"] = scalar_stats(ratios)
        aggregate.append(
            {
                "rank": rank,
                "factor_width": width,
                "seed_count": len(items),
                "per_seed_median_seconds": path_stats,
                "dense_same_output_speedups": speedup_stats,
                "maximum_relative_error": max(
                    metric["maximum_relative_error"]
                    for item in items
                    for metric in item["accuracy"].values()
                ),
            }
        )

    crossovers = []
    for rank in sorted({row["rank"] for row in rows}):
        widths = sorted(
            row["factor_width"] for row in aggregate if row["rank"] == rank
        )
        by_width = {
            row["factor_width"]: row for row in aggregate if row["rank"] == rank
        }
        dense_crossovers = {}
        for competitor in ("drag_materialize", "fresh_materialize"):
            dense_crossovers[competitor] = next(
                (
                    width
                    for width in widths
                    if by_width[width]["per_seed_median_seconds"][
                        "metric_only_specialized"
                    ]["median"]
                    <= by_width[width]["per_seed_median_seconds"][competitor][
                        "median"
                    ]
                ),
                None,
            )
        per_seed = []
        for seed in sorted({row["seed"] for row in rows if row["rank"] == rank}):
            seed_rows = {
                row["factor_width"]: row
                for row in rows
                if row["rank"] == rank and row["seed"] == seed
            }
            per_seed.append(
                {
                    "seed": seed,
                    "metric_only_vs_drag_materialize": next(
                        (
                            width
                            for width in widths
                            if seed_rows[width]["timings_seconds"][
                                "metric_only_specialized"
                            ]["median"]
                            <= seed_rows[width]["timings_seconds"][
                                "drag_materialize"
                            ]["median"]
                        ),
                        None,
                    ),
                    "metric_only_vs_fresh_materialize": next(
                        (
                            width
                            for width in widths
                            if seed_rows[width]["timings_seconds"][
                                "metric_only_specialized"
                            ]["median"]
                            <= seed_rows[width]["timings_seconds"][
                                "fresh_materialize"
                            ]["median"]
                        ),
                        None,
                    ),
                }
            )
        crossovers.append(
            {
                "rank": rank,
                "aggregate_crossover_widths": dense_crossovers,
                "per_seed_crossover_widths": per_seed,
                "thin_output_boundary": (
                    "No crossover is asserted between dense metric-only output "
                    "and drag/fresh factor-only output because the contracts differ."
                ),
            }
        )
    return {"rows": aggregate, "crossovers": crossovers}


def provenance(protocol_path: Path) -> dict[str, Any]:
    metric_binary = find_unique_binary("e3_metric_only*.so")
    one_sided_binary = find_unique_binary("one_sided_cholesky*.so")
    files = [
        protocol_path,
        EXPERIMENTS / "e3_metric_only_ext.cpp",
        EXPERIMENTS / "setup_e3_metric_only.py",
        EXPERIMENTS / "e3_metric_only_pilot.py",
        EXPERIMENTS / "representation_crossover.py",
        EXPERIMENTS / "one_sided_ext.cpp",
        metric_binary,
        one_sided_binary,
    ]
    protocol = read_json(protocol_path)
    for source in protocol["reference_source_jsons"]:
        files.append(ROOT / source)
    return {
        "git_commit": git_commit(),
        "files": {
            relative_path(path): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        },
        "metric_only_binary": relative_path(metric_binary),
        "one_sided_binary": relative_path(one_sided_binary),
        "build_flags": [
            "-O3",
            "-march=native",
            "-ffast-math",
            "-fno-math-errno",
            "-DNDEBUG",
        ],
    }


def run(protocol_path: Path, output_override: Path | None = None) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    validate_protocol(protocol, protocol_path)
    declared_output = ROOT / protocol["result_path"]
    output = output_override or declared_output
    if output.resolve() != declared_output.resolve():
        raise ValueError(
            "output override differs from the frozen protocol result path"
        )
    if output.exists():
        raise FileExistsError(f"pilot result already exists: {output}")

    execution = configure_execution(protocol)
    e3_metric_only, one_sided_cholesky = load_extensions()
    names = list(protocol["paths"])
    if names != list(DENSE_PATHS + THIN_PATHS):
        raise ValueError(
            "JSON path order changed; timing rotation requires the frozen order"
        )

    started = datetime.now(timezone.utc)
    load_before = os.getloadavg()
    rows = []
    rank_map = {
        int(rank): [int(width) for width in widths]
        for rank, widths in protocol["ranks_and_widths"].items()
    }
    for rank, widths in rank_map.items():
        for width_index, width in enumerate(widths):
            for seed in protocol["seeds"]:
                row = run_cell(
                    rank=rank,
                    width=width,
                    width_index=width_index,
                    seed=int(seed),
                    warmups=int(protocol["warmups_per_path"]),
                    repeats=int(protocol["timed_repeats_per_path"]),
                    tolerance=float(protocol["relative_error_tolerance"]),
                    names=names,
                    e3_metric_only=e3_metric_only,
                    one_sided_cholesky=one_sided_cholesky,
                )
                rows.append(row)
                metric_ms = (
                    1e3
                    * row["timings_seconds"]["metric_only_specialized"]["median"]
                )
                drag_ms = 1e3 * row["timings_seconds"]["drag_materialize"]["median"]
                print(
                    f"r={rank} p={width} seed={seed} "
                    f"metric={metric_ms:.3f}ms drag+mat={drag_ms:.3f}ms",
                    flush=True,
                )
    ended = datetime.now(timezone.utc)
    payload = {
        "experiment": "e3_metric_only_pilot",
        "pilot_id": protocol["pilot_id"],
        "status": "complete",
        "protocol": protocol,
        "protocol_path": relative_path(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "rows": rows,
        "aggregate": aggregate_rows(rows, names),
        "integrity": {
            "declared_cell_count": sum(len(widths) for widths in rank_map.values())
            * len(protocol["seeds"]),
            "observed_cell_count": len(rows),
            "all_rows_retained": True,
            "timing_paths_per_cell": len(names),
            "samples_per_path_per_cell": int(
                protocol["timed_repeats_per_path"]
            ),
            "report_modified": False,
            "canonical_results_modified": False,
            "pilot_tuned_from_outcomes": False,
        },
        "contract_boundary": {
            "dense_same_output_paths": list(DENSE_PATHS),
            "thin_factor_only_paths": list(THIN_PATHS),
            "claim_scope": (
                "Only paths within dense_same_output are performance-comparable. "
                "Thin factor-only paths are reported as the honest boundary and "
                "are not treated as losing to a method that materializes Aw."
            ),
        },
        "environment": {
            **execution,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "cpu_model": first_cpu_model(),
            "torch": torch.__version__,
            "torch_config": torch.__config__.show(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "started_utc": started.isoformat(),
            "ended_utc": ended.isoformat(),
            "elapsed_wall_seconds": (ended - started).total_seconds(),
            "load_average_before": list(load_before),
            "load_average_after": list(os.getloadavg()),
        },
        "provenance": provenance(protocol_path),
        "source_result_context": {
            "canonical_e3_driver": (
                "rebuttal_experiments/representation_crossover.py"
            ),
            "requested_but_absent_path": (
                "rebuttal_experiments/representation.py"
            ),
            "source_json_precision": (
                "FP32, inferred from the canonical driver default dtype and "
                "four-byte storage accounting; this FP64 pilot does not splice "
                "its timings into those source rows."
            ),
        },
    }
    if payload["integrity"]["declared_cell_count"] != len(rows):
        raise AssertionError("pilot did not retain every declared cell")
    atomic_json_dump(payload, output)
    print(f"wrote {output}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.protocol, args.output)


if __name__ == "__main__":
    main()
