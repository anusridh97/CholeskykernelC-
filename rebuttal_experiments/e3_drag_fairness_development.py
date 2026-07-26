"""Run the frozen, development-only E3 drag fairness selection pilot."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import gc
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
from .e3_metric_only_pilot import (
    atomic_json_dump,
    descriptive,
    first_cpu_model,
    relative_path,
    scalar_stats,
    sha256,
)


EXPERIMENTS = ROOT / "rebuttal_experiments"
PROTOCOL_PATH = EXPERIMENTS / "e3_drag_fairness_development_protocol.json"
PATH_NAMES = (
    "metric_only_specialized",
    "original_drag_factor",
    "original_drag_materialize_alloc",
    "original_drag_materialize_out",
    "fair_drag_factor",
    "fair_drag_materialize_mkl_out",
    "fair_drag_materialize_candidate",
)
REQUIRED_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("pilot_id") != "e3-drag-fairness-dev-v1":
        raise ValueError("unexpected E3 drag development pilot ID")
    if protocol.get("status") != "frozen_before_target_cpu_development":
        raise ValueError("E3 drag development protocol is not frozen")
    if protocol.get("report_use") != "development_only_not_reportable":
        raise ValueError("development result must be marked non-reportable")
    if protocol.get("dtype") != "float64" or protocol.get("threads") != 1:
        raise ValueError("development pilot must be single-thread FP64")
    if protocol.get("seeds") != [75001, 75002, 75003]:
        raise ValueError("development seeds changed")
    reserved = protocol.get("reserved_untouched_confirmation_seeds")
    if reserved != list(range(76001, 76011)):
        raise ValueError("reserved confirmation seeds changed")
    if set(protocol["seeds"]) & set(reserved):
        raise ValueError("development and confirmation seeds overlap")
    if protocol.get("warmups_per_path") != 3:
        raise ValueError("development warmup count changed")
    if protocol.get("timed_repeats_per_path") != 21:
        raise ValueError("development repeat count changed")
    if list(protocol.get("paths", {})) != list(PATH_NAMES):
        raise ValueError("development path order changed")
    if not protocol.get("retain_all_rows_and_candidates"):
        raise ValueError("development protocol must retain every candidate")
    resource = protocol.get("resource_contract", {})
    if (
        resource.get("account") != "marlowe-m000151-pm06"
        or resource.get("partition") != "batch"
        or resource.get("physical_cpus_per_process") != 1
        or not resource.get("exclusive_node")
    ):
        raise ValueError(f"invalid development resource contract: {resource}")


def _configure_runtime() -> dict[str, Any]:
    required = {
        **{name: "1" for name in REQUIRED_THREAD_ENV},
        "PYTHONNOUSERSITE": "1",
    }
    invalid = {
        name: os.environ.get(name)
        for name, expected in required.items()
        if os.environ.get(name) != expected
    }
    if invalid:
        raise RuntimeError(f"invalid hermetic environment: {invalid}")
    if os.environ.get("SLURM_JOB_ACCOUNT") != "marlowe-m000151-pm06":
        raise RuntimeError("development pilot is on the wrong account")
    if os.environ.get("SLURM_JOB_PARTITION") != "batch":
        raise RuntimeError("development pilot is on the wrong partition")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 1:
        raise RuntimeError(f"development pilot requires one CPU: {affinity}")
    configure_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("PyTorch thread contract failed")
    declared_commit = os.environ.get("SOURCE_COMMIT")
    actual_commit = _git_output("rev-parse", "HEAD")
    if not declared_commit or declared_commit != actual_commit:
        raise RuntimeError(
            f"source-lock mismatch: declared={declared_commit}, actual={actual_commit}"
        )
    if subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode:
        raise RuntimeError("tracked worktree differs from source commit")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        raise RuntimeError("index differs from source commit")
    return {
        "thread_environment": {
            name: os.environ.get(name) for name in REQUIRED_THREAD_ENV
        },
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "process_affinity": affinity,
        "source_commit": actual_commit,
    }


def _load_extensions():
    path = str(EXPERIMENTS)
    if path not in sys.path:
        sys.path.insert(0, path)
    import e3_metric_only
    import one_sided_cholesky
    import one_sided_fair

    return e3_metric_only, one_sided_cholesky, one_sided_fair


def _whiten(L: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    left = torch.linalg.solve_triangular(L, M, upper=False)
    return torch.linalg.solve_triangular(
        L, left.mT, upper=False
    ).mT.contiguous()


def _make_cases(
    *,
    L: torch.Tensor,
    Aw: torch.Tensor,
    z: torch.Tensor,
    Vb: torch.Tensor,
    Vc: torch.Tensor,
    e3_metric_only,
    one_sided_cholesky,
    one_sided_fair,
) -> dict[str, tuple[Callable[[], tuple[Any, ...]], Callable[..., Any]]]:
    rank, width = Vb.shape

    def prepare_metric():
        return (
            L.clone(),
            Aw.clone(),
            z.clone(),
            torch.empty(5 * rank, dtype=L.dtype),
        )

    def execute_metric(L_work, Aw_work, z_work, workspace):
        return e3_metric_only.metric_only_update_(
            L_work, Aw_work, z_work, workspace
        )

    def prepare_original_factor():
        return L.clone(), Vb.clone(), Vc.clone(), z.clone()

    def execute_original_factor(L_work, left, right, z_work):
        left_out, right_out = one_sided_cholesky.one_sided_update(
            L_work, left, right, z_work
        )
        return L_work, left_out, right_out

    def prepare_original_out():
        return (*prepare_original_factor(), torch.empty_like(Aw))

    def execute_original_alloc(L_work, left, right, z_work):
        left_out, right_out = one_sided_cholesky.one_sided_update(
            L_work, left, right, z_work
        )
        return L_work, left_out @ right_out.mT

    def execute_original_out(L_work, left, right, z_work, output):
        left_out, right_out = one_sided_cholesky.one_sided_update(
            L_work, left, right, z_work
        )
        torch.mm(left_out, right_out.mT, out=output)
        return L_work, output

    def prepare_fair_factor():
        return (
            L.clone(),
            Vb.clone(),
            Vc.clone(),
            z.clone(),
            torch.empty(2 * rank + 2 * width, dtype=L.dtype),
        )

    def execute_fair_factor(L_work, left, right, z_work, workspace):
        return one_sided_fair.one_sided_update_(
            L_work, left, right, z_work, workspace
        )

    def prepare_fair_out():
        return (*prepare_fair_factor(), torch.empty_like(Aw))

    def execute_fair_mkl(L_work, left, right, z_work, workspace, output):
        one_sided_fair.one_sided_update_(
            L_work, left, right, z_work, workspace
        )
        torch.mm(left, right.mT, out=output)
        return L_work, output

    def execute_fair_candidate(
        L_work, left, right, z_work, workspace, output
    ):
        one_sided_fair.one_sided_update_(
            L_work, left, right, z_work, workspace
        )
        one_sided_fair.materialize_candidate_(left, right, output)
        return L_work, output

    return {
        "metric_only_specialized": (prepare_metric, execute_metric),
        "original_drag_factor": (
            prepare_original_factor,
            execute_original_factor,
        ),
        "original_drag_materialize_alloc": (
            prepare_original_factor,
            execute_original_alloc,
        ),
        "original_drag_materialize_out": (
            prepare_original_out,
            execute_original_out,
        ),
        "fair_drag_factor": (prepare_fair_factor, execute_fair_factor),
        "fair_drag_materialize_mkl_out": (
            prepare_fair_out,
            execute_fair_mkl,
        ),
        "fair_drag_materialize_candidate": (
            prepare_fair_out,
            execute_fair_candidate,
        ),
    }


def _validate_outputs(
    cases,
    *,
    L_fresh: torch.Tensor,
    Vb_fresh: torch.Tensor,
    Vc_fresh: torch.Tensor,
    Aw_fresh: torch.Tensor,
    tolerance: float,
) -> dict[str, dict[str, float]]:
    output = {}
    for name, (prepare, execute) in cases.items():
        result = execute(*prepare())
        if name in ("original_drag_factor", "fair_drag_factor"):
            L_actual, left_actual, right_actual = result
            metrics = {
                "L_relative_error": tensor_relative_error(L_actual, L_fresh),
                "left_relative_error": tensor_relative_error(
                    left_actual, Vb_fresh
                ),
                "right_relative_error": tensor_relative_error(
                    right_actual, Vc_fresh
                ),
                "materialized_relative_error": tensor_relative_error(
                    left_actual @ right_actual.mT, Aw_fresh
                ),
            }
        else:
            L_actual, Aw_actual = result
            metrics = {
                "L_relative_error": tensor_relative_error(L_actual, L_fresh),
                "Aw_relative_error": tensor_relative_error(Aw_actual, Aw_fresh),
            }
        metrics["maximum_relative_error"] = max(metrics.values())
        if metrics["maximum_relative_error"] > tolerance:
            raise RuntimeError(f"correctness failure for {name}: {metrics}")
        output[name] = metrics
    return output


def _time_cases(cases, *, seed: int, width_index: int, warmups: int, repeats: int):
    for name in PATH_NAMES:
        prepare, execute = cases[name]
        for _ in range(warmups):
            result = execute(*prepare())
            del result
    samples = {name: [] for name in PATH_NAMES}
    schedule = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for repetition in range(repeats):
            offset = (seed + width_index + repetition) % len(PATH_NAMES)
            order = PATH_NAMES[offset:] + PATH_NAMES[:offset]
            schedule.append({"repetition": repetition, "order": list(order)})
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
        if was_enabled:
            gc.enable()
    return {
        name: descriptive(values) for name, values in samples.items()
    }, schedule


def _run_cell(
    *,
    rank: int,
    width: int,
    width_index: int,
    seed: int,
    warmups: int,
    repeats: int,
    tolerance: float,
    e3_metric_only,
    one_sided_cholesky,
    one_sided_fair,
) -> dict[str, Any]:
    dtype = torch.float64
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(rank, rank, dtype=dtype, generator=generator)
    G = raw @ raw.mT + rank * torch.eye(rank, dtype=dtype)
    L = torch.linalg.cholesky(G).contiguous()
    z = 0.1 * torch.randn(rank, dtype=dtype, generator=generator)
    widths = (
        [1, 4, 8, 16, 32, 64, 128, 256, 512]
        if rank == 256
        else [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    )
    B = C = None
    for current_width in widths:
        B_candidate = torch.randn(
            rank, current_width, dtype=dtype, generator=generator
        ) / math.sqrt(current_width)
        C_candidate = torch.randn(
            rank, current_width, dtype=dtype, generator=generator
        ) / math.sqrt(current_width)
        if current_width == width:
            B, C = B_candidate, C_candidate
            break
    if B is None or C is None:
        raise ValueError(f"undeclared width: {width}")
    Vb = torch.linalg.solve_triangular(L, B, upper=False).contiguous()
    Vc = torch.linalg.solve_triangular(L, C, upper=False).contiguous()
    Aw = (Vb @ Vc.mT).contiguous()
    L_fresh = torch.linalg.cholesky(G + torch.outer(z, z)).contiguous()
    Vb_fresh = torch.linalg.solve_triangular(L_fresh, B, upper=False)
    Vc_fresh = torch.linalg.solve_triangular(L_fresh, C, upper=False)
    Aw_fresh = _whiten(L_fresh, B @ C.mT)
    cases = _make_cases(
        L=L,
        Aw=Aw,
        z=z,
        Vb=Vb,
        Vc=Vc,
        e3_metric_only=e3_metric_only,
        one_sided_cholesky=one_sided_cholesky,
        one_sided_fair=one_sided_fair,
    )
    accuracy = _validate_outputs(
        cases,
        L_fresh=L_fresh,
        Vb_fresh=Vb_fresh,
        Vc_fresh=Vc_fresh,
        Aw_fresh=Aw_fresh,
        tolerance=tolerance,
    )
    timings, schedule = _time_cases(
        cases,
        seed=seed,
        width_index=width_index,
        warmups=warmups,
        repeats=repeats,
    )
    return {
        "rank": rank,
        "factor_width": width,
        "seed": seed,
        "dtype": "float64",
        "threads": 1,
        "accuracy": accuracy,
        "timings_seconds": timings,
        "timing_schedule": schedule,
        "output_contracts": {
            name: (
                "thin_factor_only"
                if name in ("original_drag_factor", "fair_drag_factor")
                else "dense_same_output"
            )
            for name in PATH_NAMES
        },
    }


def _geometric_mean(values: list[float]) -> float:
    if not values or not all(value > 0 and math.isfinite(value) for value in values):
        raise ValueError(f"invalid geometric-mean values: {values}")
    return math.exp(statistics.mean(math.log(value) for value in values))


def _aggregate_and_select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["rank"], row["factor_width"])].append(row)
    aggregate_rows = []
    for (rank, width), items in sorted(grouped.items()):
        if len(items) != 3:
            raise ValueError(f"development cell {(rank, width)} lacks 3 seeds")
        aggregate_rows.append(
            {
                "rank": rank,
                "factor_width": width,
                "seed_count": len(items),
                "per_seed_median_seconds": {
                    name: scalar_stats(
                        [item["timings_seconds"][name]["median"] for item in items]
                    )
                    for name in PATH_NAMES
                },
                "maximum_relative_error": max(
                    metrics["maximum_relative_error"]
                    for item in items
                    for metrics in item["accuracy"].values()
                ),
            }
        )

    update_ratios = [
        row["timings_seconds"]["fair_drag_factor"]["median"]
        / row["timings_seconds"]["original_drag_factor"]["median"]
        for row in rows
    ]
    update_cell_ratios = {}
    for key, items in grouped.items():
        update_cell_ratios[key] = statistics.median(
            item["timings_seconds"]["fair_drag_factor"]["median"]
            / item["timings_seconds"]["original_drag_factor"]["median"]
            for item in items
        )
    update_score = _geometric_mean(update_ratios)
    update_maximum_cell_ratio = max(update_cell_ratios.values())
    select_fair_update = (
        update_score <= 0.98 and update_maximum_cell_ratio <= 1.02
    )

    all_mkl = {
        (row["rank"], row["factor_width"], row["seed"]):
        row["timings_seconds"]["fair_drag_materialize_mkl_out"]["median"]
        for row in rows
    }
    thresholds = sorted({row["factor_width"] for row in rows})
    materializer_families = [
        {
            "name": "all_mkl",
            "threshold": None,
            "geometric_mean_ratio_to_all_mkl": 1.0,
            "maximum_cell_median_ratio_to_all_mkl": 1.0,
            "eligible": True,
        }
    ]
    for threshold in thresholds:
        ratios = []
        cell_ratios: dict[tuple[int, int], list[float]] = defaultdict(list)
        for row in rows:
            identity = (row["rank"], row["factor_width"], row["seed"])
            chosen_name = (
                "fair_drag_materialize_candidate"
                if row["factor_width"] <= threshold
                else "fair_drag_materialize_mkl_out"
            )
            ratio = row["timings_seconds"][chosen_name]["median"] / all_mkl[identity]
            ratios.append(ratio)
            cell_ratios[(row["rank"], row["factor_width"])].append(ratio)
        score = _geometric_mean(ratios)
        maximum = max(statistics.median(values) for values in cell_ratios.values())
        materializer_families.append(
            {
                "name": f"custom_if_p_le_{threshold}_else_mkl",
                "threshold": threshold,
                "geometric_mean_ratio_to_all_mkl": score,
                "maximum_cell_median_ratio_to_all_mkl": maximum,
                "eligible": score <= 0.95 and maximum <= 1.02,
            }
        )
    eligible_custom = [
        family
        for family in materializer_families[1:]
        if family["eligible"]
    ]
    selected_materializer = (
        min(
            eligible_custom,
            key=lambda family: (
                family["geometric_mean_ratio_to_all_mkl"],
                family["threshold"],
            ),
        )
        if eligible_custom
        else materializer_families[0]
    )
    return {
        "rows": aggregate_rows,
        "selection": {
            "update": {
                "geometric_mean_fair_over_original": update_score,
                "maximum_cell_median_fair_over_original": (
                    update_maximum_cell_ratio
                ),
                "selected": (
                    "fair_drag_factor"
                    if select_fair_update
                    else "original_drag_factor"
                ),
            },
            "materializer_families": materializer_families,
            "selected_materializer": selected_materializer,
            "confirmation_pipeline": {
                "update": (
                    "one_sided_fair.one_sided_update_"
                    if select_fair_update
                    else "one_sided_cholesky.one_sided_update"
                ),
                "materialization": (
                    "torch.mm_out_all_widths"
                    if selected_materializer["threshold"] is None
                    else selected_materializer["name"]
                ),
            },
        },
    }


def _provenance(e3_metric_only, one_sided_cholesky, one_sided_fair):
    files = [
        PROTOCOL_PATH,
        Path(__file__).resolve(),
        EXPERIMENTS / "e3_metric_only_ext.cpp",
        EXPERIMENTS / "setup_e3_metric_only.py",
        EXPERIMENTS / "one_sided_ext.cpp",
        EXPERIMENTS / "setup_one_sided.py",
        EXPERIMENTS / "one_sided_fair_ext.cpp",
        EXPERIMENTS / "setup_one_sided_fair.py",
        Path(e3_metric_only.__file__).resolve(),
        Path(one_sided_cholesky.__file__).resolve(),
        Path(one_sided_fair.__file__).resolve(),
    ]
    return {
        "files": {
            relative_path(path): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        },
        "binaries": {
            "metric": str(Path(e3_metric_only.__file__).resolve()),
            "original_drag": str(Path(one_sided_cholesky.__file__).resolve()),
            "fair_drag": str(Path(one_sided_fair.__file__).resolve()),
        },
        "build_flags": {
            "metric": [
                "-O3",
                "-march=native",
                "-ffast-math",
                "-fno-math-errno",
                "-DNDEBUG",
            ],
            "original_drag": ["-O3", "-march=native", "-ffast-math"],
            "fair_drag": [
                "-O3",
                "-march=native",
                "-ffast-math",
                "-fno-math-errno",
                "-DNDEBUG",
            ],
        },
    }


def run() -> dict[str, Any]:
    protocol = _read(PROTOCOL_PATH)
    _validate_protocol(protocol)
    output = ROOT / protocol["result_path"]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite development result: {output}")
    runtime = _configure_runtime()
    e3_metric_only, one_sided_cholesky, one_sided_fair = _load_extensions()
    started = datetime.now(timezone.utc)
    rows = []
    rank_map = {
        int(rank): [int(width) for width in widths]
        for rank, widths in protocol["ranks_and_widths"].items()
    }
    for rank, widths in rank_map.items():
        for width_index, width in enumerate(widths):
            for seed in protocol["seeds"]:
                row = _run_cell(
                    rank=rank,
                    width=width,
                    width_index=width_index,
                    seed=seed,
                    warmups=protocol["warmups_per_path"],
                    repeats=protocol["timed_repeats_per_path"],
                    tolerance=protocol["relative_error_tolerance"],
                    e3_metric_only=e3_metric_only,
                    one_sided_cholesky=one_sided_cholesky,
                    one_sided_fair=one_sided_fair,
                )
                rows.append(row)
                print(
                    f"r={rank} p={width} seed={seed} "
                    f"old={1e3 * row['timings_seconds']['original_drag_materialize_out']['median']:.3f}ms "
                    f"fair_mkl={1e3 * row['timings_seconds']['fair_drag_materialize_mkl_out']['median']:.3f}ms "
                    f"fair_candidate={1e3 * row['timings_seconds']['fair_drag_materialize_candidate']['median']:.3f}ms",
                    flush=True,
                )
    ended = datetime.now(timezone.utc)
    aggregated = _aggregate_and_select(rows)
    payload = {
        "schema_version": 1,
        "experiment": "e3_drag_fairness_development",
        "pilot_id": protocol["pilot_id"],
        "status": "complete",
        "report_use": "development_only_not_reportable",
        "protocol": protocol,
        "protocol_path": relative_path(PROTOCOL_PATH),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "rows": rows,
        "aggregate": aggregated,
        "integrity": {
            "declared_cell_count": sum(len(widths) for widths in rank_map.values())
            * len(protocol["seeds"]),
            "observed_cell_count": len(rows),
            "all_rows_and_candidates_retained": True,
            "confirmation_seeds_evaluated": False,
            "development_outcomes_used_for_report_claims": False,
        },
        "environment": {
            **runtime,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "cpu_model": first_cpu_model(),
            "torch": torch.__version__,
            "torch_config": torch.__config__.show(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_account": os.environ.get("SLURM_JOB_ACCOUNT"),
            "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "started_utc": started.isoformat(),
            "ended_utc": ended.isoformat(),
            "elapsed_wall_seconds": (ended - started).total_seconds(),
        },
        "provenance": _provenance(
            e3_metric_only, one_sided_cholesky, one_sided_fair
        ),
    }
    if payload["integrity"]["declared_cell_count"] != len(rows):
        raise AssertionError("development pilot did not retain every cell")
    atomic_json_dump(payload, output)
    print(json.dumps(aggregated["selection"], indent=2, sort_keys=True))
    print(output, flush=True)
    return payload


if __name__ == "__main__":
    run()
