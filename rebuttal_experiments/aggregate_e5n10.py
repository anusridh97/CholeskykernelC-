"""Strict audit and aggregation for the frozen ten-seed E5 campaign.

The campaign runners append ``<campaign-tag>-task<seed>`` to every JSON
filename and payload.  This module deliberately discovers only files carrying
that exact tag, rejects unexpected rows, and requires one result for every
seed in 1..10 before it writes either output.

Example
-------

.. code-block:: bash

   python -m rebuttal_experiments.aggregate_e5n10 \
     --results-root rebuttal_results \
     --campaign-tag e5n10-2ee46ba \
     --output-json /tmp/e5n10-summary.json \
     --output-rows /tmp/e5n10-rows.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CAMPAIGN_TAG = "e5n10-2ee46ba"
EXPECTED_SEEDS = tuple(range(1, 11))
EXPECTED_GPU = "NVIDIA H100 80GB HBM3"
EXPECTED_GPU_TOTAL_MEMORY = 85_045_477_376
EXPECTED_REFRESH_INTERVALS = (0, 10, 50, 100, 200)
LARGE_RANKS = (2048, 4096, 8192, 16384, 32768, 49152, 65536)
INVERSE_LAYOUT_CONTRACT = (
    "row-major contiguous lower triangle with zero upper triangle"
)
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class AuditError(RuntimeError):
    """Raised when the tagged campaign is incomplete or internally invalid."""


def _main_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for rank in (64, 128):
        for batch in (1, 4, 16, 64):
            specs.append(
                {
                    "id": f"fused-r{rank}-b{batch}",
                    "impl": "fused",
                    "rank": rank,
                    "batch": batch,
                    "num_warps": 4,
                    "sweep_num_warps": None,
                    "tile_rows": None,
                    "tile_cols": None,
                    "solve_width": None,
                    "solve_right": None,
                    "cached_inverse": False,
                    "materializes_L": True,
                    "inverse_layout_contract": None,
                    "inverse_initial_stride": None,
                    "inverse_initial_contiguous": None,
                    "inverse_gemv_backend": None,
                    "repeats": 60,
                }
            )
    for rank, num_warps, tile_rows in (
        (256, 8, 2),
        (512, 4, 1),
        (1024, 4, 4),
    ):
        specs.append(
            {
                "id": f"parallel-r{rank}-b1",
                "impl": "parallel",
                "rank": rank,
                "batch": 1,
                "num_warps": num_warps,
                "sweep_num_warps": 8,
                "tile_rows": tile_rows,
                "tile_cols": 8,
                "solve_width": 2,
                "solve_right": False,
                "cached_inverse": False,
                "materializes_L": True,
                "inverse_layout_contract": None,
                "inverse_initial_stride": None,
                "inverse_initial_contiguous": None,
                "inverse_gemv_backend": None,
                "repeats": 60,
            }
        )
    return specs


def _inverse_specs() -> list[dict[str, Any]]:
    specs = []
    for rank, num_warps, tile_rows in (
        (256, 8, 2),
        (512, 1, 1),
        (1024, 2, 4),
    ):
        specs.append(
            {
                "id": f"inverse-factor-r{rank}-b1",
                "impl": "inverse_factor",
                "rank": rank,
                "batch": 1,
                "num_warps": num_warps,
                "sweep_num_warps": 8,
                "tile_rows": tile_rows,
                "tile_cols": 8,
                "solve_width": None,
                "solve_right": None,
                "cached_inverse": True,
                "materializes_L": False,
                "inverse_layout_contract": INVERSE_LAYOUT_CONTRACT,
                "inverse_initial_stride": [rank * rank, rank, 1],
                "inverse_initial_contiguous": True,
                "inverse_gemv_backend": "matmul",
                "repeats": 60,
            }
        )
    return specs


def _stability_specs() -> list[dict[str, Any]]:
    return [
        {
            "id": "single-block-r256",
            "implementation": "single_block",
            "rank": 256,
            "steps": 500,
            "target_condition": None,
            "tile_rows": 2,
            "tile_cols": 8,
            "setup_num_warps": 8,
            "sweep_num_warps": 8,
        },
        {
            "id": "single-block-r512",
            "implementation": "single_block",
            "rank": 512,
            "steps": 500,
            "target_condition": None,
            "tile_rows": 1,
            "tile_cols": 8,
            "setup_num_warps": 4,
            "sweep_num_warps": 8,
        },
        {
            "id": "single-block-r1024",
            "implementation": "single_block",
            "rank": 1024,
            "steps": 500,
            "target_condition": None,
            "tile_rows": 4,
            "tile_cols": 8,
            "setup_num_warps": 4,
            "sweep_num_warps": 8,
        },
        {
            "id": "inplace-r1024",
            "implementation": "inplace",
            "rank": 1024,
            "steps": 500,
            "target_condition": None,
            "tile_rows": 8,
            "tile_cols": 8,
            "setup_num_warps": 4,
            "sweep_num_warps": 4,
        },
    ]


def _condition_specs() -> list[dict[str, Any]]:
    return [
        {
            "id": f"single-block-r256-kappa{condition:g}",
            "implementation": "single_block",
            "rank": 256,
            "steps": 200,
            "target_condition": condition,
            "tile_rows": 2,
            "tile_cols": 8,
            "setup_num_warps": 8,
            "sweep_num_warps": 8,
        }
        for condition in (1e2, 1e4, 1e6)
    ]


def _refresh_specs() -> list[dict[str, Any]]:
    specs = []
    for rank, tile_rows, setup_num_warps in (
        (256, 2, 8),
        (512, 1, 4),
        (1024, 4, 4),
    ):
        specs.append(
            {
                "id": f"refresh-r{rank}",
                "rank": rank,
                "steps": 200,
                "refresh_intervals": list(EXPECTED_REFRESH_INTERVALS),
                "tile_rows": tile_rows,
                "tile_cols": 8,
                "setup_num_warps": setup_num_warps,
                "sweep_num_warps": 8,
            }
        )
    return specs


def _large_specs() -> list[dict[str, Any]]:
    specs = []
    for rank in LARGE_RANKS:
        repeats = 10 if rank <= 8192 else 5 if rank <= 32768 else 3
        tile = 8 if rank <= 8192 else 16 if rank <= 16384 else 32
        specs.append(
            {
                "id": f"large-r{rank}",
                "rank": rank,
                "repeats": repeats,
                "warmup": 1,
                "inverse": {
                    "mode": "inverse_inplace_recurrent",
                    "tile_rows": tile,
                    "tile_cols": tile,
                    "setup_num_warps": 4,
                    "sweep_num_warps": 1,
                    "correctness_checked": rank <= 32768,
                },
                "refact": {"mode": "fresh_refact_recurrent"},
            }
        )
    return specs


MAIN_SPECS = _main_specs()
INVERSE_SPECS = _inverse_specs()
STABILITY_SPECS = _stability_specs()
CONDITION_SPECS = _condition_specs()
REFRESH_SPECS = _refresh_specs()
LARGE_SPECS = _large_specs()

MAIN_BY_KEY = {
    (item["impl"], item["rank"], item["batch"]): item for item in MAIN_SPECS
}
INVERSE_BY_KEY = {
    (item["impl"], item["rank"], item["batch"]): item
    for item in INVERSE_SPECS
}
STABILITY_BY_KEY = {
    (item["implementation"], item["rank"], item["steps"]): item
    for item in STABILITY_SPECS
}
CONDITION_BY_KEY = {
    (
        item["implementation"],
        item["rank"],
        item["steps"],
        item["target_condition"],
    ): item
    for item in CONDITION_SPECS
}
REFRESH_BY_RANK = {item["rank"]: item for item in REFRESH_SPECS}
LARGE_BY_RANK = {item["rank"]: item for item in LARGE_SPECS}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"{path}: cannot read JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuditError(f"{path}: top-level JSON value must be an object")
    return payload


def _discover(results_root: Path, campaign_tag: str) -> list[Path]:
    if not TAG_RE.fullmatch(campaign_tag):
        raise AuditError(
            "campaign tag must contain only letters, digits, '.', '_' and '-'"
        )
    paths: set[Path] = set()
    for directory_name in ("cuda", "cuda_tuning"):
        directory = results_root / directory_name
        if not directory.is_dir():
            continue
        paths.update(
            path
            for path in directory.rglob(f"*{campaign_tag}*.json")
            if path.is_file()
        )
    return sorted(paths)


def _require_equal(
    payload: dict[str, Any],
    expected: dict[str, Any],
    context: str,
) -> None:
    mismatches = []
    for key, expected_value in expected.items():
        if key not in payload:
            mismatches.append(f"{key}=<missing> (expected {expected_value!r})")
        elif payload[key] != expected_value:
            mismatches.append(
                f"{key}={payload[key]!r} (expected {expected_value!r})"
            )
    if mismatches:
        raise AuditError(f"{context}: frozen configuration mismatch: " + "; ".join(mismatches))


def _number(
    value: Any,
    context: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditError(f"{context}: expected a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise AuditError(f"{context}: expected a finite number, got {value!r}")
    if positive and result <= 0:
        raise AuditError(f"{context}: expected a positive number, got {value!r}")
    if nonnegative and result < 0:
        raise AuditError(f"{context}: expected a nonnegative number, got {value!r}")
    return result


def _numbers(
    values: Any,
    context: str,
    *,
    expected_length: int,
    positive: bool = False,
) -> list[float]:
    if not isinstance(values, list):
        raise AuditError(f"{context}: expected a list")
    if len(values) != expected_length:
        raise AuditError(
            f"{context}: found {len(values)} samples, expected {expected_length}"
        )
    return [
        _number(value, f"{context}[{index}]", positive=positive)
        for index, value in enumerate(values)
    ]


def _assert_close(
    actual: Any,
    expected: float,
    context: str,
    *,
    rel_tol: float = 1e-7,
    abs_tol: float = 1e-12,
) -> None:
    actual_number = _number(actual, context)
    if not math.isclose(
        actual_number,
        expected,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    ):
        raise AuditError(
            f"{context}: {actual_number!r} is inconsistent with {expected!r}"
        )


def _seed_and_suffix(
    path: Path,
    payload: dict[str, Any],
    campaign_tag: str,
) -> int:
    seed = payload.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise AuditError(f"{path}: seed must be an integer, got {seed!r}")
    if seed not in EXPECTED_SEEDS:
        raise AuditError(
            f"{path}: seed {seed!r} is outside the required set 1..10"
        )
    suffix = f"{campaign_tag}-task{seed}"
    if payload.get("output_suffix") != suffix:
        raise AuditError(
            f"{path}: output_suffix={payload.get('output_suffix')!r}, "
            f"expected {suffix!r}"
        )
    if not path.stem.endswith(f"_{suffix}"):
        raise AuditError(
            f"{path}: filename must end with the exact tagged suffix _{suffix}"
        )
    if f"_seed{seed}_" not in path.stem:
        raise AuditError(f"{path}: filename does not encode seed {seed}")
    return seed


def _task_id_matches(value: Any, seed: int) -> bool:
    return str(value) == str(seed)


def _validate_environment(
    env: Any,
    context: str,
    seed: int,
    *,
    require_array_job_id: bool,
    require_precision: bool,
) -> dict[str, Any]:
    if not isinstance(env, dict):
        raise AuditError(f"{context}: environment metadata must be an object")
    required = (
        "host",
        "platform",
        "gpu",
        "gpu_total_memory",
        "torch",
        "cuda",
        "slurm_job_id",
        "slurm_array_task_id",
    )
    missing = [key for key in required if key not in env]
    if require_array_job_id and "slurm_array_job_id" not in env:
        missing.append("slurm_array_job_id")
    if missing:
        raise AuditError(
            f"{context}: environment metadata misses {', '.join(missing)}"
        )
    for key in ("host", "platform", "torch", "cuda", "slurm_job_id"):
        if not isinstance(env[key], str) or not env[key]:
            raise AuditError(f"{context}: environment.{key} must be nonempty")
    if require_array_job_id and (
        not isinstance(env["slurm_array_job_id"], str)
        or not env["slurm_array_job_id"]
    ):
        raise AuditError(
            f"{context}: environment.slurm_array_job_id must be nonempty"
        )
    if not _task_id_matches(env["slurm_array_task_id"], seed):
        raise AuditError(
            f"{context}: slurm_array_task_id={env['slurm_array_task_id']!r}, "
            f"expected {seed!r}"
        )
    if env["gpu"] != EXPECTED_GPU:
        raise AuditError(
            f"{context}: gpu={env['gpu']!r}, expected {EXPECTED_GPU!r}"
        )
    if env["gpu_total_memory"] != EXPECTED_GPU_TOTAL_MEMORY:
        raise AuditError(
            f"{context}: gpu_total_memory={env['gpu_total_memory']!r}, "
            f"expected {EXPECTED_GPU_TOTAL_MEMORY}"
        )
    if require_precision:
        if env.get("float32_matmul_precision") != "highest":
            raise AuditError(
                f"{context}: float32_matmul_precision must be 'highest'"
            )
        if env.get("cuda_matmul_allow_tf32") is not False:
            raise AuditError(
                f"{context}: cuda_matmul_allow_tf32 must be false"
            )
    return dict(env)


def _validate_timing_samples(
    payload: dict[str, Any],
    context: str,
    *,
    incremental_prefix: str,
    fresh_prefix: str,
    repeats: int,
) -> None:
    incremental_samples = _numbers(
        payload.get(f"{incremental_prefix}_event_seconds_samples"),
        f"{context}.{incremental_prefix}_event_seconds_samples",
        expected_length=repeats,
        positive=True,
    )
    fresh_samples = _numbers(
        payload.get(f"{fresh_prefix}_event_seconds_samples"),
        f"{context}.{fresh_prefix}_event_seconds_samples",
        expected_length=repeats,
        positive=True,
    )
    incremental_wall_samples = _numbers(
        payload.get(f"{incremental_prefix}_wall_seconds_samples"),
        f"{context}.{incremental_prefix}_wall_seconds_samples",
        expected_length=repeats,
        positive=True,
    )
    fresh_wall_samples = _numbers(
        payload.get(f"{fresh_prefix}_wall_seconds_samples"),
        f"{context}.{fresh_prefix}_wall_seconds_samples",
        expected_length=repeats,
        positive=True,
    )
    for prefix, samples, wall_samples in (
        (incremental_prefix, incremental_samples, incremental_wall_samples),
        (fresh_prefix, fresh_samples, fresh_wall_samples),
    ):
        _assert_close(
            payload.get(f"{prefix}_event_seconds"),
            statistics.median(samples),
            f"{context}.{prefix}_event_seconds",
        )
        _assert_close(
            payload.get(f"{prefix}_wall_seconds"),
            statistics.median(wall_samples),
            f"{context}.{prefix}_wall_seconds",
        )


def _validate_scaling(
    path: Path,
    payload: dict[str, Any],
    spec: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    context = str(path)
    _require_equal(
        payload,
        {"experiment": "cuda_scaling", **{k: v for k, v in spec.items() if k != "id"}},
        context,
    )
    _validate_timing_samples(
        payload,
        context,
        incremental_prefix="triton",
        fresh_prefix="refact",
        repeats=spec["repeats"],
    )
    triton_event = _number(
        payload.get("triton_event_seconds"),
        f"{context}.triton_event_seconds",
        positive=True,
    )
    refact_event = _number(
        payload.get("refact_event_seconds"),
        f"{context}.refact_event_seconds",
        positive=True,
    )
    triton_wall = _number(
        payload.get("triton_wall_seconds"),
        f"{context}.triton_wall_seconds",
        positive=True,
    )
    refact_wall = _number(
        payload.get("refact_wall_seconds"),
        f"{context}.refact_wall_seconds",
        positive=True,
    )
    _assert_close(
        payload.get("event_speedup"),
        refact_event / triton_event,
        f"{context}.event_speedup",
    )
    _assert_close(
        payload.get("wall_speedup"),
        refact_wall / triton_wall,
        f"{context}.wall_speedup",
    )
    _assert_close(
        payload.get("triton_updates_per_second"),
        spec["batch"] / triton_event,
        f"{context}.triton_updates_per_second",
    )
    _assert_close(
        payload.get("refact_updates_per_second"),
        spec["batch"] / refact_event,
        f"{context}.refact_updates_per_second",
    )
    for field in ("Aw_relative_error", "minimum_effective_bandwidth_gb_s"):
        _number(payload.get(field), f"{context}.{field}", nonnegative=True)
    for field in (
        "triton_peak_extra_allocated_bytes",
        "triton_workspace_bytes",
        "refact_peak_extra_allocated_bytes",
    ):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AuditError(
                f"{context}.{field}: expected a nonnegative integer byte count, "
                f"got {value!r}"
            )
    correctness_field = (
        "Linv_relative_error"
        if spec["impl"] == "inverse_factor"
        else "L_relative_error"
    )
    _number(
        payload.get(correctness_field),
        f"{context}.{correctness_field}",
        nonnegative=True,
    )
    absent_field = (
        "L_relative_error"
        if correctness_field == "Linv_relative_error"
        else "Linv_relative_error"
    )
    if payload.get(absent_field) is not None:
        raise AuditError(f"{context}.{absent_field}: expected null")
    return _validate_environment(
        payload.get("environment"),
        context,
        seed,
        require_array_job_id=False,
        require_precision=True,
    )


def _validate_stability(
    path: Path,
    payload: dict[str, Any],
    spec: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    context = str(path)
    _require_equal(
        payload,
        {
            "experiment": "cuda_inverse_stability",
            **{k: v for k, v in spec.items() if k != "id"},
        },
        context,
    )
    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise AuditError(f"{context}.checkpoints: expected a list")
    expected_steps = sorted(
        {
            step
            for step in (1, 10, 50, 100, 200, spec["steps"])
            if step <= spec["steps"]
        }
    )
    actual_steps = [
        checkpoint.get("step") if isinstance(checkpoint, dict) else None
        for checkpoint in checkpoints
    ]
    if actual_steps != expected_steps:
        raise AuditError(
            f"{context}.checkpoints: steps={actual_steps!r}, "
            f"expected {expected_steps!r}"
        )
    metric_fields = (
        "Linv_relative_error",
        "Aw_relative_error",
        "inverse_residual",
    )
    for index, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, dict):
            raise AuditError(f"{context}.checkpoints[{index}]: expected an object")
        for field in metric_fields:
            _number(
                checkpoint.get(field),
                f"{context}.checkpoints[{index}].{field}",
                nonnegative=True,
            )
    for field in ("Linv_relative_error", "Aw_relative_error"):
        _assert_close(
            payload.get(field),
            float(checkpoints[-1][field]),
            f"{context}.{field}",
        )
    for field, checkpoint_field in (
        ("max_Linv_relative_error", "Linv_relative_error"),
        ("max_Aw_relative_error", "Aw_relative_error"),
        ("max_inverse_residual", "inverse_residual"),
    ):
        expected_max = max(float(row[checkpoint_field]) for row in checkpoints)
        _assert_close(payload.get(field), expected_max, f"{context}.{field}")
    _number(
        payload.get("initial_measured_condition"),
        f"{context}.initial_measured_condition",
        positive=True,
    )
    job_fields = {}
    for field in (
        "slurm_job_id",
        "slurm_array_job_id",
        "slurm_array_task_id",
    ):
        value = payload.get(field)
        if value is None or value == "":
            raise AuditError(f"{context}.{field}: must be nonempty")
        job_fields[field] = value
    if not _task_id_matches(job_fields["slurm_array_task_id"], seed):
        raise AuditError(
            f"{context}.slurm_array_task_id="
            f"{job_fields['slurm_array_task_id']!r}, expected {seed}"
        )
    return job_fields


def _validate_refresh(
    path: Path,
    payload: dict[str, Any],
    spec: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    context = str(path)
    _require_equal(
        payload,
        {
            "experiment": "cuda_inverse_refresh",
            **{k: v for k, v in spec.items() if k != "id"},
        },
        context,
    )
    fresh_event_total = _number(
        payload.get("fresh_event_seconds_total"),
        f"{context}.fresh_event_seconds_total",
        positive=True,
    )
    fresh_wall_total = _number(
        payload.get("fresh_wall_seconds_total"),
        f"{context}.fresh_wall_seconds_total",
        positive=True,
    )
    fresh_event_per_update = _number(
        payload.get("fresh_event_seconds_per_update"),
        f"{context}.fresh_event_seconds_per_update",
        positive=True,
    )
    fresh_wall_per_update = _number(
        payload.get("fresh_wall_seconds_per_update"),
        f"{context}.fresh_wall_seconds_per_update",
        positive=True,
    )
    _assert_close(
        fresh_event_per_update,
        fresh_event_total / spec["steps"],
        f"{context}.fresh_event_seconds_per_update",
    )
    _assert_close(
        fresh_wall_per_update,
        fresh_wall_total / spec["steps"],
        f"{context}.fresh_wall_seconds_per_update",
    )
    _number(
        payload.get("fresh_peak_allocated_bytes_including_harness"),
        f"{context}.fresh_peak_allocated_bytes_including_harness",
        nonnegative=True,
    )
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise AuditError(f"{context}.rows: expected a list")
    intervals = [
        row.get("refresh_interval") if isinstance(row, dict) else None
        for row in rows
    ]
    if intervals != list(EXPECTED_REFRESH_INTERVALS):
        raise AuditError(
            f"{context}.rows: refresh intervals={intervals!r}, "
            f"expected {list(EXPECTED_REFRESH_INTERVALS)!r}"
        )
    by_interval = {}
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise AuditError(f"{context}.rows[{row_index}]: expected an object")
        interval = int(row["refresh_interval"])
        expected_count = 0 if interval == 0 else spec["steps"] // interval
        if row.get("refresh_count") != expected_count:
            raise AuditError(
                f"{context}.rows[{row_index}].refresh_count="
                f"{row.get('refresh_count')!r}, expected {expected_count}"
            )
        expected_tail = spec["steps"] if interval == 0 else 0
        if row.get("final_tail_age") != expected_tail:
            raise AuditError(
                f"{context}.rows[{row_index}].final_tail_age="
                f"{row.get('final_tail_age')!r}, expected {expected_tail}"
            )
        if row.get("algorithm_persistent_dense_matrices") != 5:
            raise AuditError(
                f"{context}.rows[{row_index}]: expected five persistent "
                "dense matrices"
            )
        if row.get("benchmark_additional_identity_matrix") is not True:
            raise AuditError(
                f"{context}.rows[{row_index}]: identity-matrix harness "
                "metadata must be true"
            )
        event_total = _number(
            row.get("event_seconds_total"),
            f"{context}.rows[{row_index}].event_seconds_total",
            positive=True,
        )
        wall_total = _number(
            row.get("wall_seconds_total"),
            f"{context}.rows[{row_index}].wall_seconds_total",
            positive=True,
        )
        event_per_update = _number(
            row.get("event_seconds_per_update"),
            f"{context}.rows[{row_index}].event_seconds_per_update",
            positive=True,
        )
        wall_per_update = _number(
            row.get("wall_seconds_per_update"),
            f"{context}.rows[{row_index}].wall_seconds_per_update",
            positive=True,
        )
        _assert_close(
            event_per_update,
            event_total / spec["steps"],
            f"{context}.rows[{row_index}].event_seconds_per_update",
        )
        _assert_close(
            wall_per_update,
            wall_total / spec["steps"],
            f"{context}.rows[{row_index}].wall_seconds_per_update",
        )
        _assert_close(
            row.get("event_speedup_vs_fresh_every_step"),
            fresh_event_total / event_total,
            f"{context}.rows[{row_index}].event_speedup_vs_fresh_every_step",
        )
        _assert_close(
            row.get("wall_speedup_vs_fresh_every_step"),
            fresh_wall_total / wall_total,
            f"{context}.rows[{row_index}].wall_speedup_vs_fresh_every_step",
        )
        for field in (
            "Linv_relative_error",
            "Aw_relative_error",
            "peak_allocated_bytes_including_harness",
        ):
            _number(
                row.get(field),
                f"{context}.rows[{row_index}].{field}",
                nonnegative=True,
            )
        pre_refresh = row.get("pre_refresh_errors")
        if not isinstance(pre_refresh, list) or not pre_refresh:
            raise AuditError(
                f"{context}.rows[{row_index}].pre_refresh_errors: "
                "expected a nonempty list"
            )
        expected_pre_steps = (
            [spec["steps"]]
            if interval == 0
            else list(range(interval, spec["steps"] + 1, interval))
        )
        actual_pre_steps = [
            item.get("step") if isinstance(item, dict) else None
            for item in pre_refresh
        ]
        if actual_pre_steps != expected_pre_steps:
            raise AuditError(
                f"{context}.rows[{row_index}].pre_refresh_errors: "
                f"steps={actual_pre_steps!r}, expected {expected_pre_steps!r}"
            )
        pre_linv = []
        pre_aw = []
        for pre_index, item in enumerate(pre_refresh):
            if not isinstance(item, dict):
                raise AuditError(
                    f"{context}.rows[{row_index}].pre_refresh_errors"
                    f"[{pre_index}]: expected an object"
                )
            expected_age = spec["steps"] if interval == 0 else interval
            if item.get("age_since_refresh") != expected_age:
                raise AuditError(
                    f"{context}.rows[{row_index}].pre_refresh_errors"
                    f"[{pre_index}].age_since_refresh="
                    f"{item.get('age_since_refresh')!r}, expected {expected_age}"
                )
            pre_linv.append(
                _number(
                    item.get("Linv_relative_error"),
                    f"{context}.rows[{row_index}].pre_refresh_errors"
                    f"[{pre_index}].Linv_relative_error",
                    nonnegative=True,
                )
            )
            pre_aw.append(
                _number(
                    item.get("Aw_relative_error"),
                    f"{context}.rows[{row_index}].pre_refresh_errors"
                    f"[{pre_index}].Aw_relative_error",
                    nonnegative=True,
                )
            )
        _assert_close(
            row.get("max_pre_refresh_Linv_relative_error"),
            max(pre_linv),
            f"{context}.rows[{row_index}].max_pre_refresh_Linv_relative_error",
        )
        _assert_close(
            row.get("max_pre_refresh_Aw_relative_error"),
            max(pre_aw),
            f"{context}.rows[{row_index}].max_pre_refresh_Aw_relative_error",
        )
        by_interval[interval] = row
    environment = _validate_environment(
        payload.get("environment"),
        context,
        seed,
        require_array_job_id=True,
        require_precision=False,
    )
    return environment, by_interval


def _validate_large(
    path: Path,
    payload: dict[str, Any],
    spec: dict[str, Any],
    mode_key: str,
    seed: int,
) -> dict[str, Any]:
    context = str(path)
    mode_spec = spec[mode_key]
    expected = {
        "experiment": "cuda_large_rank_compare",
        "rank": spec["rank"],
        "repeats": spec["repeats"],
        "warmup": spec["warmup"],
        "mode": mode_spec["mode"],
        "dense_state_matrices": 4,
        "includes_G_M_outer_updates": True,
        "finite_full": True,
        "matrix_bytes": spec["rank"] * spec["rank"] * 4,
    }
    if mode_key == "inverse":
        expected.update(
            {
                "tile_rows": mode_spec["tile_rows"],
                "tile_cols": mode_spec["tile_cols"],
                "setup_num_warps": mode_spec["setup_num_warps"],
                "sweep_num_warps": mode_spec["sweep_num_warps"],
                "correctness_checked": mode_spec["correctness_checked"],
                "maintains_exact_G_M_for_refresh": True,
            }
        )
    else:
        expected.update(
            {
                "cholesky_info": 0,
                "cholesky_output_layout": "column-major",
                "destructive_inputs": False,
                "maintains_exact_G_M_for_next_update": True,
            }
        )
    _require_equal(payload, expected, context)
    event_samples = _numbers(
        payload.get("event_seconds_samples"),
        f"{context}.event_seconds_samples",
        expected_length=spec["repeats"],
        positive=True,
    )
    wall_samples = _numbers(
        payload.get("wall_seconds_samples"),
        f"{context}.wall_seconds_samples",
        expected_length=spec["repeats"],
        positive=True,
    )
    event_median = statistics.median(event_samples)
    wall_median = statistics.median(wall_samples)
    _assert_close(
        payload.get("event_seconds_median"),
        event_median,
        f"{context}.event_seconds_median",
    )
    _assert_close(
        payload.get("wall_seconds_median"),
        wall_median,
        f"{context}.wall_seconds_median",
    )
    _assert_close(
        payload.get("updates_per_second"),
        1.0 / event_median,
        f"{context}.updates_per_second",
    )
    for field in (
        "allocated_before_timing_bytes",
        "reserved_before_timing_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    ):
        _number(payload.get(field), f"{context}.{field}", nonnegative=True)
    if mode_key == "inverse":
        _number(
            payload.get("workspace_bytes"),
            f"{context}.workspace_bytes",
            nonnegative=True,
        )
        if mode_spec["correctness_checked"]:
            for field in ("Linv_relative_error", "Aw_relative_error"):
                _number(
                    payload.get(field),
                    f"{context}.{field}",
                    nonnegative=True,
                )
        else:
            for field in ("Linv_relative_error", "Aw_relative_error"):
                if payload.get(field) is not None:
                    raise AuditError(f"{context}.{field}: expected null")
    environment = {
        key: payload.get(key)
        for key in (
            "host",
            "platform",
            "gpu",
            "gpu_total_memory",
            "torch",
            "cuda",
            "slurm_job_id",
            "slurm_array_job_id",
            "slurm_array_task_id",
        )
    }
    return _validate_environment(
        environment,
        context,
        seed,
        require_array_job_id=True,
        require_precision=False,
    )


def _insert(
    destination: dict[int, dict[str, Any]],
    seed: int,
    record: dict[str, Any],
    context: str,
) -> None:
    if seed in destination:
        first = destination[seed]["relative_path"]
        raise AuditError(
            f"{context}: duplicate seed {seed}; first result is {first}"
        )
    destination[seed] = record


def _missing_seed_issues(
    name: str,
    rows: dict[str, dict[int, dict[str, Any]]],
) -> list[str]:
    issues = []
    expected = set(EXPECTED_SEEDS)
    for row_id, by_seed in rows.items():
        actual = set(by_seed)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            issues.append(
                f"{name}/{row_id}: seeds={sorted(actual)}, "
                f"missing={missing}, unexpected={extra}"
            )
    return issues


def _stats(values: Iterable[float]) -> dict[str, Any]:
    materialized = [float(value) for value in values]
    if len(materialized) != len(EXPECTED_SEEDS):
        raise AuditError(
            f"internal aggregation error: expected ten values, "
            f"found {len(materialized)}"
        )
    return {
        "mean": statistics.mean(materialized),
        "sample_sd": statistics.stdev(materialized),
        "n": len(materialized),
    }


def _relative_path(path: Path, results_root: Path) -> str:
    try:
        return str(path.relative_to(results_root))
    except ValueError:
        return str(path)


def _aggregate_scaling_row(
    spec: dict[str, Any],
    records: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    per_seed = []
    for seed in EXPECTED_SEEDS:
        record = records[seed]
        payload = record["payload"]
        triton_event = float(payload["triton_event_seconds"])
        refact_event = float(payload["refact_event_seconds"])
        triton_wall = float(payload["triton_wall_seconds"])
        refact_wall = float(payload["refact_wall_seconds"])
        per_seed.append(
            {
                "seed": seed,
                "source": record["relative_path"],
                "incremental_event_seconds": triton_event,
                "fresh_event_seconds": refact_event,
                "paired_event_speedup": refact_event / triton_event,
                "incremental_wall_seconds": triton_wall,
                "fresh_wall_seconds": refact_wall,
                "paired_wall_speedup": refact_wall / triton_wall,
                "incremental_updates_per_second": float(
                    payload["triton_updates_per_second"]
                ),
                "fresh_updates_per_second": float(
                    payload["refact_updates_per_second"]
                ),
                "minimum_effective_bandwidth_gb_s": float(
                    payload["minimum_effective_bandwidth_gb_s"]
                ),
                "reusable_workspace_bytes": int(
                    payload["triton_workspace_bytes"]
                ),
                "timed_extra_allocated_bytes": int(
                    payload["triton_peak_extra_allocated_bytes"]
                ),
                "effective_workspace_bytes": int(
                    payload["triton_workspace_bytes"]
                    + payload["triton_peak_extra_allocated_bytes"]
                ),
                "L_relative_error": payload["L_relative_error"],
                "Linv_relative_error": payload["Linv_relative_error"],
                "Aw_relative_error": float(payload["Aw_relative_error"]),
            }
        )
    metric_names = (
        "incremental_event_seconds",
        "fresh_event_seconds",
        "paired_event_speedup",
        "incremental_wall_seconds",
        "fresh_wall_seconds",
        "paired_wall_speedup",
        "incremental_updates_per_second",
        "fresh_updates_per_second",
        "minimum_effective_bandwidth_gb_s",
    )
    aggregate = {
        metric: _stats(item[metric] for item in per_seed)
        for metric in metric_names
    }
    correctness_fields = (
        ("Linv_relative_error", "Aw_relative_error")
        if spec["impl"] == "inverse_factor"
        else ("L_relative_error", "Aw_relative_error")
    )
    correctness_max = {
        field: max(float(item[field]) for item in per_seed)
        for field in correctness_fields
    }
    correctness_aggregate = {
        field: _stats(float(item[field]) for item in per_seed)
        for field in correctness_fields
    }
    workspace_components = {}
    for field in (
        "reusable_workspace_bytes",
        "timed_extra_allocated_bytes",
        "effective_workspace_bytes",
    ):
        values = {int(item[field]) for item in per_seed}
        if len(values) != 1:
            raise AuditError(
                f"{spec['id']}: {field} changed across seeds: "
                f"{sorted(values)}"
            )
        workspace_components[field] = next(iter(values))
    if workspace_components["effective_workspace_bytes"] != (
        workspace_components["reusable_workspace_bytes"]
        + workspace_components["timed_extra_allocated_bytes"]
    ):
        raise AuditError(
            f"{spec['id']}: effective workspace does not equal reusable "
            "workspace plus timed extra allocation"
        )
    return {
        "row_id": spec["id"],
        "configuration": {k: v for k, v in spec.items() if k != "id"},
        "seed_level": per_seed,
        "aggregate": aggregate,
        "correctness_aggregate": correctness_aggregate,
        "correctness_max": correctness_max,
        "workspace": {
            **workspace_components,
            "deterministic": True,
            "definition": (
                "triton_workspace_bytes + "
                "triton_peak_extra_allocated_bytes"
            ),
        },
    }


def _aggregate_stability_row(
    spec: dict[str, Any],
    records: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    per_seed = []
    for seed in EXPECTED_SEEDS:
        record = records[seed]
        payload = record["payload"]
        per_seed.append(
            {
                "seed": seed,
                "source": record["relative_path"],
                "initial_measured_condition": float(
                    payload["initial_measured_condition"]
                ),
                "final_Linv_relative_error": float(
                    payload["Linv_relative_error"]
                ),
                "final_Aw_relative_error": float(payload["Aw_relative_error"]),
                "final_inverse_residual": float(
                    payload["checkpoints"][-1]["inverse_residual"]
                ),
                "max_Linv_relative_error": float(
                    payload["max_Linv_relative_error"]
                ),
                "max_Aw_relative_error": float(
                    payload["max_Aw_relative_error"]
                ),
                "max_inverse_residual": float(payload["max_inverse_residual"]),
            }
        )
    return {
        "row_id": spec["id"],
        "configuration": {k: v for k, v in spec.items() if k != "id"},
        "seed_level": per_seed,
        "aggregate": {
            field: _stats(item[field] for item in per_seed)
            for field in (
                "initial_measured_condition",
                "final_Linv_relative_error",
                "final_Aw_relative_error",
                "final_inverse_residual",
            )
        },
        "drift_max": {
            field: max(item[field] for item in per_seed)
            for field in (
                "max_Linv_relative_error",
                "max_Aw_relative_error",
                "max_inverse_residual",
            )
        },
    }


def _aggregate_refresh_row(
    rank: int,
    interval: int,
    records: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    per_seed = []
    for seed in EXPECTED_SEEDS:
        record = records[seed]
        payload = record["payload"]
        row = record["row"]
        event = float(row["event_seconds_per_update"])
        fresh_event = float(payload["fresh_event_seconds_per_update"])
        wall = float(row["wall_seconds_per_update"])
        fresh_wall = float(payload["fresh_wall_seconds_per_update"])
        per_seed.append(
            {
                "seed": seed,
                "source": record["relative_path"],
                "event_seconds_per_update": event,
                "fresh_event_seconds_per_update": fresh_event,
                "paired_event_speedup": fresh_event / event,
                "wall_seconds_per_update": wall,
                "fresh_wall_seconds_per_update": fresh_wall,
                "paired_wall_speedup": fresh_wall / wall,
                "final_Linv_relative_error": float(
                    row["Linv_relative_error"]
                ),
                "final_Aw_relative_error": float(row["Aw_relative_error"]),
                "max_pre_refresh_Linv_relative_error": float(
                    row["max_pre_refresh_Linv_relative_error"]
                ),
                "max_pre_refresh_Aw_relative_error": float(
                    row["max_pre_refresh_Aw_relative_error"]
                ),
            }
        )
    return {
        "row_id": f"refresh-r{rank}-q{interval}",
        "rank": rank,
        "refresh_interval": interval,
        "refresh_count": 0 if interval == 0 else 200 // interval,
        "seed_level": per_seed,
        "aggregate": {
            field: _stats(item[field] for item in per_seed)
            for field in (
                "event_seconds_per_update",
                "fresh_event_seconds_per_update",
                "paired_event_speedup",
                "wall_seconds_per_update",
                "fresh_wall_seconds_per_update",
                "paired_wall_speedup",
                "final_Linv_relative_error",
                "final_Aw_relative_error",
                "max_pre_refresh_Linv_relative_error",
                "max_pre_refresh_Aw_relative_error",
            )
        },
        "drift_max": {
            field: max(item[field] for item in per_seed)
            for field in (
                "final_Linv_relative_error",
                "final_Aw_relative_error",
                "max_pre_refresh_Linv_relative_error",
                "max_pre_refresh_Aw_relative_error",
            )
        },
    }


def _aggregate_large_row(
    spec: dict[str, Any],
    inverse_records: dict[int, dict[str, Any]],
    refact_records: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    per_seed = []
    for seed in EXPECTED_SEEDS:
        inverse_record = inverse_records[seed]
        refact_record = refact_records[seed]
        inverse = inverse_record["payload"]
        refact = refact_record["payload"]
        inverse_event = float(inverse["event_seconds_median"])
        refact_event = float(refact["event_seconds_median"])
        inverse_wall = float(inverse["wall_seconds_median"])
        refact_wall = float(refact["wall_seconds_median"])
        per_seed.append(
            {
                "seed": seed,
                "inverse_source": inverse_record["relative_path"],
                "fresh_source": refact_record["relative_path"],
                "inverse_event_seconds": inverse_event,
                "fresh_event_seconds": refact_event,
                "paired_event_speedup": refact_event / inverse_event,
                "inverse_wall_seconds": inverse_wall,
                "fresh_wall_seconds": refact_wall,
                "paired_wall_speedup": refact_wall / inverse_wall,
                "inverse_updates_per_second": float(
                    inverse["updates_per_second"]
                ),
                "fresh_updates_per_second": float(
                    refact["updates_per_second"]
                ),
                "Linv_relative_error": inverse["Linv_relative_error"],
                "Aw_relative_error": inverse["Aw_relative_error"],
                "inverse_peak_allocated_bytes": int(
                    inverse["peak_allocated_bytes"]
                ),
                "fresh_peak_allocated_bytes": int(
                    refact["peak_allocated_bytes"]
                ),
            }
        )
    correctness_checked = spec["inverse"]["correctness_checked"]
    correctness_max = {
        "checked_against_fresh": correctness_checked,
        "finite_full_all_seeds": True,
    }
    if correctness_checked:
        correctness_aggregate = {
            field: _stats(float(item[field]) for item in per_seed)
            for field in ("Linv_relative_error", "Aw_relative_error")
        }
        correctness_max.update(
            {
                field: max(float(item[field]) for item in per_seed)
                for field in ("Linv_relative_error", "Aw_relative_error")
            }
        )
    else:
        correctness_aggregate = {
            "Linv_relative_error": None,
            "Aw_relative_error": None,
        }
        correctness_max.update(
            {"Linv_relative_error": None, "Aw_relative_error": None}
        )
    return {
        "row_id": spec["id"],
        "configuration": {
            "rank": spec["rank"],
            "repeats": spec["repeats"],
            "warmup": spec["warmup"],
            "inverse": spec["inverse"],
            "refact": spec["refact"],
        },
        "minimum_four_matrix_state_bytes": 4 * spec["rank"] * spec["rank"] * 4,
        "seed_level": per_seed,
        "aggregate": {
            field: _stats(item[field] for item in per_seed)
            for field in (
                "inverse_event_seconds",
                "fresh_event_seconds",
                "paired_event_speedup",
                "inverse_wall_seconds",
                "fresh_wall_seconds",
                "paired_wall_speedup",
                "inverse_updates_per_second",
                "fresh_updates_per_second",
            )
        },
        "correctness_aggregate": correctness_aggregate,
        "correctness_max": correctness_max,
        "memory_max": {
            "inverse_peak_allocated_bytes": max(
                item["inverse_peak_allocated_bytes"] for item in per_seed
            ),
            "fresh_peak_allocated_bytes": max(
                item["fresh_peak_allocated_bytes"] for item in per_seed
            ),
        },
    }


def _check_common_environment(
    environment_by_path: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not environment_by_path:
        raise AuditError("no environment-bearing campaign records were found")
    common_fields = ("platform", "torch", "cuda", "gpu", "gpu_total_memory")
    common = {}
    for field in common_fields:
        values = {
            json.dumps(env[field], sort_keys=True): env[field]
            for env in environment_by_path.values()
        }
        if len(values) != 1:
            rendered = sorted(repr(value) for value in values.values())
            raise AuditError(
                f"environment field {field!r} changed across campaign files: "
                f"{rendered}"
            )
        common[field] = next(iter(values.values()))
    return {
        **common,
        "hosts": sorted({env["host"] for env in environment_by_path.values()}),
        "slurm_job_ids": sorted(
            {str(env["slurm_job_id"]) for env in environment_by_path.values()}
        ),
        "slurm_array_job_ids": sorted(
            {
                str(env["slurm_array_job_id"])
                for env in environment_by_path.values()
                if env.get("slurm_array_job_id")
            }
        ),
    }


def audit_and_aggregate(
    results_root: Path,
    campaign_tag: str = DEFAULT_CAMPAIGN_TAG,
) -> dict[str, Any]:
    """Audit the exact tagged campaign and return its aggregate.

    No output is written by this function.  Any missing, duplicate, unexpected,
    malformed, or environment-inconsistent record raises :class:`AuditError`.
    """

    results_root = results_root.resolve()
    paths = _discover(results_root, campaign_tag)
    issues: list[str] = []
    main_records = {spec["id"]: {} for spec in MAIN_SPECS}
    inverse_records = {spec["id"]: {} for spec in INVERSE_SPECS}
    stability_records = {spec["id"]: {} for spec in STABILITY_SPECS}
    condition_records = {spec["id"]: {} for spec in CONDITION_SPECS}
    refresh_records = {
        f"refresh-r{rank}-q{interval}": {}
        for rank in REFRESH_BY_RANK
        for interval in EXPECTED_REFRESH_INTERVALS
    }
    large_inverse_records = {spec["id"]: {} for spec in LARGE_SPECS}
    large_refact_records = {spec["id"]: {} for spec in LARGE_SPECS}
    environment_by_path: dict[str, dict[str, Any]] = {}
    stability_job_by_seed: dict[int, set[tuple[str, str, str]]] = defaultdict(set)
    refresh_job_by_seed: dict[int, set[tuple[str, str, str]]] = defaultdict(set)
    scaling_context_by_seed: dict[int, set[tuple[str, str, str]]] = defaultdict(set)
    large_context_by_seed: dict[int, set[tuple[str, str, str, str]]] = defaultdict(set)

    for path in paths:
        relative_path = _relative_path(path, results_root)
        try:
            payload = _read_json(path)
            seed = _seed_and_suffix(path, payload, campaign_tag)
            experiment = payload.get("experiment")
            record = {
                "path": path,
                "relative_path": relative_path,
                "payload": payload,
            }
            if experiment == "cuda_scaling":
                key = (payload.get("impl"), payload.get("rank"), payload.get("batch"))
                spec = MAIN_BY_KEY.get(key)
                destination = main_records
                if spec is None:
                    spec = INVERSE_BY_KEY.get(key)
                    destination = inverse_records
                if spec is None:
                    raise AuditError(
                        f"{path}: unexpected tagged cuda_scaling row {key!r}"
                    )
                env = _validate_scaling(path, payload, spec, seed)
                _insert(destination[spec["id"]], seed, record, str(path))
                environment_by_path[relative_path] = env
                scaling_context_by_seed[seed].add(
                    (
                        str(env["slurm_job_id"]),
                        str(env["slurm_array_task_id"]),
                        str(env["host"]),
                    )
                )
            elif experiment == "cuda_inverse_stability":
                target_condition = payload.get("target_condition")
                if target_condition is None:
                    key = (
                        payload.get("implementation"),
                        payload.get("rank"),
                        payload.get("steps"),
                    )
                    spec = STABILITY_BY_KEY.get(key)
                    destination = stability_records
                else:
                    key = (
                        payload.get("implementation"),
                        payload.get("rank"),
                        payload.get("steps"),
                        target_condition,
                    )
                    spec = CONDITION_BY_KEY.get(key)
                    destination = condition_records
                if spec is None:
                    raise AuditError(
                        f"{path}: unexpected tagged stability row {key!r}"
                    )
                jobs = _validate_stability(path, payload, spec, seed)
                _insert(destination[spec["id"]], seed, record, str(path))
                stability_job_by_seed[seed].add(
                    (
                        str(jobs["slurm_job_id"]),
                        str(jobs["slurm_array_job_id"]),
                        str(jobs["slurm_array_task_id"]),
                    )
                )
            elif experiment == "cuda_inverse_refresh":
                rank = payload.get("rank")
                spec = REFRESH_BY_RANK.get(rank)
                if spec is None:
                    raise AuditError(
                        f"{path}: unexpected tagged refresh rank {rank!r}"
                    )
                env, rows = _validate_refresh(path, payload, spec, seed)
                environment_by_path[relative_path] = env
                refresh_job_by_seed[seed].add(
                    (
                        str(env["slurm_job_id"]),
                        str(env["slurm_array_job_id"]),
                        str(env["slurm_array_task_id"]),
                    )
                )
                for interval, row in rows.items():
                    expanded = {**record, "row": row}
                    row_id = f"refresh-r{rank}-q{interval}"
                    _insert(
                        refresh_records[row_id],
                        seed,
                        expanded,
                        f"{path} interval={interval}",
                    )
            elif experiment == "cuda_large_rank_compare":
                rank = payload.get("rank")
                spec = LARGE_BY_RANK.get(rank)
                if spec is None:
                    raise AuditError(
                        f"{path}: unexpected tagged large rank {rank!r}"
                    )
                mode = payload.get("mode")
                if mode == spec["inverse"]["mode"]:
                    mode_key = "inverse"
                    destination = large_inverse_records
                elif mode == spec["refact"]["mode"]:
                    mode_key = "refact"
                    destination = large_refact_records
                else:
                    raise AuditError(
                        f"{path}: unexpected tagged large mode {mode!r}"
                    )
                env = _validate_large(path, payload, spec, mode_key, seed)
                _insert(destination[spec["id"]], seed, record, str(path))
                environment_by_path[relative_path] = env
                large_context_by_seed[seed].add(
                    (
                        str(env["slurm_job_id"]),
                        str(env["slurm_array_job_id"]),
                        str(env["slurm_array_task_id"]),
                        str(env["host"]),
                    )
                )
            else:
                raise AuditError(
                    f"{path}: unexpected tagged experiment {experiment!r}"
                )
        except AuditError as exc:
            issues.append(str(exc))

    expected_file_count = (
        (len(MAIN_SPECS) + len(INVERSE_SPECS)) * len(EXPECTED_SEEDS)
        + (len(STABILITY_SPECS) + len(CONDITION_SPECS)) * len(EXPECTED_SEEDS)
        + len(REFRESH_SPECS) * len(EXPECTED_SEEDS)
        + 2 * len(LARGE_SPECS) * len(EXPECTED_SEEDS)
    )
    if len(paths) != expected_file_count:
        issues.append(
            f"tagged JSON count={len(paths)}, expected {expected_file_count}"
        )
    for name, rows in (
        ("main11", main_records),
        ("inverse-heldout3", inverse_records),
        ("stability4", stability_records),
        ("condition3", condition_records),
        ("refresh15", refresh_records),
        ("large7/inverse", large_inverse_records),
        ("large7/refact", large_refact_records),
    ):
        issues.extend(_missing_seed_issues(name, rows))

    for seed in EXPECTED_SEEDS:
        if len(scaling_context_by_seed[seed]) != 1:
            issues.append(
                f"timing seed {seed}: expected one job/task/host context, "
                f"found {sorted(scaling_context_by_seed[seed])}"
            )
        if len(stability_job_by_seed[seed]) != 1:
            issues.append(
                f"stability seed {seed}: expected one job/task context, "
                f"found {sorted(stability_job_by_seed[seed])}"
            )
        if len(refresh_job_by_seed[seed]) != 1:
            issues.append(
                f"refresh seed {seed}: expected one job/task context, "
                f"found {sorted(refresh_job_by_seed[seed])}"
            )
        if (
            len(stability_job_by_seed[seed]) == 1
            and len(refresh_job_by_seed[seed]) == 1
            and stability_job_by_seed[seed] != refresh_job_by_seed[seed]
        ):
            issues.append(
                f"stability/refresh seed {seed}: job contexts do not match: "
                f"{sorted(stability_job_by_seed[seed])} versus "
                f"{sorted(refresh_job_by_seed[seed])}"
            )
        if len(large_context_by_seed[seed]) != 1:
            issues.append(
                f"large seed {seed}: expected one job/task/host context, "
                f"found {sorted(large_context_by_seed[seed])}"
            )

    environment_summary = None
    try:
        environment_summary = _check_common_environment(environment_by_path)
    except AuditError as exc:
        issues.append(str(exc))
    if issues:
        numbered = "\n".join(
            f"  {index}. {issue}" for index, issue in enumerate(issues, 1)
        )
        raise AuditError(
            f"E5 campaign {campaign_tag!r} failed audit with "
            f"{len(issues)} issue(s):\n{numbered}"
        )

    main_rows = [
        _aggregate_scaling_row(spec, main_records[spec["id"]])
        for spec in MAIN_SPECS
    ]
    inverse_rows = [
        _aggregate_scaling_row(spec, inverse_records[spec["id"]])
        for spec in INVERSE_SPECS
    ]
    stability_rows = [
        _aggregate_stability_row(spec, stability_records[spec["id"]])
        for spec in STABILITY_SPECS
    ]
    condition_rows = [
        _aggregate_stability_row(spec, condition_records[spec["id"]])
        for spec in CONDITION_SPECS
    ]
    refresh_rows = [
        _aggregate_refresh_row(
            spec["rank"],
            interval,
            refresh_records[f"refresh-r{spec['rank']}-q{interval}"],
        )
        for spec in REFRESH_SPECS
        for interval in EXPECTED_REFRESH_INTERVALS
    ]
    large_rows = [
        _aggregate_large_row(
            spec,
            large_inverse_records[spec["id"]],
            large_refact_records[spec["id"]],
        )
        for spec in LARGE_SPECS
    ]
    assert environment_summary is not None
    return {
        "schema_version": 1,
        "campaign_tag": campaign_tag,
        "audit": {
            "status": "pass",
            "results_root": str(results_root),
            "required_seeds": list(EXPECTED_SEEDS),
            "tagged_json_count": len(paths),
            "expected_json_count": expected_file_count,
            "claim_row_counts": {
                "main": len(main_rows),
                "inverse_heldout": len(inverse_rows),
                "stability": len(stability_rows),
                "condition": len(condition_rows),
                "refresh": len(refresh_rows),
                "large_pairs": len(large_rows),
            },
            "environment": environment_summary,
        },
        "groups": {
            "main": main_rows,
            "inverse_heldout": inverse_rows,
            "stability": stability_rows,
            "condition": condition_rows,
            "refresh": refresh_rows,
            "large_pairs": large_rows,
        },
    }


def _pm(stats: dict[str, Any], scale: float = 1.0) -> str:
    return (
        f"{stats['mean'] * scale:.1f} "
        f"± {stats['sample_sd'] * scale:.1f}"
    )


def _sci_pm(stats: dict[str, Any]) -> str:
    sample_sd = float(stats["sample_sd"])
    rendered_sd = "0.0" if sample_sd == 0.0 else f"{sample_sd:.1e}"
    return f"{stats['mean']:.1e} ± {rendered_sd}"


def _optional_sci_pm(stats: dict[str, Any] | None) -> str:
    return "n/a" if stats is None else _sci_pm(stats)


def render_markdown(summary: dict[str, Any]) -> str:
    """Render concise one-decimal, claim-ready Markdown rows."""

    audit = summary["audit"]
    groups = summary["groups"]
    lines = [
        f"# E5 ten-seed campaign: `{summary['campaign_tag']}`",
        "",
        (
            f"Audit: **PASS** — {audit['tagged_json_count']} tagged JSONs; "
            f"seeds 1–10; GPU `{audit['environment']['gpu']}`. "
            "Columns marked mean ± SD are arithmetic means and sample "
            "standard deviations across seed-level measurements. Paired "
            "speedups are formed within seed before aggregation. Columns "
            "marked fixed are deterministic and identical across runs. "
            "Every stochastic correctness/drift cell shown also reports "
            "mean ± SD; for refresh rows, the seed-level statistic being "
            "aggregated is that seed's maximum pre-refresh error. The JSON "
            "additionally preserves global worst-case bounds over all ten "
            "seeds and all applicable checkpoints."
        ),
        "",
        "## Main timing (11 frozen rows)",
        "",
        "| Impl. | r (fixed) | T (fixed) | Incremental µs (mean ± SD) | Fresh µs (mean ± SD) | Incremental k/s (mean ± SD) | Fresh k/s (mean ± SD) | Paired speedup (mean ± SD) | BW GB/s (mean ± SD) | Workspace KiB (fixed) | L err. (mean ± SD) | Aw err. (mean ± SD) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in groups["main"]:
        config = row["configuration"]
        aggregate = row["aggregate"]
        correctness = row["correctness_aggregate"]
        lines.append(
            f"| {config['impl']} | {config['rank']} | {config['batch']} | "
            f"{_pm(aggregate['incremental_event_seconds'], 1e6)} | "
            f"{_pm(aggregate['fresh_event_seconds'], 1e6)} | "
            f"{_pm(aggregate['incremental_updates_per_second'], 1e-3)} | "
            f"{_pm(aggregate['fresh_updates_per_second'], 1e-3)} | "
            f"{_pm(aggregate['paired_event_speedup'])}× | "
            f"{_pm(aggregate['minimum_effective_bandwidth_gb_s'])} | "
            f"{row['workspace']['effective_workspace_bytes'] / 2**10:.1f} | "
            f"{_sci_pm(correctness['L_relative_error'])} | "
            f"{_sci_pm(correctness['Aw_relative_error'])} |"
        )
    lines.extend(
        [
            "",
            "## Cached inverse-factor timing (3 frozen rows)",
            "",
            "| r (fixed) | Inverse µs (mean ± SD) | Fresh µs (mean ± SD) | Paired speedup (mean ± SD) | Linv err. (mean ± SD) | Aw err. (mean ± SD) |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in groups["inverse_heldout"]:
        config = row["configuration"]
        aggregate = row["aggregate"]
        correctness = row["correctness_aggregate"]
        lines.append(
            f"| {config['rank']} | "
            f"{_pm(aggregate['incremental_event_seconds'], 1e6)} | "
            f"{_pm(aggregate['fresh_event_seconds'], 1e6)} | "
            f"{_pm(aggregate['paired_event_speedup'])}× | "
            f"{_sci_pm(correctness['Linv_relative_error'])} | "
            f"{_sci_pm(correctness['Aw_relative_error'])} |"
        )
    lines.extend(
        [
            "",
            "## Repeated-update stability (4 frozen rows)",
            "",
            "| Path | r (fixed) | Initial κ(G) (mean ± SD) | Final Linv err. (mean ± SD) | Final Aw err. (mean ± SD) | Final inverse residual (mean ± SD) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in groups["stability"]:
        config = row["configuration"]
        aggregate = row["aggregate"]
        lines.append(
            f"| {config['implementation']} | {config['rank']} | "
            f"{_pm(aggregate['initial_measured_condition'])} | "
            f"{_sci_pm(aggregate['final_Linv_relative_error'])} | "
            f"{_sci_pm(aggregate['final_Aw_relative_error'])} | "
            f"{_sci_pm(aggregate['final_inverse_residual'])} |"
        )
    lines.extend(
        [
            "",
            "## Condition stress (3 frozen rows)",
            "",
            "| Target κ(G) (fixed) | Measured κ(G) (mean ± SD) | Final Linv err. (mean ± SD) | Final Aw err. (mean ± SD) | Final inverse residual (mean ± SD) |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in groups["condition"]:
        config = row["configuration"]
        aggregate = row["aggregate"]
        lines.append(
            f"| {config['target_condition']:.1e} | "
            f"{_sci_pm(aggregate['initial_measured_condition'])} | "
            f"{_sci_pm(aggregate['final_Linv_relative_error'])} | "
            f"{_sci_pm(aggregate['final_Aw_relative_error'])} | "
            f"{_sci_pm(aggregate['final_inverse_residual'])} |"
        )
    lines.extend(
        [
            "",
            "## Refresh cost and drift (15 frozen rows)",
            "",
            "| r (fixed) | q (fixed) | Refreshes (fixed) | Event µs/update (mean ± SD) | Fresh µs/update (mean ± SD) | Paired speedup (mean ± SD) | Per-seed max pre-refresh Linv err. (mean ± SD) | Per-seed max pre-refresh Aw err. (mean ± SD) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in groups["refresh"]:
        aggregate = row["aggregate"]
        interval = "none" if row["refresh_interval"] == 0 else row["refresh_interval"]
        lines.append(
            f"| {row['rank']} | {interval} | {row['refresh_count']} | "
            f"{_pm(aggregate['event_seconds_per_update'], 1e6)} | "
            f"{_pm(aggregate['fresh_event_seconds_per_update'], 1e6)} | "
            f"{_pm(aggregate['paired_event_speedup'])}× | "
            f"{_sci_pm(aggregate['max_pre_refresh_Linv_relative_error'])} | "
            f"{_sci_pm(aggregate['max_pre_refresh_Aw_relative_error'])} |"
        )
    lines.extend(
        [
            "",
            "## Self-contained large-rank pairs (7 frozen rows)",
            "",
            "| r (fixed) | Inverse ms (mean ± SD) | Fresh ms (mean ± SD) | Paired speedup (mean ± SD) | State MiB (fixed) | Linv err. (mean ± SD) | Aw err. (mean ± SD) |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in groups["large_pairs"]:
        config = row["configuration"]
        aggregate = row["aggregate"]
        correctness = row["correctness_aggregate"]
        lines.append(
            f"| {config['rank']} | "
            f"{_pm(aggregate['inverse_event_seconds'], 1e3)} | "
            f"{_pm(aggregate['fresh_event_seconds'], 1e3)} | "
            f"{_pm(aggregate['paired_event_speedup'])}× | "
            f"{row['minimum_four_matrix_state_bytes'] // 2**20:,} | "
            f"{_optional_sci_pm(correctness['Linv_relative_error'])} | "
            f"{_optional_sci_pm(correctness['Aw_relative_error'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Strictly audit and aggregate the frozen ten-seed E5 campaign."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=repository_root / "rebuttal_results",
        help="root containing cuda/ and cuda_tuning/ (default: %(default)s)",
    )
    parser.add_argument(
        "--campaign-tag",
        default=DEFAULT_CAMPAIGN_TAG,
        help="exact output-suffix tag to discover (default: %(default)s)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="machine-readable audited aggregate",
    )
    parser.add_argument(
        "--output-rows",
        type=Path,
        required=True,
        help="concise Markdown claim rows",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_json.resolve() == args.output_rows.resolve():
        print("output paths must be different", file=sys.stderr)
        return 2
    try:
        summary = audit_and_aggregate(args.results_root, args.campaign_tag)
        markdown = render_markdown(summary)
    except AuditError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    _atomic_write(
        args.output_json,
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(args.output_rows, markdown)
    print(args.output_json.resolve())
    print(args.output_rows.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
