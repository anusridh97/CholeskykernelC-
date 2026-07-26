"""Fail-closed audit and aggregation for the frozen black-hole campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .cuda_blackhole_stability import (
    CAMPAIGN_TAG,
    CHECKPOINT_STEPS,
    EVALUATION_TRANSITIONS,
    EXPECTED_GPU_MEMORY,
    EXPECTED_GPU_PREFIX,
    FEATURE_SCHEMA_VERSION,
    HF_REPOSITORY,
    HF_REVISION,
    INITIAL_TRANSITIONS,
    LADDER_RANKS,
    METHOD_ORDER as STABILITY_METHOD_ORDER,
    RAW_BYTES,
    RAW_RELATIVE_PATH,
    RAW_SHA256,
    REFRESH_INTERVAL,
    RIDGE,
    ROOT,
    STABILITY_RANKS,
    STABILITY_TRANSITIONS,
    STUDENT_SOURCE,
    TRACE_SHAPE,
    WARMUP_TRANSITION,
    _load_and_validate_manifest,
)


RESULTS_ROOT = ROOT / "rebuttal_results" / "blackhole"
FEATURE_SOURCE = ROOT / "rebuttal_experiments" / "blackhole_features.py"
LADDER_SOURCE = (
    ROOT / "rebuttal_experiments" / "cuda_blackhole_rank_scaling.py"
)
STABILITY_SOURCE = (
    ROOT / "rebuttal_experiments" / "cuda_blackhole_stability.py"
)

SCHEMA_VERSION = "blackhole_aggregate_v1"
LADDER_SCHEMA_VERSION = "cuda_blackhole_rank_scaling_v1"
LADDER_EXPERIMENT = "cuda_blackhole_rank_scaling"
LADDER_PROTOCOL_VERSION = "blackhole_s4_ladder_v1"
STABILITY_SCHEMA_VERSION = "blackhole_stability_v1"
STABILITY_EXPERIMENT = "cuda_blackhole_stability"
LADDER_METHODS = ("inverse", "fresh_refact")
LADDER_REPEATS = {
    2_048: 10,
    4_096: 10,
    8_192: 10,
    16_384: 5,
    32_768: 5,
    49_152: 3,
    65_536: 3,
}
TIMING_REGION = {
    "includes": ["G += z z^T", "M += z w^T", "method core"],
    "excludes": [
        "feature I/O",
        "host-to-device feature transfer",
        "common-state construction",
        "state restoration",
        "CUDA synchronization before timing",
        "correctness checks",
    ],
    "cuda_event_unit": "seconds",
    "wall_unit": "seconds",
}
T_CRITICAL_975 = {
    9: 2.2621571627409915,
    99: 1.9842169515086827,
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class AuditError(RuntimeError):
    """Raised when a campaign is incomplete or violates the frozen protocol."""


def _atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise AuditError(f"non-finite JSON constant {value!r}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditError(f"{path}: could not load strict JSON: {error}") from error
    if not isinstance(payload, dict):
        raise AuditError(f"{path}: top-level JSON value must be an object")
    return payload


def _sha256(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise AuditError(f"{context}={actual!r}; expected {expected!r}")


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AuditError(f"{context} must be an object")
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise AuditError(f"{context} must be a list")
    return value


def _number(
    value: Any,
    context: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditError(f"{context} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise AuditError(f"{context} must be finite, got {result!r}")
    if strictly_positive and result <= 0:
        raise AuditError(f"{context} must be positive, got {result!r}")
    if minimum is not None and result < minimum:
        raise AuditError(f"{context} must be >= {minimum}, got {result!r}")
    return result


def _close(
    actual: Any,
    expected: float,
    context: str,
    *,
    rel_tol: float = 2e-9,
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
            f"{context}={actual_number:.17g}; recomputed={expected:.17g}"
        )


def _resolve_local_source(value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AuditError(f"{context} must be a nonempty string")
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise AuditError(f"{context} escapes the repository: {value!r}") from error
    if not resolved.is_file():
        raise AuditError(f"{context} does not exist: {resolved}")
    return resolved


def _sha(value: Any, context: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise AuditError(f"{context} must be a lowercase SHA-256")
    return value


def _commit(value: Any, context: str) -> str:
    if not isinstance(value, str) or not COMMIT_RE.fullmatch(value):
        raise AuditError(f"{context} must be a full lowercase git commit")
    return value


def _validate_hardware(
    value: Any,
    context: str,
) -> tuple[str, int, str, str]:
    hardware = _object(value, context)
    gpu = hardware.get("gpu")
    if not isinstance(gpu, str) or not gpu.startswith(EXPECTED_GPU_PREFIX):
        raise AuditError(
            f"{context}.gpu={gpu!r}; expected prefix {EXPECTED_GPU_PREFIX!r}"
        )
    _equal(
        hardware.get("gpu_total_memory"),
        EXPECTED_GPU_MEMORY,
        f"{context}.gpu_total_memory",
    )
    torch_version = hardware.get("torch")
    cuda_version = hardware.get("cuda")
    if not isinstance(torch_version, str) or not torch_version:
        raise AuditError(f"{context}.torch must be a nonempty string")
    if not isinstance(cuda_version, str) or not cuda_version:
        raise AuditError(f"{context}.cuda must be a nonempty string")
    _equal(
        hardware.get("compute_capability"),
        [9, 0],
        f"{context}.compute_capability",
    )
    _equal(
        hardware.get("visible_device_count"),
        1,
        f"{context}.visible_device_count",
    )
    _equal(
        hardware.get("float32_matmul_precision"),
        "highest",
        f"{context}.float32_matmul_precision",
    )
    _equal(
        hardware.get("allow_tf32_matmul"),
        False,
        f"{context}.allow_tf32_matmul",
    )
    if isinstance(hardware.get("cudnn"), bool) or not isinstance(
        hardware.get("cudnn"), int
    ):
        raise AuditError(f"{context}.cudnn must be an integer")
    return gpu, EXPECTED_GPU_MEMORY, torch_version, cuda_version


def _validate_dataset(value: Any, context: str) -> None:
    dataset = _object(value, context)
    expected = {
        "repository": HF_REPOSITORY,
        "revision": HF_REVISION,
        "relative_path": RAW_RELATIVE_PATH,
        "sha256": RAW_SHA256,
    }
    for key, frozen in expected.items():
        _equal(dataset.get(key), frozen, f"{context}.{key}")
    byte_value = dataset.get("bytes", dataset.get("actual_bytes"))
    _equal(byte_value, RAW_BYTES, f"{context}.bytes")


def _validate_trace(
    value: Any,
    *,
    expected_sha256: str,
    context: str,
) -> None:
    trace = _object(value, context)
    _equal(trace.get("sha256"), expected_sha256, f"{context}.sha256")
    _equal(trace.get("shape"), list(TRACE_SHAPE), f"{context}.shape")
    _equal(trace.get("dtype"), "float32", f"{context}.dtype")


def _sample_stats(values: Iterable[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    if len(samples) not in (10, 100):
        raise AuditError(
            f"95% t interval is not frozen for sample count {len(samples)}"
        )
    if not all(math.isfinite(value) for value in samples):
        raise AuditError("cannot aggregate non-finite samples")
    mean = statistics.fmean(samples)
    sample_sd = statistics.stdev(samples)
    degrees_freedom = len(samples) - 1
    critical = T_CRITICAL_975[degrees_freedom]
    half_width = critical * sample_sd / math.sqrt(len(samples))
    return {
        "n": len(samples),
        "mean": mean,
        "sample_sd": sample_sd,
        "t_critical_95": critical,
        "degrees_freedom": degrees_freedom,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "ci95_half_width": half_width,
        "median": statistics.median(samples),
        "minimum": min(samples),
        "maximum": max(samples),
        "values": samples,
    }


def _validate_manifest(
    path: Path,
) -> tuple[dict[str, Any], str, str, str]:
    try:
        manifest, trace_path, manifest_sha256, _ = _load_and_validate_manifest(
            path,
            STABILITY_RANKS[0],
        )
    except Exception as error:
        raise AuditError(f"{path}: invalid frozen feature manifest: {error}") from error

    source = _object(manifest.get("source"), "feature_manifest.source")
    source_path = _resolve_local_source(
        source.get("path"),
        "feature_manifest.source.path",
    )
    if source_path != FEATURE_SOURCE.resolve():
        raise AuditError(
            "feature manifest producer path is not the current frozen producer: "
            f"{source_path}"
        )
    source_sha256 = _sha(
        source.get("sha256"),
        "feature_manifest.source.sha256",
    )
    _equal(
        source_sha256,
        _sha256(FEATURE_SOURCE),
        "feature producer current-source SHA-256",
    )
    trace_sha256 = _sha(
        _object(manifest.get("trace"), "feature_manifest.trace").get("sha256"),
        "feature_manifest.trace.sha256",
    )
    trace = np.load(trace_path, mmap_mode="r", allow_pickle=False)
    _equal(list(trace.shape), list(TRACE_SHAPE), "feature trace shape")
    _equal(str(trace.dtype), "float32", "feature trace dtype")
    if not trace.flags.c_contiguous:
        raise AuditError("feature trace is not C contiguous")
    if not bool(np.isfinite(trace).all()):
        raise AuditError("feature trace contains a non-finite value")
    calibration = np.asarray(trace[:64, :], dtype=np.float64)
    cumulative_energy = np.cumsum(
        calibration * calibration,
        axis=1,
        dtype=np.float64,
    )
    scales = manifest["rank_scaling"]["scales"]
    for rank in LADDER_RANKS:
        recomputed = math.sqrt(
            float(np.mean(cumulative_energy[:, rank - 1], dtype=np.float64))
        )
        _close(
            scales[str(rank)],
            recomputed,
            f"feature manifest rank {rank} scale",
            rel_tol=1e-13,
            abs_tol=0.0,
        )
    del calibration, cumulative_energy, trace
    return manifest, manifest_sha256, trace_sha256, source_sha256


def _validate_ladder_source(
    payload: dict[str, Any],
    context: str,
    feature_source_sha256: str,
) -> tuple[str, str, str]:
    source = _object(payload.get("source"), f"{context}.source")
    expected_keys = {
        "git_commit",
        "git_dirty",
        "driver_path",
        "driver_sha256",
        "kernel_path",
        "kernel_sha256",
        "feature_source_path",
        "feature_source_sha256",
    }
    _equal(set(source), expected_keys, f"{context}.source keys")
    commit = _commit(source.get("git_commit"), f"{context}.source.git_commit")
    _equal(source.get("git_dirty"), False, f"{context}.source.git_dirty")

    driver_path = _resolve_local_source(
        source.get("driver_path"),
        f"{context}.source.driver_path",
    )
    if driver_path != LADDER_SOURCE.resolve():
        raise AuditError(f"{context}: unexpected ladder driver {driver_path}")
    driver_sha = _sha(
        source.get("driver_sha256"),
        f"{context}.source.driver_sha256",
    )
    _equal(driver_sha, _sha256(driver_path), f"{context} current driver SHA-256")

    kernel_path = _resolve_local_source(
        source.get("kernel_path"),
        f"{context}.source.kernel_path",
    )
    if kernel_path != STUDENT_SOURCE.resolve():
        raise AuditError(f"{context}: unexpected kernel source {kernel_path}")
    kernel_sha = _sha(
        source.get("kernel_sha256"),
        f"{context}.source.kernel_sha256",
    )
    _equal(kernel_sha, _sha256(kernel_path), f"{context} current kernel SHA-256")

    feature_path = _resolve_local_source(
        source.get("feature_source_path"),
        f"{context}.source.feature_source_path",
    )
    if feature_path != FEATURE_SOURCE.resolve():
        raise AuditError(f"{context}: unexpected feature source {feature_path}")
    feature_sha = _sha(
        source.get("feature_source_sha256"),
        f"{context}.source.feature_source_sha256",
    )
    _equal(
        feature_sha,
        feature_source_sha256,
        f"{context}.source.feature_source_sha256",
    )
    return commit, driver_sha, kernel_sha


def _validate_ladder_validation(
    value: Any,
    *,
    rank: int,
    method: str,
    context: str,
) -> bool:
    validation = _object(value, context)
    _equal(
        set(validation),
        {"threshold", "passed", "finite_full", "probe", "full_dense"},
        f"{context} keys",
    )
    _close(validation.get("threshold"), 1e-3, f"{context}.threshold")
    finite_full = validation.get("finite_full")
    if not isinstance(finite_full, bool):
        raise AuditError(f"{context}.finite_full must be Boolean")
    diagnostic_values: list[float] = []
    probe = _object(validation.get("probe"), f"{context}.probe")
    _equal(
        set(probe),
        {
            "seed",
            "count",
            "distribution",
            "factor_identity",
            "operator_identity",
            "factor",
            "operator",
        },
        f"{context}.probe keys",
    )
    _equal(probe.get("seed"), 29421, f"{context}.probe.seed")
    _equal(probe.get("count"), 16, f"{context}.probe.count")
    distribution = probe.get("distribution")
    _equal(
        distribution,
        (
            "NumPy PCG64 Rademacher (+/-1/sqrt(rank)), one fixed rank-by-16 "
            "matrix reused for every transition and method"
        ),
        f"{context}.probe.distribution",
    )
    factor_identity = (
        "Linv G Linv^T Q = Q"
        if method == "inverse"
        else "L L^T Q = G Q"
    )
    operator_identity = (
        "Aw Q = Linv M Linv^T Q"
        if method == "inverse"
        else "L Aw L^T Q = M Q"
    )
    _equal(
        probe.get("factor_identity"),
        factor_identity,
        f"{context}.probe.factor_identity",
    )
    _equal(
        probe.get("operator_identity"),
        operator_identity,
        f"{context}.probe.operator_identity",
    )
    for family in ("factor", "operator"):
        metrics = _object(probe.get(family), f"{context}.probe.{family}")
        _equal(
            set(metrics),
            {
                "per_probe_relative_errors",
                "max_relative_error",
                "median_relative_error",
                "frobenius_relative_error",
            },
            f"{context}.probe.{family} keys",
        )
        errors = _list(
            metrics.get("per_probe_relative_errors"),
            f"{context}.probe.{family}.per_probe_relative_errors",
        )
        _equal(len(errors), 16, f"{context}.probe.{family} error count")
        numbers = [
            _number(
                item,
                f"{context}.probe.{family}.per_probe_relative_errors[{index}]",
                minimum=0.0,
            )
            for index, item in enumerate(errors)
        ]
        _close(
            metrics.get("max_relative_error"),
            max(numbers),
            f"{context}.probe.{family}.max_relative_error",
        )
        _close(
            metrics.get("median_relative_error"),
            statistics.median(numbers),
            f"{context}.probe.{family}.median_relative_error",
        )
        frobenius = _number(
            metrics.get("frobenius_relative_error"),
            f"{context}.probe.{family}.frobenius_relative_error",
            minimum=0.0,
        )
        diagnostic_values.extend((max(numbers), frobenius))
    full_dense = validation.get("full_dense")
    if rank <= 32_768:
        dense = _object(full_dense, f"{context}.full_dense")
        _equal(
            set(dense),
            {
                "kind",
                "factor_metric",
                "factor_relative_error",
                "operator_metric",
                "operator_relative_error",
            },
            f"{context}.full_dense keys",
        )
        expected_kind = (
            "exact_fresh_factor_and_whiten"
            if method == "inverse"
            else "full_backward_residual"
        )
        _equal(
            dense.get("kind"),
            expected_kind,
            f"{context}.full_dense.kind",
        )
        for key in ("factor_metric", "operator_metric"):
            if not isinstance(dense.get(key), str) or not dense[key]:
                raise AuditError(f"{context}.full_dense.{key} must be nonempty")
        for key in ("factor_relative_error", "operator_relative_error"):
            error = _number(
                dense.get(key),
                f"{context}.full_dense.{key}",
                minimum=0.0,
            )
            diagnostic_values.append(error)
    elif full_dense is not None:
        raise AuditError(
            f"{context}.full_dense must be null above rank 32768"
        )
    expected_passed = finite_full and max(diagnostic_values) <= 1e-3
    _equal(validation.get("passed"), expected_passed, f"{context}.passed")
    return expected_passed


def _validate_ladder_method(
    value: Any,
    *,
    rank: int,
    repeats: int,
    method: str,
    context: str,
) -> dict[str, Any]:
    row = _object(value, context)
    _equal(
        set(row),
        {
            "mode",
            "repeats",
            "event_seconds_samples",
            "wall_seconds_samples",
            "state_restoration_seconds_samples",
            "event_seconds_median",
            "wall_seconds_median",
            "state_restoration_seconds_median",
            "updates_per_second_from_event_median",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "includes_G_M_outer_updates",
            "state_restoration_timed",
            "validation",
        },
        f"{context} keys",
    )
    _equal(row.get("mode"), method, f"{context}.mode")
    _equal(row.get("repeats"), repeats, f"{context}.repeats")
    event_raw = _list(
        row.get("event_seconds_samples"),
        f"{context}.event_seconds_samples",
    )
    wall_raw = _list(
        row.get("wall_seconds_samples"),
        f"{context}.wall_seconds_samples",
    )
    _equal(len(event_raw), repeats, f"{context} event sample count")
    _equal(len(wall_raw), repeats, f"{context} wall sample count")
    event = [
        _number(item, f"{context}.event_seconds_samples[{index}]", strictly_positive=True)
        for index, item in enumerate(event_raw)
    ]
    wall = [
        _number(item, f"{context}.wall_seconds_samples[{index}]", strictly_positive=True)
        for index, item in enumerate(wall_raw)
    ]
    restoration_raw = _list(
        row.get("state_restoration_seconds_samples"),
        f"{context}.state_restoration_seconds_samples",
    )
    _equal(len(restoration_raw), repeats, f"{context} restoration sample count")
    restoration = [
        _number(
            item,
            f"{context}.state_restoration_seconds_samples[{index}]",
            strictly_positive=True,
        )
        for index, item in enumerate(restoration_raw)
    ]
    _close(
        row.get("event_seconds_median"),
        statistics.median(event),
        f"{context}.event_seconds_median",
    )
    _close(
        row.get("wall_seconds_median"),
        statistics.median(wall),
        f"{context}.wall_seconds_median",
    )
    _close(
        row.get("state_restoration_seconds_median"),
        statistics.median(restoration),
        f"{context}.state_restoration_seconds_median",
    )
    _close(
        row.get("updates_per_second_from_event_median"),
        1.0 / statistics.median(event),
        f"{context}.updates_per_second_from_event_median",
    )
    for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise AuditError(f"{context}.{key} must be a positive integer")
    _equal(
        row.get("includes_G_M_outer_updates"),
        True,
        f"{context}.includes_G_M_outer_updates",
    )
    _equal(
        row.get("state_restoration_timed"),
        False,
        f"{context}.state_restoration_timed",
    )
    validation_passed = _validate_ladder_validation(
        row.get("validation"),
        rank=rank,
        method=method,
        context=f"{context}.validation",
    )
    return {
        "event_seconds_median": statistics.median(event),
        "wall_seconds_median": statistics.median(wall),
        "event_seconds_samples": event,
        "wall_seconds_samples": wall,
        "state_restoration_seconds_samples": restoration,
        "validation_passed": validation_passed,
    }


def _validate_cross_method_probe(
    value: Any,
    context: str,
) -> dict[str, Any]:
    row = _object(value, context)
    _equal(
        set(row),
        {
            "threshold",
            "passed",
            "seed",
            "count",
            "distribution",
            "factor_identity",
            "operator_identity",
            "factor",
            "operator",
        },
        f"{context} keys",
    )
    _close(row.get("threshold"), 1e-3, f"{context}.threshold")
    _equal(row.get("seed"), 29421, f"{context}.seed")
    _equal(row.get("count"), 16, f"{context}.count")
    _equal(
        row.get("distribution"),
        "NumPy PCG64 Rademacher (+/-1/sqrt(rank))",
        f"{context}.distribution",
    )
    _equal(
        row.get("factor_identity"),
        "Linv_incremental P vs solve(L_fresh, P)",
        f"{context}.factor_identity",
    )
    _equal(
        row.get("operator_identity"),
        "Aw_incremental P vs Aw_fresh P",
        f"{context}.operator_identity",
    )
    diagnostic_values = []
    for family in ("factor", "operator"):
        metrics = _object(row.get(family), f"{context}.{family}")
        _equal(
            set(metrics),
            {
                "per_probe_relative_errors",
                "max_relative_error",
                "median_relative_error",
                "frobenius_relative_error",
            },
            f"{context}.{family} keys",
        )
        raw_errors = _list(
            metrics.get("per_probe_relative_errors"),
            f"{context}.{family}.per_probe_relative_errors",
        )
        _equal(len(raw_errors), 16, f"{context}.{family} error count")
        errors = [
            _number(
                item,
                f"{context}.{family}.per_probe_relative_errors[{index}]",
                minimum=0.0,
            )
            for index, item in enumerate(raw_errors)
        ]
        _close(
            metrics.get("max_relative_error"),
            max(errors),
            f"{context}.{family}.max_relative_error",
        )
        _close(
            metrics.get("median_relative_error"),
            statistics.median(errors),
            f"{context}.{family}.median_relative_error",
        )
        frobenius = _number(
            metrics.get("frobenius_relative_error"),
            f"{context}.{family}.frobenius_relative_error",
            minimum=0.0,
        )
        diagnostic_values.extend((max(errors), frobenius))
    expected_passed = max(diagnostic_values) <= 1e-3
    _equal(row.get("passed"), expected_passed, f"{context}.passed")
    audited = dict(row)
    audited["_validation_passed"] = expected_passed
    return audited


def _validate_ladder(
    path: Path,
    *,
    rank: int,
    manifest_sha256: str,
    trace_sha256: str,
    feature_source_sha256: str,
    expected_rank_scale: float,
) -> tuple[dict[str, Any], tuple[str, str, str, str, str, str]]:
    payload = _load_json(path)
    context = str(path)
    expected_top_level_keys = {
        "schema_version",
        "experiment",
        "protocol_version",
        "campaign_tag",
        "created_utc",
        "rank",
        "repeats_per_method_per_case",
        "case_count",
        "method_order_policy",
        "tile_config",
        "timing_region",
        "feature_manifest",
        "trace",
        "dataset",
        "initial_state",
        "warmup",
        "probe",
        "full_dense_correctness_max_rank",
        "source",
        "hardware",
        "host",
        "platform",
        "python",
        "slurm",
        "cases",
    }
    _equal(set(payload), expected_top_level_keys, f"{context} top-level keys")
    _equal(
        payload.get("schema_version"),
        LADDER_SCHEMA_VERSION,
        f"{context}.schema_version",
    )
    _equal(payload.get("experiment"), LADDER_EXPERIMENT, f"{context}.experiment")
    _equal(
        payload.get("protocol_version"),
        LADDER_PROTOCOL_VERSION,
        f"{context}.protocol_version",
    )
    _equal(payload.get("campaign_tag"), CAMPAIGN_TAG, f"{context}.campaign_tag")
    _equal(payload.get("rank"), rank, f"{context}.rank")
    repeats = LADDER_REPEATS[rank]
    _equal(
        payload.get("repeats_per_method_per_case"),
        repeats,
        f"{context}.repeats_per_method_per_case",
    )
    _equal(
        payload.get("case_count"),
        len(EVALUATION_TRANSITIONS),
        f"{context}.case_count",
    )
    _equal(
        payload.get("method_order_policy"),
        (
            "zero-based even case indices run inverse then fresh_refact; "
            "odd case indices reverse that order"
        ),
        f"{context}.method_order_policy",
    )
    expected_tile = 8 if rank <= 8_192 else 16 if rank == 16_384 else 32
    _equal(
        payload.get("tile_config"),
        {
            "tile_rows": expected_tile,
            "tile_cols": expected_tile,
            "setup_num_warps": 4,
            "sweep_num_warps": 1,
        },
        f"{context}.tile_config",
    )
    _equal(payload.get("timing_region"), TIMING_REGION, f"{context}.timing_region")

    feature_manifest = _object(
        payload.get("feature_manifest"),
        f"{context}.feature_manifest",
    )
    _equal(
        feature_manifest.get("sha256"),
        manifest_sha256,
        f"{context}.feature_manifest.sha256",
    )
    _equal(
        feature_manifest.get("schema_version"),
        FEATURE_SCHEMA_VERSION,
        f"{context}.feature_manifest.schema_version",
    )
    _equal(
        feature_manifest.get("producer_source_sha256"),
        feature_source_sha256,
        f"{context}.feature_manifest.producer_source_sha256",
    )
    _validate_trace(
        payload.get("trace"),
        expected_sha256=trace_sha256,
        context=f"{context}.trace",
    )
    trace = _object(payload.get("trace"), f"{context}.trace")
    _close(
        trace.get("rank_scale"),
        expected_rank_scale,
        f"{context}.trace.rank_scale",
        rel_tol=2e-12,
        abs_tol=0.0,
    )
    _equal(
        trace.get("consumer_operation"),
        "trace[t, :rank] / rank_scale",
        f"{context}.trace.consumer_operation",
    )
    _equal(
        trace.get("consumer_scaling_dtype"),
        "float32",
        f"{context}.trace.consumer_scaling_dtype",
    )
    _validate_dataset(payload.get("dataset"), f"{context}.dataset")
    source_context = _validate_ladder_source(
        payload,
        context,
        feature_source_sha256,
    )
    hardware_context = _validate_hardware(
        payload.get("hardware"),
        f"{context}.hardware",
    )
    hardware = _object(payload.get("hardware"), f"{context}.hardware")
    _equal(
        set(hardware),
        {
            "gpu",
            "gpu_total_memory",
            "compute_capability",
            "visible_device_count",
            "torch",
            "cuda",
            "cudnn",
            "float32_matmul_precision",
            "allow_tf32_matmul",
        },
        f"{context}.hardware keys",
    )
    _equal(
        hardware.get("compute_capability"),
        [9, 0],
        f"{context}.hardware.compute_capability",
    )
    _equal(
        hardware.get("visible_device_count"),
        1,
        f"{context}.hardware.visible_device_count",
    )
    _equal(
        hardware.get("float32_matmul_precision"),
        "highest",
        f"{context}.hardware.float32_matmul_precision",
    )
    _equal(
        hardware.get("allow_tf32_matmul"),
        False,
        f"{context}.hardware.allow_tf32_matmul",
    )
    if isinstance(hardware.get("cudnn"), bool) or not isinstance(
        hardware.get("cudnn"), int
    ):
        raise AuditError(f"{context}.hardware.cudnn must be an integer")
    host = payload.get("host")
    if not isinstance(host, str) or not host:
        raise AuditError(f"{context}.host must be a nonempty string")
    slurm = _object(payload.get("slurm"), f"{context}.slurm")
    _equal(
        set(slurm),
        {
            "job_id",
            "array_job_id",
            "array_task_id",
            "cpus_per_task",
            "job_partition",
        },
        f"{context}.slurm keys",
    )
    job_id = slurm.get("job_id")
    if not isinstance(job_id, str) or not job_id.isdigit():
        raise AuditError(f"{context}.slurm.job_id must be a numeric string")
    _equal(slurm.get("array_job_id"), None, f"{context}.slurm.array_job_id")
    _equal(slurm.get("array_task_id"), None, f"{context}.slurm.array_task_id")
    _equal(slurm.get("cpus_per_task"), "4", f"{context}.slurm.cpus_per_task")
    _equal(slurm.get("job_partition"), "preempt", f"{context}.slurm.job_partition")

    initial = _object(payload.get("initial_state"), f"{context}.initial_state")
    _equal(
        set(initial),
        {
            "ridge",
            "transitions",
            "definition_G",
            "definition_M",
            "condition_number_method",
            "small_gram_rows",
            "small_gram_columns",
            "lambda_min_exact",
            "lambda_max_from_small_gram",
            "condition_number_exact",
            "small_gram_min_eigenvalue",
            "small_gram_max_eigenvalue",
            "construction_wall_seconds",
            "inverse_factor_host_backup_bytes",
            "operator_host_backup_bytes",
            "factor_storage_stride",
            "cholesky_output_stride",
            "operator_stride",
        },
        f"{context}.initial_state keys",
    )
    _close(initial.get("ridge"), RIDGE, f"{context}.initial_state.ridge")
    _equal(
        initial.get("transitions"),
        list(INITIAL_TRANSITIONS),
        f"{context}.initial_state.transitions",
    )
    _equal(
        initial.get("definition_G"),
        "I + sum_{t=0}^{63} z_t z_t^T",
        f"{context}.initial_state.definition_G",
    )
    _equal(
        initial.get("definition_M"),
        "sum_{t=0}^{63} z_t w_t^T",
        f"{context}.initial_state.definition_M",
    )
    _equal(
        initial.get("small_gram_rows"),
        64,
        f"{context}.initial_state.small_gram_rows",
    )
    _equal(
        initial.get("small_gram_columns"),
        rank,
        f"{context}.initial_state.small_gram_columns",
    )
    _close(
        initial.get("lambda_min_exact"),
        RIDGE,
        f"{context}.initial_state.lambda_min_exact",
    )
    lambda_max = _number(
        initial.get("lambda_max_from_small_gram"),
        f"{context}.initial_state.lambda_max_from_small_gram",
        strictly_positive=True,
    )
    _close(
        initial.get("condition_number_exact"),
        lambda_max / RIDGE,
        f"{context}.initial_state.condition_number_exact",
    )
    for key in (
        "small_gram_min_eigenvalue",
        "small_gram_max_eigenvalue",
    ):
        _number(initial.get(key), f"{context}.initial_state.{key}")
    _number(
        initial.get("construction_wall_seconds"),
        f"{context}.initial_state.construction_wall_seconds",
        strictly_positive=True,
    )
    expected_matrix_bytes = rank * rank * 4
    for key in (
        "inverse_factor_host_backup_bytes",
        "operator_host_backup_bytes",
    ):
        _equal(
            initial.get(key),
            expected_matrix_bytes,
            f"{context}.initial_state.{key}",
        )
    _equal(
        initial.get("factor_storage_stride"),
        [rank, 1],
        f"{context}.initial_state.factor_storage_stride",
    )
    _equal(
        initial.get("cholesky_output_stride"),
        [1, rank],
        f"{context}.initial_state.cholesky_output_stride",
    )
    _equal(
        initial.get("operator_stride"),
        [rank, 1],
        f"{context}.initial_state.operator_stride",
    )

    warmup = _object(payload.get("warmup"), f"{context}.warmup")
    _equal(
        warmup.get("transition_index"),
        WARMUP_TRANSITION,
        f"{context}.warmup.transition_index",
    )
    _equal(
        warmup.get("snapshot_pair"),
        [WARMUP_TRANSITION, WARMUP_TRANSITION + 1],
        f"{context}.warmup.snapshot_pair",
    )
    _equal(
        warmup.get("excluded_from_timing"),
        True,
        f"{context}.warmup.excluded_from_timing",
    )
    _equal(
        warmup.get("method_order"),
        ["inverse", "fresh_refact"],
        f"{context}.warmup.method_order",
    )
    warmup_methods = _object(
        warmup.get("methods"),
        f"{context}.warmup.methods",
    )
    _equal(set(warmup_methods), set(LADDER_METHODS), f"{context}.warmup methods")
    for method in LADDER_METHODS:
        method_warmup = _object(
            warmup_methods[method],
            f"{context}.warmup.methods.{method}",
        )
        _equal(
            set(method_warmup),
            {"execution_wall_seconds"},
            f"{context}.warmup.methods.{method} keys",
        )
        _number(
            method_warmup.get("execution_wall_seconds"),
            f"{context}.warmup.methods.{method}.execution_wall_seconds",
            strictly_positive=True,
        )

    probe = _object(payload.get("probe"), f"{context}.probe")
    _equal(
        probe,
        {
            "seed": 29421,
            "count": 16,
            "distribution": "NumPy PCG64 Rademacher (+/-1/sqrt(rank))",
            "threshold": 1e-3,
            "cross_method_comparison_required": True,
            "cross_method_factor_identity": (
                "Linv_incremental P vs solve(L_fresh, P)"
            ),
            "cross_method_operator_identity": (
                "Aw_incremental P vs Aw_fresh P"
            ),
        },
        f"{context}.probe",
    )
    _equal(
        payload.get("full_dense_correctness_max_rank"),
        32_768,
        f"{context}.full_dense_correctness_max_rank",
    )

    cases = _list(payload.get("cases"), f"{context}.cases")
    _equal(len(cases), len(EVALUATION_TRANSITIONS), f"{context} case count")
    audited_cases: list[dict[str, Any]] = []
    for index, (case_value, transition) in enumerate(
        zip(cases, EVALUATION_TRANSITIONS)
    ):
        case_context = f"{context}.cases[{index}]"
        case = _object(case_value, case_context)
        _equal(
            set(case),
            {
                "case_index",
                "transition_index",
                "snapshot_pair",
                "z_l2_norm",
                "w_l2_norm",
                "small_gram_rows",
                "small_gram_columns",
                "lambda_min_exact",
                "lambda_max_from_small_gram",
                "condition_number_exact",
                "small_gram_min_eigenvalue",
                "small_gram_max_eigenvalue",
                "method_order",
                "methods",
                "cross_method_probe_comparison",
            },
            f"{case_context} keys",
        )
        _equal(case.get("case_index"), index, f"{case_context}.case_index")
        _equal(
            case.get("transition_index"),
            transition,
            f"{case_context}.transition_index",
        )
        _number(
            case.get("condition_number_exact"),
            f"{case_context}.condition_number_exact",
            strictly_positive=True,
        )
        _equal(
            case.get("snapshot_pair"),
            [transition, transition + 1],
            f"{case_context}.snapshot_pair",
        )
        for key in ("z_l2_norm", "w_l2_norm"):
            _number(
                case.get(key),
                f"{case_context}.{key}",
                strictly_positive=True,
            )
        _equal(
            case.get("small_gram_rows"),
            65,
            f"{case_context}.small_gram_rows",
        )
        _equal(
            case.get("small_gram_columns"),
            rank,
            f"{case_context}.small_gram_columns",
        )
        _close(
            case.get("lambda_min_exact"),
            RIDGE,
            f"{case_context}.lambda_min_exact",
        )
        case_lambda_max = _number(
            case.get("lambda_max_from_small_gram"),
            f"{case_context}.lambda_max_from_small_gram",
            strictly_positive=True,
        )
        _close(
            case.get("condition_number_exact"),
            case_lambda_max / RIDGE,
            f"{case_context}.condition_number_exact",
        )
        for key in (
            "small_gram_min_eigenvalue",
            "small_gram_max_eigenvalue",
        ):
            _number(case.get(key), f"{case_context}.{key}")
        expected_order = (
            ["inverse", "fresh_refact"]
            if index % 2 == 0
            else ["fresh_refact", "inverse"]
        )
        _equal(
            case.get("method_order"),
            expected_order,
            f"{case_context}.method_order",
        )
        methods = _object(case.get("methods"), f"{case_context}.methods")
        _equal(set(methods), set(LADDER_METHODS), f"{case_context}.methods keys")
        audited_methods = {
            method: _validate_ladder_method(
                methods[method],
                rank=rank,
                repeats=repeats,
                method=method,
                context=f"{case_context}.methods.{method}",
            )
            for method in LADDER_METHODS
        }
        cross_method = _validate_cross_method_probe(
            case.get("cross_method_probe_comparison"),
            f"{case_context}.cross_method_probe_comparison",
        )
        headline_valid = (
            all(
                audited_methods[method]["validation_passed"]
                for method in LADDER_METHODS
            )
            and cross_method["_validation_passed"]
        )
        audited_cases.append(
            {
                "case_index": index,
                "transition_index": transition,
                "condition_number_exact": float(
                    case["condition_number_exact"]
                ),
                "methods": audited_methods,
                "cross_method_probe_comparison": cross_method,
                "headline_valid": headline_valid,
            }
        )
    return (
        {
            "rank": rank,
            "repeats": repeats,
            "cases": audited_cases,
            "tile_config": payload.get("tile_config"),
        },
        (
            source_context[0],
            source_context[1],
            source_context[2],
            "|".join(str(value) for value in hardware_context),
            host,
            job_id,
        ),
    )


def _validate_stability_source(
    payload: dict[str, Any],
    context: str,
) -> tuple[str, str, str]:
    implementation = _object(
        payload.get("implementation"),
        f"{context}.implementation",
    )
    commit = _commit(
        implementation.get("git_commit"),
        f"{context}.implementation.git_commit",
    )
    _equal(
        implementation.get("git_dirty"),
        False,
        f"{context}.implementation.git_dirty",
    )
    hashes = _object(
        implementation.get("source_hashes"),
        f"{context}.implementation.source_hashes",
    )
    expected_keys = {
        "rebuttal_experiments/cuda_blackhole_stability.py",
        "student_pkg/doublesided_cholesky.py",
        "rebuttal_experiments/blackhole_features.py",
    }
    _equal(set(hashes), expected_keys, f"{context}.source_hashes keys")
    driver_sha = _sha(
        hashes["rebuttal_experiments/cuda_blackhole_stability.py"],
        f"{context} stability source SHA-256",
    )
    kernel_sha = _sha(
        hashes["student_pkg/doublesided_cholesky.py"],
        f"{context} kernel source SHA-256",
    )
    _equal(
        driver_sha,
        _sha256(STABILITY_SOURCE),
        f"{context} current stability source SHA-256",
    )
    _equal(
        kernel_sha,
        _sha256(STUDENT_SOURCE),
        f"{context} current kernel source SHA-256",
    )
    feature_sha = _sha(
        hashes["rebuttal_experiments/blackhole_features.py"],
        f"{context} feature source SHA-256",
    )
    _equal(
        feature_sha,
        _sha256(FEATURE_SOURCE),
        f"{context} current feature source SHA-256",
    )
    return commit, driver_sha, kernel_sha


def _validate_stability_samples(
    value: Any,
    *,
    method: str,
    context: str,
) -> list[dict[str, Any]]:
    samples = _list(value, context)
    _equal(len(samples), len(STABILITY_TRANSITIONS), f"{context} sample count")
    audited = []
    for index, (raw, transition) in enumerate(
        zip(samples, STABILITY_TRANSITIONS)
    ):
        row_context = f"{context}[{index}]"
        row = _object(raw, row_context)
        _equal(row.get("sample_index"), index, f"{row_context}.sample_index")
        _equal(row.get("step"), index + 1, f"{row_context}.step")
        _equal(row.get("transition"), transition, f"{row_context}.transition")
        expected_refreshed = (
            method == "inverse_refresh_q50"
            and (index + 1) % REFRESH_INTERVAL == 0
        )
        _equal(
            row.get("refreshed"),
            expected_refreshed,
            f"{row_context}.refreshed",
        )
        event = _number(
            row.get("event_seconds"),
            f"{row_context}.event_seconds",
            strictly_positive=True,
        )
        wall = _number(
            row.get("wall_seconds"),
            f"{row_context}.wall_seconds",
            strictly_positive=True,
        )
        update_event = _number(
            row.get("update_event_seconds"),
            f"{row_context}.update_event_seconds",
            strictly_positive=True,
        )
        update_wall = _number(
            row.get("update_wall_seconds"),
            f"{row_context}.update_wall_seconds",
            strictly_positive=True,
        )
        refresh_event = _number(
            row.get("refresh_event_seconds"),
            f"{row_context}.refresh_event_seconds",
            minimum=0.0,
        )
        refresh_wall = _number(
            row.get("refresh_wall_seconds"),
            f"{row_context}.refresh_wall_seconds",
            minimum=0.0,
        )
        if expected_refreshed:
            if refresh_event <= 0 or refresh_wall <= 0:
                raise AuditError(f"{row_context}: refresh timing must be positive")
        else:
            _close(refresh_event, 0.0, f"{row_context}.refresh_event_seconds")
            _close(refresh_wall, 0.0, f"{row_context}.refresh_wall_seconds")
        _close(
            event,
            update_event + refresh_event,
            f"{row_context}.event_seconds decomposition",
        )
        _close(
            wall,
            update_wall + refresh_wall,
            f"{row_context}.wall_seconds decomposition",
        )
        audited.append(
            {
                "sample_index": index,
                "step": index + 1,
                "transition": transition,
                "event_seconds": event,
                "wall_seconds": wall,
                "update_event_seconds": update_event,
                "update_wall_seconds": update_wall,
                "refresh_event_seconds": refresh_event,
                "refresh_wall_seconds": refresh_wall,
                "refreshed": expected_refreshed,
            }
        )
    return audited


def _validate_stability_checkpoints(
    value: Any,
    *,
    method: str,
    context: str,
) -> list[dict[str, Any]]:
    rows = _list(value, context)
    _equal(len(rows), len(CHECKPOINT_STEPS), f"{context} checkpoint count")
    audited = []
    for index, (raw, step) in enumerate(zip(rows, CHECKPOINT_STEPS)):
        row_context = f"{context}[{index}]"
        row = _object(raw, row_context)
        transition = STABILITY_TRANSITIONS[step - 1]
        _equal(row.get("step"), step, f"{row_context}.step")
        _equal(row.get("transition"), transition, f"{row_context}.transition")
        expected_phase = (
            "pre_refresh"
            if method == "inverse_refresh_q50" and step % REFRESH_INTERVAL == 0
            else "post_step"
        )
        _equal(
            row.get("measurement_phase"),
            expected_phase,
            f"{row_context}.measurement_phase",
        )
        _equal(row.get("finite_full"), True, f"{row_context}.finite_full")
        for key in ("factor_probe_residual", "operator_probe_residual"):
            _number(row.get(key), f"{row_context}.{key}", minimum=0.0)
        if method != "fresh_refact":
            expected_age = (
                step
                if method == "inverse_no_refresh"
                else step % REFRESH_INTERVAL or REFRESH_INTERVAL
            )
            _equal(
                row.get("age_since_refresh"),
                expected_age,
                f"{row_context}.age_since_refresh",
            )
            for key in (
                "Linv_relative_error",
                "Aw_relative_error",
                "Linv_probe_relative_error",
                "Aw_probe_relative_error",
            ):
                _number(row.get(key), f"{row_context}.{key}", minimum=0.0)
            if expected_phase == "pre_refresh":
                _equal(
                    row.get("post_refresh_finite_full"),
                    True,
                    f"{row_context}.post_refresh_finite_full",
                )
                for key in (
                    "post_refresh_Linv_relative_error",
                    "post_refresh_Aw_relative_error",
                ):
                    _number(
                        row.get(key),
                        f"{row_context}.{key}",
                        minimum=0.0,
                    )
        audited.append(row)
    return audited


def _validate_stability_method(
    value: Any,
    *,
    rank: int,
    method: str,
    context: str,
) -> dict[str, Any]:
    row = _object(value, context)
    _equal(row.get("method"), method, f"{context}.method")
    expected_interval = (
        1
        if method == "fresh_refact"
        else REFRESH_INTERVAL
        if method == "inverse_refresh_q50"
        else 0
    )
    expected_count = (
        len(STABILITY_TRANSITIONS)
        if method == "fresh_refact"
        else len(STABILITY_TRANSITIONS) // REFRESH_INTERVAL
        if method == "inverse_refresh_q50"
        else 0
    )
    _equal(
        row.get("refresh_interval"),
        expected_interval,
        f"{context}.refresh_interval",
    )
    _equal(row.get("refresh_count"), expected_count, f"{context}.refresh_count")
    _equal(
        row.get("includes_G_M_outer_updates"),
        True,
        f"{context}.includes_G_M_outer_updates",
    )
    _equal(
        row.get("refresh_in_timed_step"),
        True,
        f"{context}.refresh_in_timed_step",
    )
    _equal(row.get("finite_full"), True, f"{context}.finite_full")
    _close(
        row.get("correctness_threshold"),
        1e-3,
        f"{context}.correctness_threshold",
    )
    if method != "fresh_refact":
        expected_tile = 8 if rank == 8_192 else 32
        _equal(row.get("tile_rows"), expected_tile, f"{context}.tile_rows")
        _equal(row.get("tile_cols"), expected_tile, f"{context}.tile_cols")
        _equal(
            row.get("setup_num_warps"),
            4,
            f"{context}.setup_num_warps",
        )
        _equal(
            row.get("sweep_num_warps"),
            1,
            f"{context}.sweep_num_warps",
        )
        workspace_bytes = row.get("workspace_bytes")
        if (
            isinstance(workspace_bytes, bool)
            or not isinstance(workspace_bytes, int)
            or workspace_bytes <= 0
        ):
            raise AuditError(f"{context}.workspace_bytes must be positive")
    samples = _validate_stability_samples(
        row.get("samples"),
        method=method,
        context=f"{context}.samples",
    )
    summary = _object(row.get("summary"), f"{context}.summary")
    event = [sample["event_seconds"] for sample in samples]
    wall = [sample["wall_seconds"] for sample in samples]
    recomputed = {
        "event_seconds_mean": statistics.fmean(event),
        "event_seconds_sample_sd": statistics.stdev(event),
        "event_seconds_median": statistics.median(event),
        "wall_seconds_mean": statistics.fmean(wall),
        "wall_seconds_sample_sd": statistics.stdev(wall),
        "wall_seconds_median": statistics.median(wall),
        "event_seconds_total": math.fsum(event),
        "wall_seconds_total": math.fsum(wall),
    }
    _equal(set(summary), set(recomputed), f"{context}.summary keys")
    for key, expected in recomputed.items():
        _close(summary.get(key), expected, f"{context}.summary.{key}")
    checkpoints = _validate_stability_checkpoints(
        row.get("checkpoints"),
        method=method,
        context=f"{context}.checkpoints",
    )
    diagnostic_keys = ("factor_probe_residual", "operator_probe_residual")
    if method != "fresh_refact":
        diagnostic_keys += (
            "Linv_relative_error",
            "Aw_relative_error",
            "Linv_probe_relative_error",
            "Aw_probe_relative_error",
        )
    diagnostics = [
        float(checkpoint[key])
        for checkpoint in checkpoints
        for key in diagnostic_keys
    ]
    diagnostics.extend(
        float(checkpoint[key])
        for checkpoint in checkpoints
        for key in (
            "post_refresh_Linv_relative_error",
            "post_refresh_Aw_relative_error",
        )
        if key in checkpoint
    )
    expected_headline_valid = max(diagnostics) <= 1e-3
    _equal(
        row.get("headline_valid"),
        expected_headline_valid,
        f"{context}.headline_valid",
    )
    return {
        "method": method,
        "samples": samples,
        "checkpoints": checkpoints,
        "headline_valid": expected_headline_valid,
        "maximum_diagnostic_error": max(diagnostics),
    }


def _validate_stability(
    path: Path,
    *,
    rank: int,
    manifest_sha256: str,
    trace_sha256: str,
    expected_rank_scale: float,
) -> tuple[dict[str, Any], tuple[str, str, str, str, str]]:
    payload = _load_json(path)
    context = str(path)
    expected_top_level_keys = {
        "schema_version",
        "experiment",
        "campaign_tag",
        "rank",
        "dtype",
        "ridge",
        "initial_transitions",
        "warmup_transition",
        "stability_transitions",
        "checkpoint_steps",
        "method_order",
        "rank_scale",
        "rank_scale_recomputed",
        "feature_manifest",
        "trace",
        "dataset",
        "implementation",
        "hardware",
        "timing_contract",
        "probe_contract",
        "peak_allocated_bytes_including_harness",
        "methods",
    }
    _equal(set(payload), expected_top_level_keys, f"{context} top-level keys")
    _equal(
        payload.get("schema_version"),
        STABILITY_SCHEMA_VERSION,
        f"{context}.schema_version",
    )
    _equal(
        payload.get("experiment"),
        STABILITY_EXPERIMENT,
        f"{context}.experiment",
    )
    _equal(payload.get("campaign_tag"), CAMPAIGN_TAG, f"{context}.campaign_tag")
    _equal(payload.get("rank"), rank, f"{context}.rank")
    _equal(payload.get("dtype"), "float32", f"{context}.dtype")
    _close(payload.get("ridge"), RIDGE, f"{context}.ridge")
    _equal(
        payload.get("initial_transitions"),
        list(INITIAL_TRANSITIONS),
        f"{context}.initial_transitions",
    )
    _equal(
        payload.get("warmup_transition"),
        WARMUP_TRANSITION,
        f"{context}.warmup_transition",
    )
    _equal(
        payload.get("stability_transitions"),
        list(STABILITY_TRANSITIONS),
        f"{context}.stability_transitions",
    )
    _equal(
        payload.get("checkpoint_steps"),
        list(CHECKPOINT_STEPS),
        f"{context}.checkpoint_steps",
    )
    _equal(
        payload.get("method_order"),
        list(STABILITY_METHOD_ORDER),
        f"{context}.method_order",
    )
    scale = _number(
        payload.get("rank_scale"),
        f"{context}.rank_scale",
        strictly_positive=True,
    )
    _close(
        scale,
        expected_rank_scale,
        f"{context}.rank_scale versus manifest",
        rel_tol=2e-12,
        abs_tol=0.0,
    )
    _close(
        payload.get("rank_scale_recomputed"),
        scale,
        f"{context}.rank_scale_recomputed",
        rel_tol=2e-12,
        abs_tol=0.0,
    )
    feature_manifest = _object(
        payload.get("feature_manifest"),
        f"{context}.feature_manifest",
    )
    _equal(
        feature_manifest.get("sha256"),
        manifest_sha256,
        f"{context}.feature_manifest.sha256",
    )
    _validate_trace(
        payload.get("trace"),
        expected_sha256=trace_sha256,
        context=f"{context}.trace",
    )
    _validate_dataset(payload.get("dataset"), f"{context}.dataset")
    _equal(
        payload.get("timing_contract"),
        {
            "unit": "seconds",
            "samples_are_per_transition": True,
            "G_M_outer_updates_included": True,
            "q50_refresh_included": True,
            "diagnostics_excluded": True,
            "cuda_event_and_synchronized_wall_samples_preserved": True,
        },
        f"{context}.timing_contract",
    )
    _equal(
        payload.get("probe_contract"),
        {
            "seed": 29421,
            "count": 16,
            "distribution": "NumPy PCG64 Rademacher (+/-1/sqrt(rank))",
        },
        f"{context}.probe_contract",
    )
    peak_bytes = payload.get("peak_allocated_bytes_including_harness")
    if (
        isinstance(peak_bytes, bool)
        or not isinstance(peak_bytes, int)
        or peak_bytes <= 0
    ):
        raise AuditError(
            f"{context}.peak_allocated_bytes_including_harness must be positive"
        )
    source_context = _validate_stability_source(payload, context)
    hardware_context = _validate_hardware(
        payload.get("hardware"),
        f"{context}.hardware",
    )
    hardware = _object(payload.get("hardware"), f"{context}.hardware")
    for key in ("python", "numpy", "triton"):
        if not isinstance(hardware.get(key), str) or not hardware[key]:
            raise AuditError(f"{context}.hardware.{key} must be nonempty")
    slurm = _object(hardware.get("slurm"), f"{context}.hardware.slurm")
    _equal(
        set(slurm),
        {
            "job_id",
            "array_job_id",
            "array_task_id",
            "cpus_per_task",
            "job_partition",
        },
        f"{context}.hardware.slurm keys",
    )
    for key in ("job_id", "array_job_id", "array_task_id"):
        value = slurm.get(key)
        if not isinstance(value, str) or not value.isdigit():
            raise AuditError(f"{context}.hardware.slurm.{key} must be numeric")
    expected_task = STABILITY_RANKS.index(rank)
    _equal(
        slurm.get("array_task_id"),
        str(expected_task),
        f"{context}.hardware.slurm.array_task_id",
    )
    _equal(
        slurm.get("cpus_per_task"),
        "4",
        f"{context}.hardware.slurm.cpus_per_task",
    )
    _equal(
        slurm.get("job_partition"),
        "preempt",
        f"{context}.hardware.slurm.job_partition",
    )
    methods_raw = _list(payload.get("methods"), f"{context}.methods")
    _equal(len(methods_raw), len(STABILITY_METHOD_ORDER), f"{context} method count")
    _equal(
        [row.get("method") if isinstance(row, dict) else None for row in methods_raw],
        list(STABILITY_METHOD_ORDER),
        f"{context} method order",
    )
    methods = {
        method: _validate_stability_method(
            value,
            rank=rank,
            method=method,
            context=f"{context}.methods[{index}]",
        )
        for index, (method, value) in enumerate(
            zip(STABILITY_METHOD_ORDER, methods_raw)
        )
    }
    return (
        {"rank": rank, "rank_scale": scale, "methods": methods},
        (
            source_context[0],
            source_context[1],
            source_context[2],
            "|".join(str(value) for value in hardware_context),
            str(slurm["array_job_id"]),
        ),
    )


def _aggregate_ladder(record: dict[str, Any]) -> dict[str, Any]:
    cases = record["cases"]
    inverse_event = [
        case["methods"]["inverse"]["event_seconds_median"] for case in cases
    ]
    fresh_event = [
        case["methods"]["fresh_refact"]["event_seconds_median"] for case in cases
    ]
    inverse_wall = [
        case["methods"]["inverse"]["wall_seconds_median"] for case in cases
    ]
    fresh_wall = [
        case["methods"]["fresh_refact"]["wall_seconds_median"] for case in cases
    ]
    headline_valid = all(case["headline_valid"] for case in cases)
    result = {
        "rank": record["rank"],
        "repeats_per_case": record["repeats"],
        "n_real_transition_cases": len(cases),
        "inverse_event_seconds": _sample_stats(inverse_event),
        "fresh_event_seconds": _sample_stats(fresh_event),
        "inverse_wall_seconds": _sample_stats(inverse_wall),
        "fresh_wall_seconds": _sample_stats(fresh_wall),
        "headline_valid": headline_valid,
        "invalid_transition_indices": [
            case["transition_index"]
            for case in cases
            if not case["headline_valid"]
        ],
        "condition_numbers": [
            case["condition_number_exact"] for case in cases
        ],
        "cross_method_probe_max_relative_errors": [
            {
                "transition_index": case["transition_index"],
                "factor": case["cross_method_probe_comparison"]["factor"][
                    "max_relative_error"
                ],
                "operator": case["cross_method_probe_comparison"]["operator"][
                    "max_relative_error"
                ],
            }
            for case in cases
        ],
        "transition_indices": [
            case["transition_index"] for case in cases
        ],
    }
    if headline_valid:
        result.update(
            {
                "paired_event_speedup_fresh_over_inverse": _sample_stats(
                    fresh / inverse
                    for fresh, inverse in zip(fresh_event, inverse_event)
                ),
                "paired_wall_speedup_fresh_over_inverse": _sample_stats(
                    fresh / inverse
                    for fresh, inverse in zip(fresh_wall, inverse_wall)
                ),
                "paired_event_seconds_saved": _sample_stats(
                    fresh - inverse
                    for fresh, inverse in zip(fresh_event, inverse_event)
                ),
            }
        )
    else:
        result["headline_withheld_reason"] = (
            "one or more direct correctness comparisons exceed the frozen "
            "1e-3 threshold"
        )
    return result


def _aggregate_stability(record: dict[str, Any]) -> dict[str, Any]:
    methods = record["methods"]
    fresh = methods["fresh_refact"]["samples"]
    result: dict[str, Any] = {
        "rank": record["rank"],
        "rank_scale": record["rank_scale"],
        "n_sequential_transitions": len(fresh),
        "transition_indices": [row["transition"] for row in fresh],
        "methods": {},
        "comparisons": {},
        "checkpoints": {
            method: methods[method]["checkpoints"]
            for method in STABILITY_METHOD_ORDER
        },
    }
    for method in STABILITY_METHOD_ORDER:
        samples = methods[method]["samples"]
        result["methods"][method] = {
            "headline_valid": methods[method]["headline_valid"],
            "maximum_diagnostic_error": methods[method][
                "maximum_diagnostic_error"
            ],
            "event_seconds": _sample_stats(
                row["event_seconds"] for row in samples
            ),
            "wall_seconds": _sample_stats(
                row["wall_seconds"] for row in samples
            ),
            "event_seconds_total": math.fsum(
                row["event_seconds"] for row in samples
            ),
            "wall_seconds_total": math.fsum(
                row["wall_seconds"] for row in samples
            ),
            "refresh_event_seconds_total": math.fsum(
                row["refresh_event_seconds"] for row in samples
            ),
            "refresh_wall_seconds_total": math.fsum(
                row["refresh_wall_seconds"] for row in samples
            ),
        }
    for method in ("inverse_no_refresh", "inverse_refresh_q50"):
        comparison_key = f"fresh_over_{method}"
        headline_eligible = (
            methods["fresh_refact"]["headline_valid"]
            and methods[method]["headline_valid"]
        )
        if not headline_eligible:
            result["comparisons"][comparison_key] = {
                "headline_eligible": False,
                "withheld_reason": (
                    "correctness diagnostics exceed the frozen 1e-3 threshold"
                ),
            }
            continue
        inverse = methods[method]["samples"]
        event_speedups = [
            fresh_row["event_seconds"] / inverse_row["event_seconds"]
            for fresh_row, inverse_row in zip(fresh, inverse)
        ]
        wall_speedups = [
            fresh_row["wall_seconds"] / inverse_row["wall_seconds"]
            for fresh_row, inverse_row in zip(fresh, inverse)
        ]
        result["comparisons"][comparison_key] = {
            "headline_eligible": True,
            "paired_event_speedup": _sample_stats(event_speedups),
            "paired_wall_speedup": _sample_stats(wall_speedups),
            "paired_event_seconds_saved": _sample_stats(
                fresh_row["event_seconds"] - inverse_row["event_seconds"]
                for fresh_row, inverse_row in zip(fresh, inverse)
            ),
            "ratio_of_event_totals": (
                result["methods"]["fresh_refact"]["event_seconds_total"]
                / result["methods"][method]["event_seconds_total"]
            ),
            "ratio_of_wall_totals": (
                result["methods"]["fresh_refact"]["wall_seconds_total"]
                / result["methods"][method]["wall_seconds_total"]
            ),
        }
    return result


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Frozen black-hole campaign",
        "",
        (
            "All values below were produced only after the strict provenance, "
            "completeness, ordering, hardware, and source-hash audit passed."
        ),
        "",
        "## Isolated real-transition rank ladder",
        "",
        (
            "| Rank | Inverse event s (mean ± SD) | Fresh event s "
            "(mean ± SD) | Paired speedup (mean ± SD) | 95% t CI |"
        ),
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["ladder"]:
        inverse = row["inverse_event_seconds"]
        fresh = row["fresh_event_seconds"]
        if not row["headline_valid"]:
            lines.append(
                f"| {row['rank']} | {inverse['mean']:.6g} ± "
                f"{inverse['sample_sd']:.3g} | {fresh['mean']:.6g} ± "
                f"{fresh['sample_sd']:.3g} | withheld | withheld |"
            )
            continue
        speedup = row["paired_event_speedup_fresh_over_inverse"]
        lines.append(
            f"| {row['rank']} | {inverse['mean']:.6g} ± "
            f"{inverse['sample_sd']:.3g} | {fresh['mean']:.6g} ± "
            f"{fresh['sample_sd']:.3g} | {speedup['mean']:.3f} ± "
            f"{speedup['sample_sd']:.3f} | "
            f"[{speedup['ci95_low']:.3f}, {speedup['ci95_high']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Sequential real-transition stability",
            "",
            (
                "| Rank | Method | Event s/step (mean ± SD) | "
                "Ratio of totals vs fresh | Paired speedup 95% t CI |"
            ),
            "|---:|:---|---:|---:|---:|",
        ]
    )
    for row in summary["stability"]:
        for method in ("inverse_no_refresh", "inverse_refresh_q50"):
            timing = row["methods"][method]["event_seconds"]
            comparison = row["comparisons"][f"fresh_over_{method}"]
            if not comparison["headline_eligible"]:
                lines.append(
                    f"| {row['rank']} | {method} | "
                    f"{timing['mean']:.6g} ± {timing['sample_sd']:.3g} | "
                    "withheld | withheld |"
                )
                continue
            speedup = comparison["paired_event_speedup"]
            lines.append(
                f"| {row['rank']} | {method} | {timing['mean']:.6g} ± "
                f"{timing['sample_sd']:.3g} | "
                f"{comparison['ratio_of_event_totals']:.3f} | "
                f"[{speedup['ci95_low']:.3f}, {speedup['ci95_high']:.3f}] |"
            )
    lines.append("")
    return "\n".join(lines)


def _expected_artifact_names() -> set[str]:
    names = {"feature_manifest.json"}
    names.update(f"ladder_r{rank}.json" for rank in LADDER_RANKS)
    names.update(f"stability_r{rank}.json" for rank in STABILITY_RANKS)
    return names


def _validate_artifact_names(observed: set[str]) -> None:
    _equal(observed, _expected_artifact_names(), "campaign JSON artifact set")


def _validate_campaign_contexts(
    ladder_contexts: list[tuple[str, str, str, str, str, str]],
    stability_contexts: list[tuple[str, str, str, str, str]],
) -> dict[str, str]:
    commits = {context[0] for context in ladder_contexts + stability_contexts}
    _equal(len(commits), 1, "implementation git commit count")
    ladder_driver_hashes = {context[1] for context in ladder_contexts}
    stability_driver_hashes = {context[1] for context in stability_contexts}
    kernel_hashes = {
        context[2] for context in ladder_contexts + stability_contexts
    }
    environments = {
        context[3] for context in ladder_contexts + stability_contexts
    }
    ladder_hosts = {context[4] for context in ladder_contexts}
    ladder_jobs = {context[5] for context in ladder_contexts}
    stability_array_jobs = {context[4] for context in stability_contexts}
    _equal(len(ladder_driver_hashes), 1, "ladder driver hash count")
    _equal(len(stability_driver_hashes), 1, "stability driver hash count")
    _equal(len(kernel_hashes), 1, "kernel source hash count")
    _equal(len(environments), 1, "GPU/Torch/CUDA environment count")
    _equal(len(ladder_hosts), 1, "serial ladder host count")
    _equal(len(ladder_jobs), 1, "serial ladder Slurm job count")
    _equal(len(stability_array_jobs), 1, "stability Slurm array job count")
    return {
        "implementation_git_commit": next(iter(commits)),
        "ladder_driver_sha256": next(iter(ladder_driver_hashes)),
        "stability_driver_sha256": next(iter(stability_driver_hashes)),
        "kernel_source_sha256": next(iter(kernel_hashes)),
        "hardware_software_context": next(iter(environments)),
        "serial_ladder_host": next(iter(ladder_hosts)),
        "serial_ladder_slurm_job_id": next(iter(ladder_jobs)),
        "stability_slurm_array_job_id": next(iter(stability_array_jobs)),
    }


def aggregate(
    *,
    results_root: Path,
    campaign_tag: str,
    output_json: Path | None,
    output_markdown: Path | None,
) -> dict[str, Any]:
    if campaign_tag != CAMPAIGN_TAG:
        raise AuditError(
            f"campaign tag {campaign_tag!r} != frozen tag {CAMPAIGN_TAG!r}"
        )
    campaign_dir = (results_root / campaign_tag).resolve()
    if not campaign_dir.is_dir():
        raise AuditError(f"campaign directory does not exist: {campaign_dir}")

    allowed_outputs = {
        path.resolve()
        for path in (output_json, output_markdown)
        if path is not None
    }
    observed_json = {
        path.name
        for path in campaign_dir.glob("*.json")
        if path.resolve() not in allowed_outputs
    }
    _validate_artifact_names(observed_json)

    manifest_path = campaign_dir / "feature_manifest.json"
    (
        manifest,
        manifest_sha256,
        trace_sha256,
        feature_source_sha256,
    ) = _validate_manifest(manifest_path)

    ladder_records = []
    ladder_contexts = []
    for rank in LADDER_RANKS:
        record, context = _validate_ladder(
            campaign_dir / f"ladder_r{rank}.json",
            rank=rank,
            manifest_sha256=manifest_sha256,
            trace_sha256=trace_sha256,
            feature_source_sha256=feature_source_sha256,
            expected_rank_scale=float(
                manifest["rank_scaling"]["scales"][str(rank)]
            ),
        )
        ladder_records.append(record)
        ladder_contexts.append(context)
    stability_records = []
    stability_contexts = []
    for rank in STABILITY_RANKS:
        record, context = _validate_stability(
            campaign_dir / f"stability_r{rank}.json",
            rank=rank,
            manifest_sha256=manifest_sha256,
            trace_sha256=trace_sha256,
            expected_rank_scale=float(
                manifest["rank_scaling"]["scales"][str(rank)]
            ),
        )
        stability_records.append(record)
        stability_contexts.append(context)

    # Every artifact must identify one implementation revision, one kernel,
    # and one software environment.  The two driver hashes differ by design.
    contexts = _validate_campaign_contexts(
        ladder_contexts,
        stability_contexts,
    )
    ladder_summary = [
        _aggregate_ladder(record) for record in ladder_records
    ]
    stability_summary = [
        _aggregate_stability(record) for record in stability_records
    ]
    all_headline_valid = (
        all(row["headline_valid"] for row in ladder_summary)
        and all(
            method["headline_valid"]
            for row in stability_summary
            for method in row["methods"].values()
        )
    )

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "blackhole_real_rank_scaling_aggregate",
        "campaign_tag": campaign_tag,
        "audit": {
            "passed": True,
            "structural_provenance_audit_passed": True,
            "all_headline_correctness_valid": all_headline_valid,
            "headline_claims_withheld": not all_headline_valid,
            "feature_manifest_count": 1,
            "ladder_artifact_count": len(ladder_records),
            "stability_artifact_count": len(stability_records),
            "expected_ladder_ranks": list(LADDER_RANKS),
            "expected_stability_ranks": list(STABILITY_RANKS),
            "expected_evaluation_transitions": list(EVALUATION_TRANSITIONS),
            "expected_stability_transitions": list(STABILITY_TRANSITIONS),
            "implementation_git_commit": contexts[
                "implementation_git_commit"
            ],
            "feature_manifest_sha256": manifest_sha256,
            "trace_sha256": trace_sha256,
            "feature_source_sha256": feature_source_sha256,
            "ladder_driver_sha256": contexts["ladder_driver_sha256"],
            "stability_driver_sha256": contexts[
                "stability_driver_sha256"
            ],
            "kernel_source_sha256": contexts["kernel_source_sha256"],
            "hardware_software_context": contexts[
                "hardware_software_context"
            ],
            "serial_ladder_host": contexts["serial_ladder_host"],
            "serial_ladder_slurm_job_id": contexts[
                "serial_ladder_slurm_job_id"
            ],
            "stability_slurm_array_job_id": contexts[
                "stability_slurm_array_job_id"
            ],
        },
        "dataset": {
            "repository": HF_REPOSITORY,
            "revision": HF_REVISION,
            "relative_path": RAW_RELATIVE_PATH,
            "bytes": RAW_BYTES,
            "sha256": RAW_SHA256,
            "scenario": manifest["protocol"]["scenario"],
            "scenario_name": manifest["protocol"]["scenario_name"],
        },
        "statistical_contract": {
            "unit_of_replication_ladder": (
                "ten frozen real transition cases; each value is the "
                "within-case timing median"
            ),
            "unit_of_replication_stability": (
                "100 frozen sequential real transitions; intervals are "
                "descriptive because successive transitions are dependent"
            ),
            "dispersion": "sample standard deviation (ddof=1)",
            "interval": "two-sided 95% Student-t interval for the mean",
            "paired_speedup": "fresh per-case time / inverse per-case time",
        },
        "ladder": ladder_summary,
        "stability": stability_summary,
    }
    if output_json is not None:
        _atomic_json_dump(summary, output_json)
    if output_markdown is not None:
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_markdown.with_suffix(
            output_markdown.suffix + ".tmp"
        )
        temporary.write_text(_markdown(summary), encoding="utf-8")
        temporary.replace(output_markdown)
    return summary


def run_synthetic_self_tests() -> None:
    """Exercise the auditor's highest-risk rejection paths without CUDA."""

    def expect_failure(function: Any, label: str) -> None:
        try:
            function()
        except AuditError:
            return
        raise AssertionError(f"synthetic audit unexpectedly accepted {label}")

    expected = _expected_artifact_names()
    _validate_artifact_names(set(expected))
    expect_failure(
        lambda: _validate_artifact_names(expected - {"ladder_r2048.json"}),
        "missing artifact",
    )
    expect_failure(
        lambda: _validate_artifact_names(expected | {"unexpected.json"}),
        "extra artifact",
    )

    probe_errors = [1e-6 * (index + 1) for index in range(16)]
    probe_metric = {
        "per_probe_relative_errors": probe_errors,
        "max_relative_error": max(probe_errors),
        "median_relative_error": statistics.median(probe_errors),
        "frobenius_relative_error": 2e-6,
    }
    cross_probe = {
        "threshold": 1e-3,
        "passed": True,
        "seed": 29421,
        "count": 16,
        "distribution": "NumPy PCG64 Rademacher (+/-1/sqrt(rank))",
        "factor_identity": "Linv_incremental P vs solve(L_fresh, P)",
        "operator_identity": "Aw_incremental P vs Aw_fresh P",
        "factor": dict(probe_metric),
        "operator": dict(probe_metric),
    }
    _validate_cross_method_probe(cross_probe, "synthetic.cross_probe")
    bad_probe = json.loads(json.dumps(cross_probe))
    bad_probe["operator"]["max_relative_error"] = 2e-3
    expect_failure(
        lambda: _validate_cross_method_probe(
            bad_probe,
            "synthetic.bad_cross_probe",
        ),
        "bad cross-method probe",
    )

    samples = []
    for index, transition in enumerate(STABILITY_TRANSITIONS):
        event = 0.01 + index * 1e-7
        wall = 0.011 + index * 1e-7
        samples.append(
            {
                "sample_index": index,
                "step": index + 1,
                "transition": transition,
                "event_seconds": event,
                "wall_seconds": wall,
                "update_event_seconds": event,
                "update_wall_seconds": wall,
                "refresh_event_seconds": 0.0,
                "refresh_wall_seconds": 0.0,
                "refreshed": False,
            }
        )
    event_values = [row["event_seconds"] for row in samples]
    wall_values = [row["wall_seconds"] for row in samples]
    stability_method = {
        "method": "fresh_refact",
        "refresh_interval": 1,
        "refresh_count": 100,
        "includes_G_M_outer_updates": True,
        "refresh_in_timed_step": True,
        "samples": samples,
        "summary": {
            "event_seconds_mean": statistics.fmean(event_values),
            "event_seconds_sample_sd": statistics.stdev(event_values),
            "event_seconds_median": statistics.median(event_values),
            "wall_seconds_mean": statistics.fmean(wall_values),
            "wall_seconds_sample_sd": statistics.stdev(wall_values),
            "wall_seconds_median": statistics.median(wall_values),
            "event_seconds_total": math.fsum(event_values),
            "wall_seconds_total": math.fsum(wall_values),
        },
        "checkpoints": [
            {
                "step": step,
                "transition": STABILITY_TRANSITIONS[step - 1],
                "measurement_phase": "post_step",
                "finite_full": True,
                "factor_probe_residual": 1e-6,
                "operator_probe_residual": 1e-6,
            }
            for step in CHECKPOINT_STEPS
        ],
        "finite_full": True,
        "correctness_threshold": 1e-3,
        "headline_valid": True,
    }
    _validate_stability_method(
        stability_method,
        rank=8_192,
        method="fresh_refact",
        context="synthetic.fresh",
    )
    tampered_method = json.loads(json.dumps(stability_method))
    tampered_method["samples"][0]["event_seconds"] += 1e-3
    tampered_method["samples"][0]["update_event_seconds"] += 1e-3
    expect_failure(
        lambda: _validate_stability_method(
            tampered_method,
            rank=8_192,
            method="fresh_refact",
            context="synthetic.tampered_summary",
        ),
        "altered raw sample with stale summary",
    )

    commit = "a" * 40
    kernel = "b" * 64
    environment = "NVIDIA H100|85045477376|torch|cuda"
    ladder_contexts = [
        (commit, "c" * 64, kernel, environment, "n01", "12345")
        for _ in LADDER_RANKS
    ]
    stability_contexts = [
        (commit, "d" * 64, kernel, environment, "67890")
        for _ in STABILITY_RANKS
    ]
    _validate_campaign_contexts(ladder_contexts, stability_contexts)
    for field_index, label, replacement in (
        (1, "source", "e" * 64),
        (4, "host", "n02"),
        (5, "Slurm job", "54321"),
    ):
        mixed = list(ladder_contexts)
        changed = list(mixed[-1])
        changed[field_index] = replacement
        mixed[-1] = tuple(changed)  # type: ignore[assignment]
        expect_failure(
            lambda rows=mixed: _validate_campaign_contexts(
                rows,  # type: ignore[arg-type]
                stability_contexts,
            ),
            f"mixed ladder {label}",
        )
    mixed_stability = list(stability_contexts)
    changed_stability = list(mixed_stability[-1])
    changed_stability[4] = "9876"
    mixed_stability[-1] = tuple(changed_stability)  # type: ignore[assignment]
    expect_failure(
        lambda: _validate_campaign_contexts(
            ladder_contexts,
            mixed_stability,  # type: ignore[arg-type]
        ),
        "mixed stability array job",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-tag")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_synthetic_self_tests()
        print("synthetic fail-closed audit tests passed", flush=True)
        return
    if args.campaign_tag is None:
        raise AuditError("--campaign-tag is required unless --self-test is used")
    campaign_dir = args.results_root / args.campaign_tag
    output_json = args.output_json or campaign_dir / "aggregate.json"
    output_markdown = (
        args.output_markdown or campaign_dir / "aggregate.md"
    )
    summary = aggregate(
        results_root=args.results_root,
        campaign_tag=args.campaign_tag,
        output_json=output_json,
        output_markdown=output_markdown,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"wrote {output_json}", flush=True)
    print(f"wrote {output_markdown}", flush=True)


if __name__ == "__main__":
    main()
