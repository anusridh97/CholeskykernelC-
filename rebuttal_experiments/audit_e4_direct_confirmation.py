"""Strict integrity audit and outcome summary for the frozen E4 confirmation."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Callable, Iterable

import torch

from .common import ROOT
from .e4_direct_confirmation import (
    CAMPAIGN_TAG,
    EXPECTED_ACCOUNT,
    EXPECTED_PARTITION,
    EXPECTED_PROTOCOL_SHA256,
    EXPECTED_QOS,
    IMPLEMENTATION_SOURCE,
    PROTOCOL_PATH,
    SETUP_SOURCE,
    validate_protocol,
    write_once_json,
)


T975 = {
    1: 12.706204736432095,
    2: 4.302652729911275,
    3: 3.182446305284263,
    4: 2.7764451051977987,
    5: 2.570581835636314,
    6: 2.4469118511449692,
    7: 2.3646242510102993,
    8: 2.3060041350333704,
    9: 2.2621571628540993,
}

HASH_FIELDS_TRAJECTORY = {
    "base_metric_fp64_sha256",
    "base_operator_fp64_sha256",
    "update_sequence_fp64_sha256",
}
HASH_FIELDS_SIGNED = {
    "base_metric_fp64_sha256",
    "base_operator_fp64_sha256",
    "z_fp64_sha256",
    "random_w_fp64_sha256",
}
DIAGNOSTIC_GROUPS = {
    "incremental_vs_fresh_same_dtype": {
        "L_relative_error",
        "Aw_relative_error",
    },
    "incremental_vs_quantized_fp64_reference": {
        "L_relative_error",
        "Aw_relative_error",
    },
    "fresh_same_dtype_vs_quantized_fp64_reference": {
        "L_relative_error",
        "Aw_relative_error",
    },
    "incremental_vs_original_fp64_target": {
        "L_relative_error",
        "Aw_relative_error",
    },
    "backward_errors": {
        "incremental_factor",
        "fresh_factor",
        "incremental_operator",
        "fresh_operator",
    },
}
DIAGNOSTIC_FIELDS = {
    "step",
    "phase",
    "condition_quantized_reference",
    *DIAGNOSTIC_GROUPS,
}
POSITIVE_ROW_FIELDS = {
    "precision",
    "rank",
    "target_condition",
    "reset_interval",
    "seed",
    "status",
    "failure_stage",
    "failure_step",
    "failure",
    "input_fingerprints",
    "records",
}
KERNEL_ROW_FIELDS = {
    "precision",
    "rank",
    "operator",
    "delta",
    "reset_interval",
    "seed",
    "status",
    "failure_stage",
    "failure_step",
    "failure",
    "input_fingerprints",
    "records",
}
SIGNED_ROW_FIELDS = {
    "precision",
    "rank",
    "operator_case",
    "operator_sign",
    "w_mode",
    "target_leverage",
    "seed",
    "status",
    "failure_stage",
    "failure",
    "original_fp64_leverage",
    "implementation_leverage",
    "mathematical_leverage",
    "theoretical_transport_bound",
    "input_fingerprints",
    "requested",
    "guard",
}
SIGNED_RESULT_FIELDS = {
    "status",
    "failure_stage",
    "failure",
    "diagnostics",
}
GUARD_FIELDS = {
    "threshold",
    "dtype_safe_target",
    "scale_applied_to_z",
    "scale_applied_to_w",
    "event_weight_beta",
    "implementation_leverage",
    "mathematical_leverage",
    "result",
}
RESULT_FIELDS = {
    "schema_version",
    "experiment",
    "campaign_tag",
    "status",
    "protocol",
    "protocol_path",
    "protocol_sha256",
    "positive_trajectories",
    "near_duplicate_kernel_trajectories",
    "signed_downdates",
    "integrity",
    "environment",
    "provenance",
}
ENVIRONMENT_FIELDS = {
    "source_commit",
    "process_affinity",
    "thread_environment",
    "python_no_user_site",
    "torch_num_threads",
    "torch_num_interop_threads",
    "host",
    "platform",
    "torch",
    "torch_config",
    "python_executable",
    "slurm_job_id",
    "slurm_job_account",
    "slurm_job_partition",
    "slurm_job_qos",
    "slurm_cpus_per_task",
    "started_utc",
    "ended_utc",
    "elapsed_wall_seconds",
}
INTEGRITY_FIELDS = {
    "positive_row_count",
    "kernel_row_count",
    "downdate_row_count",
    "all_rows_and_failures_retained",
    "confirmation_tuned_from_outcomes",
    "timing_claims",
    "positive_failure_count",
    "kernel_failure_count",
    "downdate_cell_failure_count",
}
PROVENANCE_FIELDS = {
    "files",
    "attempt_claim",
    "row_journal",
    "implementation_extension",
    "build_flags",
}
CLAIM_FIELDS = {
    "schema_version",
    "campaign_tag",
    "status",
    "source_commit",
    "protocol_sha256",
    "implementation_source_sha256",
    "implementation_extension",
    "implementation_extension_sha256",
    "slurm_job_id",
    "claimed_utc",
}
SOURCE_PATHS = (
    PROTOCOL_PATH,
    Path(__file__).resolve(),
    Path(__file__).with_name("e4_direct_confirmation.py"),
    Path(__file__).with_name("common.py"),
    IMPLEMENTATION_SOURCE,
    SETUP_SOURCE,
    Path(__file__).with_name("stability.py"),
    Path(__file__).with_name("kernel_stability.py"),
    Path(__file__).with_name("tests") / "test_e4_direct_confirmation.py",
    Path(__file__).with_name("slurm") / "e4_direct_confirmation.sbatch",
)
SCIENTIFIC_SCOPE_DECLARATIONS = {
    "backward_error_scope": (
        "Maximum over incremental_factor and incremental_operator in every "
        "positive and near-duplicate-kernel diagnostic record. Fresh-reference "
        "errors are reported but are not thresholded by this criterion; any "
        "incomplete trajectory makes the all-complete criterion false."
    ),
    "guarded_error_scope": (
        "Maximum incremental_vs_fresh_same_dtype Aw relative error over every "
        "guarded signed-downdate success. A guarded failure makes the "
        "all-complete criterion false and is never omitted from the "
        "maximum/error accounting."
    ),
}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _expect_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    context: str,
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: expected object, got {type(value).__name__}")
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{context}: field mismatch; "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _finite_nonnegative(value: Any, context: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{context}: expected finite nonnegative value, got {value}")
    return result


def _finite_positive(value: Any, context: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{context}: expected finite positive value, got {value}")
    return result


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_fingerprints(
    value: Any,
    fields: set[str],
    context: str,
    *,
    required: bool,
) -> dict[str, str] | None:
    if value is None:
        if required:
            raise ValueError(f"{context}: successful row lacks input fingerprints")
        return None
    _expect_exact_keys(value, fields, f"{context}.input_fingerprints")
    for field in fields:
        if not _is_sha256(value[field]):
            raise ValueError(
                f"{context}.input_fingerprints.{field}: invalid SHA-256"
            )
    return value


def _stats(values: list[float]) -> dict[str, Any]:
    if not values or len(values) > 10:
        raise ValueError(f"invalid E4 statistical sample size: {len(values)}")
    clean = [
        _finite_nonnegative(value, f"statistical value {index}")
        for index, value in enumerate(values)
    ]
    mean = statistics.mean(clean)
    sample_sd = statistics.stdev(clean) if len(clean) > 1 else 0.0
    output = {
        "mean": mean,
        "sample_sd": sample_sd,
        "median": statistics.median(clean),
        "minimum": min(clean),
        "maximum": max(clean),
        "n": len(clean),
        "values": clean,
    }
    if len(clean) > 1:
        half_width = T975[len(clean) - 1] * sample_sd / math.sqrt(len(clean))
        output["confidence_interval_95"] = [
            mean - half_width,
            mean + half_width,
        ]
    else:
        output["confidence_interval_95"] = None
    return output


def _stats_or_none(values: list[float]) -> dict[str, Any] | None:
    return _stats(values) if values else None


def expected_record_sequence(
    spec: dict[str, Any],
    reset_interval: int,
) -> list[tuple[int, str]]:
    """Return the exact record order produced by a successful trajectory."""
    checkpoints = {int(value) for value in spec["checkpoints"]}
    pre_reset_steps = {int(value) for value in spec["pre_reset_steps"]}
    steps = int(spec["steps"])
    sequence: list[tuple[int, str]] = []
    for step in range(steps + 1):
        if step in checkpoints:
            phase = (
                "post_reset"
                if reset_interval and step > 0 and step % reset_interval == 0
                else "checkpoint"
            )
            sequence.append((step, phase))
        if step == steps:
            break
        next_step = step + 1
        if (
            reset_interval
            and next_step % reset_interval == 0
            and next_step in pre_reset_steps
        ):
            sequence.append((next_step, "pre_reset"))
    return sequence


def validate_diagnostics(
    record: dict[str, Any],
    context: str,
    *,
    expected_step: int | None = None,
    expected_phase: str | None = None,
) -> None:
    _expect_exact_keys(record, DIAGNOSTIC_FIELDS, context)
    step = int(record["step"])
    phase = record["phase"]
    if expected_step is not None and step != expected_step:
        raise ValueError(f"{context}: expected step {expected_step}, got {step}")
    if expected_phase is not None and phase != expected_phase:
        raise ValueError(f"{context}: expected phase {expected_phase}, got {phase}")
    if phase not in {
        "checkpoint",
        "pre_reset",
        "post_reset",
        "requested_or_guarded_downdate",
    }:
        raise ValueError(f"{context}: invalid phase {phase!r}")
    condition = float(record["condition_quantized_reference"])
    if not math.isfinite(condition) or condition < 1.0:
        raise ValueError(f"{context}: invalid condition {condition}")
    for group, fields in DIAGNOSTIC_GROUPS.items():
        values = record[group]
        _expect_exact_keys(values, fields, f"{context}.{group}")
        for field in fields:
            _finite_nonnegative(values[field], f"{context}.{group}.{field}")


def _validate_status_and_failure(
    row: dict[str, Any],
    context: str,
    *,
    maximum_step: int | None,
) -> bool:
    status = row["status"]
    if status not in {"ok", "failed"}:
        raise ValueError(f"{context}: invalid status {status!r}")
    if status == "ok":
        if (
            row["failure_stage"] is not None
            or row.get("failure_step") is not None
            or row["failure"] is not None
        ):
            raise ValueError(f"{context}: successful row has failure metadata")
        return True
    if not isinstance(row["failure_stage"], str) or not row["failure_stage"]:
        raise ValueError(f"{context}: failed row lacks failure_stage")
    if not isinstance(row["failure"], str) or not row["failure"]:
        raise ValueError(f"{context}: failed row lacks failure")
    if maximum_step is not None:
        failure_step = row.get("failure_step")
        if not isinstance(failure_step, int) or not 0 <= failure_step <= maximum_step:
            raise ValueError(f"{context}: invalid failure_step {failure_step!r}")
    return False


def _validate_trajectory_rows(
    *,
    rows: list[dict[str, Any]],
    expected: set[tuple[Any, ...]],
    identity: Callable[[dict[str, Any]], tuple[Any, ...]],
    spec: dict[str, Any],
    row_fields: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(rows, list):
        raise ValueError(f"{label}: rows must be a list")
    observed = [identity(row) for row in rows]
    if len(observed) != len(set(observed)):
        raise ValueError(f"{label}: duplicate row identities")
    if set(observed) != expected:
        raise ValueError(
            f"{label}: missing={sorted(expected - set(observed))}, "
            f"extras={sorted(set(observed) - expected)}"
        )

    failures = 0
    for row in rows:
        row_id = identity(row)
        context = f"{label} {row_id}"
        _expect_exact_keys(row, row_fields, context)
        if row["rank"] != spec["rank"]:
            raise ValueError(
                f"{context}: expected rank {spec['rank']}, got {row['rank']}"
            )
        successful = _validate_status_and_failure(
            row,
            context,
            maximum_step=int(spec["steps"]),
        )
        if not successful:
            failures += 1
        _validate_fingerprints(
            row["input_fingerprints"],
            HASH_FIELDS_TRAJECTORY,
            context,
            required=successful,
        )
        expected_records = expected_record_sequence(
            spec,
            int(row["reset_interval"]),
        )
        records = row["records"]
        if not isinstance(records, list):
            raise ValueError(f"{context}: records must be a list")
        observed_records: list[tuple[int, str]] = []
        for index, record in enumerate(records):
            record_context = f"{context}.records[{index}]"
            validate_diagnostics(record, record_context)
            observed_records.append((int(record["step"]), record["phase"]))
            if record["phase"] == "post_reset":
                direct = record["incremental_vs_fresh_same_dtype"]
                if direct["L_relative_error"] != 0.0:
                    raise ValueError(f"{record_context}: factor reset is not exact")
                if direct["Aw_relative_error"] != 0.0:
                    raise ValueError(f"{record_context}: operator reset is not exact")
        if successful and observed_records != expected_records:
            raise ValueError(
                f"{context}: record sequence mismatch; "
                f"expected={expected_records}, observed={observed_records}"
            )
        if not successful and observed_records != expected_records[: len(records)]:
            raise ValueError(
                f"{context}: failed trajectory records are not an exact prefix; "
                f"expected_prefix={expected_records[:len(records)]}, "
                f"observed={observed_records}"
            )
    return {"row_count": len(rows), "failure_count": failures}


def validate_fingerprint_pairing(
    rows: Iterable[dict[str, Any]],
    *,
    group_fields: tuple[str, ...],
    fingerprint_fields: tuple[str, ...],
    label: str,
) -> dict[str, Any]:
    """Verify that every available common-input hash agrees within a group."""
    groups: dict[tuple[Any, ...], dict[str, set[str]]] = defaultdict(
        lambda: {field: set() for field in fingerprint_fields}
    )
    rows_with_fingerprints: dict[tuple[Any, ...], int] = defaultdict(int)
    for row in rows:
        fingerprints = row.get("input_fingerprints")
        if fingerprints is None:
            continue
        key = tuple(row[field] for field in group_fields)
        rows_with_fingerprints[key] += 1
        for field in fingerprint_fields:
            groups[key][field].add(fingerprints[field])
    for key, values in groups.items():
        for field, hashes in values.items():
            if len(hashes) != 1:
                raise ValueError(
                    f"{label} pairing mismatch for {key}.{field}: {sorted(hashes)}"
                )
    return {
        "group_count_with_fingerprints": len(groups),
        "rows_with_fingerprints": sum(rows_with_fingerprints.values()),
        "groups": [
            {
                "identity": list(key),
                "row_count": rows_with_fingerprints[key],
            }
            for key in sorted(groups, key=str)
        ],
    }


def _validate_signed_result(
    result: dict[str, Any],
    context: str,
) -> bool:
    _expect_exact_keys(result, SIGNED_RESULT_FIELDS, context)
    status = result["status"]
    if status == "ok":
        if result["failure_stage"] is not None or result["failure"] is not None:
            raise ValueError(f"{context}: successful signed result has failure data")
        validate_diagnostics(
            result["diagnostics"],
            f"{context}.diagnostics",
            expected_step=1,
            expected_phase="requested_or_guarded_downdate",
        )
        return True
    if status != "failed":
        raise ValueError(f"{context}: invalid status {status!r}")
    if result["failure_stage"] not in {
        "signed_incremental_implementation",
        "fresh_reference_or_diagnostics",
    }:
        raise ValueError(f"{context}: invalid failure_stage")
    if not isinstance(result["failure"], str) or not result["failure"]:
        raise ValueError(f"{context}: failure lacks an error message")
    if result["diagnostics"] is not None:
        raise ValueError(f"{context}: failed result unexpectedly has diagnostics")
    return False


def _dtype_safe_guard_target(precision: str, threshold: float) -> float:
    dtype = torch.float32 if precision == "float32" else torch.float64
    return float(
        torch.nextafter(
            torch.tensor(threshold, dtype=dtype),
            torch.tensor(0.0, dtype=dtype),
        ).item()
    )


def _validate_signed_rows(
    rows: list[dict[str, Any]],
    spec: dict[str, Any],
) -> dict[str, Any]:
    cases = {
        case["name"]: (int(case["operator_sign"]), case["w_mode"])
        for case in spec["operator_cases"]
    }
    expected = {
        (
            precision,
            rank,
            case,
            sign,
            mode,
            leverage,
            seed,
        )
        for precision in spec["precisions"]
        for rank in spec["ranks"]
        for case, (sign, mode) in cases.items()
        for leverage in spec["leverages"]
        for seed in spec["seeds"]
    }
    observed = [
        (
            row["precision"],
            row["rank"],
            row["operator_case"],
            row["operator_sign"],
            row["w_mode"],
            row["target_leverage"],
            row["seed"],
        )
        for row in rows
    ]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError(
            "downdate identity grid is incomplete or duplicated: "
            f"missing={sorted(expected - set(observed), key=str)}, "
            f"extras={sorted(set(observed) - expected, key=str)}"
        )

    cell_failures = 0
    requested_failures = 0
    guarded_failures = 0
    for row, row_id in zip(rows, observed):
        context = f"downdate {row_id}"
        _expect_exact_keys(row, SIGNED_ROW_FIELDS, context)
        status = row["status"]
        if status not in {"ok", "failed"}:
            raise ValueError(f"{context}: invalid status {status!r}")
        fingerprints = _validate_fingerprints(
            row["input_fingerprints"],
            HASH_FIELDS_SIGNED,
            context,
            required=status == "ok",
        )
        if status == "failed":
            cell_failures += 1
            if row["failure_stage"] != "signed_cell_setup_or_guard":
                raise ValueError(f"{context}: invalid cell failure stage")
            if not isinstance(row["failure"], str) or not row["failure"]:
                raise ValueError(f"{context}: cell failure lacks an error")
            for field in (
                "original_fp64_leverage",
                "implementation_leverage",
                "mathematical_leverage",
                "theoretical_transport_bound",
                "requested",
                "guard",
            ):
                if row[field] is not None:
                    raise ValueError(f"{context}: failed cell has non-null {field}")
            continue
        if row["failure_stage"] is not None or row["failure"] is not None:
            raise ValueError(f"{context}: successful cell has failure metadata")
        if fingerprints is None:  # Kept explicit for static type narrowing.
            raise AssertionError("successful signed row lost fingerprints")

        target = float(row["target_leverage"])
        original = _finite_positive(
            row["original_fp64_leverage"],
            f"{context}.original_fp64_leverage",
        )
        implementation = _finite_positive(
            row["implementation_leverage"],
            f"{context}.implementation_leverage",
        )
        mathematical = _finite_positive(
            row["mathematical_leverage"],
            f"{context}.mathematical_leverage",
        )
        if not 0.0 < implementation < 1.0 or not 0.0 < mathematical < 1.0:
            raise ValueError(f"{context}: requested downdate is not admissible")
        if not math.isclose(original, target, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"{context}: common FP64 construction misses target leverage; "
                f"target={target}, observed={original}"
            )
        tolerance = 1e-4 if row["precision"] == "float32" else 1e-10
        for name, value in (
            ("implementation", implementation),
            ("mathematical", mathematical),
        ):
            if not math.isclose(value, target, rel_tol=tolerance, abs_tol=tolerance):
                raise ValueError(
                    f"{context}: {name} leverage misses declared cell; "
                    f"target={target}, observed={value}"
                )
        bound = _finite_positive(
            row["theoretical_transport_bound"],
            f"{context}.theoretical_transport_bound",
        )
        expected_bound = 1.0 / (1.0 - mathematical)
        if not math.isclose(bound, expected_bound, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(f"{context}: transport bound mismatch")

        requested_ok = _validate_signed_result(
            row["requested"],
            f"{context}.requested",
        )
        if not requested_ok:
            requested_failures += 1

        guard = row["guard"]
        _expect_exact_keys(guard, GUARD_FIELDS, f"{context}.guard")
        threshold = float(spec["guard_leverage"])
        if guard["threshold"] != threshold:
            raise ValueError(f"{context}: guard threshold mismatch")
        safe_target = _dtype_safe_guard_target(row["precision"], threshold)
        if guard["dtype_safe_target"] != safe_target:
            raise ValueError(f"{context}: dtype-safe guard target mismatch")
        scale_z = _finite_positive(
            guard["scale_applied_to_z"],
            f"{context}.guard.scale_applied_to_z",
        )
        scale_w = _finite_positive(
            guard["scale_applied_to_w"],
            f"{context}.guard.scale_applied_to_w",
        )
        if scale_z > 1.0 + 1e-15:
            raise ValueError(f"{context}: guard amplified z")
        expected_initial_scale = (
            math.sqrt(safe_target / max(implementation, mathematical))
            if max(implementation, mathematical) > safe_target
            else 1.0
        )
        if scale_z > expected_initial_scale * (1.0 + 1e-12):
            raise ValueError(f"{context}: guard scale is not conservative")
        expected_w_scale = 1.0 if row["w_mode"] == "zero" else scale_z
        if not math.isclose(
            scale_w,
            expected_w_scale,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(f"{context}: guard w-scaling violates case semantics")
        beta = _finite_positive(
            guard["event_weight_beta"],
            f"{context}.guard.event_weight_beta",
        )
        if not math.isclose(beta, scale_z**2, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f"{context}: guard event weight is not scale_z^2")
        guarded_impl = _finite_positive(
            guard["implementation_leverage"],
            f"{context}.guard.implementation_leverage",
        )
        guarded_math = _finite_positive(
            guard["mathematical_leverage"],
            f"{context}.guard.mathematical_leverage",
        )
        if guarded_impl > threshold or guarded_math > threshold:
            raise ValueError(f"{context}: guard did not enforce both leverages")
        if scale_z == 1.0:
            if not math.isclose(
                guarded_impl,
                implementation,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ) or not math.isclose(
                guarded_math,
                mathematical,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ValueError(f"{context}: no-op guard changed leverage")
        guarded_ok = _validate_signed_result(
            guard["result"],
            f"{context}.guard.result",
        )
        if not guarded_ok:
            guarded_failures += 1

    pairing = validate_fingerprint_pairing(
        rows,
        group_fields=("rank", "target_leverage", "seed"),
        fingerprint_fields=tuple(sorted(HASH_FIELDS_SIGNED)),
        label="signed common-input",
    )
    return {
        "row_count": len(rows),
        "cell_failure_count": cell_failures,
        "requested_failure_count": requested_failures,
        "guarded_failure_count": guarded_failures,
        "pairing": pairing,
    }


def validate_journal_replay(
    journal_path: Path,
    metadata: dict[str, Any],
    payload: dict[str, Any],
    *,
    expected_path: Path,
) -> dict[str, Any]:
    _expect_exact_keys(metadata, {"path", "sha256", "line_count"}, "row_journal")
    if metadata["path"] != _relative(expected_path):
        raise ValueError("E4 row-journal path mismatch")
    if journal_path.resolve() != expected_path.resolve():
        raise ValueError("E4 auditor opened an unexpected journal path")
    if not journal_path.is_file():
        raise FileNotFoundError(journal_path)
    if metadata["sha256"] != _sha256(journal_path):
        raise ValueError("E4 row-journal hash mismatch")
    raw = journal_path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("E4 row journal lacks a final newline")
    lines = raw.splitlines()
    expected_entries = [
        *(
            ("positive", row)
            for row in payload["positive_trajectories"]
        ),
        *(
            ("kernel", row)
            for row in payload["near_duplicate_kernel_trajectories"]
        ),
        *(("downdate", row) for row in payload["signed_downdates"]),
    ]
    if metadata["line_count"] != len(expected_entries):
        raise ValueError("E4 row-journal declared line count mismatch")
    if len(lines) != len(expected_entries):
        raise ValueError(
            f"E4 row-journal physical line count mismatch: "
            f"expected={len(expected_entries)}, observed={len(lines)}"
        )
    for index, (line, (expected_kind, expected_row)) in enumerate(
        zip(lines, expected_entries)
    ):
        try:
            entry = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid E4 journal line {index + 1}") from error
        _expect_exact_keys(entry, {"kind", "row"}, f"journal line {index + 1}")
        if entry["kind"] != expected_kind or entry["row"] != expected_row:
            raise ValueError(f"E4 row-journal replay mismatch at line {index + 1}")
    return {
        "path": _relative(journal_path),
        "sha256": metadata["sha256"],
        "line_count": len(lines),
        "replay_exact": True,
    }


def _git_blob_bytes(commit: str, relative_path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def git_blob_metadata(commit: str, relative_path: str) -> dict[str, Any]:
    blob = _git_blob_bytes(commit, relative_path)
    return {"sha256": _sha256_bytes(blob), "bytes": len(blob)}


def _validate_file_metadata(
    path: Path,
    metadata: dict[str, Any],
    context: str,
) -> None:
    _expect_exact_keys(metadata, {"sha256", "bytes"}, context)
    if not path.is_file():
        raise FileNotFoundError(path)
    if metadata["sha256"] != _sha256(path):
        raise ValueError(f"{context}: current-file hash mismatch")
    if metadata["bytes"] != path.stat().st_size:
        raise ValueError(f"{context}: current-file size mismatch")


def validate_provenance(
    payload: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    provenance = payload["provenance"]
    _expect_exact_keys(provenance, PROVENANCE_FIELDS, "provenance")
    files = provenance["files"]
    if not isinstance(files, dict):
        raise ValueError("provenance.files must be an object")
    environment = payload["environment"]
    source_commit = environment["source_commit"]
    current_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if current_head != source_commit:
        raise ValueError(
            f"E4 audit checkout is not the source commit: "
            f"source={source_commit}, HEAD={current_head}"
        )

    source_keys = {_relative(path) for path in SOURCE_PATHS}
    extension = Path(provenance["implementation_extension"]).resolve()
    expected_extension = (
        Path("/scratch/m000151-pm06/arjo/job-builds")
        / str(environment["slurm_job_id"])
        / "e4-direct"
        / "lib"
        / "fast_cholesky.so"
    ).resolve()
    if extension != expected_extension:
        raise ValueError(
            f"E4 extension path mismatch: expected {expected_extension}, got {extension}"
        )
    python_executable = Path(environment["python_executable"]).resolve()
    if python_executable != Path(sys.executable).resolve():
        raise ValueError("E4 Python executable differs from the audit runtime")
    torch_module = Path(torch.__file__).resolve()
    torch_cpu = torch_module.parent / "lib" / "libtorch_cpu.so"
    expected_paths = {
        *SOURCE_PATHS,
        extension,
        python_executable,
        torch_module,
    }
    if torch_cpu.is_file():
        expected_paths.add(torch_cpu)
    expected_keys = {_relative(path) for path in expected_paths}
    if set(files) != expected_keys:
        raise ValueError(
            "E4 provenance file-set mismatch: "
            f"missing={sorted(expected_keys - set(files))}, "
            f"extra={sorted(set(files) - expected_keys)}"
        )

    for key, metadata in files.items():
        path = Path(key)
        if not path.is_absolute():
            path = ROOT / path
        _validate_file_metadata(path, metadata, f"provenance.files[{key!r}]")
    for key in source_keys:
        committed = git_blob_metadata(source_commit, key)
        if files[key] != committed:
            raise ValueError(
                f"E4 source file does not match git blob {source_commit}:{key}"
            )
    if provenance["build_flags"] != ["-O3", "-std=c++17", "-march=native"]:
        raise ValueError("E4 build flags differ from the locked extension build")

    claim_path = ROOT / protocol["attempt_claim_path"]
    claim = _read(claim_path)
    _expect_exact_keys(claim, CLAIM_FIELDS, "attempt claim")
    if claim["schema_version"] != 1:
        raise ValueError("E4 attempt claim schema mismatch")
    expected_claim = {
        "campaign_tag": CAMPAIGN_TAG,
        "status": "claimed_before_science",
        "source_commit": source_commit,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "implementation_source_sha256": files[
            protocol["implementation_source"]
        ]["sha256"],
        "implementation_extension": str(extension),
        "implementation_extension_sha256": files[str(extension)]["sha256"],
        "slurm_job_id": environment["slurm_job_id"],
    }
    for key, expected_value in expected_claim.items():
        if claim[key] != expected_value:
            raise ValueError(
                f"E4 attempt claim mismatch for {key}: "
                f"expected {expected_value}, got {claim[key]}"
            )
    try:
        claimed = datetime.fromisoformat(claim["claimed_utc"])
        started = datetime.fromisoformat(environment["started_utc"])
        ended = datetime.fromisoformat(environment["ended_utc"])
    except (TypeError, ValueError) as error:
        raise ValueError("E4 timestamps are not valid ISO-8601 values") from error
    if not claimed <= started <= ended:
        raise ValueError("E4 attempt/start/end timestamps are not monotonic")
    expected_claim_provenance = {
        "path": _relative(claim_path),
        "sha256": _sha256(claim_path),
    }
    if provenance["attempt_claim"] != expected_claim_provenance:
        raise ValueError("E4 attempt-claim provenance mismatch")
    return {
        "source_commit": source_commit,
        "source_file_count": len(source_keys),
        "provenance_file_count": len(files),
        "git_blob_verification": True,
        "attempt_claim_verified": True,
    }


def _validate_environment(environment: dict[str, Any]) -> None:
    _expect_exact_keys(environment, ENVIRONMENT_FIELDS, "environment")
    expected = {
        "slurm_job_account": EXPECTED_ACCOUNT,
        "slurm_job_partition": EXPECTED_PARTITION,
        "slurm_job_qos": EXPECTED_QOS,
        "slurm_cpus_per_task": "1",
        "python_no_user_site": "1",
        "thread_environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
    }
    for key, expected_value in expected.items():
        if environment[key] != expected_value:
            raise ValueError(
                f"E4 environment mismatch for {key}: "
                f"expected {expected_value}, got {environment[key]}"
            )
    affinity = environment["process_affinity"]
    if (
        not isinstance(affinity, list)
        or len(affinity) != 1
        or not isinstance(affinity[0], int)
    ):
        raise ValueError("E4 process was not bound to exactly one CPU")
    source_commit = environment["source_commit"]
    if not (
        isinstance(source_commit, str)
        and len(source_commit) == 40
        and all(character in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError("E4 source commit is not a full Git object ID")
    job_id = environment["slurm_job_id"]
    if not isinstance(job_id, str) or not job_id.isdigit():
        raise ValueError("E4 Slurm job ID is invalid")
    for field in ("host", "platform", "torch", "torch_config", "python_executable"):
        if not isinstance(environment[field], str) or not environment[field]:
            raise ValueError(f"E4 environment field {field} is empty")
    elapsed = _finite_nonnegative(
        environment["elapsed_wall_seconds"],
        "environment.elapsed_wall_seconds",
    )
    if elapsed == 0.0:
        raise ValueError("E4 elapsed wall time is zero")


def _validate_integrity_declaration(
    integrity: dict[str, Any],
    *,
    positive_rows: list[dict[str, Any]],
    kernel_rows: list[dict[str, Any]],
    downdate_rows: list[dict[str, Any]],
) -> None:
    _expect_exact_keys(integrity, INTEGRITY_FIELDS, "integrity")
    expected = {
        "positive_row_count": len(positive_rows),
        "kernel_row_count": len(kernel_rows),
        "downdate_row_count": len(downdate_rows),
        "positive_failure_count": sum(
            row["status"] != "ok" for row in positive_rows
        ),
        "kernel_failure_count": sum(
            row["status"] != "ok" for row in kernel_rows
        ),
        "downdate_cell_failure_count": sum(
            row["status"] != "ok" for row in downdate_rows
        ),
        "all_rows_and_failures_retained": True,
        "confirmation_tuned_from_outcomes": False,
        "timing_claims": False,
    }
    if integrity != expected:
        raise ValueError(
            f"E4 integrity declaration mismatch: expected={expected}, "
            f"observed={integrity}"
        )


def _trajectory_summary(
    rows: list[dict[str, Any]],
    spec: dict[str, Any],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        group_rows.sort(key=lambda row: row["seed"])
        reset_interval = int(group_rows[0]["reset_interval"])
        expected_sequence = expected_record_sequence(spec, reset_interval)
        for step, phase in expected_sequence:
            observed: list[tuple[int, dict[str, Any]]] = []
            missing_seeds = []
            for row in group_rows:
                match = next(
                    (
                        record
                        for record in row["records"]
                        if record["step"] == step and record["phase"] == phase
                    ),
                    None,
                )
                if match is None:
                    missing_seeds.append(row["seed"])
                else:
                    observed.append((row["seed"], match))
            records = [record for _, record in observed]
            output.append(
                {
                    **dict(zip(group_fields, key)),
                    "step": step,
                    "phase": phase,
                    "expected_seed_count": len(group_rows),
                    "observed_seed_count": len(records),
                    "observed_seeds": [seed for seed, _ in observed],
                    "missing_seeds": missing_seeds,
                    "failed_row_seeds": [
                        row["seed"]
                        for row in group_rows
                        if row["status"] != "ok"
                    ],
                    "incremental_vs_fresh_L": _stats_or_none(
                        [
                            float(record["incremental_vs_fresh_same_dtype"][
                                "L_relative_error"
                            ])
                            for record in records
                        ]
                    ),
                    "incremental_vs_fresh_Aw": _stats_or_none(
                        [
                            float(record["incremental_vs_fresh_same_dtype"][
                                "Aw_relative_error"
                            ])
                            for record in records
                        ]
                    ),
                    "incremental_forward_Aw": _stats_or_none(
                        [
                            float(
                                record[
                                    "incremental_vs_quantized_fp64_reference"
                                ]["Aw_relative_error"]
                            )
                            for record in records
                        ]
                    ),
                    "fresh_forward_Aw": _stats_or_none(
                        [
                            float(
                                record[
                                    "fresh_same_dtype_vs_quantized_fp64_reference"
                                ]["Aw_relative_error"]
                            )
                            for record in records
                        ]
                    ),
                    "incremental_backward_operator": _stats_or_none(
                        [
                            float(record["backward_errors"][
                                "incremental_operator"
                            ])
                            for record in records
                        ]
                    ),
                    "fresh_backward_operator": _stats_or_none(
                        [
                            float(record["backward_errors"]["fresh_operator"])
                            for record in records
                        ]
                    ),
                }
            )
    return output


def _downdate_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["precision"],
                row["rank"],
                row["operator_case"],
                row["operator_sign"],
                row["w_mode"],
                row["target_leverage"],
            )
        ].append(row)
    output = []
    for key, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        group_rows.sort(key=lambda row: row["seed"])
        usable = [row for row in group_rows if row["status"] == "ok"]
        requested_ok = [
            row for row in usable if row["requested"]["status"] == "ok"
        ]
        guarded_ok = [
            row
            for row in usable
            if row["guard"]["result"]["status"] == "ok"
        ]
        output.append(
            {
                "precision": key[0],
                "rank": key[1],
                "operator_case": key[2],
                "operator_sign": key[3],
                "w_mode": key[4],
                "target_leverage": key[5],
                "expected_seed_count": len(group_rows),
                "cell_success_count": len(usable),
                "cell_failure_seeds": [
                    row["seed"] for row in group_rows if row["status"] != "ok"
                ],
                "requested_success_count": len(requested_ok),
                "requested_failure_seeds": [
                    row["seed"]
                    for row in usable
                    if row["requested"]["status"] != "ok"
                ],
                "guarded_success_count": len(guarded_ok),
                "guarded_failure_seeds": [
                    row["seed"]
                    for row in usable
                    if row["guard"]["result"]["status"] != "ok"
                ],
                "original_fp64_leverage": _stats_or_none(
                    [float(row["original_fp64_leverage"]) for row in usable]
                ),
                "implementation_leverage": _stats_or_none(
                    [float(row["implementation_leverage"]) for row in usable]
                ),
                "mathematical_leverage": _stats_or_none(
                    [float(row["mathematical_leverage"]) for row in usable]
                ),
                "requested_incremental_vs_fresh_Aw": _stats_or_none(
                    [
                        float(
                            row["requested"]["diagnostics"][
                                "incremental_vs_fresh_same_dtype"
                            ]["Aw_relative_error"]
                        )
                        for row in requested_ok
                    ]
                ),
                "guarded_incremental_vs_fresh_Aw": _stats_or_none(
                    [
                        float(
                            row["guard"]["result"]["diagnostics"][
                                "incremental_vs_fresh_same_dtype"
                            ]["Aw_relative_error"]
                        )
                        for row in guarded_ok
                    ]
                ),
            }
        )
    return output


def evaluate_scientific_criteria(
    protocol: dict[str, Any],
    positive_rows: list[dict[str, Any]],
    kernel_rows: list[dict[str, Any]],
    downdate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    criteria = protocol["predeclared_correctness_criteria"]
    observed_scopes = {
        name: criteria.get(name) for name in SCIENTIFIC_SCOPE_DECLARATIONS
    }
    if observed_scopes != SCIENTIFIC_SCOPE_DECLARATIONS:
        raise ValueError(
            "E4 scientific-criterion scope declarations differ from the "
            f"frozen definitions: {observed_scopes}"
        )
    trajectory_rows = [*positive_rows, *kernel_rows]
    trajectory_failures = sum(row["status"] != "ok" for row in trajectory_rows)

    backward_by_precision: dict[str, list[float]] = defaultdict(list)
    for row in trajectory_rows:
        for record in row["records"]:
            backward_by_precision[row["precision"]].extend(
                [
                    float(record["backward_errors"]["incremental_factor"]),
                    float(record["backward_errors"]["incremental_operator"]),
                ]
            )
    maximum_backward = {
        precision: (
            max(values) if values else None
        )
        for precision, values in backward_by_precision.items()
    }
    for precision in ("float32", "float64"):
        maximum_backward.setdefault(precision, None)

    fp64_rows = [
        row for row in downdate_rows if row["precision"] == "float64"
    ]
    requested_fp64_complete = all(
        row["status"] == "ok" and row["requested"]["status"] == "ok"
        for row in fp64_rows
    )
    guarded_complete = all(
        row["status"] == "ok" and row["guard"]["result"]["status"] == "ok"
        for row in downdate_rows
    )
    guarded_aw: dict[str, list[float]] = defaultdict(list)
    for row in downdate_rows:
        if (
            row["status"] == "ok"
            and row["guard"]["result"]["status"] == "ok"
        ):
            guarded_aw[row["precision"]].append(
                float(
                    row["guard"]["result"]["diagnostics"][
                        "incremental_vs_fresh_same_dtype"
                    ]["Aw_relative_error"]
                )
            )
    maximum_guarded_aw = {
        precision: (max(values) if values else None)
        for precision, values in guarded_aw.items()
    }
    for precision in ("float32", "float64"):
        maximum_guarded_aw.setdefault(precision, None)

    checks: dict[str, dict[str, Any]] = {}

    def add_boolean(name: str, observed: bool, expected: bool) -> None:
        checks[name] = {
            "observed": observed,
            "expected": expected,
            "passed": observed is expected,
        }

    def add_maximum(name: str, observed: float | None, threshold: float) -> None:
        checks[name] = {
            "observed": observed,
            "maximum_allowed": threshold,
            "passed": observed is not None and observed <= threshold,
        }

    add_boolean(
        "positive_and_kernel_all_trajectories_complete",
        trajectory_failures == 0,
        bool(criteria["positive_and_kernel_all_trajectories_complete"]),
    )
    add_maximum(
        "maximum_backward_error_float32",
        maximum_backward["float32"],
        float(criteria["maximum_backward_error_float32"]),
    )
    add_maximum(
        "maximum_backward_error_float64",
        maximum_backward["float64"],
        float(criteria["maximum_backward_error_float64"]),
    )
    add_boolean(
        "requested_fp64_downdates_all_complete",
        requested_fp64_complete,
        bool(criteria["requested_fp64_downdates_all_complete"]),
    )
    add_boolean(
        "guarded_downdates_all_complete",
        guarded_complete,
        bool(criteria["guarded_downdates_all_complete"]),
    )
    add_maximum(
        "maximum_guarded_incremental_vs_fresh_Aw_float32",
        maximum_guarded_aw["float32"],
        float(criteria["maximum_guarded_incremental_vs_fresh_Aw_float32"]),
    )
    add_maximum(
        "maximum_guarded_incremental_vs_fresh_Aw_float64",
        maximum_guarded_aw["float64"],
        float(criteria["maximum_guarded_incremental_vs_fresh_Aw_float64"]),
    )
    executable_criteria = set(criteria) - set(SCIENTIFIC_SCOPE_DECLARATIONS)
    if set(checks) != executable_criteria:
        raise ValueError("E4 scientific-criterion implementation is incomplete")
    return {
        "status": (
            "criteria_passed"
            if all(check["passed"] for check in checks.values())
            else "criteria_failed"
        ),
        "all_passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
        "scope_declarations": observed_scopes,
        "outcome_does_not_affect_integrity_status": True,
    }


def audit() -> tuple[dict[str, Any], Path]:
    protocol = _read(PROTOCOL_PATH)
    validate_protocol(protocol)
    if _sha256(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("E4 protocol differs from its frozen SHA-256")
    result_path = ROOT / protocol["result_path"]
    audit_path = ROOT / protocol["audit_path"]
    payload = _read(result_path)
    _expect_exact_keys(payload, RESULT_FIELDS, "result")
    if payload["schema_version"] != 1:
        raise ValueError("E4 result schema mismatch")
    identity = {
        "experiment": payload["experiment"],
        "campaign_tag": payload["campaign_tag"],
        "status": payload["status"],
    }
    if identity != {
        "experiment": "e4_direct_confirmation",
        "campaign_tag": CAMPAIGN_TAG,
        "status": "complete",
    }:
        raise ValueError(f"E4 result identity mismatch: {identity}")
    if payload["protocol"] != protocol:
        raise ValueError("E4 embedded protocol differs from frozen protocol")
    if payload["protocol_path"] != _relative(PROTOCOL_PATH):
        raise ValueError("E4 protocol path mismatch")
    if payload["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("E4 result protocol hash mismatch")

    environment = payload["environment"]
    _validate_environment(environment)
    provenance_summary = validate_provenance(payload, protocol)

    positive_spec = protocol["positive_trajectories"]
    positive_expected = {
        (precision, positive_spec["rank"], condition, reset, seed)
        for precision in positive_spec["precisions"]
        for condition in positive_spec["conditions"]
        for reset in positive_spec["reset_intervals"]
        for seed in positive_spec["seeds"]
    }
    positive_rows = payload["positive_trajectories"]
    positive_integrity = _validate_trajectory_rows(
        rows=positive_rows,
        expected=positive_expected,
        identity=lambda row: (
            row["precision"],
            row["rank"],
            row["target_condition"],
            row["reset_interval"],
            row["seed"],
        ),
        spec=positive_spec,
        row_fields=POSITIVE_ROW_FIELDS,
        label="positive",
    )
    positive_pairing = validate_fingerprint_pairing(
        positive_rows,
        group_fields=("target_condition", "seed"),
        fingerprint_fields=tuple(sorted(HASH_FIELDS_TRAJECTORY)),
        label="positive common-input",
    )

    kernel_spec = protocol["near_duplicate_kernels"]
    kernel_expected = {
        (precision, kernel_spec["rank"], operator, delta, reset, seed)
        for precision in kernel_spec["precisions"]
        for operator in kernel_spec["operators"]
        for delta in kernel_spec["deltas"]
        for reset in kernel_spec["reset_intervals"]
        for seed in kernel_spec["seeds"]
    }
    kernel_rows = payload["near_duplicate_kernel_trajectories"]
    kernel_integrity = _validate_trajectory_rows(
        rows=kernel_rows,
        expected=kernel_expected,
        identity=lambda row: (
            row["precision"],
            row["rank"],
            row["operator"],
            row["delta"],
            row["reset_interval"],
            row["seed"],
        ),
        spec=kernel_spec,
        row_fields=KERNEL_ROW_FIELDS,
        label="kernel",
    )
    kernel_metric_update_pairing = validate_fingerprint_pairing(
        kernel_rows,
        group_fields=("delta", "seed"),
        fingerprint_fields=(
            "base_metric_fp64_sha256",
            "update_sequence_fp64_sha256",
        ),
        label="kernel metric/update common-input",
    )
    kernel_operator_pairing = validate_fingerprint_pairing(
        kernel_rows,
        group_fields=("operator", "delta", "seed"),
        fingerprint_fields=("base_operator_fp64_sha256",),
        label="kernel operator common-input",
    )

    downdate_rows = payload["signed_downdates"]
    signed_integrity = _validate_signed_rows(
        downdate_rows,
        protocol["signed_downdates"],
    )
    _validate_integrity_declaration(
        payload["integrity"],
        positive_rows=positive_rows,
        kernel_rows=kernel_rows,
        downdate_rows=downdate_rows,
    )

    journal_path = ROOT / protocol["journal_path"]
    journal_summary = validate_journal_replay(
        journal_path,
        payload["provenance"]["row_journal"],
        payload,
        expected_path=journal_path,
    )
    scientific = evaluate_scientific_criteria(
        protocol,
        positive_rows,
        kernel_rows,
        downdate_rows,
    )

    expected_counts = {
        "positive_row_count": len(positive_expected),
        "kernel_row_count": len(kernel_expected),
        "downdate_row_count": (
            len(protocol["signed_downdates"]["precisions"])
            * len(protocol["signed_downdates"]["ranks"])
            * len(protocol["signed_downdates"]["operator_cases"])
            * len(protocol["signed_downdates"]["leverages"])
            * len(protocol["signed_downdates"]["seeds"])
        ),
    }
    summary = {
        "schema_version": 2,
        "experiment": "e4_direct_confirmation_audit",
        "campaign_tag": CAMPAIGN_TAG,
        "status": "integrity_passed",
        "scientific_status": scientific["status"],
        "source_result": _relative(result_path),
        "source_result_sha256": _sha256(result_path),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "source_commit": environment["source_commit"],
        "slurm_job_id": environment["slurm_job_id"],
        "positive": _trajectory_summary(
            positive_rows,
            positive_spec,
            ("precision", "target_condition", "reset_interval"),
        ),
        "near_duplicate_kernels": _trajectory_summary(
            kernel_rows,
            kernel_spec,
            ("precision", "operator", "delta", "reset_interval"),
        ),
        "signed_downdates": _downdate_summary(downdate_rows),
        "scientific_criteria": scientific,
        "integrity": {
            **expected_counts,
            "positive_failure_count": positive_integrity["failure_count"],
            "kernel_failure_count": kernel_integrity["failure_count"],
            "downdate_cell_failure_count": signed_integrity[
                "cell_failure_count"
            ],
            "requested_downdate_failure_count": signed_integrity[
                "requested_failure_count"
            ],
            "guarded_downdate_failure_count": signed_integrity[
                "guarded_failure_count"
            ],
            "all_declared_rows_retained": True,
            "journal": journal_summary,
            "provenance": provenance_summary,
            "pairing": {
                "positive": positive_pairing,
                "kernel_metric_and_update": kernel_metric_update_pairing,
                "kernel_operator": kernel_operator_pairing,
                "signed": signed_integrity["pairing"],
            },
            "timing_claims": False,
        },
    }
    return summary, audit_path


def main() -> None:
    summary, path = audit()
    write_once_json(summary, path)
    print(
        "E4 direct confirmation integrity audit passed: "
        f"{summary['integrity']['positive_row_count']} positive, "
        f"{summary['integrity']['kernel_row_count']} kernel, "
        f"{summary['integrity']['downdate_row_count']} downdate rows; "
        f"scientific_status={summary['scientific_status']}; wrote {path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
