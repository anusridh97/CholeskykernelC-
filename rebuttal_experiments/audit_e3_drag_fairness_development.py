"""Strictly audit the frozen E3 drag fairness development result."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .common import ROOT
from .e3_drag_fairness_development import (
    PATH_NAMES,
    PROTOCOL_PATH,
    _aggregate_and_select,
    _validate_protocol,
)
from .e3_metric_only_pilot import atomic_json_dump


EXPECTED_SOURCE_COMMIT = "0050862f8980bf442694be50f1fef1a02829c2a5"


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit() -> tuple[dict[str, Any], Path]:
    protocol = _read(PROTOCOL_PATH)
    _validate_protocol(protocol)
    result_path = ROOT / protocol["result_path"]
    payload = _read(result_path)
    if {
        "experiment": payload.get("experiment"),
        "pilot_id": payload.get("pilot_id"),
        "status": payload.get("status"),
        "report_use": payload.get("report_use"),
    } != {
        "experiment": "e3_drag_fairness_development",
        "pilot_id": "e3-drag-fairness-dev-v1",
        "status": "complete",
        "report_use": "development_only_not_reportable",
    }:
        raise ValueError("development result identity or status is invalid")
    if payload.get("protocol") != protocol:
        raise ValueError("embedded development protocol differs from frozen file")
    if payload.get("protocol_sha256") != _sha256(PROTOCOL_PATH):
        raise ValueError("development protocol hash mismatch")

    environment = payload.get("environment", {})
    expected_environment = {
        "slurm_job_account": "marlowe-m000151-pm06",
        "slurm_job_partition": "batch",
        "thread_environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        "python_no_user_site": "1",
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "source_commit": EXPECTED_SOURCE_COMMIT,
    }
    for key, expected in expected_environment.items():
        if environment.get(key) != expected:
            raise ValueError(
                f"development environment mismatch for {key}: "
                f"expected {expected}, got {environment.get(key)}"
            )
    if len(environment.get("process_affinity") or []) != 1:
        raise ValueError("development process did not use one physical CPU")
    if not environment.get("slurm_job_id"):
        raise ValueError("development result lacks a Slurm job ID")

    rank_map = {
        int(rank): [int(width) for width in widths]
        for rank, widths in protocol["ranks_and_widths"].items()
    }
    expected_keys = {
        (rank, width, seed)
        for rank, widths in rank_map.items()
        for width in widths
        for seed in protocol["seeds"]
    }
    rows = payload.get("rows", [])
    observed_keys = {
        (int(row["rank"]), int(row["factor_width"]), int(row["seed"]))
        for row in rows
    }
    if len(rows) != len(expected_keys) or observed_keys != expected_keys:
        raise ValueError(
            f"development row mismatch: expected={len(expected_keys)}, "
            f"observed={len(rows)}, missing={sorted(expected_keys-observed_keys)}, "
            f"extra={sorted(observed_keys-expected_keys)}"
        )

    expected_contracts = {
        name: protocol["paths"][name]["output_contract"] for name in PATH_NAMES
    }
    expected_positions = Counter({position: 3 for position in range(7)})
    maximum_error = 0.0
    for row in rows:
        identity = (row["rank"], row["factor_width"], row["seed"])
        if row.get("dtype") != "float64" or row.get("threads") != 1:
            raise ValueError(f"invalid development run contract: {identity}")
        if row.get("output_contracts") != expected_contracts:
            raise ValueError(f"output contract mismatch: {identity}")
        for name in PATH_NAMES:
            accuracy = row["accuracy"][name]
            error = float(accuracy["maximum_relative_error"])
            if not math.isfinite(error) or error > protocol["relative_error_tolerance"]:
                raise ValueError(
                    f"development correctness failure: {identity}, {name}, {error}"
                )
            maximum_error = max(maximum_error, error)
            timing = row["timings_seconds"][name]
            values = [float(value) for value in timing["values"]]
            if timing.get("n") != 21 or len(values) != 21:
                raise ValueError(f"timing count mismatch: {identity}, {name}")
            if not all(math.isfinite(value) and value > 0 for value in values):
                raise ValueError(f"invalid timing: {identity}, {name}")
        schedule = row.get("timing_schedule", [])
        if len(schedule) != 21:
            raise ValueError(f"schedule length mismatch: {identity}")
        for name in PATH_NAMES:
            positions = Counter(
                entry["order"].index(name) for entry in schedule
            )
            if positions != expected_positions:
                raise ValueError(
                    f"unbalanced path positions: {identity}, {name}, {positions}"
                )

    integrity = payload.get("integrity", {})
    if integrity != {
        "declared_cell_count": len(expected_keys),
        "observed_cell_count": len(expected_keys),
        "all_rows_and_candidates_retained": True,
        "confirmation_seeds_evaluated": False,
        "development_outcomes_used_for_report_claims": False,
    }:
        raise ValueError(f"invalid development integrity declaration: {integrity}")

    recomputed = _aggregate_and_select(rows)
    if payload.get("aggregate") != recomputed:
        raise ValueError("development aggregate/selection does not reconcile")

    for path_text, metadata in payload.get("provenance", {}).get("files", {}).items():
        path = Path(path_text)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(path)
        if metadata.get("sha256") != _sha256(path):
            raise ValueError(f"development provenance hash mismatch: {path}")
        if metadata.get("bytes") != path.stat().st_size:
            raise ValueError(f"development provenance size mismatch: {path}")

    selection = recomputed["selection"]
    selected_update = selection["update"]["selected"]
    if selected_update not in ("fair_drag_factor", "original_drag_factor"):
        raise ValueError(f"invalid selected update: {selected_update}")
    selected_materializer = selection["selected_materializer"]
    if not selected_materializer.get("eligible"):
        raise ValueError("selected materializer is not eligible")
    if selected_materializer["name"] != "all_mkl":
        eligible_custom = [
            family
            for family in selection["materializer_families"][1:]
            if family["eligible"]
        ]
        if not eligible_custom:
            raise ValueError("custom materializer selected without eligibility")
        minimum = min(
            eligible_custom,
            key=lambda family: (
                family["geometric_mean_ratio_to_all_mkl"],
                family["threshold"],
            ),
        )
        if selected_materializer != minimum:
            raise ValueError("materializer selection does not minimize frozen score")

    summary = {
        "schema_version": 1,
        "experiment": "e3_drag_fairness_development_audit",
        "pilot_id": protocol["pilot_id"],
        "status": "passed",
        "report_use": "development_only_not_reportable",
        "source_result": str(result_path.relative_to(ROOT)),
        "source_result_sha256": _sha256(result_path),
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "source_commit": environment["source_commit"],
        "slurm_job_id": environment["slurm_job_id"],
        "cell_count": len(rows),
        "path_count": len(PATH_NAMES),
        "timing_samples_per_path_per_cell": 21,
        "maximum_relative_error": maximum_error,
        "all_candidates_retained": True,
        "balanced_timing_order": True,
        "reserved_confirmation_seeds_evaluated": False,
        "selection": selection,
    }
    audit_path = result_path.with_name("audit_summary.json")
    return summary, audit_path


def main() -> None:
    summary, path = audit()
    atomic_json_dump(summary, path)
    print(
        f"E3 drag development audit passed: {summary['cell_count']} cells; "
        f"wrote {path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
