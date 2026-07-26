"""Strictly audit and summarize the frozen E3 confirmation artifact."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from .common import ROOT
from .e3_metric_only_confirmation import (
    CAMPAIGN_TAG,
    PROTOCOL_PATH,
    RANKS_AND_WIDTHS,
    REPEATS,
    SEEDS,
    TOLERANCE,
    validate_protocol,
)
from .e3_metric_only_pilot import (
    DENSE_PATHS,
    THIN_PATHS,
    aggregate_rows,
    atomic_json_dump,
)


T9_975 = 2.2621571628540993


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(values: list[float]) -> dict[str, Any]:
    if len(values) != 10 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid ten-seed sample: {values}")
    mean = statistics.mean(values)
    sample_sd = statistics.stdev(values)
    half_width = T9_975 * sample_sd / math.sqrt(len(values))
    return {
        "mean": mean,
        "sample_sd": sample_sd,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "confidence_interval_95": [mean - half_width, mean + half_width],
        "n": len(values),
        "values": values,
    }


def audit() -> tuple[dict[str, Any], Path]:
    protocol = _read(PROTOCOL_PATH)
    validate_protocol(protocol)
    result_path = ROOT / protocol["result_path"]
    audit_path = ROOT / protocol["audit_path"]
    payload = _read(result_path)
    if {
        "experiment": payload.get("experiment"),
        "campaign_tag": payload.get("campaign_tag"),
        "status": payload.get("status"),
    } != {
        "experiment": "e3_metric_only_confirmation",
        "campaign_tag": CAMPAIGN_TAG,
        "status": "complete",
    }:
        raise ValueError("E3 result identity or completion status is invalid")
    if payload.get("protocol") != protocol:
        raise ValueError("embedded E3 protocol differs from the frozen file")
    if payload.get("protocol_sha256") != _sha256(PROTOCOL_PATH):
        raise ValueError("E3 protocol hash mismatch")

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
    }
    for key, expected in expected_environment.items():
        if environment.get(key) != expected:
            raise ValueError(
                f"E3 environment mismatch for {key}: "
                f"expected {expected}, got {environment.get(key)}"
            )
    if len(environment.get("process_affinity") or []) != 1:
        raise ValueError("E3 process was not restricted to one physical CPU")
    if not environment.get("slurm_job_id"):
        raise ValueError("E3 result lacks a Slurm job ID")

    names = list(DENSE_PATHS + THIN_PATHS)
    rows = payload.get("rows", [])
    expected_keys = {
        (rank, width, seed)
        for rank, widths in RANKS_AND_WIDTHS.items()
        for width in widths
        for seed in SEEDS
    }
    observed_keys = {
        (int(row["rank"]), int(row["factor_width"]), int(row["seed"]))
        for row in rows
    }
    if len(rows) != len(expected_keys) or observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extras = sorted(observed_keys - expected_keys)
        raise ValueError(
            f"E3 cell mismatch: rows={len(rows)}, missing={missing}, extras={extras}"
        )

    for row in rows:
        identity = (row["rank"], row["factor_width"], row["seed"])
        if (
            row.get("dtype") != "float64"
            or row.get("threads") != 1
            or row.get("warmups_per_path") != 5
            or row.get("timed_repeats_per_path") != REPEATS
        ):
            raise ValueError(f"E3 run contract mismatch in {identity}")
        if row.get("output_contracts") != {
            name: (
                "dense_same_output" if name in DENSE_PATHS else "thin_factor_only"
            )
            for name in names
        }:
            raise ValueError(f"E3 output contract mismatch in {identity}")
        for name in names:
            accuracy = row["accuracy"][name]
            if (
                not math.isfinite(float(accuracy["maximum_relative_error"]))
                or float(accuracy["maximum_relative_error"]) > TOLERANCE
            ):
                raise ValueError(
                    f"E3 correctness failure in {identity}, {name}: {accuracy}"
                )
            timing = row["timings_seconds"][name]
            values = [float(value) for value in timing["values"]]
            if len(values) != REPEATS or timing.get("n") != REPEATS:
                raise ValueError(
                    f"E3 timing count mismatch in {identity}, {name}"
                )
            if not all(math.isfinite(value) and value > 0.0 for value in values):
                raise ValueError(f"E3 invalid timing in {identity}, {name}")
        schedule = row.get("timing_schedule", [])
        if len(schedule) != REPEATS:
            raise ValueError(f"E3 schedule length mismatch in {identity}")
        position_counts = {
            name: Counter(
                entry["order"].index(name)
                for entry in schedule
            )
            for name in names
        }
        expected_positions = Counter({position: 5 for position in range(6)})
        if any(counts != expected_positions for counts in position_counts.values()):
            raise ValueError(
                f"E3 timing order is not balanced in {identity}: {position_counts}"
            )

    recomputed = aggregate_rows(
        rows,
        names,
        expected_seed_count=len(SEEDS),
    )
    if payload.get("aggregate") != recomputed:
        raise ValueError("E3 embedded aggregate does not reconcile with raw rows")
    integrity = payload.get("integrity", {})
    if (
        integrity.get("declared_cell_count") != len(expected_keys)
        or integrity.get("observed_cell_count") != len(expected_keys)
        or not integrity.get("all_rows_retained")
        or not integrity.get("pilot_excluded_from_statistics")
        or integrity.get("confirmation_tuned_from_outcomes")
    ):
        raise ValueError(f"E3 integrity declaration is invalid: {integrity}")

    for path_text, metadata in payload.get("provenance", {}).get("files", {}).items():
        path = Path(path_text)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(path)
        if metadata.get("sha256") != _sha256(path):
            raise ValueError(f"E3 provenance hash mismatch: {path}")
        if metadata.get("bytes") != path.stat().st_size:
            raise ValueError(f"E3 provenance size mismatch: {path}")

    grouped = {
        (rank, width): [
            row
            for row in rows
            if row["rank"] == rank and row["factor_width"] == width
        ]
        for rank, widths in RANKS_AND_WIDTHS.items()
        for width in widths
    }
    summary_rows = []
    for (rank, width), selected in grouped.items():
        selected.sort(key=lambda item: item["seed"])
        specialized = [
            float(row["timings_seconds"]["metric_only_specialized"]["median"])
            for row in selected
        ]
        drag = [
            float(row["timings_seconds"]["drag_materialize"]["median"])
            for row in selected
        ]
        fresh = [
            float(row["timings_seconds"]["fresh_materialize"]["median"])
            for row in selected
        ]
        summary_rows.append(
            {
                "rank": rank,
                "factor_width": width,
                "metric_only_specialized_seconds": _stats(specialized),
                "drag_materialize_seconds": _stats(drag),
                "fresh_materialize_seconds": _stats(fresh),
                "drag_over_metric_only_speedup": _stats(
                    [left / right for left, right in zip(drag, specialized)]
                ),
                "fresh_over_metric_only_speedup": _stats(
                    [left / right for left, right in zip(fresh, specialized)]
                ),
                "maximum_relative_error": max(
                    float(metrics["maximum_relative_error"])
                    for row in selected
                    for metrics in row["accuracy"].values()
                ),
            }
        )
    summary = {
        "schema_version": 1,
        "experiment": "e3_metric_only_confirmation_audit",
        "campaign_tag": CAMPAIGN_TAG,
        "status": "passed",
        "source_result": str(result_path.relative_to(ROOT)),
        "source_result_sha256": _sha256(result_path),
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "slurm_job_id": environment["slurm_job_id"],
        "source_commit": environment["source_commit"],
        "cell_count": len(rows),
        "seed_count_per_cell": len(SEEDS),
        "timing_samples_per_path_per_seed": REPEATS,
        "all_rows_retained": True,
        "rows": summary_rows,
        "crossovers": recomputed["crossovers"],
    }
    return summary, audit_path


def main() -> None:
    summary, path = audit()
    atomic_json_dump(summary, path)
    print(
        f"E3 confirmation audit passed: {summary['cell_count']} cells; "
        f"wrote {path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
