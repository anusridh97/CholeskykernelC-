"""Strict aggregation for the frozen compact rank-k E6 confirmation."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any

from .common import ROOT
from .rank_k import (
    CONFIRMATION_CAMPAIGN,
    CONFIRMATION_KS,
    CONFIRMATION_PROTOCOL,
    CONFIRMATION_RANKS,
    CONFIRMATION_REPEATS,
    CONFIRMATION_SEEDS,
)


RESULT_ROOT = (
    ROOT / "rebuttal_results" / "rank_k_confirmation" / CONFIRMATION_CAMPAIGN
)
SUMMARY_PATH = RESULT_ROOT / "summary.json"
T9_975 = 2.2621571628540993


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def stats(values: list[float], *, interval: bool = False) -> dict[str, Any]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid statistical sample: {values}")
    mean = statistics.mean(values)
    sample_sd = statistics.stdev(values) if len(values) > 1 else 0.0
    output = {
        "mean": mean,
        "sample_sd": sample_sd,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "n": len(values),
        "values": values,
    }
    if interval:
        if len(values) != 10:
            raise ValueError("the preregistered E6 interval requires ten seeds")
        half_width = T9_975 * sample_sd / math.sqrt(len(values))
        output["confidence_interval_95"] = [
            mean - half_width,
            mean + half_width,
        ]
    return output


def expected_paths() -> dict[tuple[int, int], Path]:
    return {
        (rank, seed): RESULT_ROOT / f"rank_k_compact_r{rank}_seed{seed}.json"
        for rank in CONFIRMATION_RANKS
        for seed in CONFIRMATION_SEEDS
    }


def validate_payload(
    payload: dict[str, Any],
    *,
    rank: int,
    seed: int,
    protocol_sha256: str,
) -> None:
    expected_scalars = {
        "experiment": "rank_k",
        "rank": rank,
        "seed": seed,
        "threads": 1,
        "implementation": "compact",
        "pilot": False,
        "confirmation": True,
        "campaign_tag": CONFIRMATION_CAMPAIGN,
        "confirmation_protocol_sha256": protocol_sha256,
    }
    actual = {key: payload.get(key) for key in expected_scalars}
    if actual != expected_scalars:
        raise ValueError(
            f"identity mismatch for r={rank}, seed={seed}: "
            f"expected {expected_scalars}, got {actual}"
        )
    environment = payload.get("environment", {})
    expected_environment = {
        "slurm_job_account": "marlowe-m000151-pm06",
        "slurm_job_partition": "batch",
        "thread_environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
    }
    if not environment.get("source_commit"):
        raise ValueError(f"missing source commit in r={rank}, seed={seed}")
    for key, expected in expected_environment.items():
        if environment.get(key) != expected:
            raise ValueError(
                f"runtime mismatch for r={rank}, seed={seed}, {key}: "
                f"expected {expected}, got {environment.get(key)}"
            )
    if len(environment.get("process_affinity") or []) != 1:
        raise ValueError(f"non-isolated CPU affinity in r={rank}, seed={seed}")
    rows = payload.get("rows", [])
    if [row.get("k") for row in rows] != list(CONFIRMATION_KS):
        raise ValueError(f"rank-k row mismatch in r={rank}, seed={seed}")
    for row in rows:
        k = int(row["k"])
        if int(row.get("repeats", -1)) != CONFIRMATION_REPEATS:
            raise ValueError(f"repeat mismatch in r={rank}, seed={seed}, k={k}")
        samples = row.get("timing_seconds_samples", {})
        if set(samples) != {"legacy", "compact", "refactor"}:
            raise ValueError(f"timing method mismatch in r={rank}, seed={seed}, k={k}")
        if any(len(values) != CONFIRMATION_REPEATS for values in samples.values()):
            raise ValueError(f"timing sample count mismatch in r={rank}, seed={seed}, k={k}")
        if any(
            not all(math.isfinite(float(value)) and float(value) > 0.0 for value in values)
            for values in samples.values()
        ):
            raise ValueError(f"invalid timing samples in r={rank}, seed={seed}, k={k}")
        schedules = row.get("timing_method_order_schedule", [])
        expected_orders = {
            tuple(order) for order in itertools.permutations(("legacy", "compact", "refactor"))
        }
        observed_orders = [tuple(order) for order in schedules]
        if (
            len(observed_orders) != CONFIRMATION_REPEATS
            or set(observed_orders) != expected_orders
            or Counter(observed_orders)
            != Counter({order: CONFIRMATION_REPEATS // 6 for order in expected_orders})
        ):
            raise ValueError(
                f"unbalanced timing order in r={rank}, seed={seed}, k={k}"
            )
        for field in (
            "L_relative_error",
            "Aw_relative_error",
            "compact_L_relative_error",
            "compact_Aw_relative_error",
            "compact_vs_legacy_L_relative_error",
            "compact_vs_legacy_Aw_relative_error",
        ):
            value = float(row[field])
            if not math.isfinite(value) or value > 1e-9:
                raise ValueError(
                    f"correctness failure in r={rank}, seed={seed}, k={k}, "
                    f"{field}={value}"
                )
        for field in (
            "sequential_seconds",
            "compact_seconds",
            "refactor_seconds",
            "refactor_over_compact_speedup",
        ):
            value = float(row[field])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"invalid timing in r={rank}, seed={seed}, k={k}, "
                    f"{field}={value}"
                )


def aggregate() -> dict[str, Any]:
    if not CONFIRMATION_PROTOCOL.is_file():
        raise FileNotFoundError(CONFIRMATION_PROTOCOL)
    protocol_sha256 = sha256(CONFIRMATION_PROTOCOL)
    paths = expected_paths()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing E6 confirmation artifacts: {missing}")
    allowed = {path.resolve() for path in paths.values()}
    if SUMMARY_PATH.is_file():
        allowed.add(SUMMARY_PATH.resolve())
    extras = [
        str(path)
        for path in RESULT_ROOT.glob("*.json")
        if path.resolve() not in allowed
    ]
    if extras:
        raise ValueError(f"unexpected E6 confirmation JSONs: {extras}")

    payloads: dict[tuple[int, int], dict[str, Any]] = {}
    source_hashes: dict[str, set[str]] = {
        "driver_sha256": set(),
        "kernel_sha256": set(),
        "extension_sha256": set(),
    }
    job_ids: set[str] = set()
    source_commits: set[str] = set()
    for (rank, seed), path in paths.items():
        payload = read_json(path)
        validate_payload(
            payload,
            rank=rank,
            seed=seed,
            protocol_sha256=protocol_sha256,
        )
        payloads[(rank, seed)] = payload
        environment = payload["environment"]
        for field in source_hashes:
            source_hashes[field].add(str(environment[field]))
        job_ids.add(str(environment["slurm_job_id"]))
        source_commits.add(str(environment["source_commit"]))
    inconsistent = {
        key: sorted(values) for key, values in source_hashes.items() if len(values) != 1
    }
    if inconsistent:
        raise ValueError(f"E6 source/binary hashes differ across runs: {inconsistent}")
    if len(job_ids) != 1:
        raise ValueError(f"E6 confirmation must use one exclusive job: {job_ids}")
    if len(source_commits) != 1:
        raise ValueError(
            f"E6 confirmation used multiple source commits: {source_commits}"
        )

    rows = []
    for rank in CONFIRMATION_RANKS:
        for k in CONFIRMATION_KS:
            selected = [
                next(
                    row
                    for row in payloads[(rank, seed)]["rows"]
                    if int(row["k"]) == k
                )
                for seed in CONFIRMATION_SEEDS
            ]
            rows.append(
                {
                    "rank": rank,
                    "k": k,
                    "seed_count": len(selected),
                    "compact_seconds": stats(
                        [float(row["compact_seconds"]) for row in selected]
                    ),
                    "legacy_seconds": stats(
                        [float(row["sequential_seconds"]) for row in selected]
                    ),
                    "refactor_seconds": stats(
                        [float(row["refactor_seconds"]) for row in selected]
                    ),
                    "refactor_over_compact_speedup": stats(
                        [
                            float(row["refactor_over_compact_speedup"])
                            for row in selected
                        ],
                        interval=True,
                    ),
                    "legacy_over_compact_speedup": stats(
                        [
                            float(row["legacy_over_compact_speedup"])
                            for row in selected
                        ],
                        interval=True,
                    ),
                    "maximum_compact_Aw_relative_error": max(
                        float(row["compact_Aw_relative_error"])
                        for row in selected
                    ),
                }
            )
    return {
        "schema_version": 1,
        "experiment": "e6_compact_rank_k_confirmation_summary",
        "campaign_tag": CONFIRMATION_CAMPAIGN,
        "status": "complete",
        "protocol_path": str(CONFIRMATION_PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": protocol_sha256,
        "source_hashes": {
            key: next(iter(values)) for key, values in source_hashes.items()
        },
        "slurm_job_id": next(iter(job_ids)),
        "source_commit": next(iter(source_commits)),
        "artifact_count": len(payloads),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    summary = aggregate()
    if args.check_only:
        print(
            f"E6 confirmation audit passed: {summary['artifact_count']} artifacts, "
            f"{len(summary['rows'])} rows"
        )
        return
    if SUMMARY_PATH.exists():
        raise FileExistsError(f"refusing to overwrite {SUMMARY_PATH}")
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = SUMMARY_PATH.with_suffix(f".json.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(SUMMARY_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
