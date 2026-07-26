"""Strict aggregate for the source-locked E5 confirmation campaign."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import aggregate_e5n10 as original
from .e5_locked_confirmation import (
    CAMPAIGN_TAG,
    EXPECTED_PROBE_FILES,
    EXPECTED_RAW_JSONS,
    PROBE_RANKS,
    SCHEMA_ATTESTATION,
    SEEDS,
    LockError,
    audit_probes,
    file_metadata,
    strict_json,
)


def atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def tagged_jsons(repository: Path, seed: int) -> list[Path]:
    suffix = f"{CAMPAIGN_TAG}-task{seed}"
    return sorted(
        {
            path
            for directory in (
                repository / "rebuttal_results" / "cuda",
                repository / "rebuttal_results" / "cuda_tuning",
            )
            if directory.is_dir()
            for path in directory.rglob(f"*{suffix}*.json")
        }
    )


def audit_attestations(
    repository: Path,
    campaign_dir: Path,
) -> dict[str, Any]:
    rows = []
    lock_hashes = set()
    commits = set()
    gpu_uuids = set()
    for seed in SEEDS:
        attestation_path = campaign_dir / f"attestation_seed{seed}.json"
        attestation = strict_json(attestation_path)
        if attestation.get("schema_version") != SCHEMA_ATTESTATION:
            raise LockError(f"{attestation_path}: wrong attestation schema")
        if attestation.get("campaign_tag") != CAMPAIGN_TAG:
            raise LockError(f"{attestation_path}: wrong campaign tag")
        if attestation.get("seed") != seed:
            raise LockError(f"{attestation_path}: wrong seed")
        if attestation.get("protocol_complete") is not True:
            raise LockError(f"{attestation_path}: protocol not complete")
        paths = tagged_jsons(repository, seed)
        if len(paths) != EXPECTED_RAW_JSONS:
            raise LockError(
                f"seed {seed}: {len(paths)} tagged raw JSONs != "
                f"{EXPECTED_RAW_JSONS}"
            )
        metadata = [file_metadata(path, repository) for path in paths]
        if metadata != attestation.get("raw_jsons"):
            raise LockError(f"seed {seed}: raw artifact manifest changed")
        payloads = [(path, strict_json(path)) for path in paths]
        recomputed_probe_rows = audit_probes(repository, payloads)
        if recomputed_probe_rows != attestation.get("probe_validation"):
            raise LockError(f"seed {seed}: probe attestation changed")
        if [row["rank"] for row in recomputed_probe_rows] != list(PROBE_RANKS):
            raise LockError(f"seed {seed}: incomplete probe ranks")
        if (
            sum(len(row["artifacts"]) for row in recomputed_probe_rows)
            != EXPECTED_PROBE_FILES
        ):
            raise LockError(f"seed {seed}: incomplete probe artifact set")
        source = attestation["source"]
        lock_hashes.add(source["lock_sha256"])
        commits.add(source["campaign_commit"])
        gpu_uuids.add(attestation["gpu"]["uuid"])
        rows.append(
            {
                "seed": seed,
                "attestation": file_metadata(attestation_path, campaign_dir),
                "gpu_uuid": attestation["gpu"]["uuid"],
                "host": attestation["host"],
                "probe_validation": recomputed_probe_rows,
            }
        )
    if len(lock_hashes) != 1:
        raise LockError(f"multiple lock hashes in attestations: {lock_hashes!r}")
    if len(commits) != 1:
        raise LockError(f"multiple campaign commits in attestations: {commits!r}")
    return {
        "status": "pass",
        "seeds": list(SEEDS),
        "attestations": rows,
        "campaign_commit": next(iter(commits)),
        "campaign_lock_sha256": next(iter(lock_hashes)),
        "gpu_uuids": sorted(gpu_uuids),
        "probe_cross_comparison_recomputed_from_npz": True,
    }


def run(
    repository: Path,
    campaign_dir: Path,
) -> dict[str, Any]:
    previous = original.EXPECTED_SEEDS
    original.EXPECTED_SEEDS = SEEDS
    try:
        summary = original.audit_and_aggregate(
            repository / "rebuttal_results",
            CAMPAIGN_TAG,
        )
    finally:
        original.EXPECTED_SEEDS = previous
    summary["locked_confirmation"] = audit_attestations(
        repository,
        campaign_dir,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-rows", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = run(args.repository.resolve(), args.campaign_dir.resolve())
        markdown = original.render_markdown(summary)
    except (LockError, original.AuditError) as error:
        print(str(error), file=sys.stderr)
        return 2
    atomic_write(
        args.output_json,
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    atomic_write(args.output_rows, markdown)
    print(args.output_json)
    print(args.output_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
