"""Audit-only repair for complete E5-v3 rows after final-attestation failure.

The v3 numerical tasks ran the full frozen protocol and then exited nonzero
because the final wrapper read array metadata from the wrong JSON level for
two driver families.  This tool never executes a numerical driver.  It binds
the preserved v3 pre-run lock, the original in-task verification record, all
raw artifacts, and a separately frozen audit-tool commit before writing the
missing write-once attestation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .e5_locked_confirmation import (
    CAMPAIGN_TAG,
    EXPECTED_PROBE_FILES,
    EXPECTED_RAW_JSONS,
    SCHEMA_ATTESTATION,
    SEEDS,
    SOURCE_PATHS,
    LockError,
    audit_probes,
    file_metadata,
    git_text,
    payload_array_task_id,
    sha256_file,
    strict_json,
    verify_lock,
    write_once_json,
)


AUDIT_LOCK_SCHEMA = "e5_posthoc_attestation_lock_v1"
AUDIT_SOURCE_PATHS = (
    "rebuttal_experiments/e5_locked_confirmation.py",
    "rebuttal_experiments/e5_posthoc_attest.py",
    "rebuttal_experiments/aggregate_e5_locked.py",
)


def make_audit_lock(
    audit_repository: Path,
    numeric_lock: Path,
    output: Path,
) -> None:
    if git_text(audit_repository, "status", "--porcelain"):
        raise LockError("audit repository must be clean before locking")
    numeric = strict_json(numeric_lock)
    if numeric.get("campaign_tag") != CAMPAIGN_TAG:
        raise LockError("numeric lock is not the frozen v3 campaign")
    sources = {}
    for relative in AUDIT_SOURCE_PATHS:
        path = audit_repository / relative
        sources[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    payload = {
        "schema_version": AUDIT_LOCK_SCHEMA,
        "campaign_tag": CAMPAIGN_TAG,
        "audit_commit": git_text(audit_repository, "rev-parse", "HEAD"),
        "sources": sources,
        "numeric_lock_sha256": sha256_file(numeric_lock),
        "numeric_implementation_commit": numeric["implementation_commit"],
        "purpose": (
            "attest already-complete v3 raw outputs after fail-closed "
            "metadata-location bug; never execute numerical code"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def verify_audit_lock(
    audit_repository: Path,
    audit_lock_path: Path,
    numeric_lock_path: Path,
) -> dict[str, Any]:
    lock = strict_json(audit_lock_path)
    if lock.get("schema_version") != AUDIT_LOCK_SCHEMA:
        raise LockError("wrong posthoc audit-lock schema")
    if lock.get("campaign_tag") != CAMPAIGN_TAG:
        raise LockError("wrong posthoc campaign tag")
    if lock.get("numeric_lock_sha256") != sha256_file(numeric_lock_path):
        raise LockError("posthoc audit lock binds a different numeric lock")
    head = git_text(audit_repository, "rev-parse", "HEAD")
    required_head = os.environ.get("E5_AUDIT_COMMIT")
    if required_head and head != required_head:
        raise LockError(f"audit HEAD={head}, expected {required_head}")
    if head != lock.get("audit_commit"):
        raise LockError("posthoc audit repository is not at the locked commit")
    if git_text(audit_repository, "diff", "--name-only"):
        raise LockError("tracked posthoc audit worktree changes detected")
    if set(lock.get("sources", {})) != set(AUDIT_SOURCE_PATHS):
        raise LockError("posthoc audit source set changed")
    for relative, expected in lock["sources"].items():
        path = audit_repository / relative
        actual = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if actual != expected:
            raise LockError(f"posthoc audit source mismatch for {relative}")
    expected_sha = os.environ.get("E5_AUDIT_LOCK_SHA256")
    actual_sha = sha256_file(audit_lock_path)
    if expected_sha and actual_sha != expected_sha:
        raise LockError("posthoc audit-lock SHA mismatch")
    return {
        "audit_commit": head,
        "audit_lock_sha256": actual_sha,
        "sources": lock["sources"],
        "purpose": lock["purpose"],
    }


def original_verification(path: Path, seed: int) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise LockError(f"cannot read original stdout {path}: {error}") from error
    for line in lines:
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("status") == "verified" and value.get("seed") == seed:
            return value
    raise LockError(f"{path}: no original in-task verification for seed {seed}")


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


def payload_environment(payload: dict[str, Any]) -> dict[str, Any]:
    environment = payload.get("environment")
    if isinstance(environment, dict):
        return environment
    return payload


def posthoc_attest(args: argparse.Namespace) -> None:
    task = args.seed - SEEDS[0]
    if args.seed not in SEEDS or task not in range(10):
        raise LockError(f"seed {args.seed} is not frozen")
    numeric_source = verify_lock(args.numeric_repository, args.numeric_lock)
    audit_source = verify_audit_lock(
        args.audit_repository,
        args.audit_lock,
        args.numeric_lock,
    )
    verification = original_verification(args.original_stdout, args.seed)
    if verification.get("source") != numeric_source:
        raise LockError("original task verified a different numeric source")
    scheduler = verification.get("scheduler")
    gpu = verification.get("gpu")
    if not isinstance(scheduler, dict) or not isinstance(gpu, dict):
        raise LockError("original verification lacks scheduler/GPU identity")
    if str(scheduler.get("array_task_id")) != str(task):
        raise LockError("original verification has the wrong array task")

    paths = tagged_jsons(args.numeric_repository, args.seed)
    if len(paths) != EXPECTED_RAW_JSONS:
        raise LockError(
            f"seed {args.seed}: {len(paths)} raw JSONs != {EXPECTED_RAW_JSONS}"
        )
    payloads = []
    hosts = set()
    for path in paths:
        payload = strict_json(path)
        if payload.get("seed") != args.seed:
            raise LockError(f"{path}: wrong seed")
        if payload.get("output_suffix") != (
            f"{CAMPAIGN_TAG}-task{args.seed}"
        ):
            raise LockError(f"{path}: wrong suffix")
        if str(payload_array_task_id(payload)) != str(task):
            raise LockError(f"{path}: wrong original array task")
        environment = payload_environment(payload)
        if environment.get("host"):
            hosts.add(environment["host"])
        payloads.append((path, payload))
    if len(hosts) != 1:
        raise LockError(f"seed {args.seed}: inconsistent raw hosts {hosts!r}")
    probes = audit_probes(args.numeric_repository, payloads)
    probe_count = sum(len(row["artifacts"]) for row in probes)
    if probe_count != EXPECTED_PROBE_FILES:
        raise LockError(f"seed {args.seed}: incomplete probe artifacts")

    attestation = {
        "schema_version": SCHEMA_ATTESTATION,
        "campaign_tag": CAMPAIGN_TAG,
        "seed": args.seed,
        "source": numeric_source,
        "audit_source": audit_source,
        "scheduler": scheduler,
        "gpu": gpu,
        "host": next(iter(hosts)),
        "raw_jsons": [
            file_metadata(path, args.numeric_repository) for path in paths
        ],
        "probe_validation": probes,
        "raw_json_count": len(paths),
        "probe_file_count": probe_count,
        "protocol_complete": True,
        "attestation_mode": "posthoc_after_fail_closed_metadata_lookup_error",
        "numerical_rerun_performed": False,
        "original_stdout": file_metadata(
            args.original_stdout,
            args.numeric_repository,
        ),
        "original_stderr": file_metadata(
            args.original_stderr,
            args.numeric_repository,
        ),
    }
    write_once_json(args.output, attestation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    make = subparsers.add_parser("make-audit-lock")
    make.add_argument("--audit-repository", type=Path, required=True)
    make.add_argument("--numeric-lock", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    make.set_defaults(
        function=lambda args: make_audit_lock(
            args.audit_repository,
            args.numeric_lock,
            args.output,
        )
    )
    attest = subparsers.add_parser("attest")
    attest.add_argument("--audit-repository", type=Path, required=True)
    attest.add_argument("--audit-lock", type=Path, required=True)
    attest.add_argument("--numeric-repository", type=Path, required=True)
    attest.add_argument("--numeric-lock", type=Path, required=True)
    attest.add_argument("--seed", type=int, required=True)
    attest.add_argument("--original-stdout", type=Path, required=True)
    attest.add_argument("--original-stderr", type=Path, required=True)
    attest.add_argument("--output", type=Path, required=True)
    attest.set_defaults(function=posthoc_attest)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
