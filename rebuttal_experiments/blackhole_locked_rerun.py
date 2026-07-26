"""Prepare, attest, and audit the immutable black-hole ladder rerun.

The numerical benchmark is deliberately *not* implemented here.  Every task
executes the unmodified ``blackhole-s4-v1`` driver from the exact detached Git
commit recorded in ``campaign_lock.json``.  This outer harness gives that
unchanged protocol a distinct rerun identity, makes array-task ownership
write-once, and audits the array metadata that the original serial aggregator
did not accept.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import patch


CONTROL_ROOT = Path(__file__).resolve().parents[1]
LOCK_SCHEMA = "blackhole_locked_rerun_lock_v1"
ATTESTATION_SCHEMA = "blackhole_locked_rerun_attestation_v1"
AGGREGATE_SCHEMA = "blackhole_locked_rerun_aggregate_v1"
OUTER_CAMPAIGN_TAG = "blackhole-s4-v1-locked-rerun2"
INNER_CAMPAIGN_TAG = "blackhole-s4-v1"
OUTER_ACCOUNT = "marlowe-m000151-pm06"
OUTER_PARTITION = "batch"
OUTER_CPUS_PER_TASK = "12"
INNER_V1_AUDIT_CPUS_PER_TASK = "4"
LOCKED_COMMIT = "96c8018cdd0b56ffe1fd93bddf48fa4affb34697"
RANKS = (2_048, 4_096, 8_192, 16_384, 32_768, 49_152, 65_536)
TASK_TO_RANK = dict(enumerate(RANKS))
MANIFEST_SHA256 = (
    "3b8c6e32da3842b8a9f21ee722e8b9fb1093899744d283d4e939427dd116a5c2"
)
TRACE_SHA256 = (
    "bdf1619a70af70585af317d25aa69452fcb8e94d8417e525de9d50f878e530b4"
)
TRACE_BYTES = 47_448_192
MANIFEST_BYTES = 24_020
MANIFEST_RELATIVE = (
    "rebuttal_results/blackhole/blackhole-s4-v1/feature_manifest.json"
)
TRACE_RELATIVE = (
    "data/the_well/processed/"
    "post_neutron_star_merger_scenario_4_dct65536.npy"
)
LOCK_RELATIVE = (
    "rebuttal_results/blackhole/"
    f"{OUTER_CAMPAIGN_TAG}/campaign_lock.json"
)
HARNESS_PATHS = (
    "rebuttal_experiments/blackhole_locked_rerun.py",
    "rebuttal_experiments/slurm/cuda_blackhole_rank_scaling_locked.sbatch",
    "rebuttal_experiments/BLACKHOLE_LOCKED_RERUN_PROTOCOL.md",
)
LOCKED_SOURCE_PATHS = (
    "rebuttal_experiments/__init__.py",
    "rebuttal_experiments/common.py",
    "rebuttal_experiments/blackhole_features.py",
    "rebuttal_experiments/cuda_blackhole_rank_scaling.py",
    "rebuttal_experiments/cuda_blackhole_stability.py",
    "rebuttal_experiments/aggregate_blackhole.py",
    "student_pkg/doublesided_cholesky.py",
)
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class LockedRerunError(RuntimeError):
    """A fail-closed protocol or provenance violation."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LockedRerunError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise LockedRerunError(f"non-finite JSON constant {value!r}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LockedRerunError(f"cannot read strict JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise LockedRerunError(f"{path}: top-level value must be an object")
    return value


def _equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise LockedRerunError(
            f"{context}={actual!r}; expected frozen value {expected!r}"
        )


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LockedRerunError(f"{context} must be an object")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise LockedRerunError(f"{context} must be a nonempty string")
    return value


def _sha(value: Any, context: str) -> str:
    result = _string(value, context)
    if not SHA_RE.fullmatch(result):
        raise LockedRerunError(f"{context} must be a lowercase SHA-256")
    return result


def _git(
    repository: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ("git", "-C", str(repository), *arguments),
            check=check,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.decode("utf-8", errors="replace").strip()
        raise LockedRerunError(
            f"git {' '.join(arguments)} failed in {repository}: {stderr}"
        ) from error


def _git_text(repository: Path, *arguments: str) -> str:
    return _git(repository, *arguments).stdout.decode("utf-8").strip()


def _git_blob(repository: Path, commit: str, relative: str) -> bytes:
    return _git(repository, "show", f"{commit}:{relative}").stdout


def _atomic_write_once_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_json_write_once(path: Path, value: dict[str, Any]) -> None:
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_write_once_bytes(path, encoded)


def _copy_exact_or_verify(source: Path, destination: Path) -> None:
    data = source.read_bytes()
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != data:
            raise LockedRerunError(
                f"existing destination differs from frozen source: {destination}"
            )
        return
    _atomic_write_once_bytes(destination, data)


def _default_lock(control_repo: Path) -> Path:
    return control_repo / LOCK_RELATIVE


def _default_manifest(control_repo: Path) -> Path:
    return control_repo / MANIFEST_RELATIVE


def _default_trace(control_repo: Path) -> Path:
    return control_repo / TRACE_RELATIVE


def _validate_campaign_directory(campaign_dir: Path) -> None:
    _equal(
        campaign_dir.name,
        OUTER_CAMPAIGN_TAG,
        "campaign directory basename",
    )


def _validate_lock_shape(lock: dict[str, Any]) -> None:
    _equal(
        set(lock),
        {
            "schema_version",
            "campaign_tag",
            "inner_campaign_tag",
            "created_utc",
            "rationale",
            "locked_git_commit",
            "harness_git_commit",
            "inputs",
            "sources",
            "harness",
            "configuration",
            "execution",
        },
        "campaign lock keys",
    )
    _equal(lock.get("schema_version"), LOCK_SCHEMA, "lock.schema_version")
    _equal(lock.get("campaign_tag"), OUTER_CAMPAIGN_TAG, "lock.campaign_tag")
    _equal(
        lock.get("inner_campaign_tag"),
        INNER_CAMPAIGN_TAG,
        "lock.inner_campaign_tag",
    )
    commit = lock.get("locked_git_commit")
    if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
        raise LockedRerunError("lock.locked_git_commit must be a full commit")
    _equal(commit, LOCKED_COMMIT, "lock.locked_git_commit")
    harness_commit = lock.get("harness_git_commit")
    if not isinstance(harness_commit, str) or not COMMIT_RE.fullmatch(
        harness_commit
    ):
        raise LockedRerunError("lock.harness_git_commit must be a full commit")

    inputs = _object(lock.get("inputs"), "lock.inputs")
    _equal(set(inputs), {"feature_manifest", "feature_trace"}, "lock.inputs keys")
    manifest = _object(inputs["feature_manifest"], "lock.inputs.feature_manifest")
    _equal(
        manifest,
        {
            "bytes": MANIFEST_BYTES,
            "embedded_campaign_tag": INNER_CAMPAIGN_TAG,
            "path": MANIFEST_RELATIVE,
            "sha256": MANIFEST_SHA256,
        },
        "lock.inputs.feature_manifest",
    )
    trace = _object(inputs["feature_trace"], "lock.inputs.feature_trace")
    _equal(
        trace,
        {
            "bytes": TRACE_BYTES,
            "dtype": "float32",
            "path": TRACE_RELATIVE,
            "sha256": TRACE_SHA256,
            "shape": [181, 65_536],
        },
        "lock.inputs.feature_trace",
    )

    sources = _object(lock.get("sources"), "lock.sources")
    _equal(set(sources), set(LOCKED_SOURCE_PATHS), "lock.sources keys")
    for relative, metadata_value in sources.items():
        metadata = _object(metadata_value, f"lock.sources.{relative}")
        _equal(set(metadata), {"bytes", "sha256"}, f"lock.sources.{relative} keys")
        if (
            isinstance(metadata["bytes"], bool)
            or not isinstance(metadata["bytes"], int)
            or metadata["bytes"] <= 0
        ):
            raise LockedRerunError(f"invalid source byte count for {relative}")
        _sha(metadata["sha256"], f"lock.sources.{relative}.sha256")

    harness = _object(lock.get("harness"), "lock.harness")
    _equal(set(harness), set(HARNESS_PATHS), "lock.harness keys")
    for relative, metadata_value in harness.items():
        metadata = _object(metadata_value, f"lock.harness.{relative}")
        _equal(set(metadata), {"bytes", "sha256"}, f"lock.harness.{relative} keys")
        if (
            isinstance(metadata["bytes"], bool)
            or not isinstance(metadata["bytes"], int)
            or metadata["bytes"] <= 0
        ):
            raise LockedRerunError(f"invalid harness byte count for {relative}")
        _sha(metadata["sha256"], f"lock.harness.{relative}.sha256")

    configuration = _object(lock.get("configuration"), "lock.configuration")
    _equal(
        configuration,
        {
            "array_task_to_rank": {
                str(task): rank for task, rank in TASK_TO_RANK.items()
            },
            "dtype": "float32",
            "evaluation_transitions": [80, 91, 102, 113, 124, 135, 146, 157, 168, 179],
            "initial_transitions": list(range(64)),
            "method_order": ["inverse", "fresh_refact"],
            "ranks": list(RANKS),
            "repeats": {
                "2048": 10,
                "4096": 10,
                "8192": 10,
                "16384": 5,
                "32768": 5,
                "49152": 3,
                "65536": 3,
            },
            "ridge": 1.0,
            "tiles": {
                "2048": 8,
                "4096": 8,
                "8192": 8,
                "16384": 16,
                "32768": 32,
                "49152": 32,
                "65536": 32,
            },
            "warmup_transition": 64,
        },
        "lock.configuration",
    )
    execution = _object(lock.get("execution"), "lock.execution")
    _equal(
        execution,
        {
            "account": OUTER_ACCOUNT,
            "array_concurrency_cap": 7,
            "cpus_per_task": int(OUTER_CPUS_PER_TASK),
            "gpus_per_task": 1,
            "immutable_mode": "detached_git_worktree",
            "inner_driver_is_unmodified": True,
            "outer_attestation_required": True,
            "partition": OUTER_PARTITION,
            "result_publication": "atomic_claim_then_inner_driver_write-once",
        },
        "lock.execution",
    )


def _verify_control_harness(lock: dict[str, Any], control_repo: Path) -> None:
    harness_commit = lock["harness_git_commit"]
    _equal(
        _git_text(control_repo, "rev-parse", f"{harness_commit}^{{commit}}"),
        harness_commit,
        "resolved harness commit",
    )
    for relative, metadata in lock["harness"].items():
        path = control_repo / relative
        if not path.is_file():
            raise LockedRerunError(f"missing locked harness file: {path}")
        _equal(path.stat().st_size, metadata["bytes"], f"{relative} byte count")
        _equal(
            _sha256_file(path),
            metadata["sha256"],
            f"{relative} SHA-256",
        )
        blob = _git_blob(control_repo, harness_commit, relative)
        _equal(len(blob), metadata["bytes"], f"committed {relative} byte count")
        _equal(
            _sha256_bytes(blob),
            metadata["sha256"],
            f"committed {relative} SHA-256",
        )
    changed = _git(
        control_repo,
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--",
        *HARNESS_PATHS,
    ).stdout
    if changed:
        raise LockedRerunError("outer rerun harness differs from its locked bytes")


def _canonical_lock_commit(
    control_repo: Path,
    lock: dict[str, Any],
    lock_path: Path | None = None,
) -> str:
    canonical = _default_lock(control_repo)
    if not canonical.is_file():
        raise LockedRerunError(f"missing canonical campaign lock: {canonical}")
    canonical_value = _load_json(canonical)
    if canonical_value != lock:
        raise LockedRerunError("campaign lock differs from the canonical lock")
    if lock_path is not None:
        if not lock_path.is_file():
            raise LockedRerunError(f"missing supplied campaign lock: {lock_path}")
        _equal(
            lock_path.stat().st_size,
            canonical.stat().st_size,
            "campaign/canonical lock byte count",
        )
        _equal(
            _sha256_file(lock_path),
            _sha256_file(canonical),
            "campaign/canonical lock SHA-256",
        )
    relative = str(canonical.relative_to(control_repo))
    tracked = _git(
        control_repo,
        "ls-files",
        "--error-unmatch",
        "--",
        relative,
        check=False,
    )
    if tracked.returncode != 0:
        raise LockedRerunError("canonical campaign lock is not tracked by Git")
    changed = _git(
        control_repo,
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--",
        relative,
    ).stdout
    if changed:
        raise LockedRerunError("canonical campaign lock is dirty")
    commit = _git_text(control_repo, "log", "-1", "--format=%H", "--", relative)
    if not COMMIT_RE.fullmatch(commit):
        raise LockedRerunError("could not resolve canonical lock commit")
    return commit


def _verify_git_sources(
    lock: dict[str, Any],
    repository: Path,
    *,
    require_head: bool,
) -> None:
    if not (repository / ".git").exists():
        # A linked worktree has a .git file rather than a directory.
        if not (repository / ".git").is_file():
            raise LockedRerunError(f"not a Git checkout: {repository}")
    actual_commit = _git_text(repository, "rev-parse", f"{LOCKED_COMMIT}^{{commit}}")
    _equal(actual_commit, LOCKED_COMMIT, "resolved locked commit")
    if require_head:
        _equal(
            _git_text(repository, "rev-parse", "HEAD"),
            LOCKED_COMMIT,
            "detached worktree HEAD",
        )
        symbolic_head = _git(
            repository,
            "symbolic-ref",
            "-q",
            "HEAD",
            check=False,
        )
        if symbolic_head.returncode == 0:
            branch = symbolic_head.stdout.decode("utf-8", errors="replace").strip()
            raise LockedRerunError(
                "locked source must have detached HEAD, but points to "
                f"{branch or 'a symbolic ref'}"
            )
    for relative, metadata in lock["sources"].items():
        blob = _git_blob(repository, LOCKED_COMMIT, relative)
        _equal(len(blob), metadata["bytes"], f"Git blob {relative} byte count")
        _equal(
            _sha256_bytes(blob),
            metadata["sha256"],
            f"Git blob {relative} SHA-256",
        )
        if require_head:
            path = repository / relative
            if not path.is_file():
                raise LockedRerunError(f"missing worktree source: {path}")
            _equal(path.stat().st_size, metadata["bytes"], f"{relative} byte count")
            _equal(
                _sha256_file(path),
                metadata["sha256"],
                f"{relative} worktree SHA-256",
            )
    if require_head:
        changed = _git(
            repository,
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            *LOCKED_SOURCE_PATHS,
        ).stdout
        if changed:
            raise LockedRerunError(
                "locked computational sources are dirty in the detached worktree"
            )


def _verify_input_file(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    label: str,
) -> None:
    if not path.is_file():
        raise LockedRerunError(f"missing {label}: {path}")
    _equal(path.stat().st_size, expected_bytes, f"{label} byte count")
    _equal(_sha256_file(path), expected_sha256, f"{label} SHA-256")


def _verify_manifest_semantics(path: Path) -> None:
    manifest = _load_json(path)
    _equal(
        manifest.get("campaign_tag"),
        INNER_CAMPAIGN_TAG,
        "v1 manifest embedded campaign tag",
    )
    trace = _object(manifest.get("trace"), "feature manifest trace")
    _equal(trace.get("sha256"), TRACE_SHA256, "feature manifest trace SHA-256")
    _equal(trace.get("path"), TRACE_RELATIVE, "feature manifest trace path")


def _load_and_verify_lock(
    *,
    lock_path: Path,
    control_repo: Path,
    source_repo: Path,
    require_source_head: bool,
) -> dict[str, Any]:
    lock = _load_json(lock_path)
    _validate_lock_shape(lock)
    _verify_control_harness(lock, control_repo)
    _canonical_lock_commit(control_repo, lock, lock_path)
    _verify_git_sources(lock, source_repo, require_head=require_source_head)
    return lock


def _rank_task(rank: int) -> int:
    if rank not in RANKS:
        raise LockedRerunError(f"rank {rank} is not frozen in {RANKS}")
    return RANKS.index(rank)


def _slurm_metadata(rank: int) -> dict[str, Any]:
    required = {
        "job_id": "SLURM_JOB_ID",
        "array_job_id": "SLURM_ARRAY_JOB_ID",
        "array_task_id": "SLURM_ARRAY_TASK_ID",
        "array_task_count": "SLURM_ARRAY_TASK_COUNT",
        "array_task_min": "SLURM_ARRAY_TASK_MIN",
        "array_task_max": "SLURM_ARRAY_TASK_MAX",
        "array_task_step": "SLURM_ARRAY_TASK_STEP",
        "account": "SLURM_JOB_ACCOUNT",
        "partition": "SLURM_JOB_PARTITION",
        "cpus_per_task": "SLURM_CPUS_PER_TASK",
        "job_name": "SLURM_JOB_NAME",
        "node_list": "SLURM_JOB_NODELIST",
    }
    values: dict[str, Any] = {}
    for key, variable in required.items():
        value = os.environ.get(variable)
        if value is None or value == "":
            raise LockedRerunError(f"{variable} must be set by Slurm")
        values[key] = value
    for key in (
        "job_id",
        "array_job_id",
        "array_task_id",
        "array_task_count",
        "array_task_min",
        "array_task_max",
        "array_task_step",
    ):
        if not str(values[key]).isdigit():
            raise LockedRerunError(f"Slurm {key} must be numeric")
    expected_task = _rank_task(rank)
    _equal(int(values["array_task_id"]), expected_task, "Slurm task-to-rank mapping")
    _equal(values["array_task_count"], "7", "Slurm array task count")
    _equal(values["array_task_min"], "0", "Slurm array task minimum")
    _equal(values["array_task_max"], "6", "Slurm array task maximum")
    _equal(values["array_task_step"], "1", "Slurm array task step")
    _equal(values["account"], OUTER_ACCOUNT, "Slurm account")
    _equal(values["partition"], OUTER_PARTITION, "Slurm partition")
    _equal(
        values["cpus_per_task"],
        OUTER_CPUS_PER_TASK,
        "Slurm CPUs per task",
    )
    values.update(
        {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "step_gpus": os.environ.get("SLURM_STEP_GPUS"),
            "gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
        }
    )
    return values


def _gpu_identity() -> dict[str, str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible or "," in visible:
        raise LockedRerunError(
            "CUDA_VISIBLE_DEVICES must identify exactly one allocated GPU"
        )
    command = [
        "nvidia-smi",
        f"--id={visible}",
        "--query-gpu=uuid,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise LockedRerunError(f"could not query allocated GPU: {error}") from error
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise LockedRerunError(f"expected one nvidia-smi row, got {rows!r}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 4:
        raise LockedRerunError(f"unexpected nvidia-smi row: {rows[0]!r}")
    uuid, name, memory_mib, driver_version = fields
    if not uuid.startswith("GPU-") or not name.startswith("NVIDIA H100"):
        raise LockedRerunError(f"unexpected allocated GPU: {rows[0]!r}")
    return {
        "uuid": uuid,
        "name": name,
        "memory_mib": memory_mib,
        "driver_version": driver_version,
    }


def _validate_result_provenance(
    payload: dict[str, Any],
    *,
    rank: int,
    lock: dict[str, Any],
    slurm: dict[str, Any] | None,
) -> None:
    _equal(payload.get("campaign_tag"), INNER_CAMPAIGN_TAG, "result campaign tag")
    _equal(payload.get("rank"), rank, "result rank")
    _equal(
        payload.get("schema_version"),
        "cuda_blackhole_rank_scaling_v1",
        "result schema",
    )
    _equal(
        payload.get("protocol_version"),
        "blackhole_s4_ladder_v1",
        "result protocol",
    )
    source = _object(payload.get("source"), "result source")
    _equal(source.get("git_commit"), LOCKED_COMMIT, "result source commit")
    _equal(source.get("git_dirty"), False, "result source dirty flag")
    expected_source_fields = {
        "driver_sha256": lock["sources"][
            "rebuttal_experiments/cuda_blackhole_rank_scaling.py"
        ]["sha256"],
        "kernel_sha256": lock["sources"][
            "student_pkg/doublesided_cholesky.py"
        ]["sha256"],
        "feature_source_sha256": lock["sources"][
            "rebuttal_experiments/blackhole_features.py"
        ]["sha256"],
    }
    for key, expected in expected_source_fields.items():
        _equal(source.get(key), expected, f"result source {key}")
    feature_manifest = _object(
        payload.get("feature_manifest"), "result feature manifest"
    )
    _equal(
        feature_manifest.get("sha256"),
        MANIFEST_SHA256,
        "result feature manifest SHA-256",
    )
    trace = _object(payload.get("trace"), "result trace")
    _equal(trace.get("sha256"), TRACE_SHA256, "result trace SHA-256")
    if slurm is not None:
        recorded = _object(payload.get("slurm"), "result Slurm metadata")
        for result_key, attestation_key in (
            ("job_id", "job_id"),
            ("array_job_id", "array_job_id"),
            ("array_task_id", "array_task_id"),
            ("cpus_per_task", "cpus_per_task"),
            ("job_partition", "partition"),
        ):
            _equal(
                recorded.get(result_key),
                slurm[attestation_key],
                f"result Slurm {result_key}",
            )
        _equal(payload.get("host"), socket.gethostname(), "result host")


def prepare(args: argparse.Namespace) -> None:
    control_repo = args.control_repo.resolve()
    source_repo = args.source_repo.resolve()
    locked_worktree = args.locked_worktree.resolve()
    campaign_dir = args.campaign_dir.resolve()
    _validate_campaign_directory(campaign_dir)
    lock_source = (args.lock_file or _default_lock(control_repo)).resolve()
    manifest_source = (args.manifest or _default_manifest(control_repo)).resolve()
    trace_source = (args.trace or _default_trace(control_repo)).resolve()

    lock = _load_and_verify_lock(
        lock_path=lock_source,
        control_repo=control_repo,
        source_repo=source_repo,
        require_source_head=False,
    )
    _verify_input_file(
        manifest_source,
        expected_bytes=MANIFEST_BYTES,
        expected_sha256=MANIFEST_SHA256,
        label="v1 feature manifest",
    )
    _verify_manifest_semantics(manifest_source)
    _verify_input_file(
        trace_source,
        expected_bytes=TRACE_BYTES,
        expected_sha256=TRACE_SHA256,
        label="feature trace",
    )

    if not locked_worktree.exists():
        locked_worktree.parent.mkdir(parents=True, exist_ok=True)
        _git(
            source_repo,
            "worktree",
            "add",
            "--detach",
            str(locked_worktree),
            LOCKED_COMMIT,
        )
    _verify_git_sources(lock, locked_worktree, require_head=True)

    worktree_trace = locked_worktree / TRACE_RELATIVE
    worktree_trace.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(worktree_trace):
        _verify_input_file(
            worktree_trace,
            expected_bytes=TRACE_BYTES,
            expected_sha256=TRACE_SHA256,
            label="locked-worktree feature trace",
        )
    else:
        worktree_trace.symlink_to(trace_source)
        _verify_input_file(
            worktree_trace,
            expected_bytes=TRACE_BYTES,
            expected_sha256=TRACE_SHA256,
            label="locked-worktree feature trace",
        )

    campaign_dir.mkdir(parents=True, exist_ok=True)
    _copy_exact_or_verify(lock_source, campaign_dir / "campaign_lock.json")
    _copy_exact_or_verify(manifest_source, campaign_dir / "feature_manifest.json")
    (campaign_dir / ".claims").mkdir(exist_ok=True)
    (campaign_dir / "logs").mkdir(exist_ok=True)
    print(f"prepared immutable worktree: {locked_worktree}", flush=True)
    print(f"prepared campaign directory: {campaign_dir}", flush=True)


def verify_task(args: argparse.Namespace) -> None:
    control_repo = args.control_repo.resolve()
    source_repo = args.source_repo.resolve()
    locked_worktree = args.locked_worktree.resolve()
    campaign_dir = args.campaign_dir.resolve()
    _validate_campaign_directory(campaign_dir)
    lock_path = campaign_dir / "campaign_lock.json"
    _load_and_verify_lock(
        lock_path=lock_path,
        control_repo=control_repo,
        source_repo=source_repo,
        require_source_head=True,
    )
    _equal(source_repo, locked_worktree, "task source/worktree identity")
    manifest = campaign_dir / "feature_manifest.json"
    _verify_input_file(
        manifest,
        expected_bytes=MANIFEST_BYTES,
        expected_sha256=MANIFEST_SHA256,
        label="campaign feature manifest",
    )
    _verify_manifest_semantics(manifest)
    _verify_input_file(
        locked_worktree / TRACE_RELATIVE,
        expected_bytes=TRACE_BYTES,
        expected_sha256=TRACE_SHA256,
        label="locked-worktree feature trace",
    )
    _slurm_metadata(args.rank)
    print(
        f"verified immutable task {_rank_task(args.rank)} -> rank {args.rank}",
        flush=True,
    )


def attest_task(args: argparse.Namespace) -> None:
    control_repo = args.control_repo.resolve()
    source_repo = args.source_repo.resolve()
    locked_worktree = args.locked_worktree.resolve()
    campaign_dir = args.campaign_dir.resolve()
    _validate_campaign_directory(campaign_dir)
    lock_path = campaign_dir / "campaign_lock.json"
    lock = _load_and_verify_lock(
        lock_path=lock_path,
        control_repo=control_repo,
        source_repo=source_repo,
        require_source_head=True,
    )
    _equal(source_repo, locked_worktree, "task source/worktree identity")
    slurm = _slurm_metadata(args.rank)
    result_path = campaign_dir / f"ladder_r{args.rank}.json"
    result = _load_json(result_path)
    _validate_result_provenance(
        result,
        rank=args.rank,
        lock=lock,
        slurm=slurm,
    )
    attestation_path = campaign_dir / f"attestation_r{args.rank}.json"
    if attestation_path.exists():
        raise LockedRerunError(
            f"refusing to replace attestation: {attestation_path}"
        )
    payload = {
        "schema_version": ATTESTATION_SCHEMA,
        "campaign_tag": OUTER_CAMPAIGN_TAG,
        "inner_campaign_tag": INNER_CAMPAIGN_TAG,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rank": args.rank,
        "array_task_id": _rank_task(args.rank),
        "campaign_lock": {
            "bytes": lock_path.stat().st_size,
            "git_commit": _canonical_lock_commit(control_repo, lock),
            "sha256": _sha256_file(lock_path),
        },
        "raw_result": {
            "bytes": result_path.stat().st_size,
            "filename": result_path.name,
            "sha256": _sha256_file(result_path),
        },
        "immutable_source": {
            "git_commit": LOCKED_COMMIT,
            "mode": "detached_git_worktree",
            "source_hashes": {
                relative: metadata["sha256"]
                for relative, metadata in lock["sources"].items()
            },
        },
        "inputs": {
            "feature_manifest_sha256": MANIFEST_SHA256,
            "feature_trace_sha256": TRACE_SHA256,
        },
        "slurm": slurm,
        "host": socket.gethostname(),
        "gpu": _gpu_identity(),
    }
    _atomic_json_write_once(attestation_path, payload)
    print(f"wrote {attestation_path}", flush=True)


def _import_locked_aggregator(locked_worktree: Path) -> Any:
    root_string = str(locked_worktree)
    sys.path.insert(0, root_string)
    try:
        for name in tuple(sys.modules):
            if name == "rebuttal_experiments" or name.startswith(
                "rebuttal_experiments."
            ):
                del sys.modules[name]
        module = importlib.import_module("rebuttal_experiments.aggregate_blackhole")
    finally:
        try:
            sys.path.remove(root_string)
        except ValueError:
            pass
    expected = (
        locked_worktree / "rebuttal_experiments" / "aggregate_blackhole.py"
    ).resolve()
    _equal(Path(module.__file__).resolve(), expected, "locked aggregator import")
    return module


def _expected_json_names() -> set[str]:
    names = {"campaign_lock.json", "feature_manifest.json"}
    names.update(f"ladder_r{rank}.json" for rank in RANKS)
    names.update(f"attestation_r{rank}.json" for rank in RANKS)
    return names


def _validate_attestation(
    path: Path,
    *,
    result_path: Path,
    rank: int,
    lock_sha256: str,
    expected_lock_git_commit: str,
    lock: dict[str, Any],
    expected_account: str,
    expected_partition: str,
) -> dict[str, Any]:
    value = _load_json(path)
    _equal(
        set(value),
        {
            "schema_version",
            "campaign_tag",
            "inner_campaign_tag",
            "created_utc",
            "rank",
            "array_task_id",
            "campaign_lock",
            "raw_result",
            "immutable_source",
            "inputs",
            "slurm",
            "host",
            "gpu",
        },
        f"{path} keys",
    )
    _equal(value.get("schema_version"), ATTESTATION_SCHEMA, f"{path} schema")
    _equal(value.get("campaign_tag"), OUTER_CAMPAIGN_TAG, f"{path} campaign")
    _equal(
        value.get("inner_campaign_tag"),
        INNER_CAMPAIGN_TAG,
        f"{path} inner campaign",
    )
    _equal(value.get("rank"), rank, f"{path} rank")
    _equal(value.get("array_task_id"), _rank_task(rank), f"{path} task")
    lock_record = _object(value.get("campaign_lock"), f"{path} campaign lock")
    _equal(lock_record.get("sha256"), lock_sha256, f"{path} lock SHA-256")
    lock_commit = _string(
        lock_record.get("git_commit"),
        f"{path} campaign lock Git commit",
    )
    if not COMMIT_RE.fullmatch(lock_commit):
        raise LockedRerunError(f"{path}: malformed campaign lock Git commit")
    _equal(
        lock_commit,
        expected_lock_git_commit,
        f"{path} canonical campaign lock Git commit",
    )
    raw = _object(value.get("raw_result"), f"{path} raw result")
    _equal(raw.get("filename"), result_path.name, f"{path} result filename")
    _equal(raw.get("bytes"), result_path.stat().st_size, f"{path} result bytes")
    _equal(
        raw.get("sha256"),
        _sha256_file(result_path),
        f"{path} result SHA-256",
    )
    source = _object(value.get("immutable_source"), f"{path} immutable source")
    _equal(source.get("git_commit"), LOCKED_COMMIT, f"{path} source commit")
    _equal(
        source.get("mode"),
        "detached_git_worktree",
        f"{path} immutable mode",
    )
    _equal(
        source.get("source_hashes"),
        {
            relative: metadata["sha256"]
            for relative, metadata in lock["sources"].items()
        },
        f"{path} source hashes",
    )
    _equal(
        value.get("inputs"),
        {
            "feature_manifest_sha256": MANIFEST_SHA256,
            "feature_trace_sha256": TRACE_SHA256,
        },
        f"{path} inputs",
    )
    slurm = _object(value.get("slurm"), f"{path} Slurm")
    for key in ("job_id", "array_job_id", "array_task_id"):
        item = slurm.get(key)
        if not isinstance(item, str) or not item.isdigit():
            raise LockedRerunError(f"{path}: Slurm {key} must be numeric")
    _equal(int(slurm["array_task_id"]), _rank_task(rank), f"{path} Slurm task")
    _equal(slurm.get("account"), expected_account, f"{path} Slurm account")
    _equal(
        slurm.get("partition"),
        expected_partition,
        f"{path} Slurm partition",
    )
    _equal(
        slurm.get("cpus_per_task"),
        OUTER_CPUS_PER_TASK,
        f"{path} Slurm CPUs",
    )
    host = _string(value.get("host"), f"{path} host")
    gpu = _object(value.get("gpu"), f"{path} GPU")
    _equal(set(gpu), {"uuid", "name", "memory_mib", "driver_version"}, f"{path} GPU keys")
    if not _string(gpu.get("uuid"), f"{path} GPU UUID").startswith("GPU-"):
        raise LockedRerunError(f"{path}: malformed GPU UUID")
    if not _string(gpu.get("name"), f"{path} GPU name").startswith("NVIDIA H100"):
        raise LockedRerunError(f"{path}: expected H100")
    _string(gpu.get("memory_mib"), f"{path} GPU memory")
    _string(gpu.get("driver_version"), f"{path} driver version")
    return {
        "slurm": slurm,
        "host": host,
        "gpu": gpu,
        "raw_result_sha256": raw["sha256"],
        "campaign_lock_git_commit": lock_commit,
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Locked black-hole ladder rerun",
        "",
        (
            f"Outer campaign `{summary['campaign_tag']}`; unchanged inner "
            f"protocol `{summary['inner_campaign_tag']}`; immutable commit "
            f"`{summary['audit']['locked_git_commit']}`."
        ),
        "",
        "| rank | inverse event, s (mean ± SD) | fresh event, s (mean ± SD) | speedup |",
        "|---:|---:|---:|---:|",
    ]
    for row in summary["ladder"]:
        inverse = row["inverse_event_seconds"]
        fresh = row["fresh_event_seconds"]
        speedup = row.get("paired_event_speedup_fresh_over_inverse")
        speedup_text = (
            f"{speedup['mean']:.3f}x" if speedup is not None else "withheld"
        )
        lines.append(
            f"| {row['rank']} | {inverse['mean']:.6g} ± {inverse['sample_sd']:.3g} "
            f"| {fresh['mean']:.6g} ± {fresh['sample_sd']:.3g} "
            f"| {speedup_text} |"
        )
    lines.append("")
    return "\n".join(lines)


def _validate_parallel_contexts(
    contexts: list[dict[str, Any]],
    *,
    expected_account: str,
    expected_partition: str,
    require_distinct_gpus: bool,
) -> None:
    _equal(len(contexts), len(RANKS), "parallel execution context count")
    _equal(
        len({row["array_job_id"] for row in contexts}),
        1,
        "Slurm array job count",
    )
    _equal(
        {int(row["array_task_id"]) for row in contexts},
        set(TASK_TO_RANK),
        "Slurm array task set",
    )
    for row in contexts:
        _equal(
            int(row["array_task_id"]),
            _rank_task(int(row["rank"])),
            f"rank {row['rank']} task mapping",
        )
    _equal(
        len({row["job_id"] for row in contexts}),
        len(RANKS),
        "unique Slurm job ID count",
    )
    for key, expected in (
        ("array_task_count", "7"),
        ("array_task_min", "0"),
        ("array_task_max", "6"),
        ("array_task_step", "1"),
    ):
        _equal(
            {row[key] for row in contexts},
            {expected},
            f"Slurm {key} set",
        )
    _equal(
        {row["account"] for row in contexts},
        {expected_account},
        "Slurm account set",
    )
    _equal(
        {row["partition"] for row in contexts},
        {expected_partition},
        "Slurm partition set",
    )
    _equal(
        len({row["campaign_lock_git_commit"] for row in contexts}),
        1,
        "campaign lock Git commit count",
    )
    environment_strings = {
        json.dumps(
            {
                "hardware": row["hardware"],
                "platform": row["platform"],
                "python": row["python"],
                "gpu_name": row["gpu"]["name"],
                "gpu_memory_mib": row["gpu"]["memory_mib"],
                "gpu_driver": row["gpu"]["driver_version"],
            },
            sort_keys=True,
        )
        for row in contexts
    }
    _equal(len(environment_strings), 1, "hardware/software environment count")
    if require_distinct_gpus:
        _equal(
            len({row["gpu"]["uuid"] for row in contexts}),
            len(RANKS),
            "distinct physical GPU count",
        )


def _preflight_output_pair(output_json: Path, output_markdown: Path) -> None:
    if output_json == output_markdown:
        raise LockedRerunError("aggregate JSON and Markdown paths must differ")
    existing = [path for path in (output_json, output_markdown) if path.exists()]
    if existing:
        raise LockedRerunError(
            "refusing a partial or replacement aggregate; existing output(s): "
            + ", ".join(str(path) for path in existing)
        )


def _validate_claim_entries(claims_dir: Path) -> None:
    claim_entries = list(claims_dir.iterdir()) if claims_dir.is_dir() else []
    _equal(
        {path.name for path in claim_entries},
        {f"ladder_r{rank}.claim" for rank in RANKS},
        "persistent task claim set",
    )
    non_directories = [path for path in claim_entries if not path.is_dir()]
    if non_directories:
        raise LockedRerunError(
            "every task claim must be a directory: "
            + ", ".join(str(path) for path in non_directories)
        )


def aggregate(args: argparse.Namespace) -> None:
    control_repo = args.control_repo.resolve()
    source_repo = args.source_repo.resolve()
    locked_worktree = args.locked_worktree.resolve()
    campaign_dir = args.campaign_dir.resolve()
    _validate_campaign_directory(campaign_dir)
    expected_account = _string(args.expected_account, "expected account")
    expected_partition = _string(args.expected_partition, "expected partition")
    lock_path = campaign_dir / "campaign_lock.json"
    lock = _load_and_verify_lock(
        lock_path=lock_path,
        control_repo=control_repo,
        source_repo=source_repo,
        require_source_head=True,
    )
    _equal(source_repo, locked_worktree, "aggregate source/worktree identity")
    lock_sha256 = _sha256_file(lock_path)
    lock_git_commit = _canonical_lock_commit(control_repo, lock)
    manifest_path = campaign_dir / "feature_manifest.json"
    _verify_input_file(
        manifest_path,
        expected_bytes=MANIFEST_BYTES,
        expected_sha256=MANIFEST_SHA256,
        label="campaign feature manifest",
    )
    _verify_manifest_semantics(manifest_path)
    _verify_input_file(
        locked_worktree / TRACE_RELATIVE,
        expected_bytes=TRACE_BYTES,
        expected_sha256=TRACE_SHA256,
        label="locked-worktree feature trace",
    )

    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else campaign_dir / "aggregate.json"
    )
    output_markdown = (
        args.output_markdown.resolve()
        if args.output_markdown is not None
        else campaign_dir / "aggregate.md"
    )
    _preflight_output_pair(output_json, output_markdown)
    excluded = {output_json, output_markdown}
    observed_json = {
        path.name
        for path in campaign_dir.glob("*.json")
        if path.resolve() not in excluded
    }
    _equal(observed_json, _expected_json_names(), "campaign JSON artifact set")
    _validate_claim_entries(campaign_dir / ".claims")

    base = _import_locked_aggregator(locked_worktree)
    (
        manifest,
        manifest_sha256,
        trace_sha256,
        feature_source_sha256,
    ) = base._validate_manifest(manifest_path)
    _equal(manifest_sha256, MANIFEST_SHA256, "base-audited manifest SHA-256")
    _equal(trace_sha256, TRACE_SHA256, "base-audited trace SHA-256")

    records: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="blackhole-locked-audit-") as temporary:
        temporary_dir = Path(temporary)
        for rank in RANKS:
            result_path = campaign_dir / f"ladder_r{rank}.json"
            attestation = _validate_attestation(
                campaign_dir / f"attestation_r{rank}.json",
                result_path=result_path,
                rank=rank,
                lock_sha256=lock_sha256,
                expected_lock_git_commit=lock_git_commit,
                lock=lock,
                expected_account=expected_account,
                expected_partition=expected_partition,
            )
            payload = _load_json(result_path)
            _validate_result_provenance(
                payload,
                rank=rank,
                lock=lock,
                slurm=None,
            )
            recorded_slurm = _object(payload.get("slurm"), f"rank {rank} Slurm")
            for result_key, attestation_key in (
                ("job_id", "job_id"),
                ("array_job_id", "array_job_id"),
                ("array_task_id", "array_task_id"),
                ("cpus_per_task", "cpus_per_task"),
                ("job_partition", "partition"),
            ):
                _equal(
                    recorded_slurm.get(result_key),
                    attestation["slurm"][attestation_key],
                    f"rank {rank} result/attestation Slurm {result_key}",
                )
            _equal(payload.get("host"), attestation["host"], f"rank {rank} host")

            # The v1 auditor intentionally requires a serial, non-array,
            # preempt job.  Sanitize only those five already-audited scheduler
            # fields in a temporary copy; the actual array values were
            # strictly checked above and remain untouched on disk.
            sanitized = copy.deepcopy(payload)
            sanitized["slurm"] = {
                "job_id": attestation["slurm"]["job_id"],
                "array_job_id": None,
                "array_task_id": None,
                "cpus_per_task": INNER_V1_AUDIT_CPUS_PER_TASK,
                "job_partition": "preempt",
            }
            sanitized_path = temporary_dir / result_path.name
            sanitized_path.write_text(
                json.dumps(sanitized, allow_nan=False),
                encoding="utf-8",
            )
            record, source_context = base._validate_ladder(
                sanitized_path,
                rank=rank,
                manifest_sha256=manifest_sha256,
                trace_sha256=trace_sha256,
                feature_source_sha256=feature_source_sha256,
                expected_rank_scale=float(
                    manifest["rank_scaling"]["scales"][str(rank)]
                ),
            )
            _equal(source_context[0], LOCKED_COMMIT, f"rank {rank} base commit")
            records.append(record)
            contexts.append(
                {
                    "rank": rank,
                    "array_job_id": attestation["slurm"]["array_job_id"],
                    "array_task_id": attestation["slurm"]["array_task_id"],
                    "job_id": attestation["slurm"]["job_id"],
                    "account": attestation["slurm"]["account"],
                    "partition": attestation["slurm"]["partition"],
                    "host": attestation["host"],
                    "gpu": attestation["gpu"],
                    "hardware": payload["hardware"],
                    "platform": payload["platform"],
                    "python": payload["python"],
                    "raw_result_sha256": attestation["raw_result_sha256"],
                    "campaign_lock_git_commit": attestation[
                        "campaign_lock_git_commit"
                    ],
                }
            )

    _validate_parallel_contexts(
        contexts,
        expected_account=expected_account,
        expected_partition=expected_partition,
        require_distinct_gpus=True,
    )

    # The original strict aggregator treats the ten predeclared real
    # transitions as n=10.  Each transition contributes its within-case
    # median; timing repeats remain retained in the raw artifact and are not
    # promoted to independent observations.
    ladder = [base._aggregate_ladder(record) for record in records]
    for record, row in zip(records, ladder):
        inverse = [
            case["methods"]["inverse"]["event_seconds_median"]
            for case in record["cases"]
        ]
        fresh = [
            case["methods"]["fresh_refact"]["event_seconds_median"]
            for case in record["cases"]
        ]
        row["ratio_of_transition_median_sums_fresh_over_inverse"] = (
            sum(fresh) / sum(inverse)
        )
    all_valid = all(row["headline_valid"] for row in ladder)
    summary = {
        "schema_version": AGGREGATE_SCHEMA,
        "experiment": "cuda_blackhole_rank_scaling_locked_rerun",
        "campaign_tag": OUTER_CAMPAIGN_TAG,
        "inner_campaign_tag": INNER_CAMPAIGN_TAG,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "audit": {
            "passed": True,
            "all_headline_correctness_valid": all_valid,
            "headline_claims_withheld": not all_valid,
            "locked_git_commit": LOCKED_COMMIT,
            "campaign_lock_sha256": lock_sha256,
            "campaign_lock_git_commit": contexts[0][
                "campaign_lock_git_commit"
            ],
            "feature_manifest_sha256": MANIFEST_SHA256,
            "feature_trace_sha256": TRACE_SHA256,
            "slurm_array_job_id": contexts[0]["array_job_id"],
            "slurm_account": expected_account,
            "slurm_partition": expected_partition,
            "distinct_gpu_count": len(RANKS),
            "task_to_rank": {
                str(task): rank for task, rank in TASK_TO_RANK.items()
            },
        },
        "execution_contexts": contexts,
        "ladder": ladder,
    }
    _atomic_json_write_once(output_json, summary)
    _atomic_write_once_bytes(output_markdown, _markdown(summary).encode("utf-8"))
    print(f"wrote {output_json}", flush=True)
    print(f"wrote {output_markdown}", flush=True)


def self_test(args: argparse.Namespace) -> None:
    control_repo = args.control_repo.resolve()
    source_repo = args.source_repo.resolve()
    lock_path = (args.lock_file or _default_lock(control_repo)).resolve()
    lock = _load_and_verify_lock(
        lock_path=lock_path,
        control_repo=control_repo,
        source_repo=source_repo,
        require_source_head=False,
    )
    _equal(_rank_task(2_048), 0, "self-test first rank")
    _equal(_rank_task(65_536), 6, "self-test last rank")
    try:
        _rank_task(1_024)
    except LockedRerunError:
        pass
    else:
        raise AssertionError("unexpected rank was accepted")

    def expect_locked_error(function: Any, label: str) -> None:
        try:
            function()
        except LockedRerunError:
            return
        raise AssertionError(f"self-test expected rejection: {label}")

    with tempfile.TemporaryDirectory(prefix="blackhole-path-self-test-") as temporary:
        root = Path(temporary)
        _validate_campaign_directory(root / OUTER_CAMPAIGN_TAG)
        expect_locked_error(
            lambda: _validate_campaign_directory(
                root / "blackhole-s4-v1-locked-rerun1"
            ),
            "rerun-1 campaign directory",
        )
        expect_locked_error(
            lambda: _validate_campaign_directory(root / "arbitrary-output"),
            "arbitrary campaign directory",
        )

    slurm_environment = {
        "SLURM_JOB_ID": "800",
        "SLURM_ARRAY_JOB_ID": "800",
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_ARRAY_TASK_COUNT": "7",
        "SLURM_ARRAY_TASK_MIN": "0",
        "SLURM_ARRAY_TASK_MAX": "6",
        "SLURM_ARRAY_TASK_STEP": "1",
        "SLURM_JOB_ACCOUNT": OUTER_ACCOUNT,
        "SLURM_JOB_PARTITION": OUTER_PARTITION,
        "SLURM_CPUS_PER_TASK": OUTER_CPUS_PER_TASK,
        "SLURM_JOB_NAME": "bh-s4-locked-r2",
        "SLURM_JOB_NODELIST": "n01",
    }
    with patch.dict(os.environ, slurm_environment):
        _equal(
            _slurm_metadata(RANKS[0])["cpus_per_task"],
            OUTER_CPUS_PER_TASK,
            "self-test accepted CPU allocation",
        )
    wrong_cpu_environment = dict(slurm_environment)
    wrong_cpu_environment["SLURM_CPUS_PER_TASK"] = INNER_V1_AUDIT_CPUS_PER_TASK
    with patch.dict(os.environ, wrong_cpu_environment):
        expect_locked_error(
            lambda: _slurm_metadata(RANKS[0]),
            "rerun-1 four-core allocation",
        )
    excessive_cpu_environment = dict(slurm_environment)
    excessive_cpu_environment["SLURM_CPUS_PER_TASK"] = "13"
    with patch.dict(os.environ, excessive_cpu_environment):
        expect_locked_error(
            lambda: _slurm_metadata(RANKS[0]),
            "non-frozen 13-core allocation",
        )

    with tempfile.TemporaryDirectory(prefix="blackhole-lock-self-test-") as temporary:
        root = Path(temporary)
        destination = root / "once.json"
        _atomic_json_write_once(destination, {"finite": 1.0})
        try:
            _atomic_json_write_once(destination, {"finite": 2.0})
        except FileExistsError:
            pass
        else:
            raise AssertionError("write-once publication was replaceable")
        duplicate = root / "duplicate.json"
        duplicate.write_text('{"a": 1, "a": 2}', encoding="utf-8")
        try:
            _load_json(duplicate)
        except LockedRerunError:
            pass
        else:
            raise AssertionError("duplicate JSON key was accepted")
        aggregate_markdown = root / "aggregate.md"
        aggregate_json = root / "aggregate.json"
        _preflight_output_pair(aggregate_json, aggregate_markdown)
        _atomic_write_once_bytes(aggregate_markdown, b"first\n")
        expect_locked_error(
            lambda: _preflight_output_pair(aggregate_json, aggregate_markdown),
            "half-published aggregate pair",
        )
        try:
            _atomic_write_once_bytes(aggregate_markdown, b"replacement\n")
        except FileExistsError:
            pass
        else:
            raise AssertionError("write-once aggregate was replaceable")
        claims = root / ".claims"
        claims.mkdir()
        for rank in RANKS:
            (claims / f"ladder_r{rank}.claim").mkdir()
        _validate_claim_entries(claims)
        corrupt_claim = claims / f"ladder_r{RANKS[-1]}.claim"
        corrupt_claim.rmdir()
        corrupt_claim.write_text("not a directory", encoding="utf-8")
        expect_locked_error(
            lambda: _validate_claim_entries(claims),
            "non-directory claim",
        )

    environment = {
        "hardware": {
            "gpu": "NVIDIA H100 80GB HBM3",
            "torch": "test",
            "cuda": "test",
            "cudnn": 1,
        },
        "platform": "test-platform",
        "python": "test-python",
    }
    contexts = [
        {
            "rank": rank,
            "array_job_id": "700",
            "array_task_id": str(task),
            "array_task_count": "7",
            "array_task_min": "0",
            "array_task_max": "6",
            "array_task_step": "1",
            "job_id": str(700 + task),
            "account": "authorized",
            "partition": "h100",
            "host": f"n{task}",
            "gpu": {
                "uuid": f"GPU-{task:032x}",
                "name": "NVIDIA H100 80GB HBM3",
                "memory_mib": "81159",
                "driver_version": "test",
            },
            "campaign_lock_git_commit": "f" * 40,
            **environment,
        }
        for task, rank in TASK_TO_RANK.items()
    ]
    _validate_parallel_contexts(
        contexts,
        expected_account="authorized",
        expected_partition="h100",
        require_distinct_gpus=True,
    )
    duplicate_task = copy.deepcopy(contexts)
    duplicate_task[-1]["array_task_id"] = "0"
    expect_locked_error(
        lambda: _validate_parallel_contexts(
            duplicate_task,
            expected_account="authorized",
            expected_partition="h100",
            require_distinct_gpus=True,
        ),
        "duplicate task",
    )
    mixed_array = copy.deepcopy(contexts)
    mixed_array[-1]["array_job_id"] = "701"
    expect_locked_error(
        lambda: _validate_parallel_contexts(
            mixed_array,
            expected_account="authorized",
            expected_partition="h100",
            require_distinct_gpus=True,
        ),
        "mixed array",
    )
    duplicate_job = copy.deepcopy(contexts)
    duplicate_job[-1]["job_id"] = duplicate_job[0]["job_id"]
    expect_locked_error(
        lambda: _validate_parallel_contexts(
            duplicate_job,
            expected_account="authorized",
            expected_partition="h100",
            require_distinct_gpus=True,
        ),
        "duplicate Slurm job ID",
    )
    wrong_array_count = copy.deepcopy(contexts)
    wrong_array_count[-1]["array_task_count"] = "6"
    expect_locked_error(
        lambda: _validate_parallel_contexts(
            wrong_array_count,
            expected_account="authorized",
            expected_partition="h100",
            require_distinct_gpus=True,
        ),
        "wrong Slurm array task count",
    )
    wrong_account = copy.deepcopy(contexts)
    wrong_account[-1]["account"] = "other"
    expect_locked_error(
        lambda: _validate_parallel_contexts(
            wrong_account,
            expected_account="authorized",
            expected_partition="h100",
            require_distinct_gpus=True,
        ),
        "mismatched account",
    )
    wrong_partition = copy.deepcopy(contexts)
    wrong_partition[-1]["partition"] = "other"
    expect_locked_error(
        lambda: _validate_parallel_contexts(
            wrong_partition,
            expected_account="authorized",
            expected_partition="h100",
            require_distinct_gpus=True,
        ),
        "mismatched partition",
    )
    wrong_mapping = copy.deepcopy(contexts)
    wrong_mapping[0]["array_task_id"], wrong_mapping[1]["array_task_id"] = (
        wrong_mapping[1]["array_task_id"],
        wrong_mapping[0]["array_task_id"],
    )
    expect_locked_error(
        lambda: _validate_parallel_contexts(
            wrong_mapping,
            expected_account="authorized",
            expected_partition="h100",
            require_distinct_gpus=True,
        ),
        "task-to-rank mismatch",
    )

    valid_result = {
        "campaign_tag": INNER_CAMPAIGN_TAG,
        "rank": 2_048,
        "schema_version": "cuda_blackhole_rank_scaling_v1",
        "protocol_version": "blackhole_s4_ladder_v1",
        "source": {
            "git_commit": LOCKED_COMMIT,
            "git_dirty": False,
            "driver_sha256": lock["sources"][
                "rebuttal_experiments/cuda_blackhole_rank_scaling.py"
            ]["sha256"],
            "kernel_sha256": lock["sources"][
                "student_pkg/doublesided_cholesky.py"
            ]["sha256"],
            "feature_source_sha256": lock["sources"][
                "rebuttal_experiments/blackhole_features.py"
            ]["sha256"],
        },
        "feature_manifest": {"sha256": MANIFEST_SHA256},
        "trace": {"sha256": TRACE_SHA256},
    }
    _validate_result_provenance(
        valid_result,
        rank=2_048,
        lock=lock,
        slurm=None,
    )
    corrupt_result = copy.deepcopy(valid_result)
    corrupt_result["source"]["driver_sha256"] = "0" * 64
    expect_locked_error(
        lambda: _validate_result_provenance(
            corrupt_result,
            rank=2_048,
            lock=lock,
            slurm=None,
        ),
        "corrupt raw result provenance",
    )
    _validate_lock_shape(lock)
    print("black-hole locked-rerun self-test passed", flush=True)


def _common_parser(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--control-repo",
        type=Path,
        default=CONTROL_ROOT,
    )
    subparser.add_argument("--source-repo", type=Path, required=True)
    subparser.add_argument("--lock-file", type=Path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    _common_parser(prepare_parser)
    prepare_parser.add_argument("--locked-worktree", type=Path, required=True)
    prepare_parser.add_argument("--campaign-dir", type=Path, required=True)
    prepare_parser.add_argument("--manifest", type=Path)
    prepare_parser.add_argument("--trace", type=Path)

    verify_parser = subparsers.add_parser("verify-task")
    _common_parser(verify_parser)
    verify_parser.add_argument("--locked-worktree", type=Path, required=True)
    verify_parser.add_argument("--campaign-dir", type=Path, required=True)
    verify_parser.add_argument("--rank", type=int, choices=RANKS, required=True)

    attest_parser = subparsers.add_parser("attest-task")
    _common_parser(attest_parser)
    attest_parser.add_argument("--locked-worktree", type=Path, required=True)
    attest_parser.add_argument("--campaign-dir", type=Path, required=True)
    attest_parser.add_argument("--rank", type=int, choices=RANKS, required=True)

    aggregate_parser = subparsers.add_parser("aggregate")
    _common_parser(aggregate_parser)
    aggregate_parser.add_argument("--locked-worktree", type=Path, required=True)
    aggregate_parser.add_argument("--campaign-dir", type=Path, required=True)
    aggregate_parser.add_argument("--expected-account", required=True)
    aggregate_parser.add_argument("--expected-partition", required=True)
    aggregate_parser.add_argument("--output-json", type=Path)
    aggregate_parser.add_argument("--output-markdown", type=Path)

    test_parser = subparsers.add_parser("self-test")
    _common_parser(test_parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    actions = {
        "prepare": prepare,
        "verify-task": verify_task,
        "attest-task": attest_task,
        "aggregate": aggregate,
        "self-test": self_test,
    }
    actions[args.action](args)


if __name__ == "__main__":
    main()
