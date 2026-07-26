"""Lock, launch, attest, and audit the frozen black-hole stability trace.

This module never implements numerical work.  It executes the unmodified
``cuda_blackhole_stability.py`` driver from the exact detached Git commit
recorded below.  The outer harness gives the rerun its own immutable identity,
records the real ``batch`` allocation, and confines the legacy ``preempt``
partition value to the inner driver's metadata check.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


CONTROL_ROOT = Path(__file__).resolve().parents[1]
LOCK_SCHEMA = "blackhole_stability_locked_lock_v1"
ATTESTATION_SCHEMA = "blackhole_stability_locked_attestation_v1"
AGGREGATE_SCHEMA = "blackhole_stability_locked_aggregate_v1"
OUTER_CAMPAIGN_TAG = "blackhole-s4-v1-stability-locked-rerun3"
INNER_CAMPAIGN_TAG = "blackhole-s4-v1"
LOCKED_COMMIT = "96c8018cdd0b56ffe1fd93bddf48fa4affb34697"
OUTER_ACCOUNT = "marlowe-m000151-pm06"
OUTER_PARTITION = "batch"
OUTER_QOS = "medium"
OUTER_CPUS_PER_TASK = "4"
INNER_PARTITION_METADATA = "preempt"
RANKS = (8_192, 32_768)
TASK_TO_RANK = dict(enumerate(RANKS))
MANIFEST_SHA256 = (
    "3b8c6e32da3842b8a9f21ee722e8b9fb1093899744d283d4e939427dd116a5c2"
)
TRACE_SHA256 = (
    "bdf1619a70af70585af317d25aa69452fcb8e94d8417e525de9d50f878e530b4"
)
MANIFEST_BYTES = 24_020
TRACE_BYTES = 47_448_192
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
    "rebuttal_experiments/blackhole_stability_locked.py",
    "rebuttal_experiments/slurm/cuda_blackhole_stability_locked.sbatch",
    "rebuttal_experiments/BLACKHOLE_STABILITY_LOCKED_PROTOCOL.md",
)
LOCKED_SOURCE_PATHS = (
    "rebuttal_experiments/__init__.py",
    "rebuttal_experiments/common.py",
    "rebuttal_experiments/blackhole_features.py",
    "rebuttal_experiments/cuda_blackhole_stability.py",
    "rebuttal_experiments/aggregate_blackhole.py",
    "student_pkg/doublesided_cholesky.py",
)
EXPECTED_SOFTWARE = {
    "python_version_prefix": "3.10.12 ",
    "python_executable": "/usr/bin/python3.10",
    "python_executable_sha256": (
        "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
    ),
    "versions": {
        "torch": "2.12.0+cu126",
        "cuda": "12.6",
        "cudnn": 91002,
        "numpy": "2.2.6",
        "triton": "3.7.0",
    },
    "distribution_files": {
        "torch": {
            "METADATA": (
                "3efd74705a5bdeeb10141945d835be7e8ea1379bf669a1bf00e3d1cd90a97a82"
            ),
            "RECORD": (
                "c820a29f87d40b05458d25e85edb5832567beb9ac7ffb26e5ee4c7e166ddd6a3"
            ),
        },
        "numpy": {
            "METADATA": (
                "22d648f53848429464ca643b40d73b49a920d47876fc77c395d4dca4e834904a"
            ),
            "RECORD": (
                "e24ddb2a10fe4461e9424b0a844bbe3000503a3c2671c49ef1cd4686a70d8868"
            ),
        },
        "triton": {
            "METADATA": (
                "9ece1cff01d51bf4935859aadddf5cd92ead252902a33a0a05dfa2e9a64b5f48"
            ),
            "RECORD": (
                "86f337a8b700273fcb8cc39e24e6c3bf0ad2380eb140c44e02c0cc2c42a03c50"
            ),
        },
    },
    "critical_files": {
        "torch/__init__.py": {
            "bytes": 109_312,
            "sha256": (
                "d9dfff4b75d46e4c75572200a3466b70231d05b0318e38ac1bd121789165fb49"
            ),
        },
        "torch/_C.cpython-310-x86_64-linux-gnu.so": {
            "bytes": 25_521,
            "sha256": (
                "ba23a10655304ddcd9f96f2520182bc2a6b3f49b427fa1ea0fe47d26a8c6bb84"
            ),
        },
        "torch/lib/libtorch_cuda.so": {
            "bytes": 1_007_731_497,
            "sha256": (
                "6cae344fdfe71512a039162ed50ddb714f57f0158a891760a4813db8fa2d3f89"
            ),
        },
        "numpy/__init__.py": {
            "bytes": 22_147,
            "sha256": (
                "6ae17b070c0f70a8e3cad89a510a256942e5a1f37ea5feb120cec167ed2a6236"
            ),
        },
        "numpy/_core/_multiarray_umath.cpython-310-x86_64-linux-gnu.so": {
            "bytes": 10_449_209,
            "sha256": (
                "c634eaf05367af02b815cf3d2020fbeee9b47426af1fe26cf3189e4798932dc5"
            ),
        },
        "triton/__init__.py": {
            "bytes": 1_683,
            "sha256": (
                "1d02da1c872f91eef29fa8fab3f8d59d69c40be662bf777c5f5a30724f35e001"
            ),
        },
        "triton/compiler/compiler.py": {
            "bytes": 20_736,
            "sha256": (
                "021a45ca590052ab6ca6d850eb0a102abf542165836ad92bfb3bb573218fc5d3"
            ),
        },
    },
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class LockedStabilityError(RuntimeError):
    """A fail-closed protocol, provenance, or execution violation."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise LockedStabilityError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> Any:
    raise LockedStabilityError(f"non-finite JSON constant {value!r}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LockedStabilityError(f"cannot read strict JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise LockedStabilityError(f"{path}: top-level JSON must be an object")
    return value


def _equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise LockedStabilityError(
            f"{context}={actual!r}; expected frozen value {expected!r}"
        )


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LockedStabilityError(f"{context} must be an object")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise LockedStabilityError(f"{context} must be a nonempty string")
    return value


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
        raise LockedStabilityError(
            f"git {' '.join(arguments)} failed in {repository}: {stderr}"
        ) from error


def _git_text(repository: Path, *arguments: str) -> str:
    return _git(repository, *arguments).stdout.decode("utf-8").strip()


def _git_blob(repository: Path, commit: str, relative: str) -> bytes:
    return _git(repository, "show", f"{commit}:{relative}").stdout


def _atomic_write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
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
    data = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_write_once(path, data)


def _copy_exact_or_verify(source: Path, destination: Path) -> None:
    data = source.read_bytes()
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != data:
            raise LockedStabilityError(
                f"existing destination differs from frozen source: {destination}"
            )
        return
    _atomic_write_once(destination, data)


def _default_lock(control_repo: Path) -> Path:
    return control_repo / LOCK_RELATIVE


def _default_manifest(control_repo: Path) -> Path:
    return control_repo / MANIFEST_RELATIVE


def _default_trace(control_repo: Path) -> Path:
    return control_repo / TRACE_RELATIVE


def _validate_campaign_directory(path: Path) -> None:
    _equal(path.name, OUTER_CAMPAIGN_TAG, "campaign directory basename")


def _rank_task(rank: int) -> int:
    if rank not in RANKS:
        raise LockedStabilityError(f"rank {rank} is not frozen in {RANKS}")
    return RANKS.index(rank)


def _raw_result_path(campaign_dir: Path, rank: int) -> Path:
    return (
        campaign_dir
        / "inner"
        / INNER_CAMPAIGN_TAG
        / f"stability_r{rank}.json"
    )


def _verify_file(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    context: str,
) -> None:
    if not path.is_file():
        raise LockedStabilityError(f"missing {context}: {path}")
    _equal(path.stat().st_size, expected_bytes, f"{context} byte count")
    _equal(_sha256_file(path), expected_sha256, f"{context} SHA-256")


def _verify_manifest_semantics(path: Path) -> None:
    manifest = _load_json(path)
    _equal(
        manifest.get("campaign_tag"),
        INNER_CAMPAIGN_TAG,
        "feature manifest campaign",
    )
    trace = _object(manifest.get("trace"), "feature manifest trace")
    _equal(trace.get("path"), TRACE_RELATIVE, "feature manifest trace path")
    _equal(trace.get("sha256"), TRACE_SHA256, "feature manifest trace hash")


def _source_records(repository: Path, commit: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for relative in LOCKED_SOURCE_PATHS:
        blob = _git_blob(repository, commit, relative)
        result[relative] = {
            "bytes": len(blob),
            "sha256": _sha256_bytes(blob),
        }
    return result


def _harness_records(repository: Path, commit: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for relative in HARNESS_PATHS:
        blob = _git_blob(repository, commit, relative)
        result[relative] = {
            "bytes": len(blob),
            "sha256": _sha256_bytes(blob),
        }
    return result


def _lock_configuration() -> dict[str, Any]:
    return {
        "array_task_to_rank": {
            str(task): rank for task, rank in TASK_TO_RANK.items()
        },
        "ranks": list(RANKS),
        "dtype": "float32",
        "ridge": 1.0,
        "initial_transitions": list(range(64)),
        "warmup_transition": 64,
        "stability_transitions": list(range(64, 164)),
        "checkpoint_steps": [1, 10, 25, 50, 75, 100],
        "method_order": [
            "fresh_refact",
            "inverse_no_refresh",
            "inverse_refresh_q50",
        ],
        "refresh_interval": 50,
        "probe_seed": 29_421,
        "probe_count": 16,
        "correctness_threshold": 1e-3,
        "tiles": {"8192": 8, "32768": 32},
        "setup_num_warps": 4,
        "sweep_num_warps": 1,
        "tf32": False,
    }


def _lock_execution() -> dict[str, Any]:
    return {
        "account": OUTER_ACCOUNT,
        "partition": OUTER_PARTITION,
        "qos": OUTER_QOS,
        "array": "0-1%2",
        "cpus_per_task": int(OUTER_CPUS_PER_TASK),
        "gpus_per_task": 1,
        "memory": "64G",
        "time_limit": "04:00:00",
        "gpu_name_prefix": "NVIDIA H100",
        "gpu_total_memory_bytes": 85_045_477_376,
        "inner_partition_metadata_shim": INNER_PARTITION_METADATA,
        "inner_driver_is_unmodified": True,
        "immutable_mode": "detached_git_worktree",
        "result_publication": "persistent_claim_then_atomic_inner_result",
    }


def make_lock(args: argparse.Namespace) -> None:
    control_repo = args.control_repo.resolve()
    source_repo = args.source_repo.resolve()
    manifest = (args.manifest or _default_manifest(control_repo)).resolve()
    trace = (args.trace or _default_trace(control_repo)).resolve()
    output = (args.output or _default_lock(control_repo)).resolve()
    if output.exists():
        raise LockedStabilityError(f"refusing to replace lock: {output}")
    _equal(
        _git_text(source_repo, "rev-parse", f"{LOCKED_COMMIT}^{{commit}}"),
        LOCKED_COMMIT,
        "resolved numerical commit",
    )
    harness_commit = _git_text(control_repo, "rev-parse", "HEAD")
    if not COMMIT_RE.fullmatch(harness_commit):
        raise LockedStabilityError("control HEAD is not a full Git commit")
    changed = _git(
        control_repo,
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--",
        *HARNESS_PATHS,
    ).stdout
    if changed:
        raise LockedStabilityError(
            "harness files must be tracked and clean before lock creation"
        )
    for relative in HARNESS_PATHS:
        tracked = _git(
            control_repo,
            "ls-files",
            "--error-unmatch",
            "--",
            relative,
            check=False,
        )
        if tracked.returncode != 0:
            raise LockedStabilityError(f"untracked harness file: {relative}")
    _verify_file(
        manifest,
        expected_bytes=MANIFEST_BYTES,
        expected_sha256=MANIFEST_SHA256,
        context="feature manifest",
    )
    _verify_manifest_semantics(manifest)
    _verify_file(
        trace,
        expected_bytes=TRACE_BYTES,
        expected_sha256=TRACE_SHA256,
        context="feature trace",
    )
    payload = {
        "schema_version": LOCK_SCHEMA,
        "campaign_tag": OUTER_CAMPAIGN_TAG,
        "inner_campaign_tag": INNER_CAMPAIGN_TAG,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rationale": (
            "Immutable provenance rerun of the predeclared 100-transition "
            "secondary stability trace. Existing unlocked outputs are "
            "preserved and were not used to alter any experimental choice. "
            "Locked rerun1 stopped during a data-free preflight because its "
            "lock named the /usr/bin/python3 symlink while verification "
            "normalized it to /usr/bin/python3.10; it created no task claim "
            "and ran no numerical code. Rerun2 corrects only that pathname."
            " Rerun2 array 394115 then exposed a Slurm identifier ambiguity: "
            "rank 8,192 completed, but rank 32,768 stopped in the outer "
            "scheduler verifier before numerical code because its bare "
            "SLURM_JOB_ID also named the array master and returned two "
            "scontrol records. Both claims and the completed rank-8,192 "
            "artifact remain preserved. Rerun3 changes only the scontrol "
            "selector to the unambiguous ArrayJobID_TaskID form and reruns "
            "both preregistered ranks."
        ),
        "locked_git_commit": LOCKED_COMMIT,
        "harness_git_commit": harness_commit,
        "inputs": {
            "feature_manifest": {
                "path": MANIFEST_RELATIVE,
                "bytes": MANIFEST_BYTES,
                "sha256": MANIFEST_SHA256,
                "embedded_campaign_tag": INNER_CAMPAIGN_TAG,
            },
            "feature_trace": {
                "path": TRACE_RELATIVE,
                "bytes": TRACE_BYTES,
                "sha256": TRACE_SHA256,
                "shape": [181, 65_536],
                "dtype": "float32",
            },
        },
        "sources": _source_records(source_repo, LOCKED_COMMIT),
        "harness": _harness_records(control_repo, harness_commit),
        "configuration": _lock_configuration(),
        "execution": _lock_execution(),
        "software": EXPECTED_SOFTWARE,
    }
    _validate_lock_shape(payload)
    _atomic_json_write_once(output, payload)
    print(f"wrote immutable campaign lock: {output}", flush=True)


def _validate_hash_records(
    value: Any,
    expected_paths: tuple[str, ...],
    context: str,
) -> None:
    records = _object(value, context)
    _equal(set(records), set(expected_paths), f"{context} keys")
    for relative, raw in records.items():
        record = _object(raw, f"{context}.{relative}")
        _equal(set(record), {"bytes", "sha256"}, f"{context}.{relative} keys")
        count = record.get("bytes")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise LockedStabilityError(f"{context}.{relative}.bytes is invalid")
        sha = record.get("sha256")
        if not isinstance(sha, str) or not SHA_RE.fullmatch(sha):
            raise LockedStabilityError(f"{context}.{relative}.sha256 is invalid")


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
            "software",
        },
        "campaign lock keys",
    )
    _equal(lock.get("schema_version"), LOCK_SCHEMA, "lock schema")
    _equal(lock.get("campaign_tag"), OUTER_CAMPAIGN_TAG, "lock campaign")
    _equal(
        lock.get("inner_campaign_tag"),
        INNER_CAMPAIGN_TAG,
        "lock inner campaign",
    )
    _equal(lock.get("locked_git_commit"), LOCKED_COMMIT, "lock numerical commit")
    harness_commit = lock.get("harness_git_commit")
    if not isinstance(harness_commit, str) or not COMMIT_RE.fullmatch(harness_commit):
        raise LockedStabilityError("lock harness commit is invalid")
    inputs = _object(lock.get("inputs"), "lock inputs")
    _equal(set(inputs), {"feature_manifest", "feature_trace"}, "lock input keys")
    _equal(
        inputs["feature_manifest"],
        {
            "path": MANIFEST_RELATIVE,
            "bytes": MANIFEST_BYTES,
            "sha256": MANIFEST_SHA256,
            "embedded_campaign_tag": INNER_CAMPAIGN_TAG,
        },
        "lock feature manifest",
    )
    _equal(
        inputs["feature_trace"],
        {
            "path": TRACE_RELATIVE,
            "bytes": TRACE_BYTES,
            "sha256": TRACE_SHA256,
            "shape": [181, 65_536],
            "dtype": "float32",
        },
        "lock feature trace",
    )
    _validate_hash_records(lock.get("sources"), LOCKED_SOURCE_PATHS, "lock sources")
    _validate_hash_records(lock.get("harness"), HARNESS_PATHS, "lock harness")
    _equal(lock.get("configuration"), _lock_configuration(), "lock configuration")
    _equal(lock.get("execution"), _lock_execution(), "lock execution")
    _equal(lock.get("software"), EXPECTED_SOFTWARE, "lock software")


def _verify_harness(lock: dict[str, Any], control_repo: Path) -> None:
    commit = lock["harness_git_commit"]
    _equal(
        _git_text(control_repo, "rev-parse", f"{commit}^{{commit}}"),
        commit,
        "resolved harness commit",
    )
    for relative, metadata in lock["harness"].items():
        path = control_repo / relative
        _verify_file(
            path,
            expected_bytes=metadata["bytes"],
            expected_sha256=metadata["sha256"],
            context=f"harness {relative}",
        )
        blob = _git_blob(control_repo, commit, relative)
        _equal(len(blob), metadata["bytes"], f"committed {relative} bytes")
        _equal(
            _sha256_bytes(blob),
            metadata["sha256"],
            f"committed {relative} hash",
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
        raise LockedStabilityError("locked outer harness is dirty")


def _verify_sources(
    lock: dict[str, Any],
    repository: Path,
    *,
    require_head: bool,
) -> None:
    _equal(
        _git_text(repository, "rev-parse", f"{LOCKED_COMMIT}^{{commit}}"),
        LOCKED_COMMIT,
        "resolved locked source commit",
    )
    if require_head:
        _equal(
            _git_text(repository, "rev-parse", "HEAD"),
            LOCKED_COMMIT,
            "locked worktree HEAD",
        )
        symbolic = _git(repository, "symbolic-ref", "-q", "HEAD", check=False)
        if symbolic.returncode == 0:
            raise LockedStabilityError("numerical source worktree is not detached")
    for relative, metadata in lock["sources"].items():
        blob = _git_blob(repository, LOCKED_COMMIT, relative)
        _equal(len(blob), metadata["bytes"], f"source blob {relative} bytes")
        _equal(
            _sha256_bytes(blob),
            metadata["sha256"],
            f"source blob {relative} hash",
        )
        if require_head:
            _verify_file(
                repository / relative,
                expected_bytes=metadata["bytes"],
                expected_sha256=metadata["sha256"],
                context=f"worktree source {relative}",
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
            raise LockedStabilityError("locked numerical sources are dirty")


def _canonical_lock_commit(control_repo: Path, lock: dict[str, Any]) -> str:
    canonical = _default_lock(control_repo)
    if not canonical.is_file():
        raise LockedStabilityError(f"missing canonical lock: {canonical}")
    _equal(_load_json(canonical), lock, "canonical/campaign lock content")
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
        raise LockedStabilityError("canonical campaign lock is not tracked")
    if _git(
        control_repo,
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--",
        relative,
    ).stdout:
        raise LockedStabilityError("canonical campaign lock is dirty")
    commit = _git_text(control_repo, "log", "-1", "--format=%H", "--", relative)
    if not COMMIT_RE.fullmatch(commit):
        raise LockedStabilityError("canonical lock commit is invalid")
    return commit


def _load_and_verify_lock(
    lock_path: Path,
    *,
    control_repo: Path,
    source_repo: Path,
    require_source_head: bool,
) -> dict[str, Any]:
    lock = _load_json(lock_path)
    _validate_lock_shape(lock)
    _verify_harness(lock, control_repo)
    _verify_sources(lock, source_repo, require_head=require_source_head)
    canonical = _default_lock(control_repo)
    _equal(
        _sha256_file(lock_path),
        _sha256_file(canonical),
        "campaign/canonical lock hash",
    )
    _canonical_lock_commit(control_repo, lock)
    return lock


def _distribution_file(package: str, filename: str) -> Path:
    distribution = importlib.metadata.distribution(package)
    return Path(distribution._path) / filename


def _environment_fingerprint() -> dict[str, Any]:
    import numpy as np
    import torch
    import triton

    roots = {
        "torch": Path(torch.__file__).resolve().parent,
        "numpy": Path(np.__file__).resolve().parent,
        "triton": Path(triton.__file__).resolve().parent,
    }
    critical: dict[str, Any] = {}
    for relative, expected in EXPECTED_SOFTWARE["critical_files"].items():
        package, remainder = relative.split("/", 1)
        path = roots[package] / remainder
        _verify_file(
            path,
            expected_bytes=expected["bytes"],
            expected_sha256=expected["sha256"],
            context=f"critical dependency {relative}",
        )
        critical[relative] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    distributions: dict[str, Any] = {}
    for package, expected_files in EXPECTED_SOFTWARE[
        "distribution_files"
    ].items():
        files: dict[str, str] = {}
        for filename, expected_hash in expected_files.items():
            path = _distribution_file(package, filename)
            actual = _sha256_file(path)
            _equal(actual, expected_hash, f"{package} {filename} hash")
            files[filename] = actual
        distributions[package] = files
    executable = Path(sys.executable).resolve()
    _equal(str(executable), EXPECTED_SOFTWARE["python_executable"], "Python path")
    _equal(
        _sha256_file(executable),
        EXPECTED_SOFTWARE["python_executable_sha256"],
        "Python executable hash",
    )
    if not sys.version.startswith(EXPECTED_SOFTWARE["python_version_prefix"]):
        raise LockedStabilityError(f"unexpected Python version: {sys.version}")
    versions = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "triton": triton.__version__,
    }
    _equal(versions, EXPECTED_SOFTWARE["versions"], "software versions")
    return {
        "python": sys.version,
        "python_executable": str(executable),
        "python_executable_sha256": _sha256_file(executable),
        "versions": versions,
        "distribution_files": distributions,
        "critical_files": critical,
        "platform": platform.platform(),
    }


def _scontrol_job_record(job_selector: str) -> dict[str, str]:
    try:
        completed = subprocess.run(
            ("scontrol", "show", "job", job_selector, "-o"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise LockedStabilityError(f"scontrol could not inspect job: {error}") from error
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise LockedStabilityError(f"expected one scontrol job row, got {rows!r}")
    fields: dict[str, str] = {}
    for token in rows[0].split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    required = {
        "Account",
        "Partition",
        "QOS",
        "TimeLimit",
        "NumNodes",
        "NumCPUs",
        "NumTasks",
        "MinMemoryNode",
        "TresPerNode",
    }
    missing = required - set(fields)
    if missing:
        raise LockedStabilityError(
            "scontrol job record is missing: " + ", ".join(sorted(missing))
        )
    _equal(fields["Account"], OUTER_ACCOUNT, "scontrol account")
    _equal(fields["Partition"], OUTER_PARTITION, "scontrol partition")
    _equal(fields["QOS"], OUTER_QOS, "scontrol QOS")
    _equal(fields["TimeLimit"], "04:00:00", "scontrol time limit")
    _equal(fields["NumNodes"], "1", "scontrol node count")
    _equal(fields["NumCPUs"], OUTER_CPUS_PER_TASK, "scontrol CPU count")
    _equal(fields["NumTasks"], "1", "scontrol task count")
    if fields["MinMemoryNode"] not in {"64G", "65536M"}:
        raise LockedStabilityError(
            "scontrol MinMemoryNode="
            f"{fields['MinMemoryNode']!r}; expected exactly 64 GiB"
        )
    if fields["TresPerNode"] not in {"gres/gpu:1", "gres/gpu=1"}:
        raise LockedStabilityError(
            f"scontrol TresPerNode={fields['TresPerNode']!r}; expected one GPU"
        )
    return {key: fields[key] for key in sorted(required)}


def _slurm_metadata(rank: int) -> dict[str, Any]:
    variables = {
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
    for key, variable in variables.items():
        value = os.environ.get(variable)
        if not value:
            raise LockedStabilityError(f"{variable} must be set by Slurm")
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
            raise LockedStabilityError(f"Slurm {key} must be numeric")
    _equal(int(values["array_task_id"]), _rank_task(rank), "task/rank mapping")
    _equal(values["array_task_count"], "2", "Slurm array task count")
    _equal(values["array_task_min"], "0", "Slurm array minimum")
    _equal(values["array_task_max"], "1", "Slurm array maximum")
    _equal(values["array_task_step"], "1", "Slurm array step")
    _equal(values["account"], OUTER_ACCOUNT, "Slurm account")
    _equal(values["partition"], OUTER_PARTITION, "Slurm partition")
    _equal(values["cpus_per_task"], OUTER_CPUS_PER_TASK, "Slurm CPUs per task")
    values.update(
        {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "step_gpus": os.environ.get("SLURM_STEP_GPUS"),
            "gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
            "scontrol": _scontrol_job_record(
                f"{values['array_job_id']}_{values['array_task_id']}"
            ),
        }
    )
    return values


def _gpu_identity() -> dict[str, str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible or "," in visible:
        raise LockedStabilityError(
            "CUDA_VISIBLE_DEVICES must identify exactly one allocated GPU"
        )
    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                f"--id={visible}",
                "--query-gpu=uuid,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise LockedStabilityError(f"nvidia-smi failed: {error}") from error
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise LockedStabilityError(f"expected one GPU row, got {rows!r}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 4:
        raise LockedStabilityError(f"unexpected GPU row: {rows[0]!r}")
    uuid, name, memory_mib, driver = fields
    if not uuid.startswith("GPU-") or not name.startswith("NVIDIA H100"):
        raise LockedStabilityError(f"unexpected allocated GPU: {rows[0]!r}")
    return {
        "uuid": uuid,
        "name": name,
        "memory_mib": memory_mib,
        "driver_version": driver,
    }


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
        lock_source,
        control_repo=control_repo,
        source_repo=source_repo,
        require_source_head=False,
    )
    _verify_file(
        manifest_source,
        expected_bytes=MANIFEST_BYTES,
        expected_sha256=MANIFEST_SHA256,
        context="feature manifest",
    )
    _verify_manifest_semantics(manifest_source)
    _verify_file(
        trace_source,
        expected_bytes=TRACE_BYTES,
        expected_sha256=TRACE_SHA256,
        context="feature trace",
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
    _verify_sources(lock, locked_worktree, require_head=True)
    worktree_trace = locked_worktree / TRACE_RELATIVE
    worktree_trace.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(worktree_trace):
        _verify_file(
            worktree_trace,
            expected_bytes=TRACE_BYTES,
            expected_sha256=TRACE_SHA256,
            context="worktree feature trace",
        )
    else:
        worktree_trace.symlink_to(trace_source)
        _verify_file(
            worktree_trace,
            expected_bytes=TRACE_BYTES,
            expected_sha256=TRACE_SHA256,
            context="worktree feature trace",
        )
    campaign_dir.mkdir(parents=True, exist_ok=True)
    _copy_exact_or_verify(lock_source, campaign_dir / "campaign_lock.json")
    _copy_exact_or_verify(manifest_source, campaign_dir / "feature_manifest.json")
    (campaign_dir / ".claims").mkdir(exist_ok=True)
    (campaign_dir / "logs").mkdir(exist_ok=True)
    print(f"prepared locked worktree: {locked_worktree}", flush=True)
    print(f"prepared write-once campaign: {campaign_dir}", flush=True)


def preflight(args: argparse.Namespace) -> None:
    control_repo = args.control_repo.resolve()
    source_repo = args.source_repo.resolve()
    lock_path = (args.lock_file or _default_lock(control_repo)).resolve()
    manifest = (args.manifest or _default_manifest(control_repo)).resolve()
    trace = (args.trace or _default_trace(control_repo)).resolve()
    _load_and_verify_lock(
        lock_path,
        control_repo=control_repo,
        source_repo=source_repo,
        require_source_head=True,
    )
    _verify_file(
        manifest,
        expected_bytes=MANIFEST_BYTES,
        expected_sha256=MANIFEST_SHA256,
        context="preflight manifest",
    )
    _verify_manifest_semantics(manifest)
    _verify_file(
        trace,
        expected_bytes=TRACE_BYTES,
        expected_sha256=TRACE_SHA256,
        context="preflight trace",
    )
    environment = _environment_fingerprint()
    print(
        json.dumps(
            {
                "passed": True,
                "locked_git_commit": LOCKED_COMMIT,
                "manifest_sha256": MANIFEST_SHA256,
                "trace_sha256": TRACE_SHA256,
                "software": environment,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


def verify_task(args: argparse.Namespace) -> None:
    control_repo = args.control_repo.resolve()
    source_repo = args.source_repo.resolve()
    campaign_dir = args.campaign_dir.resolve()
    _validate_campaign_directory(campaign_dir)
    lock = _load_and_verify_lock(
        campaign_dir / "campaign_lock.json",
        control_repo=control_repo,
        source_repo=source_repo,
        require_source_head=True,
    )
    _verify_file(
        campaign_dir / "feature_manifest.json",
        expected_bytes=MANIFEST_BYTES,
        expected_sha256=MANIFEST_SHA256,
        context="campaign feature manifest",
    )
    _verify_manifest_semantics(campaign_dir / "feature_manifest.json")
    _verify_file(
        source_repo / TRACE_RELATIVE,
        expected_bytes=TRACE_BYTES,
        expected_sha256=TRACE_SHA256,
        context="worktree feature trace",
    )
    _slurm_metadata(args.rank)
    _environment_fingerprint()
    _equal(
        lock["execution"]["inner_partition_metadata_shim"],
        INNER_PARTITION_METADATA,
        "inner partition metadata shim",
    )
    print(f"verified locked task {_rank_task(args.rank)} -> {args.rank}", flush=True)


def _validate_result_provenance(
    payload: dict[str, Any],
    *,
    rank: int,
    lock: dict[str, Any],
    slurm: dict[str, Any] | None,
) -> None:
    _equal(payload.get("schema_version"), "blackhole_stability_v1", "result schema")
    _equal(
        payload.get("experiment"),
        "cuda_blackhole_stability",
        "result experiment",
    )
    _equal(payload.get("campaign_tag"), INNER_CAMPAIGN_TAG, "result campaign")
    _equal(payload.get("rank"), rank, "result rank")
    implementation = _object(payload.get("implementation"), "result implementation")
    _equal(
        implementation.get("git_commit"),
        LOCKED_COMMIT,
        "result source commit",
    )
    _equal(implementation.get("git_dirty"), False, "result source dirty flag")
    expected_hashes = {
        relative: lock["sources"][relative]["sha256"]
        for relative in (
            "rebuttal_experiments/cuda_blackhole_stability.py",
            "rebuttal_experiments/blackhole_features.py",
            "student_pkg/doublesided_cholesky.py",
        )
    }
    _equal(
        implementation.get("source_hashes"),
        expected_hashes,
        "result source hashes",
    )
    manifest = _object(payload.get("feature_manifest"), "result manifest")
    _equal(manifest.get("sha256"), MANIFEST_SHA256, "result manifest hash")
    trace = _object(payload.get("trace"), "result trace")
    _equal(trace.get("sha256"), TRACE_SHA256, "result trace hash")
    hardware = _object(payload.get("hardware"), "result hardware")
    _equal(
        {
            "torch": hardware.get("torch"),
            "cuda": hardware.get("cuda"),
            "cudnn": hardware.get("cudnn"),
            "numpy": hardware.get("numpy"),
            "triton": hardware.get("triton"),
        },
        EXPECTED_SOFTWARE["versions"],
        "result software versions",
    )
    if not _string(hardware.get("python"), "result Python").startswith(
        EXPECTED_SOFTWARE["python_version_prefix"]
    ):
        raise LockedStabilityError("result Python version differs from lock")
    _equal(hardware.get("gpu"), "NVIDIA H100 80GB HBM3", "result GPU")
    _equal(
        hardware.get("gpu_total_memory"),
        85_045_477_376,
        "result GPU memory",
    )
    _equal(hardware.get("allow_tf32_matmul"), False, "result TF32")
    recorded = _object(hardware.get("slurm"), "result inner Slurm metadata")
    _equal(recorded.get("cpus_per_task"), OUTER_CPUS_PER_TASK, "inner CPUs")
    _equal(
        recorded.get("job_partition"),
        INNER_PARTITION_METADATA,
        "inner partition metadata shim",
    )
    _equal(recorded.get("array_task_id"), str(_rank_task(rank)), "inner task")
    if slurm is not None:
        for result_key, outer_key in (
            ("job_id", "job_id"),
            ("array_job_id", "array_job_id"),
            ("array_task_id", "array_task_id"),
            ("cpus_per_task", "cpus_per_task"),
        ):
            _equal(
                recorded.get(result_key),
                slurm[outer_key],
                f"result/outer Slurm {result_key}",
            )
        _equal(hardware.get("host"), socket.gethostname(), "result host")


def attest_task(args: argparse.Namespace) -> None:
    control_repo = args.control_repo.resolve()
    source_repo = args.source_repo.resolve()
    campaign_dir = args.campaign_dir.resolve()
    _validate_campaign_directory(campaign_dir)
    lock_path = campaign_dir / "campaign_lock.json"
    lock = _load_and_verify_lock(
        lock_path,
        control_repo=control_repo,
        source_repo=source_repo,
        require_source_head=True,
    )
    slurm = _slurm_metadata(args.rank)
    environment = _environment_fingerprint()
    result_path = _raw_result_path(campaign_dir, args.rank)
    result = _load_json(result_path)
    _validate_result_provenance(result, rank=args.rank, lock=lock, slurm=slurm)
    attestation_path = campaign_dir / f"attestation_stability_r{args.rank}.json"
    payload = {
        "schema_version": ATTESTATION_SCHEMA,
        "campaign_tag": OUTER_CAMPAIGN_TAG,
        "inner_campaign_tag": INNER_CAMPAIGN_TAG,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rank": args.rank,
        "array_task_id": _rank_task(args.rank),
        "campaign_lock": {
            "bytes": lock_path.stat().st_size,
            "sha256": _sha256_file(lock_path),
            "git_commit": _canonical_lock_commit(control_repo, lock),
        },
        "raw_result": {
            "relative_path": str(result_path.relative_to(campaign_dir)),
            "bytes": result_path.stat().st_size,
            "sha256": _sha256_file(result_path),
        },
        "immutable_source": {
            "git_commit": LOCKED_COMMIT,
            "source_hashes": {
                relative: metadata["sha256"]
                for relative, metadata in lock["sources"].items()
            },
        },
        "inputs": {
            "feature_manifest_sha256": MANIFEST_SHA256,
            "feature_trace_sha256": TRACE_SHA256,
        },
        "partition_metadata_contract": {
            "actual_outer_partition": OUTER_PARTITION,
            "inner_recorded_partition": INNER_PARTITION_METADATA,
            "reason": (
                "The unmodified v1 driver validates legacy scheduler metadata; "
                "the outer attestation independently records the actual batch "
                "allocation. Partition metadata is not read by numerical code."
            ),
        },
        "slurm": slurm,
        "host": socket.gethostname(),
        "gpu": _gpu_identity(),
        "software": environment,
    }
    _atomic_json_write_once(attestation_path, payload)
    print(f"wrote {attestation_path}", flush=True)


def _import_locked_aggregator(locked_worktree: Path) -> Any:
    root = str(locked_worktree)
    sys.path.insert(0, root)
    try:
        for name in tuple(sys.modules):
            if name == "rebuttal_experiments" or name.startswith(
                "rebuttal_experiments."
            ):
                del sys.modules[name]
        module = importlib.import_module("rebuttal_experiments.aggregate_blackhole")
    finally:
        sys.path.remove(root)
    expected = (
        locked_worktree / "rebuttal_experiments" / "aggregate_blackhole.py"
    ).resolve()
    _equal(Path(module.__file__).resolve(), expected, "locked aggregator import")
    return module


def _validate_attestation(
    path: Path,
    *,
    result_path: Path,
    rank: int,
    lock: dict[str, Any],
    lock_sha256: str,
    lock_git_commit: str,
) -> dict[str, Any]:
    value = _load_json(path)
    _equal(value.get("schema_version"), ATTESTATION_SCHEMA, f"{path} schema")
    _equal(value.get("campaign_tag"), OUTER_CAMPAIGN_TAG, f"{path} campaign")
    _equal(value.get("rank"), rank, f"{path} rank")
    _equal(value.get("array_task_id"), _rank_task(rank), f"{path} task")
    lock_record = _object(value.get("campaign_lock"), f"{path} lock")
    _equal(lock_record.get("sha256"), lock_sha256, f"{path} lock hash")
    _equal(lock_record.get("git_commit"), lock_git_commit, f"{path} lock commit")
    raw = _object(value.get("raw_result"), f"{path} raw result")
    _equal(
        raw.get("relative_path"),
        str(result_path.relative_to(path.parent)),
        f"{path} result path",
    )
    _equal(raw.get("bytes"), result_path.stat().st_size, f"{path} result bytes")
    _equal(raw.get("sha256"), _sha256_file(result_path), f"{path} result hash")
    _equal(
        value.get("inputs"),
        {
            "feature_manifest_sha256": MANIFEST_SHA256,
            "feature_trace_sha256": TRACE_SHA256,
        },
        f"{path} inputs",
    )
    contract = _object(
        value.get("partition_metadata_contract"),
        f"{path} partition contract",
    )
    _equal(
        contract.get("actual_outer_partition"),
        OUTER_PARTITION,
        f"{path} actual partition",
    )
    _equal(
        contract.get("inner_recorded_partition"),
        INNER_PARTITION_METADATA,
        f"{path} inner partition",
    )
    slurm = _object(value.get("slurm"), f"{path} Slurm")
    _equal(slurm.get("account"), OUTER_ACCOUNT, f"{path} account")
    _equal(slurm.get("partition"), OUTER_PARTITION, f"{path} partition")
    _equal(slurm.get("cpus_per_task"), OUTER_CPUS_PER_TASK, f"{path} CPUs")
    _equal(slurm.get("array_task_count"), "2", f"{path} task count")
    _equal(slurm.get("array_task_min"), "0", f"{path} task min")
    _equal(slurm.get("array_task_max"), "1", f"{path} task max")
    _equal(slurm.get("array_task_step"), "1", f"{path} task step")
    _equal(int(slurm.get("array_task_id")), _rank_task(rank), f"{path} task map")
    gpu = _object(value.get("gpu"), f"{path} GPU")
    if not _string(gpu.get("uuid"), f"{path} GPU UUID").startswith("GPU-"):
        raise LockedStabilityError(f"{path}: malformed GPU UUID")
    if not _string(gpu.get("name"), f"{path} GPU name").startswith("NVIDIA H100"):
        raise LockedStabilityError(f"{path}: expected H100")
    software = _object(value.get("software"), f"{path} software")
    _equal(
        software.get("versions"),
        EXPECTED_SOFTWARE["versions"],
        f"{path} versions",
    )
    _equal(
        software.get("python_executable_sha256"),
        EXPECTED_SOFTWARE["python_executable_sha256"],
        f"{path} Python hash",
    )
    _equal(
        software.get("distribution_files"),
        EXPECTED_SOFTWARE["distribution_files"],
        f"{path} distribution hashes",
    )
    _equal(
        software.get("critical_files"),
        EXPECTED_SOFTWARE["critical_files"],
        f"{path} critical dependency hashes",
    )
    source = _object(value.get("immutable_source"), f"{path} source")
    _equal(source.get("git_commit"), LOCKED_COMMIT, f"{path} source commit")
    _equal(
        source.get("source_hashes"),
        {
            relative: metadata["sha256"]
            for relative, metadata in lock["sources"].items()
        },
        f"{path} source hashes",
    )
    return {
        "slurm": slurm,
        "host": _string(value.get("host"), f"{path} host"),
        "gpu": gpu,
        "software": software,
        "raw_result_sha256": raw["sha256"],
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Locked black-hole stability trace",
        "",
        (
            f"Outer campaign `{OUTER_CAMPAIGN_TAG}`; unmodified inner protocol "
            f"`{INNER_CAMPAIGN_TAG}` at `{LOCKED_COMMIT}`."
        ),
        "",
        "| rank | method | event seconds, mean ± SD | total event seconds | valid |",
        "|---:|:---|---:|---:|:---:|",
    ]
    for row in summary["stability"]:
        for method, value in row["methods"].items():
            stats = value["event_seconds"]
            lines.append(
                f"| {row['rank']} | {method} | {stats['mean']:.6g} ± "
                f"{stats['sample_sd']:.3g} | "
                f"{value['event_seconds_total']:.6g} | "
                f"{'yes' if value['headline_valid'] else 'no'} |"
            )
    lines.extend(["", "Paired speedups are emitted only when the frozen "
                  "correctness gate passes.", ""])
    return "\n".join(lines)


def aggregate(args: argparse.Namespace) -> None:
    control_repo = args.control_repo.resolve()
    source_repo = args.source_repo.resolve()
    campaign_dir = args.campaign_dir.resolve()
    _validate_campaign_directory(campaign_dir)
    lock_path = campaign_dir / "campaign_lock.json"
    lock = _load_and_verify_lock(
        lock_path,
        control_repo=control_repo,
        source_repo=source_repo,
        require_source_head=True,
    )
    lock_sha256 = _sha256_file(lock_path)
    lock_git_commit = _canonical_lock_commit(control_repo, lock)
    manifest_path = campaign_dir / "feature_manifest.json"
    _verify_file(
        manifest_path,
        expected_bytes=MANIFEST_BYTES,
        expected_sha256=MANIFEST_SHA256,
        context="campaign manifest",
    )
    _verify_manifest_semantics(manifest_path)
    _verify_file(
        source_repo / TRACE_RELATIVE,
        expected_bytes=TRACE_BYTES,
        expected_sha256=TRACE_SHA256,
        context="worktree trace",
    )
    output_json = campaign_dir / "aggregate.json"
    output_markdown = campaign_dir / "aggregate.md"
    if output_json.exists() or output_markdown.exists():
        raise LockedStabilityError("refusing to replace an aggregate")
    claims = campaign_dir / ".claims"
    claim_entries = list(claims.iterdir()) if claims.is_dir() else []
    _equal(
        {path.name for path in claim_entries},
        {f"stability_r{rank}.claim" for rank in RANKS},
        "persistent claim set",
    )
    if any(not path.is_dir() for path in claim_entries):
        raise LockedStabilityError("every task claim must be a directory")
    inner_dir = campaign_dir / "inner" / INNER_CAMPAIGN_TAG
    _equal(
        {path.name for path in inner_dir.glob("*.json")},
        {"feature_manifest.json", *(f"stability_r{rank}.json" for rank in RANKS)},
        "inner JSON artifact set",
    )
    base = _import_locked_aggregator(source_repo)
    (
        manifest,
        manifest_sha256,
        trace_sha256,
        _feature_source_sha256,
    ) = base._validate_manifest(manifest_path)
    _equal(manifest_sha256, MANIFEST_SHA256, "base manifest hash")
    _equal(trace_sha256, TRACE_SHA256, "base trace hash")
    records: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    for rank in RANKS:
        result_path = _raw_result_path(campaign_dir, rank)
        attestation = _validate_attestation(
            campaign_dir / f"attestation_stability_r{rank}.json",
            result_path=result_path,
            rank=rank,
            lock=lock,
            lock_sha256=lock_sha256,
            lock_git_commit=lock_git_commit,
        )
        payload = _load_json(result_path)
        _validate_result_provenance(payload, rank=rank, lock=lock, slurm=None)
        recorded = payload["hardware"]["slurm"]
        for key in ("job_id", "array_job_id", "array_task_id", "cpus_per_task"):
            _equal(
                recorded.get(key),
                attestation["slurm"].get(key),
                f"rank {rank} result/attestation {key}",
            )
        _equal(
            recorded.get("job_partition"),
            INNER_PARTITION_METADATA,
            f"rank {rank} inner partition",
        )
        _equal(
            payload["hardware"].get("host"),
            attestation["host"],
            f"rank {rank} host",
        )
        record, source_context = base._validate_stability(
            result_path,
            rank=rank,
            manifest_sha256=manifest_sha256,
            trace_sha256=trace_sha256,
            expected_rank_scale=float(
                manifest["rank_scaling"]["scales"][str(rank)]
            ),
        )
        _equal(source_context[0], LOCKED_COMMIT, f"rank {rank} base source")
        records.append(record)
        contexts.append(
            {
                "rank": rank,
                "slurm": attestation["slurm"],
                "host": attestation["host"],
                "gpu": attestation["gpu"],
                "software": attestation["software"],
                "raw_result_sha256": attestation["raw_result_sha256"],
            }
        )
    _equal(
        len({row["slurm"]["array_job_id"] for row in contexts}),
        1,
        "Slurm array job count",
    )
    _equal(
        {int(row["slurm"]["array_task_id"]) for row in contexts},
        {0, 1},
        "Slurm task set",
    )
    _equal(
        len({row["slurm"]["job_id"] for row in contexts}),
        2,
        "distinct element job IDs",
    )
    environment_keys = {
        json.dumps(
            {
                "gpu_name": row["gpu"]["name"],
                "gpu_memory": row["gpu"]["memory_mib"],
                "driver": row["gpu"]["driver_version"],
                "software": row["software"],
            },
            sort_keys=True,
        )
        for row in contexts
    }
    _equal(len(environment_keys), 1, "execution environment count")
    stability = [base._aggregate_stability(record) for record in records]
    all_valid = all(
        method["headline_valid"]
        for row in stability
        for method in row["methods"].values()
    )
    summary = {
        "schema_version": AGGREGATE_SCHEMA,
        "experiment": "cuda_blackhole_stability_locked_rerun",
        "campaign_tag": OUTER_CAMPAIGN_TAG,
        "inner_campaign_tag": INNER_CAMPAIGN_TAG,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "audit": {
            "passed": True,
            "all_headline_correctness_valid": all_valid,
            "headline_claims_withheld": not all_valid,
            "locked_git_commit": LOCKED_COMMIT,
            "campaign_lock_sha256": lock_sha256,
            "campaign_lock_git_commit": lock_git_commit,
            "feature_manifest_sha256": MANIFEST_SHA256,
            "feature_trace_sha256": TRACE_SHA256,
            "slurm_array_job_id": contexts[0]["slurm"]["array_job_id"],
            "slurm_account": OUTER_ACCOUNT,
            "actual_slurm_partition": OUTER_PARTITION,
            "inner_recorded_partition": INNER_PARTITION_METADATA,
            "task_to_rank": {
                str(task): rank for task, rank in TASK_TO_RANK.items()
            },
        },
        "execution_contexts": contexts,
        "stability": stability,
    }
    _atomic_json_write_once(output_json, summary)
    _atomic_write_once(output_markdown, _markdown(summary).encode("utf-8"))
    print(f"wrote {output_json}", flush=True)
    print(f"wrote {output_markdown}", flush=True)


def self_test(args: argparse.Namespace) -> None:
    _equal(_rank_task(8_192), 0, "first task")
    _equal(_rank_task(32_768), 1, "last task")
    try:
        _strict_pairs([("a", 1), ("a", 2)])
    except LockedStabilityError:
        pass
    else:
        raise LockedStabilityError("duplicate-key rejection test failed")
    try:
        _reject_constant("NaN")
    except LockedStabilityError:
        pass
    else:
        raise LockedStabilityError("non-finite JSON rejection test failed")
    _equal(_lock_configuration()["stability_transitions"], list(range(64, 164)),
           "100-transition schedule")
    lock_path = args.lock_file or _default_lock(args.control_repo.resolve())
    if lock_path.exists():
        lock = _load_json(lock_path)
        _validate_lock_shape(lock)
    print("locked stability harness self-test passed", flush=True)


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--control-repo", type=Path, default=CONTROL_ROOT)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    make_parser = subparsers.add_parser("make-lock")
    _common_parser(make_parser)
    make_parser.add_argument("--manifest", type=Path)
    make_parser.add_argument("--trace", type=Path)
    make_parser.add_argument("--output", type=Path)

    prepare_parser = subparsers.add_parser("prepare")
    _common_parser(prepare_parser)
    prepare_parser.add_argument("--locked-worktree", type=Path, required=True)
    prepare_parser.add_argument("--campaign-dir", type=Path, required=True)
    prepare_parser.add_argument("--manifest", type=Path)
    prepare_parser.add_argument("--trace", type=Path)

    preflight_parser = subparsers.add_parser("preflight")
    _common_parser(preflight_parser)
    preflight_parser.add_argument("--manifest", type=Path)
    preflight_parser.add_argument("--trace", type=Path)

    verify_parser = subparsers.add_parser("verify-task")
    _common_parser(verify_parser)
    verify_parser.add_argument("--campaign-dir", type=Path, required=True)
    verify_parser.add_argument("--rank", type=int, choices=RANKS, required=True)

    attest_parser = subparsers.add_parser("attest-task")
    _common_parser(attest_parser)
    attest_parser.add_argument("--campaign-dir", type=Path, required=True)
    attest_parser.add_argument("--rank", type=int, choices=RANKS, required=True)

    aggregate_parser = subparsers.add_parser("aggregate")
    _common_parser(aggregate_parser)
    aggregate_parser.add_argument("--campaign-dir", type=Path, required=True)

    test_parser = subparsers.add_parser("self-test")
    test_parser.add_argument("--control-repo", type=Path, default=CONTROL_ROOT)
    test_parser.add_argument("--lock-file", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    {
        "make-lock": make_lock,
        "prepare": prepare,
        "preflight": preflight,
        "verify-task": verify_task,
        "attest-task": attest_task,
        "aggregate": aggregate,
        "self-test": self_test,
    }[args.action](args)


if __name__ == "__main__":
    main()
