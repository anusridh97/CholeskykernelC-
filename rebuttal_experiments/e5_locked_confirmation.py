"""Fail-closed provenance harness for the fresh ten-seed E5 confirmation.

The numerical drivers remain separate modules so their timed regions stay
identical to the original E5 campaign.  This harness freezes and verifies the
source/runtime/protocol, records a data-free GPU preflight, and writes one
cryptographic attestation after each seed task has produced all expected raw
artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCHEMA_LOCK = "e5_locked_confirmation_lock_v1"
SCHEMA_PREFLIGHT = "e5_locked_confirmation_preflight_v1"
SCHEMA_ATTESTATION = "e5_locked_confirmation_attestation_v1"
CAMPAIGN_TAG = "e5-locked-confirm-v1"
SEEDS = tuple(range(5001, 5011))
RANKS = (2048, 4096, 8192, 16384, 32768, 49152, 65536)
PROBE_RANKS = (49152, 65536)
PROBE_WIDTH = 16
PROBE_SEED = 29421
PROBE_TOLERANCE = 1e-3
EXPECTED_RAW_JSONS = 38
EXPECTED_PROBE_FILES = 4
EXPECTED_ACCOUNT = "marlowe-m000151-pm06"
EXPECTED_PARTITION = "batch"
EXPECTED_QOS = "medium"
EXPECTED_CPUS = 4
EXPECTED_GPU = "NVIDIA H100 80GB HBM3"
EXPECTED_GPU_MEMORY = 85_045_477_376
EXPECTED_DRIVER = "565.57.01"
EXPECTED_RUNTIME = {
    "python_executable": "/usr/bin/python3",
    "python_sha256": (
        "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
    ),
    "python_version": "3.10.12",
    "torch": "2.12.0+cu126",
    "torch_cuda": "12.6",
    "cudnn": 91002,
    "numpy": "2.2.6",
    "triton": "3.7.0",
    "matmul_allow_tf32": False,
    "cudnn_allow_tf32": True,
    "float32_matmul_precision": "highest",
}
SOURCE_PATHS = (
    "rebuttal_experiments/__init__.py",
    "rebuttal_experiments/common.py",
    "rebuttal_experiments/cuda_scaling.py",
    "rebuttal_experiments/cuda_inverse_stability.py",
    "rebuttal_experiments/cuda_inverse_refresh.py",
    "rebuttal_experiments/cuda_large_rank_compare.py",
    "rebuttal_experiments/aggregate_e5n10.py",
    "rebuttal_experiments/aggregate_e5_locked.py",
    "rebuttal_experiments/e5_locked_confirmation.py",
    "rebuttal_experiments/slurm/cuda_e5_locked_confirmation.sbatch",
    "rebuttal_experiments/slurm/cuda_e5_locked_preflight.sbatch",
    "rebuttal_experiments/E5_LOCKED_CONFIRMATION_PROTOCOL.md",
    "student_pkg/doublesided_cholesky.py",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class LockError(RuntimeError):
    """A source, runtime, scheduler, or artifact invariant was violated."""


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise LockError(f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject(value: str) -> Any:
        raise LockError(f"{path}: non-finite JSON constant {value!r}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LockError(f"cannot read strict JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise LockError(f"{path}: top-level JSON value must be an object")
    return value


def write_once_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def run_text(arguments: tuple[str, ...], cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            arguments,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise LockError(
            f"{' '.join(arguments)} failed: {error.stderr.strip()}"
        ) from error
    return completed.stdout.strip()


def git_text(repository: Path, *arguments: str) -> str:
    return run_text(("git", "-C", str(repository), *arguments))


def current_runtime() -> dict[str, Any]:
    import triton

    executable = Path(sys.executable).resolve()
    return {
        "python_executable": str(executable),
        "python_sha256": sha256_file(executable),
        "python_version": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "triton": triton.__version__,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def gpu_identity() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise LockError("CUDA is unavailable")
    props = torch.cuda.get_device_properties(0)
    query = run_text(
        (
            "nvidia-smi",
            "--query-gpu=uuid,driver_version,name,memory.total",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        )
    ).splitlines()
    if len(query) != 1:
        raise LockError(f"expected one visible GPU, nvidia-smi returned {query!r}")
    fields = [item.strip() for item in query[0].split(",", 3)]
    if len(fields) != 4:
        raise LockError(f"cannot parse nvidia-smi row {query[0]!r}")
    uuid, driver, name, memory_mib = fields
    result = {
        "torch_name": torch.cuda.get_device_name(0),
        "torch_total_memory": props.total_memory,
        "uuid": uuid,
        "driver": driver,
        "nvidia_smi_name": name,
        "nvidia_smi_memory_mib": int(memory_mib),
    }
    if result["torch_name"] != EXPECTED_GPU:
        raise LockError(f"GPU name {result['torch_name']!r} != {EXPECTED_GPU!r}")
    if result["torch_total_memory"] != EXPECTED_GPU_MEMORY:
        raise LockError(
            f"GPU memory {result['torch_total_memory']} != {EXPECTED_GPU_MEMORY}"
        )
    if result["driver"] != EXPECTED_DRIVER:
        raise LockError(f"driver {result['driver']!r} != {EXPECTED_DRIVER!r}")
    return result


def scheduler_identity(seed_required: bool) -> dict[str, Any]:
    required = {
        "SLURM_JOB_ID": None,
        "SLURM_JOB_ACCOUNT": EXPECTED_ACCOUNT,
        "SLURM_JOB_PARTITION": EXPECTED_PARTITION,
        "SLURM_JOB_QOS": EXPECTED_QOS,
        "SLURM_CPUS_PER_TASK": str(EXPECTED_CPUS),
    }
    if seed_required:
        required["SLURM_ARRAY_JOB_ID"] = None
        required["SLURM_ARRAY_TASK_ID"] = None
    for key, expected in required.items():
        actual = os.environ.get(key)
        if not actual:
            raise LockError(f"required scheduler variable {key} is missing")
        if expected is not None and actual != expected:
            raise LockError(f"{key}={actual!r}, expected {expected!r}")
    if seed_required and int(os.environ["SLURM_ARRAY_TASK_ID"]) not in SEEDS:
        raise LockError(
            f"array task must be one of {SEEDS}, got "
            f"{os.environ['SLURM_ARRAY_TASK_ID']!r}"
        )
    job_id = os.environ["SLURM_JOB_ID"]
    return {
        "job_id": job_id,
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "account": os.environ["SLURM_JOB_ACCOUNT"],
        "partition": os.environ["SLURM_JOB_PARTITION"],
        "qos": os.environ["SLURM_JOB_QOS"],
        "cpus_per_task": int(os.environ["SLURM_CPUS_PER_TASK"]),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "scontrol_show_job": run_text(("scontrol", "show", "job", "-o", job_id)),
    }


def expected_configuration() -> dict[str, Any]:
    return {
        "seeds": list(SEEDS),
        "cuda_scaling": {
            "fused": {
                "ranks": [64, 128],
                "batches": [1, 4, 16, 64],
                "warmup": 10,
                "repeats": 60,
            },
            "parallel": {
                "ranks": [256, 512, 1024],
                "batch": 1,
                "warmup": 10,
                "repeats": 60,
            },
            "cached_inverse": {
                "ranks": [256, 512, 1024],
                "batch": 1,
                "warmup": 10,
                "repeats": 60,
            },
        },
        "stability": {
            "single_block": {"ranks": [256, 512, 1024], "steps": 500},
            "inplace": {"ranks": [1024], "steps": 500},
            "condition": {
                "rank": 256,
                "steps": 200,
                "target_conditions": [100.0, 10000.0, 1000000.0],
            },
        },
        "refresh": {
            "ranks": [256, 512, 1024],
            "steps": 200,
            "intervals": [0, 10, 50, 100, 200],
        },
        "large_rank": {
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
            "warmup": 1,
            "method_order": "odd seeds inverse-first; even seeds fresh-first",
            "timing_region_changed_from_original_e5": False,
            "dense_correctness_through_rank": 32768,
            "probe_validation": {
                "ranks": list(PROBE_RANKS),
                "generator": "numpy.PCG64 Rademacher",
                "seed_plus_rank": PROBE_SEED,
                "width": PROBE_WIDTH,
                "tolerance": PROBE_TOLERANCE,
                "storage": "O(r*q)",
                "excluded_from_timing": True,
                "cross_method_actions": [
                    "inverse_factor_action",
                    "whitened_operator_action",
                ],
            },
        },
        "expected_raw_jsons_per_seed": EXPECTED_RAW_JSONS,
        "expected_probe_files_per_seed": EXPECTED_PROBE_FILES,
    }


def make_lock(repository: Path, output: Path) -> None:
    if git_text(repository, "status", "--porcelain"):
        raise LockError("repository must be clean before creating the lock")
    commit = git_text(repository, "rev-parse", "HEAD")
    if not COMMIT_RE.fullmatch(commit):
        raise LockError(f"invalid Git commit {commit!r}")
    sources: dict[str, Any] = {}
    for relative in SOURCE_PATHS:
        path = repository / relative
        if not path.is_file():
            raise LockError(f"missing frozen source {relative}")
        sources[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    payload = {
        "schema_version": SCHEMA_LOCK,
        "campaign_tag": CAMPAIGN_TAG,
        "created_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "implementation_commit": commit,
        "sources": sources,
        "runtime": EXPECTED_RUNTIME,
        "gpu": {
            "name": EXPECTED_GPU,
            "total_memory_bytes": EXPECTED_GPU_MEMORY,
            "driver": EXPECTED_DRIVER,
        },
        "scheduler": {
            "account": EXPECTED_ACCOUNT,
            "partition": EXPECTED_PARTITION,
            "qos": EXPECTED_QOS,
            "array": "5001-5010%10",
            "cpus_per_task": EXPECTED_CPUS,
            "gpus_per_task": 1,
            "memory": "64G",
            "time_limit": "01:00:00",
        },
        "configuration": expected_configuration(),
        "analysis_rule": (
            "No tuning, early stopping, seed replacement, or outcome-dependent "
            "rerun; aggregate all ten predeclared seeds."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def verify_lock(repository: Path, lock_path: Path) -> dict[str, Any]:
    lock = strict_json(lock_path)
    expected_top = {
        "schema_version",
        "campaign_tag",
        "created_utc",
        "implementation_commit",
        "sources",
        "runtime",
        "gpu",
        "scheduler",
        "configuration",
        "analysis_rule",
    }
    if set(lock) != expected_top:
        raise LockError(f"campaign lock keys are {set(lock)!r}, expected {expected_top!r}")
    if lock["schema_version"] != SCHEMA_LOCK:
        raise LockError("wrong lock schema")
    if lock["campaign_tag"] != CAMPAIGN_TAG:
        raise LockError("wrong campaign tag")
    if lock["runtime"] != EXPECTED_RUNTIME:
        raise LockError("runtime lock differs from compiled protocol")
    if lock["configuration"] != expected_configuration():
        raise LockError("configuration lock differs from compiled protocol")
    if set(lock["sources"]) != set(SOURCE_PATHS):
        raise LockError("source path set differs from compiled protocol")
    commit = lock["implementation_commit"]
    if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
        raise LockError("implementation_commit is not a full Git commit")
    for relative, expected in lock["sources"].items():
        path = repository / relative
        actual = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if actual != expected:
            raise LockError(
                f"source mismatch for {relative}: {actual!r} != {expected!r}"
            )
    actual_runtime = current_runtime()
    if actual_runtime != EXPECTED_RUNTIME:
        raise LockError(
            f"runtime mismatch: {actual_runtime!r} != {EXPECTED_RUNTIME!r}"
        )
    required_head = os.environ.get("E5_CAMPAIGN_COMMIT")
    head = git_text(repository, "rev-parse", "HEAD")
    if required_head and head != required_head:
        raise LockError(f"HEAD={head}, expected E5_CAMPAIGN_COMMIT={required_head}")
    if git_text(repository, "diff", "--name-only"):
        raise LockError("tracked worktree changes detected")
    expected_lock_sha = os.environ.get("E5_LOCK_SHA256")
    actual_lock_sha = sha256_file(lock_path)
    if expected_lock_sha and actual_lock_sha != expected_lock_sha:
        raise LockError(
            f"lock SHA={actual_lock_sha}, expected E5_LOCK_SHA256={expected_lock_sha}"
        )
    return {
        "campaign_commit": head,
        "implementation_commit": commit,
        "lock_path": str(lock_path),
        "lock_sha256": actual_lock_sha,
        "runtime": actual_runtime,
    }


def file_metadata(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(expected)), np.finfo(np.float32).tiny)
    return float(np.linalg.norm(actual - expected) / denominator)


def sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(item) for item in contiguous.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def audit_probes(
    repository: Path,
    payloads: list[tuple[Path, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for rank in PROBE_RANKS:
        by_mode = {
            payload["mode"]: (path, payload)
            for path, payload in payloads
            if payload.get("experiment") == "cuda_large_rank_compare"
            and payload.get("rank") == rank
        }
        expected_modes = {"inverse_inplace_recurrent", "fresh_refact_recurrent"}
        if set(by_mode) != expected_modes:
            raise LockError(
                f"rank {rank}: probe modes {set(by_mode)!r} != {expected_modes!r}"
            )
        arrays: dict[str, dict[str, np.ndarray]] = {}
        artifacts = []
        identity_residuals = {}
        for mode, (_, payload) in by_mode.items():
            if payload.get("probe_validation") is not True:
                raise LockError(f"rank {rank} mode {mode}: probe validation absent")
            if payload.get("probe_width") != PROBE_WIDTH:
                raise LockError(f"rank {rank} mode {mode}: wrong probe width")
            if payload.get("probe_seed") != PROBE_SEED + rank:
                raise LockError(f"rank {rank} mode {mode}: wrong probe seed")
            if payload.get("probe_tolerance") != PROBE_TOLERANCE:
                raise LockError(f"rank {rank} mode {mode}: wrong probe tolerance")
            residual = payload.get("probe_identity_residuals")
            if not isinstance(residual, dict) or not residual:
                raise LockError(f"rank {rank} mode {mode}: residuals absent")
            if max(float(value) for value in residual.values()) > PROBE_TOLERANCE:
                raise LockError(f"rank {rank} mode {mode}: identity gate failed")
            identity_residuals[mode] = residual
            artifact = repository / payload["probe_artifact"]
            metadata = file_metadata(artifact, repository)
            if metadata["sha256"] != payload.get("probe_artifact_sha256"):
                raise LockError(f"rank {rank} mode {mode}: probe SHA mismatch")
            if metadata["bytes"] != payload.get("probe_artifact_bytes"):
                raise LockError(f"rank {rank} mode {mode}: probe size mismatch")
            with np.load(artifact, allow_pickle=False) as archive:
                expected_keys = {
                    "probe",
                    "inverse_factor_action",
                    "whitened_operator_action",
                }
                if set(archive.files) != expected_keys:
                    raise LockError(
                        f"rank {rank} mode {mode}: NPZ keys {set(archive.files)!r}"
                    )
                arrays[mode] = {key: archive[key] for key in expected_keys}
            hash_fields = {
                "probe": "probe_array_sha256",
                "inverse_factor_action": "inverse_factor_action_sha256",
                "whitened_operator_action": (
                    "whitened_operator_action_sha256"
                ),
            }
            for array_name, field in hash_fields.items():
                actual_hash = sha256_array(arrays[mode][array_name])
                if payload.get(field) != actual_hash:
                    raise LockError(
                        f"rank {rank} mode {mode}: {field} mismatch"
                    )
            artifacts.append(metadata)
        inverse = arrays["inverse_inplace_recurrent"]
        fresh = arrays["fresh_refact_recurrent"]
        if not np.array_equal(inverse["probe"], fresh["probe"]):
            raise LockError(f"rank {rank}: cross-method probes differ")
        probe_sha256 = sha256_array(inverse["probe"])
        cross = {
            key: relative_error(inverse[key], fresh[key])
            for key in ("inverse_factor_action", "whitened_operator_action")
        }
        if max(cross.values()) > PROBE_TOLERANCE:
            raise LockError(f"rank {rank}: cross-method probe gate failed: {cross}")
        rows.append(
            {
                "rank": rank,
                "cross_method_relative_errors": cross,
                "identical_probe_array_sha256": probe_sha256,
                "identity_residuals": identity_residuals,
                "artifacts": sorted(artifacts, key=lambda value: value["path"]),
            }
        )
    return rows


def command_preflight(args: argparse.Namespace) -> None:
    source = verify_lock(args.repository, args.lock)
    scheduler = scheduler_identity(seed_required=False)
    gpu = gpu_identity()
    payload = {
        "schema_version": SCHEMA_PREFLIGHT,
        "campaign_tag": CAMPAIGN_TAG,
        "source": source,
        "scheduler": scheduler,
        "gpu": gpu,
        "host": socket.gethostname(),
        "data_free": True,
        "numerical_driver_imported": False,
    }
    write_once_json(args.output, payload)


def command_verify_task(args: argparse.Namespace) -> None:
    seed = int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
    if seed != args.seed or seed not in SEEDS:
        raise LockError(f"task seed {seed} != requested seed {args.seed}")
    source = verify_lock(args.repository, args.lock)
    scheduler = scheduler_identity(seed_required=True)
    gpu = gpu_identity()
    print(
        json.dumps(
            {
                "status": "verified",
                "seed": seed,
                "source": source,
                "scheduler": scheduler,
                "gpu": gpu,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def command_attest_task(args: argparse.Namespace) -> None:
    seed = int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
    if seed != args.seed or seed not in SEEDS:
        raise LockError(f"task seed {seed} != requested seed {args.seed}")
    source = verify_lock(args.repository, args.lock)
    scheduler = scheduler_identity(seed_required=True)
    gpu = gpu_identity()
    suffix = f"{CAMPAIGN_TAG}-task{seed}"
    raw_paths = sorted(
        {
            path
            for directory in (
                args.repository / "rebuttal_results" / "cuda",
                args.repository / "rebuttal_results" / "cuda_tuning",
            )
            if directory.is_dir()
            for path in directory.rglob(f"*{suffix}*.json")
        }
    )
    if len(raw_paths) != EXPECTED_RAW_JSONS:
        raise LockError(
            f"seed {seed}: found {len(raw_paths)} tagged JSONs, "
            f"expected {EXPECTED_RAW_JSONS}"
        )
    payloads = []
    for path in raw_paths:
        payload = strict_json(path)
        if payload.get("seed") != seed:
            raise LockError(f"{path}: seed {payload.get('seed')!r} != {seed}")
        if payload.get("output_suffix") != suffix:
            raise LockError(f"{path}: wrong output_suffix")
        if str(payload.get("slurm_array_task_id")) != str(seed):
            raise LockError(f"{path}: wrong Slurm array task")
        payloads.append((path, payload))
    probe_rows = audit_probes(args.repository, payloads)
    probe_count = sum(len(row["artifacts"]) for row in probe_rows)
    if probe_count != EXPECTED_PROBE_FILES:
        raise LockError(
            f"seed {seed}: {probe_count} probe files != {EXPECTED_PROBE_FILES}"
        )
    attestation = {
        "schema_version": SCHEMA_ATTESTATION,
        "campaign_tag": CAMPAIGN_TAG,
        "seed": seed,
        "source": source,
        "scheduler": scheduler,
        "gpu": gpu,
        "host": socket.gethostname(),
        "raw_jsons": [
            file_metadata(path, args.repository) for path in raw_paths
        ],
        "probe_validation": probe_rows,
        "raw_json_count": len(raw_paths),
        "probe_file_count": probe_count,
        "protocol_complete": True,
    }
    write_once_json(args.output, attestation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    make = subparsers.add_parser("make-lock")
    make.add_argument("--repository", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    make.set_defaults(function=lambda args: make_lock(args.repository, args.output))

    for name, function in (
        ("preflight", command_preflight),
        ("verify-task", command_verify_task),
        ("attest-task", command_attest_task),
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--repository", type=Path, required=True)
        command.add_argument("--lock", type=Path, required=True)
        command.add_argument("--output", type=Path)
        if name != "preflight":
            command.add_argument("--seed", type=int, required=True)
        command.set_defaults(function=function)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
