"""Source-locked paired confirmation campaign for E1.

This file intentionally exposes no experimental hyperparameter overrides.
The protocol is frozen in ``FIXED_CONFIG`` and task indices map bijectively
to a phase, seed, rank, and thread count.  Each task runs Ours and Refact
sequentially from identical initial state and minibatch permutations.

The intended workflow is:

1. Commit this driver, its protocol, and its two Slurm scripts.
2. Create a write-once source lock with ``--action freeze-lock``.
3. Run and audit the excluded pilots.
4. Explicitly approve the pilot audit.
5. Run the untouched confirmation seeds and aggregate them.

The driver never overwrites an artifact.  Confirmation tasks refuse to run
until the pilot audit has been explicitly approved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CORRECTED_DIR = ROOT / "Cholesky-main-2" / "corrected_incremental_cholesky"
DATA_PATH = ROOT / "Cholesky-main-2" / "data" / "protein.mat"
DEFAULT_RESULTS = ROOT / "rebuttal_results" / "e1_confirmation"
DEFAULT_CAMPAIGN = "e1-confirm-fastpositive-v2"
EXPECTED_SLURM_ACCOUNT = "marlowe-m000151-pm06"
EXPECTED_SLURM_PARTITION = "batch"
EXPECTED_TORCH_SITE = Path(
    "/users/arjo/.local/lib/python3.10/site-packages"
)

# Resolve data_utils and fast_cholesky from the corrected package, not from
# student_pkg or an extension elsewhere on PYTHONPATH.
if str(CORRECTED_DIR) not in sys.path:
    sys.path.insert(0, str(CORRECTED_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

import numpy
import scipy
import torch
from scipy.io import loadmat
from rebuttal_experiments.ng_sweep import (
    apply_first_layer_step,
    build_model,
    flatten_first_layer_gradient,
    whiten,
)


PILOT_SEEDS = (1000, 1001, 1002)
CONFIRMATION_SEEDS = tuple(range(2000, 2010))
RANKS = (256, 512, 1024)
THREAD_COUNTS = (1, 10)
METHODS = ("ours", "refact")
EXPECTED_DATA_PARTITION = {
    "train_examples": 36584,
    "test_examples": 9146,
    "features": 9,
    "train_index_sha256": (
        "6d8fd4e5f016310c1d005c6963740ec8dc9ff57886c9cf9a92077239b77bd301"
    ),
    "test_index_sha256": (
        "8a6bdd65ada4b99d2e2ddc2ac368c8ce99fc17c64001c27928cc73f039b5f4a7"
    ),
    "training_tensor_sha256": (
        "e17d77c799877dd560905f0a88217e703d16973af55368ed8e7a34897c987b95"
    ),
}

FIXED_CONFIG: dict[str, Any] = {
    "dataset": "UCI Protein protein.mat",
    "split_seed": 42,
    "train_fraction": 0.8,
    "dtype": "float32",
    "architecture": [9, 512, 512, 128, 1],
    "epochs": 10,
    "batch_size": 256,
    "learning_rate": 0.1,
    "damping": 1.0,
    "momentum": 0.9,
    "refresh_threshold": 0.1,
    "refactor_every": 50,
    "ranks": list(RANKS),
    "thread_counts": list(THREAD_COUNTS),
    "pilot_seeds": list(PILOT_SEEDS),
    "confirmation_seeds": list(CONFIRMATION_SEEDS),
    "method_order": (
        "parity counterbalanced by seed_index + rank_index + thread_index"
    ),
    "test_policy": (
        "zero test evaluations in pilots; exactly one final test evaluation "
        "per method in each confirmation pair, after both methods finish "
        "training and all pre-test checks pass"
    ),
    "outcome_adaptive_changes": False,
    "expected_data_partition": EXPECTED_DATA_PARTITION,
}

LOCKED_FILES = {
    "driver": "rebuttal_experiments/e1_confirmation.py",
    "protocol": "rebuttal_experiments/E1_CONFIRMATION_PROTOCOL.md",
    "prior_outcome_audit": (
        "rebuttal_experiments/E1_PRIOR_OUTCOME_AUDIT.md"
    ),
    "pilot_sbatch": "rebuttal_experiments/slurm/e1_confirmation_pilot.sbatch",
    "preflight_sbatch": (
        "rebuttal_experiments/slurm/e1_confirmation_preflight.sbatch"
    ),
    "confirmation_sbatch": (
        "rebuttal_experiments/slurm/e1_confirmation.sbatch"
    ),
    "ng_reference_driver": "rebuttal_experiments/ng_sweep.py",
    "common_helpers": "rebuttal_experiments/common.py",
    "corrected_data_loader": (
        "Cholesky-main-2/corrected_incremental_cholesky/data_utils.py"
    ),
    "incremental_source": (
        "Cholesky-main-2/corrected_incremental_cholesky/fast_update.cpp"
    ),
    "extension_build": (
        "Cholesky-main-2/corrected_incremental_cholesky/setup_ext.py"
    ),
    "incremental_binary": (
        "Cholesky-main-2/corrected_incremental_cholesky/fast_cholesky.so"
    ),
    "protein_data": "Cholesky-main-2/data/protein.mat",
}
TRACKED_LOCK_KEYS = tuple(
    key for key in LOCKED_FILES if key != "incremental_binary"
)

TIMING_COMPONENTS = (
    "batch_fetch_seconds",
    "forward_backward_seconds",
    "projection_decision_seconds",
    "refresh_seconds",
    "nonrefresh_core_seconds",
    "reset_seconds",
    "direction_seconds",
    "parameter_update_seconds",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _module_record(module: Any) -> dict[str, Any]:
    path = Path(module.__file__).resolve()
    return {
        "version": str(module.__version__),
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _cpu_isa_manifest() -> dict[str, str]:
    first_processor: dict[str, str] = {}
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            break
        if ":" in line:
            name, value = line.split(":", 1)
            first_processor[name.strip()] = value.strip()
    required = (
        "vendor_id",
        "cpu family",
        "model",
        "model name",
        "stepping",
        "flags",
    )
    missing = [name for name in required if name not in first_processor]
    if missing:
        raise RuntimeError(f"CPU ISA manifest omitted {missing}")
    return {name: first_processor[name] for name in required}


def _runtime_manifest() -> dict[str, Any]:
    """Return stable runtime fields that must match on every campaign task."""
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("runtime lock requires PYTHONNOUSERSITE=1")
    configured_torch_site = os.environ.get("E1_TORCH_SITE")
    if configured_torch_site != str(EXPECTED_TORCH_SITE):
        raise RuntimeError(
            f"E1_TORCH_SITE must be {str(EXPECTED_TORCH_SITE)!r}"
        )
    torch_root = Path(torch.__file__).resolve().parent
    try:
        torch_root.relative_to(EXPECTED_TORCH_SITE)
    except ValueError as error:
        raise RuntimeError(
            f"torch loaded outside pinned site: {torch_root}"
        ) from error
    dependency_root = (ROOT / "rebuttal_deps").resolve()
    for name, module in (("numpy", numpy), ("scipy", scipy)):
        try:
            Path(module.__file__).resolve().relative_to(dependency_root)
        except ValueError as error:
            raise RuntimeError(
                f"{name} loaded outside campaign dependency root"
            ) from error

    import numpy._core._multiarray_umath as numpy_core
    import scipy.linalg._fblas as scipy_blas
    import scipy.linalg._flapack as scipy_lapack

    python = Path(sys.executable).resolve()
    extension = (ROOT / LOCKED_FILES["incremental_binary"]).resolve()
    ldd_raw = subprocess.run(
        ("ldd", str(extension)),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    ldd_lines = [
        re.sub(r"\s+\(0x[0-9a-fA-F]+\)$", "", line.strip())
        for line in ldd_raw.splitlines()
        if line.strip()
    ]
    ldd = "\n".join(ldd_lines)
    return {
        "schema_version": 1,
        "python": {
            "executable": str(python),
            "bytes": python.stat().st_size,
            "sha256": _sha256(python),
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "packages": {
            "torch": _module_record(torch),
            "numpy": _module_record(numpy),
            "scipy": _module_record(scipy),
        },
        "critical_binary_files": {
            "torch_C": _file_record(Path(torch._C.__file__)),
            "torch_cpu": _file_record(
                torch_root / "lib" / "libtorch_cpu.so"
            ),
            "numpy_core": _file_record(Path(numpy_core.__file__)),
            "scipy_blas": _file_record(Path(scipy_blas.__file__)),
            "scipy_lapack": _file_record(Path(scipy_lapack.__file__)),
        },
        "path_policy": {
            "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
            "E1_TORCH_SITE": configured_torch_site,
            "dependency_root": str(dependency_root),
        },
        "torch_config_sha256": _text_sha256(torch.__config__.show()),
        "platform": {
            "machine": platform.machine(),
            "libc": list(platform.libc_ver()),
            "cpu_isa": _cpu_isa_manifest(),
        },
        "extension_ldd_sha256": _text_sha256(ldd),
        "extension_ldd": ldd_lines,
    }


def _git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git",) + arguments,
        cwd=ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _scoped_git_clean(relative_paths: list[str]) -> None:
    for relative in relative_paths:
        tracked = _git("ls-files", "--error-unmatch", "--", relative, check=False)
        if tracked.returncode != 0:
            raise RuntimeError(
                f"source-lock prerequisite is not tracked by git: {relative}"
            )
    unstaged = _git(
        "diff", "--quiet", "HEAD", "--", *relative_paths, check=False
    )
    staged = _git(
        "diff", "--cached", "--quiet", "HEAD", "--", *relative_paths,
        check=False,
    )
    if unstaged.returncode != 0 or staged.returncode != 0:
        raise RuntimeError(
            "source-lock prerequisite has staged or unstaged changes; commit "
            "the E1 harness and all locked sources first"
        )


def _write_once_json(payload: dict[str, Any], path: Path) -> None:
    """Atomically publish JSON without replacing an existing artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link is atomic and fails if the target already exists.
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path}")
    return payload


def _campaign_dir(args: argparse.Namespace) -> Path:
    return Path(args.results_dir).resolve() / args.campaign_tag


def _lock_path(args: argparse.Namespace) -> Path:
    if args.lock_file:
        return Path(args.lock_file).resolve()
    return _campaign_dir(args) / "source_lock.json"


def _phase_seeds(phase: str) -> tuple[int, ...]:
    if phase == "pilot":
        return PILOT_SEEDS
    if phase == "confirmation":
        return CONFIRMATION_SEEDS
    raise ValueError(f"unknown phase {phase!r}")


def _task_specs(phase: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(_phase_seeds(phase)):
        for rank_index, rank in enumerate(RANKS):
            for thread_index, threads in enumerate(THREAD_COUNTS):
                if (seed_index + rank_index + thread_index) % 2 == 0:
                    order = ["ours", "refact"]
                else:
                    order = ["refact", "ours"]
                specs.append(
                    {
                        "phase": phase,
                        "task_index": len(specs),
                        "seed": seed,
                        "seed_index": seed_index,
                        "rank": rank,
                        "rank_index": rank_index,
                        "threads": threads,
                        "thread_index": thread_index,
                        "method_order": order,
                    }
                )
    return specs


def _task_spec(phase: str, task_index: int) -> dict[str, Any]:
    specs = _task_specs(phase)
    if not 0 <= task_index < len(specs):
        raise ValueError(
            f"{phase} task index must be in [0, {len(specs) - 1}], "
            f"got {task_index}"
        )
    return specs[task_index]


def _artifact_path(directory: Path, spec: dict[str, Any]) -> Path:
    return directory / (
        f"{spec['phase']}_pair_r{spec['rank']}_t{spec['threads']}"
        f"_seed{spec['seed']}.json"
    )


def freeze_lock(args: argparse.Namespace) -> dict[str, Any]:
    if args.campaign_tag != DEFAULT_CAMPAIGN:
        raise ValueError(
            f"the reviewed campaign tag is frozen as {DEFAULT_CAMPAIGN!r}"
        )
    relative_tracked = [LOCKED_FILES[key] for key in TRACKED_LOCK_KEYS]
    _scoped_git_clean(relative_tracked)
    head = _git("rev-parse", "HEAD").stdout.strip()
    files: dict[str, Any] = {}
    for key, relative in LOCKED_FILES.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing locked file: {path}")
        files[key] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "git_tracked": key in TRACKED_LOCK_KEYS,
        }

    payload = {
        "schema_version": 2,
        "experiment": "e1_paired_confirmation_source_lock",
        "campaign_tag": args.campaign_tag,
        "created_utc": _utc_now(),
        "git_commit": head,
        "fixed_config": FIXED_CONFIG,
        "task_counts": {
            "pilot": len(_task_specs("pilot")),
            "confirmation": len(_task_specs("confirmation")),
        },
        "implementation": {
            "ours": (
                "explicit metric_sign=+1/operator_sign=+1 dispatch to the "
                "row-major virtual-padding SIMD positive-update kernel"
            ),
            "refact": (
                "G += aa^T, fresh Cholesky, complete rewhitening, and two "
                "triangular solves"
            ),
            "precision": "both methods use CPU torch.float32 throughout",
        },
        "runtime": _runtime_manifest(),
        "files": files,
    }
    output = _lock_path(args)
    try:
        output.relative_to(ROOT)
    except ValueError as error:
        raise RuntimeError(
            "the source lock must be created inside the repository so it "
            "can be committed before pilots"
        ) from error
    _write_once_json(payload, output)
    print(f"wrote immutable source lock {output}", flush=True)
    return payload


def _verify_lock(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str, str]:
    path = _lock_path(args)
    lock = _read_json(path)
    if lock.get("schema_version") != 2:
        raise RuntimeError("unsupported or missing source-lock schema")
    if lock.get("campaign_tag") != args.campaign_tag:
        raise RuntimeError("campaign tag does not match source lock")
    if lock.get("fixed_config") != FIXED_CONFIG:
        raise RuntimeError("compiled driver configuration differs from source lock")
    locked_commit = lock.get("git_commit")
    if not isinstance(locked_commit, str):
        raise RuntimeError("source lock has no git commit")
    current_head = _git("rev-parse", "HEAD").stdout.strip()
    ancestor = _git(
        "merge-base",
        "--is-ancestor",
        locked_commit,
        current_head,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(
            f"locked commit {locked_commit} is not an ancestor of run "
            f"HEAD {current_head}"
        )

    try:
        relative_lock = str(path.relative_to(ROOT))
    except ValueError as error:
        raise RuntimeError(
            "the immutable source lock must live inside the repository"
        ) from error
    relative_tracked = [
        *[LOCKED_FILES[key] for key in TRACKED_LOCK_KEYS],
        relative_lock,
    ]
    _scoped_git_clean(relative_tracked)
    locked_files = lock.get("files")
    if not isinstance(locked_files, dict):
        raise RuntimeError("source lock has no file manifest")
    for key, relative in LOCKED_FILES.items():
        record = locked_files.get(key)
        if not isinstance(record, dict) or record.get("path") != relative:
            raise RuntimeError(f"source-lock path mismatch for {key}")
        path_to_check = ROOT / relative
        if not path_to_check.is_file():
            raise FileNotFoundError(f"locked file disappeared: {path_to_check}")
        if path_to_check.stat().st_size != record.get("bytes"):
            raise RuntimeError(f"locked byte size changed for {relative}")
        if _sha256(path_to_check) != record.get("sha256"):
            raise RuntimeError(f"locked SHA-256 changed for {relative}")
    observed_runtime = _runtime_manifest()
    if lock.get("runtime") != observed_runtime:
        raise RuntimeError(
            "Python/PyTorch/NumPy/SciPy/ABI runtime differs from source lock"
        )
    return lock, _sha256(path), current_head


def _parse_parallel_value(report: str, name: str) -> int:
    match = re.search(rf"{re.escape(name)}\s*:\s*(\d+)", report)
    if match is None:
        raise RuntimeError(f"runtime thread report omitted {name!r}")
    return int(match.group(1))


def _cpu_topology(cpu: int) -> dict[str, int]:
    topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    try:
        package = int((topology / "physical_package_id").read_text().strip())
        core = int((topology / "core_id").read_text().strip())
    except (FileNotFoundError, ValueError) as error:
        raise RuntimeError(f"cannot verify physical topology for CPU {cpu}") from error
    return {"logical_cpu": cpu, "physical_package": package, "core_id": core}


def configure_and_verify_threads(threads: int) -> dict[str, Any]:
    required_environment = {
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
    }
    for name, expected in required_environment.items():
        actual = os.environ.get(name)
        if actual is None or actual.upper() != expected:
            raise RuntimeError(
                f"{name} must be {expected!r} before Python starts, got {actual!r}"
            )

    affinity_before = sorted(os.sched_getaffinity(0))
    if len(affinity_before) < threads:
        raise RuntimeError(
            f"allocation exposes only {len(affinity_before)} CPUs for {threads} threads"
        )
    affinity_topology = [_cpu_topology(cpu) for cpu in affinity_before]
    by_package: dict[int, list[dict[str, int]]] = {}
    for entry in affinity_topology:
        by_package.setdefault(entry["physical_package"], []).append(entry)
    eligible_packages = [
        (package, entries)
        for package, entries in by_package.items()
        if len(
            {
                (entry["physical_package"], entry["core_id"])
                for entry in entries
            }
        )
        >= threads
    ]
    if not eligible_packages:
        raise RuntimeError(
            f"allocation does not expose {threads} distinct cores on one socket"
        )
    selected_package, package_entries = min(
        eligible_packages,
        key=lambda item: (-len(item[1]), item[0]),
    )
    selected_entries: list[dict[str, int]] = []
    seen_cores: set[int] = set()
    for entry in sorted(
        package_entries,
        key=lambda item: (item["core_id"], item["logical_cpu"]),
    ):
        if entry["core_id"] in seen_cores:
            continue
        seen_cores.add(entry["core_id"])
        selected_entries.append(entry)
        if len(selected_entries) == threads:
            break
    selected_cpus = sorted(entry["logical_cpu"] for entry in selected_entries)
    os.sched_setaffinity(0, set(selected_cpus))
    affinity_after = sorted(os.sched_getaffinity(0))
    if affinity_after != selected_cpus:
        raise RuntimeError("failed to establish exact process CPU affinity")
    selected_topology = [_cpu_topology(cpu) for cpu in selected_cpus]
    physical_cores = {
        (entry["physical_package"], entry["core_id"])
        for entry in selected_topology
    }
    if len(physical_cores) != threads:
        raise RuntimeError(
            "selected affinity includes simultaneous hardware threads from "
            "the same physical core"
        )

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as error:
        raise RuntimeError(
            "PyTorch interop threads were initialized before E1 pinning"
        ) from error
    report = torch.__config__.parallel_info()
    runtime_values = {
        "aten_intraop": _parse_parallel_value(
            report, "at::get_num_threads()"
        ),
        "aten_interop": _parse_parallel_value(
            report, "at::get_num_interop_threads()"
        ),
        "openmp_max": _parse_parallel_value(report, "omp_get_max_threads()"),
        "mkl_max": _parse_parallel_value(report, "mkl_get_max_threads()"),
    }
    for name in ("aten_intraop", "openmp_max", "mkl_max"):
        if runtime_values[name] != threads:
            raise RuntimeError(
                f"runtime {name}={runtime_values[name]}, expected {threads}"
            )
    if runtime_values["aten_interop"] != 1:
        raise RuntimeError(
            f"runtime interop threads={runtime_values['aten_interop']}, expected 1"
        )
    return {
        "requested_threads": threads,
        "environment": {
            name: os.environ.get(name) for name in required_environment
        },
        "affinity_before": affinity_before,
        "affinity_before_topology": affinity_topology,
        "selected_cpus": selected_cpus,
        "affinity_after": affinity_after,
        "selected_cpu_topology": selected_topology,
        "selected_physical_package": selected_package,
        "unique_physical_cores": len(physical_cores),
        "runtime_values": runtime_values,
        "torch_parallel_info": report,
    }


def _validate_slurm_task(spec: dict[str, Any]) -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if not job_id:
        raise RuntimeError("run-pair is restricted to a reviewed Slurm allocation")
    if array_task != str(spec["task_index"]):
        raise RuntimeError(
            f"SLURM_ARRAY_TASK_ID={array_task!r} does not match task "
            f"{spec['task_index']}"
        )
    if cpus != "10":
        raise RuntimeError(
            f"each task must reserve exactly 10 CPUs, got {cpus!r}"
        )
    if os.environ.get("SLURM_RESTART_COUNT", "0") != "0":
        raise RuntimeError(
            "requeued/restarted tasks are forbidden by the exactly-once policy"
        )
    required = {
        "SLURM_JOB_ACCOUNT": EXPECTED_SLURM_ACCOUNT,
        "SLURM_JOB_PARTITION": EXPECTED_SLURM_PARTITION,
        "SLURM_NTASKS": "1",
        "SLURM_JOB_NUM_NODES": "1",
    }
    for name, expected in required.items():
        actual = os.environ.get(name)
        if actual != expected:
            raise RuntimeError(
                f"{name} must be {expected!r}, got {actual!r}"
            )
    allocation = subprocess.run(
        ("scontrol", "show", "job", "-o", job_id),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    if not re.search(
        r"(?:^|\s)(?:Shared=(?:0|EXCLUSIVE)|OverSubscribe=NO)(?:\s|$)",
        allocation,
    ):
        raise RuntimeError(
            "Slurm allocation is not exclusive; expected #SBATCH --exclusive"
        )
    return {
        "job_id": job_id,
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": array_task,
        "cpus_per_task": cpus,
        "account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "ntasks": os.environ.get("SLURM_NTASKS"),
        "job_num_nodes": os.environ.get("SLURM_JOB_NUM_NODES"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "step_id": os.environ.get("SLURM_STEP_ID"),
        "scontrol_job_sha256": _text_sha256(allocation),
        "exclusive_allocation_verified": True,
        "restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"),
    }


def _validate_preflight_slurm() -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        raise RuntimeError("preflight requires one non-array Slurm job")
    required = {
        "SLURM_JOB_ACCOUNT": EXPECTED_SLURM_ACCOUNT,
        "SLURM_JOB_PARTITION": EXPECTED_SLURM_PARTITION,
        "SLURM_NTASKS": "1",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_CPUS_PER_TASK": "10",
    }
    for name, expected in required.items():
        actual = os.environ.get(name)
        if actual != expected:
            raise RuntimeError(
                f"{name} must be {expected!r}, got {actual!r}"
            )
    allocation = subprocess.run(
        ("scontrol", "show", "job", "-o", job_id),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    if not re.search(
        r"(?:^|\s)(?:Shared=(?:0|EXCLUSIVE)|OverSubscribe=NO)(?:\s|$)",
        allocation,
    ):
        raise RuntimeError("preflight allocation is not exclusive")
    return {
        "job_id": job_id,
        "account": os.environ["SLURM_JOB_ACCOUNT"],
        "partition": os.environ["SLURM_JOB_PARTITION"],
        "exclusive_allocation_verified": True,
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
    }


def _tensor_digest_entries(
    entries: list[tuple[str, torch.Tensor]],
) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(entries, key=lambda item: item[0]):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _fixed_split_indices(
    examples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(FIXED_CONFIG["split_seed"])
    permutation = torch.randperm(examples, generator=generator)
    split = int(FIXED_CONFIG["train_fraction"] * examples)
    return permutation[:split], permutation[split:]


def _load_training_partition(
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
    """Load and transform training rows without constructing held-out tensors."""
    raw = loadmat(DATA_PATH)
    if "data" not in raw:
        raise KeyError(f"{DATA_PATH} does not contain a 'data' matrix")
    data = torch.as_tensor(raw["data"], dtype=torch.float32, device="cpu")
    train_index, test_index = _fixed_split_indices(data.shape[0])
    requested_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        train = data[train_index]
        X_train, y_train = train[:, :-1], train[:, -1]
        x_min = X_train.amin(dim=0)
        x_range = (X_train.amax(dim=0) - x_min).clamp_min(1e-8)
        y_mean = y_train.mean()
        y_std = y_train.std(unbiased=False).clamp_min(1e-8)
        X_train = ((X_train - x_min) / x_range).contiguous()
        y_train = ((y_train - y_mean) / y_std).contiguous()
    finally:
        torch.set_num_threads(requested_threads)

    metadata = {
        "train_examples": int(train_index.numel()),
        "test_examples": int(test_index.numel()),
        "features": int(X_train.shape[1]),
        "train_index_sha256": _tensor_digest_entries(
            [("train_index", train_index)]
        ),
        "test_index_sha256": _tensor_digest_entries(
            [("test_index", test_index)]
        ),
        "training_tensor_sha256": _tensor_digest_entries(
            [("X_train", X_train), ("y_train", y_train)]
        ),
    }
    if metadata != EXPECTED_DATA_PARTITION:
        raise RuntimeError(
            "fixed Protein split/preprocessing differs from predeclared manifest"
        )
    preprocessing = {
        "x_min": x_min,
        "x_range": x_range,
        "y_mean": y_mean,
        "y_std": y_std,
        "test_index": test_index,
    }
    return X_train, y_train, metadata, preprocessing


def _materialize_confirmation_test(
    preprocessing: dict[str, torch.Tensor],
    expected_metadata: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize held-out rows only after both confirmation trainings end."""
    raw = loadmat(DATA_PATH)
    data = torch.as_tensor(raw["data"], dtype=torch.float32, device="cpu")
    train_index, test_index = _fixed_split_indices(data.shape[0])
    if _tensor_digest_entries(
        [("train_index", train_index)]
    ) != expected_metadata["train_index_sha256"]:
        raise RuntimeError("training split changed before test materialization")
    if _tensor_digest_entries(
        [("test_index", test_index)]
    ) != expected_metadata["test_index_sha256"]:
        raise RuntimeError("held-out split changed before test materialization")
    if not torch.equal(test_index, preprocessing["test_index"]):
        raise RuntimeError("held-out indices differ from predeclared split")
    test = data[test_index]
    X_test = (
        (test[:, :-1] - preprocessing["x_min"])
        / preprocessing["x_range"]
    ).contiguous()
    y_test = (
        (test[:, -1] - preprocessing["y_mean"])
        / preprocessing["y_std"]
    ).contiguous()
    return X_test, y_test


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.linalg.norm(expected).clamp_min(
        torch.finfo(expected.dtype).tiny
    )
    return float((torch.linalg.norm(actual - expected) / denominator).item())


def _prepare_initial_state(
    X_train: torch.Tensor,
    seed: int,
    rank: int,
) -> tuple[dict[str, Any], list[torch.Tensor], dict[str, str]]:
    torch.manual_seed(seed)
    model = build_model(X_train.shape[1])
    d_param = model[0].weight.numel() + model[0].bias.numel()
    if rank > d_param:
        raise ValueError(f"rank {rank} exceeds parameter dimension {d_param}")

    U, _ = torch.linalg.qr(torch.randn(d_param, rank), mode="reduced")
    identity = torch.eye(rank, dtype=torch.float32)
    model_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    initial: dict[str, Any] = {
        "model_state": model_state,
        "U": U.contiguous(),
        "G": identity.clone().contiguous(),
        "L": identity.clone().contiguous(),
        "Aw": torch.zeros(rank, rank, dtype=torch.float32).contiguous(),
        "first_velocity": torch.zeros(d_param, dtype=torch.float32),
        "other_velocities": {
            layer_index: {
                "weight": torch.zeros_like(model[layer_index].weight),
                "bias": torch.zeros_like(model[layer_index].bias),
            }
            for layer_index in (2, 4, 6)
        },
        "identity": identity,
        "d_param": d_param,
    }
    permutations = []
    for epoch in range(FIXED_CONFIG["epochs"]):
        generator = torch.Generator().manual_seed(seed * 100003 + epoch)
        permutations.append(
            torch.randperm(X_train.shape[0], generator=generator)
        )

    initial_entries = [
        (f"model.{key}", value) for key, value in model_state.items()
    ]
    for key in ("U", "G", "L", "Aw", "first_velocity"):
        initial_entries.append((key, initial[key]))
    for layer_index, velocities in initial["other_velocities"].items():
        for name, value in velocities.items():
            initial_entries.append(
                (f"velocity.{layer_index}.{name}", value)
            )
    hashes = {
        "initial_state_sha256": _tensor_digest_entries(initial_entries),
        "permutations_sha256": _tensor_digest_entries(
            [
                (f"epoch.{epoch}", permutation)
                for epoch, permutation in enumerate(permutations)
            ]
        ),
    }
    return initial, permutations, hashes


def _clone_training_state(initial: dict[str, Any]) -> dict[str, Any]:
    model = build_model(FIXED_CONFIG["architecture"][0])
    model.load_state_dict(
        {key: value.clone() for key, value in initial["model_state"].items()}
    )
    return {
        "model": model,
        "U": initial["U"].clone(),
        "G": initial["G"].clone(),
        "L": initial["L"].clone(),
        "Aw": initial["Aw"].clone(),
        "first_velocity": initial["first_velocity"].clone(),
        "other_velocities": {
            layer_index: {
                name: value.clone() for name, value in velocities.items()
            }
            for layer_index, velocities in initial[
                "other_velocities"
            ].items()
        },
        "identity": initial["identity"].clone(),
    }


def _warm_up_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    initial: dict[str, Any],
) -> float:
    start = time.perf_counter()
    state = _clone_training_state(initial)
    model = state["model"]
    prediction = model(X_train[:32]).squeeze(-1)
    loss = torch.nn.functional.mse_loss(prediction, y_train[:32])
    model.zero_grad(set_to_none=True)
    loss.backward()
    _ = flatten_first_layer_gradient(model)
    return time.perf_counter() - start


def _kernel_equivalence_probe(
    rank: int,
    fast_cholesky: Any,
) -> dict[str, Any]:
    # This probe is outside the measured region.  It both warms the two core
    # backends and verifies that they return the same mathematical FP32
    # objects at the experiment's actual rank.
    coordinate = torch.arange(1, rank + 1, dtype=torch.float32)
    a = (0.01 * torch.sin(coordinate)).contiguous()
    identity = torch.eye(rank, dtype=torch.float32)
    L_update = identity.clone()
    Aw_update, vz_update = fast_cholesky.fast_incremental_update(
        L_update,
        torch.zeros_like(identity),
        a.clone(),
        a.clone(),
        metric_sign=1,
        operator_sign=1,
    )
    G_fresh = identity + torch.outer(a, a)
    L_fresh = torch.linalg.cholesky(G_fresh).contiguous()
    Aw_fresh = whiten(L_fresh, torch.outer(a, a))
    vz_fresh = torch.linalg.solve_triangular(
        L_fresh, a.unsqueeze(1), upper=False
    ).squeeze(1)
    errors = {
        "L_relative_error": _relative_error(L_update, L_fresh),
        "Aw_relative_error": _relative_error(Aw_update, Aw_fresh),
        "vz_relative_error": _relative_error(vz_update, vz_fresh),
    }
    tolerances = {
        "L_relative_error": 5e-6,
        "Aw_relative_error": 5e-5,
        "vz_relative_error": 5e-5,
    }
    for name, tolerance in tolerances.items():
        if not math.isfinite(errors[name]) or errors[name] > tolerance:
            raise RuntimeError(
                f"positive-kernel equivalence probe failed: {name}="
                f"{errors[name]:.6g} > {tolerance:.6g}"
            )
    return {
        "rank": rank,
        "dtype": str(L_update.dtype),
        "explicit_metric_sign": 1,
        "explicit_operator_sign": 1,
        "errors": errors,
        "predeclared_tolerances": tolerances,
        "passed": True,
    }


def _new_raw_samples() -> dict[str, list[Any]]:
    raw: dict[str, list[Any]] = {
        "step": [],
        "epoch": [],
        "batch_offset": [],
        "batch_mse": [],
        "unexplained_energy": [],
        "refreshed": [],
        "step_total_seconds": [],
        "unaccounted_seconds": [],
    }
    for name in TIMING_COMPONENTS:
        raw[name] = []
    return raw


def _run_method(
    method: str,
    phase: str,
    rank: int,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    initial: dict[str, Any],
    permutations: list[torch.Tensor],
    input_hashes: dict[str, str],
    fast_cholesky: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    clone_start = time.perf_counter()
    state = _clone_training_state(initial)
    clone_seconds = time.perf_counter() - clone_start
    model = state["model"]
    U = state["U"]
    G = state["G"]
    L = state["L"]
    Aw = state["Aw"]
    first_velocity = state["first_velocity"]
    other_velocities = state["other_velocities"]
    identity = state["identity"]

    raw = _new_raw_samples()
    reset_drift: list[float] = []
    epoch_summaries: list[dict[str, Any]] = []
    refresh_count = 0
    total_steps = 0
    training_start = time.perf_counter()

    for epoch, permutation in enumerate(permutations):
        epoch_start = time.perf_counter()
        epoch_loss_start = len(raw["batch_mse"])
        for offset in range(
            0,
            X_train.shape[0] - FIXED_CONFIG["batch_size"] + 1,
            FIXED_CONFIG["batch_size"],
        ):
            step_start = time.perf_counter()

            component_start = time.perf_counter()
            index = permutation[offset : offset + FIXED_CONFIG["batch_size"]]
            xb, yb = X_train[index], y_train[index]
            batch_fetch_seconds = time.perf_counter() - component_start

            component_start = time.perf_counter()
            prediction = model(xb).squeeze(-1)
            loss = torch.nn.functional.mse_loss(prediction, yb)
            model.zero_grad(set_to_none=True)
            loss.backward()
            g = flatten_first_layer_gradient(model)
            forward_backward_seconds = time.perf_counter() - component_start

            component_start = time.perf_counter()
            a = U.mT @ g
            residual = g - U @ a
            g_energy = float(torch.dot(g, g).item())
            residual_energy = float(torch.dot(residual, residual).item())
            unexplained = residual_energy / max(g_energy, 1e-30)
            refreshed = (
                unexplained > FIXED_CONFIG["refresh_threshold"]
                and residual_energy > 1e-24
            )
            projection_decision_seconds = (
                time.perf_counter() - component_start
            )

            refresh_seconds = 0.0
            nonrefresh_core_seconds = 0.0
            if refreshed:
                component_start = time.perf_counter()
                rho = residual_energy**0.5
                U_bar = torch.cat(
                    (U, (residual / rho).unsqueeze(1)), dim=1
                )
                c_bar = torch.cat(
                    (a, torch.tensor([rho], dtype=torch.float32))
                )
                fisher_bar = torch.zeros(
                    rank + 1, rank + 1, dtype=torch.float32
                )
                fisher_bar[:rank, :rank] = (
                    G - FIXED_CONFIG["damping"] * identity
                )
                fisher_bar.add_(torch.outer(c_bar, c_bar))
                eigenvalues, eigenvectors = torch.linalg.eigh(fisher_bar)
                top = torch.arange(rank, 0, -1)
                rotation = eigenvectors[:, top]
                fisher = torch.diag(
                    eigenvalues[top].clamp_min(0.0)
                ).contiguous()
                U = (U_bar @ rotation).contiguous()
                G = (
                    FIXED_CONFIG["damping"] * identity + fisher
                ).contiguous()
                L = torch.linalg.cholesky(G).contiguous()
                Aw = whiten(L, fisher)
                # Refresh rotates coordinates, so projection and both solves
                # are part of the complete refresh path.
                a = U.mT @ g
                residual = g - U @ a
                y_solve = torch.linalg.solve_triangular(
                    L, a.unsqueeze(1), upper=False
                ).squeeze(1)
                x = torch.linalg.solve_triangular(
                    L.mT, y_solve.unsqueeze(1), upper=True
                ).squeeze(1)
                refresh_seconds = time.perf_counter() - component_start
                refresh_count += 1
            else:
                component_start = time.perf_counter()
                if method == "ours":
                    Aw, vz = fast_cholesky.fast_incremental_update(
                        L,
                        Aw,
                        a.clone(),
                        a.clone(),
                        metric_sign=1,
                        operator_sign=1,
                    )
                    G.add_(torch.outer(a, a))
                    x = torch.linalg.solve_triangular(
                        L.mT, vz.unsqueeze(1), upper=True
                    ).squeeze(1)
                else:
                    G.add_(torch.outer(a, a))
                    L = torch.linalg.cholesky(G).contiguous()
                    fisher = G - FIXED_CONFIG["damping"] * identity
                    Aw = whiten(L, fisher)
                    y_solve = torch.linalg.solve_triangular(
                        L, a.unsqueeze(1), upper=False
                    ).squeeze(1)
                    x = torch.linalg.solve_triangular(
                        L.mT, y_solve.unsqueeze(1), upper=True
                    ).squeeze(1)
                nonrefresh_core_seconds = (
                    time.perf_counter() - component_start
                )

            total_steps += 1
            reset_seconds = 0.0
            if (
                method == "ours"
                and FIXED_CONFIG["refactor_every"] > 0
                and total_steps % FIXED_CONFIG["refactor_every"] == 0
            ):
                component_start = time.perf_counter()
                L_exact = torch.linalg.cholesky(G).contiguous()
                Aw_exact = whiten(
                    L_exact, G - FIXED_CONFIG["damping"] * identity
                )
                reset_drift.append(_relative_error(Aw, Aw_exact))
                L = L_exact
                Aw = Aw_exact
                reset_seconds = time.perf_counter() - component_start

            component_start = time.perf_counter()
            direction = residual / FIXED_CONFIG["damping"] + U @ x
            direction_seconds = time.perf_counter() - component_start

            component_start = time.perf_counter()
            with torch.no_grad():
                first_velocity.mul_(FIXED_CONFIG["momentum"]).add_(direction)
                apply_first_layer_step(
                    model,
                    first_velocity,
                    FIXED_CONFIG["learning_rate"],
                )
                for layer_index in (2, 4, 6):
                    layer = model[layer_index]
                    velocity = other_velocities[layer_index]
                    velocity["weight"].mul_(
                        FIXED_CONFIG["momentum"]
                    ).add_(layer.weight.grad)
                    velocity["bias"].mul_(
                        FIXED_CONFIG["momentum"]
                    ).add_(layer.bias.grad)
                    layer.weight.add_(
                        velocity["weight"],
                        alpha=-FIXED_CONFIG["learning_rate"],
                    )
                    layer.bias.add_(
                        velocity["bias"],
                        alpha=-FIXED_CONFIG["learning_rate"],
                    )
            parameter_update_seconds = time.perf_counter() - component_start
            step_total_seconds = time.perf_counter() - step_start

            components = {
                "batch_fetch_seconds": batch_fetch_seconds,
                "forward_backward_seconds": forward_backward_seconds,
                "projection_decision_seconds": projection_decision_seconds,
                "refresh_seconds": refresh_seconds,
                "nonrefresh_core_seconds": nonrefresh_core_seconds,
                "reset_seconds": reset_seconds,
                "direction_seconds": direction_seconds,
                "parameter_update_seconds": parameter_update_seconds,
            }
            raw["step"].append(total_steps)
            raw["epoch"].append(epoch + 1)
            raw["batch_offset"].append(offset)
            raw["batch_mse"].append(float(loss.detach().item()))
            raw["unexplained_energy"].append(unexplained)
            raw["refreshed"].append(bool(refreshed))
            raw["step_total_seconds"].append(step_total_seconds)
            for name, value in components.items():
                raw[name].append(value)
            raw["unaccounted_seconds"].append(
                step_total_seconds - sum(components.values())
            )

        epoch_losses = raw["batch_mse"][epoch_loss_start:]
        epoch_summaries.append(
            {
                "epoch": epoch + 1,
                "steps": total_steps,
                "training_wall_seconds": time.perf_counter()
                - training_start,
                "epoch_seconds": time.perf_counter() - epoch_start,
                "mean_training_batch_mse": statistics.mean(epoch_losses),
                "refreshes": refresh_count,
            }
        )
        print(
            f"phase={phase} method={method} rank={rank} "
            f"epoch={epoch + 1}/{FIXED_CONFIG['epochs']} "
            f"steps={total_steps} refreshes={refresh_count}",
            flush=True,
        )

    training_wall_seconds = time.perf_counter() - training_start

    train_eval_start = time.perf_counter()
    with torch.no_grad():
        train_prediction = model(X_train).squeeze(-1)
        final_train_mse = float(
            torch.nn.functional.mse_loss(
                train_prediction, y_train
            ).item()
        )
    train_evaluation_seconds = time.perf_counter() - train_eval_start

    if not math.isfinite(final_train_mse):
        raise RuntimeError(f"{method} produced non-finite final_train_mse")

    model_entries = [
        (f"model.{key}", value)
        for key, value in model.state_dict().items()
    ]
    final_model_vector = torch.cat(
        [value.detach().reshape(-1) for _, value in model_entries]
    )
    component_totals = {
        name: math.fsum(raw[name]) for name in TIMING_COMPONENTS
    }
    method_payload = {
        "method": method,
        "dtype": "torch.float32",
        "initial_state_sha256": input_hashes["initial_state_sha256"],
        "permutations_sha256": input_hashes["permutations_sha256"],
        "steps": total_steps,
        "epochs": FIXED_CONFIG["epochs"],
        "refresh_count": refresh_count,
        "refresh_fraction": refresh_count / total_steps,
        "final_train_mse": final_train_mse,
        "final_test_mse": None,
        "test_evaluations": 0,
        "timing": {
            "state_clone_seconds": clone_seconds,
            "training_wall_seconds": training_wall_seconds,
            "sum_step_seconds": math.fsum(raw["step_total_seconds"]),
            "application_step_seconds": math.fsum(
                raw["step_total_seconds"]
            ),
            "component_totals": component_totals,
            "sum_unaccounted_seconds": math.fsum(
                raw["unaccounted_seconds"]
            ),
            "train_evaluation_seconds": train_evaluation_seconds,
            "test_evaluation_seconds": 0.0,
        },
        "reset_drift_values": reset_drift,
        "max_reset_drift": max(reset_drift, default=None),
        "mean_reset_drift": (
            statistics.mean(reset_drift) if reset_drift else None
        ),
        "final_state": {
            "model_sha256": _tensor_digest_entries(model_entries),
            "U_sha256": _tensor_digest_entries([("U", U)]),
            "G_sha256": _tensor_digest_entries([("G", G)]),
            "L_sha256": _tensor_digest_entries([("L", L)]),
            "Aw_sha256": _tensor_digest_entries([("Aw", Aw)]),
        },
        "epoch_training_only": epoch_summaries,
        "raw_step_samples": raw,
    }
    private = {
        "model_vector": final_model_vector,
        "model": model,
        "train_prediction": train_prediction,
        "test_prediction": None,
        "refresh_trace": raw["refreshed"],
        "unexplained_trace": raw["unexplained_energy"],
    }
    return method_payload, private


def _evaluate_confirmation_test_once(
    methods: dict[str, dict[str, Any]],
    private: dict[str, dict[str, Any]],
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> None:
    """Perform the predeclared terminal test pass exactly once per method."""
    for method in METHODS:
        if methods[method]["test_evaluations"] != 0:
            raise RuntimeError(f"{method} test set was accessed before final pass")
        start = time.perf_counter()
        with torch.no_grad():
            prediction = private[method]["model"](X_test).squeeze(-1)
            mse = float(
                torch.nn.functional.mse_loss(prediction, y_test).item()
            )
        if not math.isfinite(mse):
            raise RuntimeError(f"{method} produced non-finite final_test_mse")
        methods[method]["final_test_mse"] = mse
        methods[method]["test_evaluations"] = 1
        methods[method]["timing"]["test_evaluation_seconds"] = (
            time.perf_counter() - start
        )
        private[method]["test_prediction"] = prediction


def _paired_diagnostics(
    methods: dict[str, dict[str, Any]],
    private: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ours = private["ours"]
    refact = private["refact"]
    refresh_mismatches = sum(
        left != right
        for left, right in zip(
            ours["refresh_trace"], refact["refresh_trace"], strict=True
        )
    )
    unexplained_differences = [
        abs(left - right)
        for left, right in zip(
            ours["unexplained_trace"],
            refact["unexplained_trace"],
            strict=True,
        )
    ]
    test_relative_error = None
    if ours["test_prediction"] is not None:
        test_relative_error = _relative_error(
            ours["test_prediction"], refact["test_prediction"]
        )
    return {
        "same_initial_state_hash": (
            methods["ours"]["initial_state_sha256"]
            == methods["refact"]["initial_state_sha256"]
        ),
        "same_permutation_hash": (
            methods["ours"]["permutations_sha256"]
            == methods["refact"]["permutations_sha256"]
        ),
        "same_step_count": (
            methods["ours"]["steps"] == methods["refact"]["steps"]
        ),
        "same_declared_precision": (
            methods["ours"]["dtype"] == methods["refact"]["dtype"]
            == "torch.float32"
        ),
        "refresh_decision_mismatches": refresh_mismatches,
        "refresh_decision_mismatch_fraction": (
            refresh_mismatches / methods["ours"]["steps"]
        ),
        "max_absolute_unexplained_energy_difference": max(
            unexplained_differences, default=0.0
        ),
        "final_model_relative_error_ours_vs_refact": _relative_error(
            ours["model_vector"], refact["model_vector"]
        ),
        "final_train_prediction_relative_error_ours_vs_refact": (
            _relative_error(
                ours["train_prediction"], refact["train_prediction"]
            )
        ),
        "final_test_prediction_relative_error_ours_vs_refact": (
            test_relative_error
        ),
        "final_train_mse_difference_ours_minus_refact": (
            methods["ours"]["final_train_mse"]
            - methods["refact"]["final_train_mse"]
        ),
        "final_test_mse_difference_ours_minus_refact": (
            None
            if methods["ours"]["final_test_mse"] is None
            else methods["ours"]["final_test_mse"]
            - methods["refact"]["final_test_mse"]
        ),
        "interpretation": (
            "The two backends implement the same algebra in FP32, but a "
            "long nonlinear training trajectory need not be bit-identical. "
            "All numerical divergences are reported rather than filtered."
        ),
    }


def _require_pilot_gate(
    directory: Path,
    lock_sha256: str,
) -> dict[str, Any]:
    gate_path = directory / "pilot_approved.json"
    if not gate_path.is_file():
        raise RuntimeError(
            "confirmation is gated: audit pilots and create an explicit "
            "pilot approval before submitting confirmation"
        )
    gate = _read_json(gate_path)
    if gate.get("source_lock_sha256") != lock_sha256:
        raise RuntimeError("pilot gate refers to a different source lock")
    summary_path = directory / "pilot_summary.json"
    if gate.get("pilot_summary_sha256") != _sha256(summary_path):
        raise RuntimeError("pilot summary changed after approval")
    if gate.get("decision") != "approve_untouched_confirmation":
        raise RuntimeError("pilot gate does not approve confirmation")
    return gate


def _claim_confirmation_attempt(
    directory: Path,
    spec: dict[str, Any],
    lock_sha256: str,
    run_git_commit: str,
    slurm: dict[str, Any],
) -> dict[str, Any]:
    """Permanently prevent a confirmation pair from touching test twice."""
    claim = {
        "schema_version": 1,
        "experiment": "e1_confirmation_exactly_once_claim",
        "campaign_tag": DEFAULT_CAMPAIGN,
        "created_utc": _utc_now(),
        "spec": spec,
        "source_lock_sha256": lock_sha256,
        "run_git_commit": run_git_commit,
        "slurm": slurm,
        "policy": (
            "This task index may never be retried in this campaign. If it "
            "fails before publishing its result, retire the campaign tag "
            "rather than risk a second held-out evaluation."
        ),
    }
    path = (
        directory
        / "confirmation_attempt_claims"
        / f"{_artifact_path(directory, spec).stem}.claim.json"
    )
    try:
        _write_once_json(claim, path)
    except FileExistsError as error:
        raise RuntimeError(
            f"confirmation task already claimed and may not be retried: {path}"
        ) from error
    return {"path": str(path), "sha256": _sha256(path), "payload": claim}


def run_pair(args: argparse.Namespace) -> dict[str, Any]:
    if args.task_index is None:
        raise ValueError("--task-index is required for run-pair")
    spec = _task_spec(args.phase, args.task_index)
    lock, lock_sha256, run_git_commit = _verify_lock(args)
    slurm = _validate_slurm_task(spec)
    thread_report = configure_and_verify_threads(spec["threads"])
    torch.use_deterministic_algorithms(True)

    directory = _campaign_dir(args)
    gate = None
    confirmation_claim = None
    if args.phase == "confirmation":
        gate = _require_pilot_gate(directory, lock_sha256)
        confirmation_claim = _claim_confirmation_attempt(
            directory, spec, lock_sha256, run_git_commit, slurm
        )

    import fast_cholesky

    extension_path = Path(fast_cholesky.__file__).resolve()
    expected_extension = (
        ROOT / LOCKED_FILES["incremental_binary"]
    ).resolve()
    if extension_path != expected_extension:
        raise RuntimeError(
            f"loaded {extension_path}, expected locked extension "
            f"{expected_extension}"
        )

    pair_start = time.perf_counter()
    data_start = time.perf_counter()
    X_train, y_train, data_metadata, preprocessing = (
        _load_training_partition()
    )
    data_load_seconds = time.perf_counter() - data_start
    if X_train.shape[1] != FIXED_CONFIG["architecture"][0]:
        raise RuntimeError(
            f"unexpected Protein feature count {X_train.shape[1]}"
        )

    setup_start = time.perf_counter()
    initial, permutations, input_hashes = _prepare_initial_state(
        X_train, spec["seed"], spec["rank"]
    )
    setup_seconds = time.perf_counter() - setup_start
    kernel_probe = _kernel_equivalence_probe(
        spec["rank"], fast_cholesky
    )
    model_warmup_seconds = _warm_up_model(X_train, y_train, initial)

    methods: dict[str, dict[str, Any]] = {}
    private: dict[str, dict[str, Any]] = {}
    for method in spec["method_order"]:
        methods[method], private[method] = _run_method(
            method,
            spec["phase"],
            spec["rank"],
            X_train,
            y_train,
            initial,
            permutations,
            input_hashes,
            fast_cholesky,
        )

    pre_test_paired = _paired_diagnostics(methods, private)
    for required in (
        "same_initial_state_hash",
        "same_permutation_hash",
        "same_step_count",
        "same_declared_precision",
    ):
        if not pre_test_paired[required]:
            raise RuntimeError(f"paired fairness invariant failed: {required}")
    test_materialization_seconds = 0.0
    if spec["phase"] == "confirmation":
        test_materialization_start = time.perf_counter()
        X_test, y_test = _materialize_confirmation_test(
            preprocessing, data_metadata
        )
        test_materialization_seconds = (
            time.perf_counter() - test_materialization_start
        )
        _evaluate_confirmation_test_once(
            methods, private, X_test, y_test
        )
    paired = _paired_diagnostics(methods, private)

    payload = {
        "schema_version": 1,
        "experiment": "e1_paired_confirmation",
        "campaign_tag": args.campaign_tag,
        "created_utc": _utc_now(),
        "spec": spec,
        "fixed_config": FIXED_CONFIG,
        "data": {
            "path": str(DATA_PATH.resolve()),
            "sha256": lock["files"]["protein_data"]["sha256"],
            **data_metadata,
            "normalization": "fit on fixed training partition only",
            "split_seed": FIXED_CONFIG["split_seed"],
            "heldout_tensors_materialized": (
                spec["phase"] == "confirmation"
            ),
        },
        "paired_inputs": {
            **input_hashes,
            "identical_data_object": True,
            "identical_model_and_curvature_clones": True,
            "identical_minibatch_permutations": True,
        },
        "source_lock": {
            "path": str(_lock_path(args)),
            "sha256": lock_sha256,
            "locked_git_commit": lock["git_commit"],
            "run_git_commit": run_git_commit,
            "files": lock["files"],
            "runtime": lock["runtime"],
        },
        "implementation": {
            "incremental_extension": str(extension_path),
            "incremental_dispatch": (
                "explicit metric_sign=1, operator_sign=1"
            ),
            "incremental_kernel": (
                "row-major virtual-padding contiguous SIMD positive path"
            ),
            "refact_backend": (
                "fresh Cholesky, complete rewhitening, and two triangular "
                "solves on every non-refresh step"
            ),
            "kernel_equivalence_probe": kernel_probe,
        },
        "methods": methods,
        "paired_diagnostics": paired,
        "test_access": {
            "per_method_evaluations": (
                0 if spec["phase"] == "pilot" else 1
            ),
            "pair_total_evaluations": (
                0 if spec["phase"] == "pilot" else 2
            ),
            "selection_using_test_values": False,
            "terminal_after_both_methods_trained": True,
        },
        "timing_outside_measured_training": {
            "data_load_seconds": data_load_seconds,
            "paired_initialization_seconds": setup_seconds,
            "model_warmup_seconds": model_warmup_seconds,
            "test_materialization_seconds": test_materialization_seconds,
            "pair_total_seconds": time.perf_counter() - pair_start,
        },
        "pilot_gate": gate,
        "confirmation_exactly_once_claim": confirmation_claim,
        "environment": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_config": torch.__config__.show(),
            "runtime_lock": lock["runtime"],
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "thread_verification": thread_report,
            "slurm": slurm,
        },
    }
    output = _artifact_path(directory, spec)
    _write_once_json(payload, output)
    print(f"wrote immutable paired result {output}", flush=True)
    return payload


def _validate_artifact(
    payload: dict[str, Any],
    expected_spec: dict[str, Any],
    lock_sha256: str,
) -> None:
    if payload.get("schema_version") != 1:
        raise RuntimeError("unexpected artifact schema")
    if payload.get("campaign_tag") != DEFAULT_CAMPAIGN:
        raise RuntimeError("artifact campaign mismatch")
    if payload.get("spec") != expected_spec:
        raise RuntimeError(f"artifact spec mismatch for {expected_spec}")
    if payload.get("fixed_config") != FIXED_CONFIG:
        raise RuntimeError("artifact fixed configuration mismatch")
    if payload["source_lock"]["sha256"] != lock_sha256:
        raise RuntimeError("artifact source lock mismatch")
    if payload["source_lock"]["files"] is None:
        raise RuntimeError("artifact omitted source and binary hashes")
    if payload["source_lock"].get("runtime") is None:
        raise RuntimeError("artifact omitted locked runtime manifest")
    expected_test_evaluations = (
        0 if expected_spec["phase"] == "pilot" else 1
    )
    steps = (
        payload["data"]["train_examples"]
        // FIXED_CONFIG["batch_size"]
        * FIXED_CONFIG["epochs"]
    )
    for method in METHODS:
        result = payload["methods"][method]
        if result["steps"] != steps:
            raise RuntimeError(f"{method} has wrong step count")
        if result["dtype"] != "torch.float32":
            raise RuntimeError(f"{method} has wrong precision")
        if result["test_evaluations"] != expected_test_evaluations:
            raise RuntimeError(f"{method} has wrong test access count")
        if len(result["raw_step_samples"]["step"]) != steps:
            raise RuntimeError(f"{method} omitted raw step samples")
        for name in TIMING_COMPONENTS:
            if len(result["raw_step_samples"][name]) != steps:
                raise RuntimeError(
                    f"{method} omitted raw timing component {name}"
                )
        if expected_spec["phase"] == "pilot":
            if result["final_test_mse"] is not None:
                raise RuntimeError("pilot artifact contains a test MSE")
        elif result["final_test_mse"] is None:
            raise RuntimeError("confirmation artifact omitted final test MSE")
    if expected_spec["phase"] == "pilot":
        if payload["data"]["heldout_tensors_materialized"]:
            raise RuntimeError("pilot materialized held-out tensors")
        if payload["confirmation_exactly_once_claim"] is not None:
            raise RuntimeError("pilot unexpectedly contains confirmation claim")
    elif payload["confirmation_exactly_once_claim"] is None:
        raise RuntimeError("confirmation omitted exactly-once attempt claim")
    paired = payload["paired_diagnostics"]
    for invariant in (
        "same_initial_state_hash",
        "same_permutation_hash",
        "same_step_count",
        "same_declared_precision",
    ):
        if not paired[invariant]:
            raise RuntimeError(f"paired invariant is false: {invariant}")
    runtime = payload["environment"]["thread_verification"][
        "runtime_values"
    ]
    threads = expected_spec["threads"]
    if not (
        runtime["aten_intraop"]
        == runtime["openmp_max"]
        == runtime["mkl_max"]
        == threads
        and runtime["aten_interop"] == 1
    ):
        raise RuntimeError("artifact thread runtime verification failed")
    affinity = payload["environment"]["thread_verification"][
        "affinity_after"
    ]
    if len(affinity) != threads:
        raise RuntimeError("artifact CPU affinity width is wrong")
    if (
        payload["environment"]["thread_verification"][
            "unique_physical_cores"
        ]
        != threads
    ):
        raise RuntimeError("artifact physical-core pinning is wrong")
    if not payload["implementation"]["kernel_equivalence_probe"]["passed"]:
        raise RuntimeError("artifact kernel equivalence probe failed")


def _audit_phase(
    args: argparse.Namespace,
    phase: str,
) -> tuple[list[dict[str, Any]], str]:
    _, lock_sha256, _ = _verify_lock(args)
    directory = _campaign_dir(args)
    payloads = []
    for spec in _task_specs(phase):
        path = _artifact_path(directory, spec)
        if not path.is_file():
            raise FileNotFoundError(f"missing expected {phase} artifact {path}")
        payload = _read_json(path)
        _validate_artifact(payload, spec, lock_sha256)
        payloads.append(payload)
    return payloads, lock_sha256


def _mean_sd(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "values": values,
    }


def _summarize_group(
    payloads: list[dict[str, Any]],
    rank: int,
    threads: int,
) -> dict[str, Any]:
    group = sorted(
        (
            payload
            for payload in payloads
            if payload["spec"]["rank"] == rank
            and payload["spec"]["threads"] == threads
        ),
        key=lambda payload: payload["spec"]["seed"],
    )
    if len(group) != len({item["spec"]["seed"] for item in group}):
        raise RuntimeError("duplicate seeds in summary group")
    method_summary: dict[str, Any] = {}
    for method in METHODS:
        method_summary[method] = {
            "training_wall_seconds": _mean_sd(
                [
                    item["methods"][method]["timing"][
                        "training_wall_seconds"
                    ]
                    for item in group
                ]
            ),
            "sum_step_seconds": _mean_sd(
                [
                    item["methods"][method]["timing"]["sum_step_seconds"]
                    for item in group
                ]
            ),
            "nonrefresh_core_seconds": _mean_sd(
                [
                    item["methods"][method]["timing"][
                        "component_totals"
                    ]["nonrefresh_core_seconds"]
                    for item in group
                ]
            ),
            "refresh_seconds": _mean_sd(
                [
                    item["methods"][method]["timing"][
                        "component_totals"
                    ]["refresh_seconds"]
                    for item in group
                ]
            ),
            "reset_seconds": _mean_sd(
                [
                    item["methods"][method]["timing"][
                        "component_totals"
                    ]["reset_seconds"]
                    for item in group
                ]
            ),
            "refresh_fraction": _mean_sd(
                [item["methods"][method]["refresh_fraction"] for item in group]
            ),
            "final_train_mse": _mean_sd(
                [item["methods"][method]["final_train_mse"] for item in group]
            ),
            "final_test_mse": (
                None
                if group[0]["spec"]["phase"] == "pilot"
                else _mean_sd(
                    [
                        item["methods"][method]["final_test_mse"]
                        for item in group
                    ]
                )
            ),
        }
    application_speedups = [
        item["methods"]["refact"]["timing"]["application_step_seconds"]
        / item["methods"]["ours"]["timing"]["application_step_seconds"]
        for item in group
    ]
    harness_wall_speedups = [
        item["methods"]["refact"]["timing"]["training_wall_seconds"]
        / item["methods"]["ours"]["timing"]["training_wall_seconds"]
        for item in group
    ]
    core_speedups = [
        item["methods"]["refact"]["timing"]["component_totals"][
            "nonrefresh_core_seconds"
        ]
        / item["methods"]["ours"]["timing"]["component_totals"][
            "nonrefresh_core_seconds"
        ]
        for item in group
    ]
    return {
        "rank": rank,
        "threads": threads,
        "seeds": [item["spec"]["seed"] for item in group],
        "method_order_counts": {
            "ours_first": sum(
                item["spec"]["method_order"][0] == "ours" for item in group
            ),
            "refact_first": sum(
                item["spec"]["method_order"][0] == "refact" for item in group
            ),
        },
        "methods": method_summary,
        "paired_application_speedup_refact_over_ours": _mean_sd(
            application_speedups
        ),
        "paired_instrumented_wall_speedup_refact_over_ours": _mean_sd(
            harness_wall_speedups
        ),
        "paired_nonrefresh_core_speedup_refact_over_ours": _mean_sd(
            core_speedups
        ),
        "paired_test_mse_difference_ours_minus_refact": (
            None
            if group[0]["spec"]["phase"] == "pilot"
            else _mean_sd(
                [
                    item["paired_diagnostics"][
                        "final_test_mse_difference_ours_minus_refact"
                    ]
                    for item in group
                ]
            )
        ),
        "refresh_decision_mismatch_count": _mean_sd(
            [
                item["paired_diagnostics"]["refresh_decision_mismatches"]
                for item in group
            ]
        ),
        "individual_pairs": [
            {
                "seed": item["spec"]["seed"],
                "method_order": item["spec"]["method_order"],
                "ours_training_wall_seconds": item["methods"]["ours"][
                    "timing"
                ]["training_wall_seconds"],
                "refact_training_wall_seconds": item["methods"]["refact"][
                    "timing"
                ]["training_wall_seconds"],
                "application_speedup_refact_over_ours": (
                    item["methods"]["refact"]["timing"][
                        "application_step_seconds"
                    ]
                    / item["methods"]["ours"]["timing"][
                        "application_step_seconds"
                    ]
                ),
                "ours_test_mse": item["methods"]["ours"]["final_test_mse"],
                "refact_test_mse": item["methods"]["refact"][
                    "final_test_mse"
                ],
                "refresh_decision_mismatches": item[
                    "paired_diagnostics"
                ]["refresh_decision_mismatches"],
            }
            for item in group
        ],
    }


def summarize_phase(args: argparse.Namespace, phase: str) -> dict[str, Any]:
    payloads, lock_sha256 = _audit_phase(args, phase)
    groups = [
        _summarize_group(payloads, rank, threads)
        for rank in RANKS
        for threads in THREAD_COUNTS
    ]
    output_name = (
        "pilot_summary.json"
        if phase == "pilot"
        else "confirmation_summary.json"
    )
    payload = {
        "schema_version": 1,
        "experiment": f"e1_paired_{phase}_summary",
        "campaign_tag": args.campaign_tag,
        "created_utc": _utc_now(),
        "source_lock_sha256": lock_sha256,
        "fixed_config": FIXED_CONFIG,
        "artifact_count": len(payloads),
        "groups": groups,
        "audit": {
            "all_expected_artifacts_present": True,
            "all_write_once_artifacts_source_locked": True,
            "all_runtime_thread_counts_verified": True,
            "all_affinities_exact": True,
            "all_paired_input_hashes_equal": True,
            "pilot_test_values_absent": phase != "pilot"
            or all(
                item["methods"][method]["final_test_mse"] is None
                for item in payloads
                for method in METHODS
            ),
            "confirmation_test_evaluations_exactly_once_per_method": (
                phase != "confirmation"
                or all(
                    item["methods"][method]["test_evaluations"] == 1
                    for item in payloads
                    for method in METHODS
                )
            ),
        },
    }
    output = _campaign_dir(args) / output_name
    _write_once_json(payload, output)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"wrote immutable summary {output}", flush=True)
    return payload


def approve_pilot(args: argparse.Namespace) -> dict[str, Any]:
    if not args.reviewer or not args.note:
        raise ValueError(
            "--reviewer and --note are required for explicit pilot approval"
        )
    _, lock_sha256, run_git_commit = _verify_lock(args)
    directory = _campaign_dir(args)
    summary_path = directory / "pilot_summary.json"
    summary = _read_json(summary_path)
    if summary.get("source_lock_sha256") != lock_sha256:
        raise RuntimeError("pilot summary refers to a different source lock")
    if summary.get("artifact_count") != len(_task_specs("pilot")):
        raise RuntimeError("pilot summary is incomplete")
    payload = {
        "schema_version": 1,
        "experiment": "e1_pilot_review_gate",
        "campaign_tag": args.campaign_tag,
        "created_utc": _utc_now(),
        "decision": "approve_untouched_confirmation",
        "reviewer": args.reviewer,
        "review_note": args.note,
        "source_lock_sha256": lock_sha256,
        "approval_git_commit": run_git_commit,
        "pilot_summary_path": str(summary_path),
        "pilot_summary_sha256": _sha256(summary_path),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "confirmation_configuration_changed_after_pilot": False,
    }
    output = directory / "pilot_approved.json"
    _write_once_json(payload, output)
    print(f"wrote explicit pilot gate {output}", flush=True)
    return payload


def self_test() -> None:
    pilot = _task_specs("pilot")
    confirmation = _task_specs("confirmation")
    if len(pilot) != 18 or len(confirmation) != 60:
        raise AssertionError("unexpected task count")
    if set(PILOT_SEEDS) & set(CONFIRMATION_SEEDS):
        raise AssertionError("pilot and confirmation seeds overlap")
    for specs, phase in (
        (pilot, "pilot"),
        (confirmation, "confirmation"),
    ):
        if [spec["task_index"] for spec in specs] != list(range(len(specs))):
            raise AssertionError("task indices are not contiguous")
        for spec in specs:
            if spec["phase"] != phase:
                raise AssertionError("phase mapping failed")
            if spec["threads"] != THREAD_COUNTS[spec["task_index"] % 2]:
                raise AssertionError("thread count is not shell-decodable")
    for rank in RANKS:
        for threads in THREAD_COUNTS:
            group = [
                spec
                for spec in confirmation
                if spec["rank"] == rank and spec["threads"] == threads
            ]
            counts = {
                method: sum(spec["method_order"][0] == method for spec in group)
                for method in METHODS
            }
            if counts != {"ours": 5, "refact": 5}:
                raise AssertionError("confirmation order is not balanced")
    train_a, test_a = _fixed_split_indices(100)
    train_b, test_b = _fixed_split_indices(100)
    if not torch.equal(train_a, train_b) or not torch.equal(test_a, test_b):
        raise AssertionError("fixed split is not deterministic")
    if set(train_a.tolist()) & set(test_a.tolist()):
        raise AssertionError("train/test split overlaps")
    for relative in (
        LOCKED_FILES["preflight_sbatch"],
        LOCKED_FILES["pilot_sbatch"],
        LOCKED_FILES["confirmation_sbatch"],
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        required = (
            "#SBATCH --account=marlowe-m000151-pm06",
            "#SBATCH --partition=batch",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=10",
            "#SBATCH --exclusive",
            "#SBATCH --hint=nomultithread",
            "#SBATCH --no-requeue",
            "PYTHONNOUSERSITE=1",
        )
        for line in required:
            if line not in text:
                raise AssertionError(f"{relative} omitted {line!r}")
        if "#SBATCH --requeue" in text:
            raise AssertionError(f"{relative} enables requeue")
    confirmation_text = (
        ROOT / LOCKED_FILES["confirmation_sbatch"]
    ).read_text(encoding="utf-8")
    if "--action approve-pilot" in confirmation_text:
        raise AssertionError("confirmation submission auto-approves pilots")
    print("E1 confirmation self-test passed", flush=True)


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    self_test()
    lock, lock_sha256, run_git_commit = _verify_lock(args)
    slurm = _validate_preflight_slurm()
    thread_report = configure_and_verify_threads(10)
    torch.use_deterministic_algorithms(True)

    directory = _campaign_dir(args)
    existing_outcomes = [
        *directory.glob("pilot_pair_*.json"),
        *directory.glob("confirmation_pair_*.json"),
        *directory.glob("confirmation_attempt_claims/*.json"),
    ]
    if existing_outcomes:
        raise RuntimeError(
            "preflight found outcomes or claims in the new campaign: "
            + ", ".join(str(path) for path in existing_outcomes)
        )

    import fast_cholesky

    extension_path = Path(fast_cholesky.__file__).resolve()
    expected_extension = (
        ROOT / LOCKED_FILES["incremental_binary"]
    ).resolve()
    if extension_path != expected_extension:
        raise RuntimeError(
            f"loaded {extension_path}, expected {expected_extension}"
        )
    X_train, y_train, data_metadata, _ = _load_training_partition()
    if data_metadata["features"] != FIXED_CONFIG["architecture"][0]:
        raise RuntimeError("Protein feature count failed preflight")
    if not torch.isfinite(X_train).all() or not torch.isfinite(y_train).all():
        raise RuntimeError("training partition contains non-finite values")
    kernel_probes = [
        _kernel_equivalence_probe(rank, fast_cholesky)
        for rank in RANKS
    ]
    payload = {
        "schema_version": 1,
        "experiment": "e1_confirmation_preflight",
        "campaign_tag": args.campaign_tag,
        "created_utc": _utc_now(),
        "source_lock_sha256": lock_sha256,
        "locked_git_commit": lock["git_commit"],
        "run_git_commit": run_git_commit,
        "runtime": lock["runtime"],
        "slurm": slurm,
        "thread_verification": thread_report,
        "data": data_metadata,
        "kernel_probes": kernel_probes,
        "pilot_outcomes_present_before_launch": False,
        "confirmation_outcomes_present_before_launch": False,
    }
    output = directory / "preflight_receipt.json"
    _write_once_json(payload, output)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"wrote immutable preflight receipt {output}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        required=True,
        choices=(
            "plan",
            "self-test",
            "freeze-lock",
            "preflight",
            "run-pair",
            "summarize-pilot",
            "approve-pilot",
            "aggregate-confirmation",
        ),
    )
    parser.add_argument(
        "--phase",
        choices=("pilot", "confirmation"),
        default="pilot",
    )
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--campaign-tag", default=DEFAULT_CAMPAIGN)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--lock-file")
    parser.add_argument("--reviewer")
    parser.add_argument("--note")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "plan":
        print(
            json.dumps(
                {
                    "fixed_config": FIXED_CONFIG,
                    "tasks": _task_specs(args.phase),
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.action == "self-test":
        self_test()
    elif args.action == "freeze-lock":
        freeze_lock(args)
    elif args.action == "preflight":
        preflight(args)
    elif args.action == "run-pair":
        run_pair(args)
    elif args.action == "summarize-pilot":
        summarize_phase(args, "pilot")
    elif args.action == "approve-pilot":
        approve_pilot(args)
    elif args.action == "aggregate-confirmation":
        _require_pilot_gate(
            _campaign_dir(args), _verify_lock(args)[1]
        )
        summarize_phase(args, "confirmation")
    else:
        raise AssertionError(f"unhandled action {args.action}")


if __name__ == "__main__":
    main()
