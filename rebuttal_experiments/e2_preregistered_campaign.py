"""Preregistered, paired E2 confirmation on UCI Protein.

This campaign has two deliberately separate confirmatory estimands:

* fixed-ten-epoch, same-rank numerical/accuracy equivalence; and
* completed training batches under a pilot-calibrated wall-time budget.

The pilot and throughput stages never construct a held-out test tensor.
The accuracy stage evaluates the fixed final state exactly once per method.
Every pair clones one initialization and consumes identical minibatch
permutations.  The driver only reads its new campaign directory and uses
atomic write-once outputs.

PyTorch is imported lazily, after Slurm affinity and thread environment
checks, so that OpenMP cannot silently narrow or expand the requested CPU
set before it is recorded.
"""

from __future__ import annotations

import argparse
import copy
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
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "e2_preregistered_protocol.json"
LOCK_PATH = HERE / "e2_preregistered_lock.json"
RUNTIME_CONTRACT_PATH = HERE / "e2_runtime_contract.json"
DEFAULT_RESULTS = ROOT / "rebuttal_results" / "e2_preregistered"
CAMPAIGN = "e2-positive-preregistered-v2"

EXPECTED_SLURM_ACCOUNT = "marlowe-m000151-pm06"
EXPECTED_SLURM_PARTITION = "batch"
SPLIT_DERIVATION = (
    "SHA256('NeurIPS-29421|e2-positive-preregistered-v2|"
    "heldout-test-split') first 8 bytes big-endian modulo 2^31-1"
)
SPLIT_DERIVATION_INPUT = (
    "NeurIPS-29421|e2-positive-preregistered-v2|heldout-test-split"
)
SPLIT_DERIVATION_SHA256 = (
    "42c28436913da4fee653e7f536c1118b535d4a44e5607e36b4f7b72d6e723b75"
)
MANUAL_GATE_ATTESTATION = (
    "I audited all five E2 v2 pilots and freeze confirmation without "
    "outcome-dependent changes"
)

PILOT_SEEDS = tuple(range(51001, 51006))
ACCURACY_SEEDS = tuple(range(52001, 52021))
THROUGHPUT_SEEDS = tuple(range(53001, 53021))
STAGE_SEEDS = {
    "pilot-pair": PILOT_SEEDS,
    "accuracy-pair": ACCURACY_SEEDS,
    "throughput-pair": THROUGHPUT_SEEDS,
}

RANK = 1024
THREADS = 10
INTEROP_THREADS = 1
EPOCHS = 10
THROUGHPUT_MAX_EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 0.1
DAMPING = 1.0
MOMENTUM = 0.9
REFRESH_THRESHOLD = 0.1
REFACTOR_EVERY = 50
SPLIT_SEED = 381857132
INNER_SPLIT_SEED = 314159
DEVELOPMENT_FRACTION = 0.8
INNER_TRAIN_FRACTION = 0.9
EQUIVALENCE_MARGIN = 0.01
PREFLIGHT_RANK_REPEATS = 64
PREFLIGHT_RESET_AFTER = 50
REQUIRED_AVX512_FLAGS = ("avx512f", "avx512dq", "avx512bw", "avx512vl")

DATA_RELATIVE = Path("Cholesky-main-2/data/protein.mat")
CANDIDATE_SOURCE_RELATIVE = Path(
    "rebuttal_experiments/e1_positive_candidate_ext.cpp"
)
CANDIDATE_BINARY_RELATIVE = Path(
    "rebuttal_experiments/e1_positive_candidate.so"
)

# No module-level import of torch; configure_runtime imports it only after
# exact affinity has been established and verified.
torch: Any = None
candidate_extension: Any = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def derived_split_seed() -> tuple[int, str]:
    digest = hashlib.sha256(SPLIT_DERIVATION_INPUT.encode("utf-8"))
    digest_hex = digest.hexdigest()
    seed = int.from_bytes(digest.digest()[:8], "big") % (2**31 - 1)
    return seed, digest_hex


def load_and_verify_runtime_contract() -> tuple[dict[str, Any], str]:
    contract = read_json(RUNTIME_CONTRACT_PATH)
    if (
        contract.get("schema_version") != 1
        or contract.get("campaign") != CAMPAIGN
        or contract.get("slurm_account") != EXPECTED_SLURM_ACCOUNT
        or contract.get("slurm_partition") != EXPECTED_SLURM_PARTITION
    ):
        raise RuntimeError("runtime contract campaign/Slurm mismatch")
    required = contract.get("required_environment", {})
    for name, expected in {
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }.items():
        if required.get(name) != expected:
            raise RuntimeError(
                f"runtime contract does not freeze {name}={expected}"
            )
    expected_flags = contract.get("required_cpu_flags")
    if expected_flags != list(REQUIRED_AVX512_FLAGS):
        raise RuntimeError("runtime contract AVX-512 flags mismatch")
    return contract, sha256_file(RUNTIME_CONTRACT_PATH)


def write_once(payload: dict[str, Any], destination: Path) -> None:
    """Atomically create destination without replacing an existing artifact."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
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
        # A hard link is an atomic create-if-absent operation.  Unlike
        # os.replace, it cannot silently overwrite a prior campaign result.
        os.link(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def claim_compute_attempt(
    results_root: Path,
    stage: str,
    task_id: int | None,
    context: dict[str, Any],
) -> Path:
    """Create a permanent claim before preflight or dataset access."""

    if os.environ.get("E2_ONE_ATTEMPT") != "1":
        raise RuntimeError("one-attempt execution contract is not enabled")
    restart_count = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    if restart_count != 0:
        raise RuntimeError("a restarted Slurm task may not claim E2 work")
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("attempt claims require a live Slurm job")
    suffix = stage
    seed = None
    if stage in STAGE_SEEDS:
        if task_id is None:
            raise RuntimeError("paired stage claim omitted task ID")
        seed = STAGE_SEEDS[stage][task_id]
        suffix = f"{stage}_task{task_id}_seed{seed}"
    destination = (
        results_root / CAMPAIGN / "claims" / f"{suffix}.json"
    )
    write_once(
        {
            "schema_version": 1,
            "experiment": "e2_preregistered_one_attempt_claim",
            "campaign": CAMPAIGN,
            "stage": stage,
            "task_id": task_id,
            "seed": seed,
            "git_commit": context["git_commit"],
            "lock_sha256": context["lock_sha256"],
            "protocol_sha256": context["protocol_sha256"],
            "slurm_job_id": job_id,
            "slurm_array_job_id": os.environ.get(
                "SLURM_ARRAY_JOB_ID"
            ),
            "slurm_array_task_id": os.environ.get(
                "SLURM_ARRAY_TASK_ID"
            ),
            "slurm_restart_count": restart_count,
            "attempt_ordinal": 1,
            "claimed_unix_ns": time.time_ns(),
        },
        destination,
    )
    return destination


def git_head() -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def verify_locked_commit(expected: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
        raise RuntimeError("--locked-commit must be a full 40-character SHA")
    actual = git_head()
    if actual != expected:
        raise RuntimeError(
            f"immutable commit mismatch: expected {expected}, got {actual}"
        )
    symbolic = subprocess.run(
        ("git", "symbolic-ref", "-q", "HEAD"),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if symbolic.returncode == 0:
        raise RuntimeError(
            "compute campaign must run from detached immutable HEAD, got "
            f"{symbolic.stdout.strip()!r}"
        )
    return actual


def verify_git_tracked_at_head(relative: str) -> None:
    """Require a locked source file to be the exact blob in detached HEAD."""

    tracked = subprocess.run(
        ("git", "ls-files", "--error-unmatch", "--", relative),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if tracked.returncode != 0:
        raise RuntimeError(
            f"locked source is not tracked by Git: {relative}"
        )
    unchanged = subprocess.run(
        ("git", "diff", "--quiet", "HEAD", "--", relative),
        cwd=ROOT,
    )
    if unchanged.returncode != 0:
        raise RuntimeError(
            f"locked source differs from detached HEAD: {relative}"
        )


def load_and_verify_protocol() -> tuple[dict[str, Any], str]:
    protocol = read_json(PROTOCOL_PATH)
    expected = {
        "campaign": CAMPAIGN,
        "schema_version": 1,
        "seeds": {
            "pilot": list(PILOT_SEEDS),
            "accuracy_confirmation": list(ACCURACY_SEEDS),
            "throughput_confirmation": list(THROUGHPUT_SEEDS),
        },
    }
    actual = {
        "campaign": protocol.get("campaign"),
        "schema_version": protocol.get("schema_version"),
        "seeds": protocol.get("seeds"),
    }
    if actual != expected:
        raise RuntimeError(
            f"protocol/driver constant mismatch: expected {expected}, got {actual}"
        )
    frozen = protocol.get("frozen_training", {})
    frozen_expected = {
        "batch_size": BATCH_SIZE,
        "damping": DAMPING,
        "drop_last_incomplete_batch": True,
        "learning_rate": LEARNING_RATE,
        "max_throughput_epochs_safety_cap": THROUGHPUT_MAX_EPOCHS,
        "momentum": MOMENTUM,
        "ours_refactor_every": REFACTOR_EVERY,
        "pilot_epochs": EPOCHS,
        "rank": RANK,
        "refresh_threshold": REFRESH_THRESHOLD,
    }
    if frozen != frozen_expected:
        raise RuntimeError(
            f"frozen training mismatch: expected {frozen_expected}, got {frozen}"
        )
    resources = protocol.get("resources", {})
    if (
        resources.get("openmp_threads") != THREADS
        or resources.get("blas_threads") != THREADS
        or resources.get("interop_threads") != INTEROP_THREADS
        or resources.get("experimental_processes_per_node") != 1
        or resources.get("exclusive_node") is not True
    ):
        raise RuntimeError("protocol resource contract does not match driver")
    if (
        protocol["analysis"]["accuracy_equivalence"][
            "equivalence_margin_absolute_normalized_mse"
        ]
        != EQUIVALENCE_MARGIN
    ):
        raise RuntimeError("equivalence margin differs from frozen driver")
    if protocol["dataset"]["sha256"] != (
        "5ce538fa05f6834de3acc727e220e777a02fb1ebc3a7bfebcb5aaad9769fed13"
    ):
        raise RuntimeError("unexpected frozen Protein data hash")
    derived_seed, derivation_sha = derived_split_seed()
    if (
        derived_seed != SPLIT_SEED
        or derivation_sha != SPLIT_DERIVATION_SHA256
        or protocol["dataset"].get("split_seed") != SPLIT_SEED
        or protocol["dataset"].get("split_seed_derivation")
        != SPLIT_DERIVATION
        or protocol["dataset"].get("split_seed_derivation_input")
        != SPLIT_DERIVATION_INPUT
        or protocol["dataset"].get("split_seed_derivation_sha256")
        != SPLIT_DERIVATION_SHA256
        or protocol["dataset"].get("historical_seed42_outcomes_viewed")
        is not True
        or protocol["dataset"].get(
            "v2_heldout_outcomes_viewed_before_preregistration"
        )
        is not False
    ):
        raise RuntimeError("untouched mechanically derived split mismatch")
    if protocol["implementation"]["candidate_source_sha256"] != (
        "a0d7bba11f178b40fb5ab5c2674640ee8b2f7c5a63836eb931ac9749db0dc8b0"
    ):
        raise RuntimeError("unexpected frozen candidate source hash")
    if protocol["implementation"].get("candidate_binary_sha256") != (
        "b94285904eb7cf7633ab955b5f3f0b63560b4b7c66ff396e59acf5b56e977c19"
    ):
        raise RuntimeError("unexpected frozen AVX-512 candidate binary")
    if (
        resources.get("slurm_account") != EXPECTED_SLURM_ACCOUNT
        or resources.get("slurm_partition") != EXPECTED_SLURM_PARTITION
        or resources.get("requeue") is not False
        or resources.get("one_attempt_per_stage_task") is not True
        or resources.get("rank1024_preflight_repeats")
        != PREFLIGHT_RANK_REPEATS
        or resources.get("rank1024_preflight_reset_after")
        != PREFLIGHT_RESET_AFTER
    ):
        raise RuntimeError("protocol scheduler/preflight contract mismatch")
    all_seeds = PILOT_SEEDS + ACCURACY_SEEDS + THROUGHPUT_SEEDS
    if len(set(all_seeds)) != len(all_seeds):
        raise RuntimeError("pilot and confirmation seed sets overlap")
    prior_or_synthetic_seeds = {
        0,
        1,
        2,
        1000,
        1001,
        1002,
        6100,
        6101,
        6102,
        7777,
    }
    if set(all_seeds) & prior_or_synthetic_seeds:
        raise RuntimeError("campaign seeds overlap prior E2 or synthetic seeds")
    return protocol, sha256_file(PROTOCOL_PATH)


def load_and_verify_lock(expected_sha256: str) -> tuple[dict[str, Any], str]:
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise RuntimeError(
            "--expected-lock-sha256 must be a full lowercase SHA-256"
        )
    actual_lock_sha = sha256_file(LOCK_PATH)
    if actual_lock_sha != expected_sha256:
        raise RuntimeError(
            f"lock hash mismatch: expected {expected_sha256}, "
            f"got {actual_lock_sha}"
        )
    lock = read_json(LOCK_PATH)
    if lock.get("campaign") != CAMPAIGN or lock.get("schema_version") != 1:
        raise RuntimeError("lock manifest campaign/schema mismatch")
    entries = lock.get("files")
    if not isinstance(entries, dict) or not entries:
        raise RuntimeError("lock manifest has no file entries")
    for relative, expected_hash in entries.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"unsafe lock path {relative!r}")
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"locked file is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"locked file changed: {relative}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
    required = {
        str(Path(__file__).resolve().relative_to(ROOT)),
        str(PROTOCOL_PATH.relative_to(ROOT)),
        str(RUNTIME_CONTRACT_PATH.relative_to(ROOT)),
        str(DATA_RELATIVE),
        str(CANDIDATE_SOURCE_RELATIVE),
        str(CANDIDATE_BINARY_RELATIVE),
    }
    if not required.issubset(entries):
        raise RuntimeError(
            f"lock omits required entries: {sorted(required - set(entries))}"
        )
    git_tracked_files = lock.get("git_tracked_files")
    if (
        not isinstance(git_tracked_files, list)
        or len(git_tracked_files) != len(set(git_tracked_files))
    ):
        raise RuntimeError("lock has an invalid git_tracked_files list")
    tracked_required = {
        str(Path(__file__).resolve().relative_to(ROOT)),
        str(PROTOCOL_PATH.relative_to(ROOT)),
        str(RUNTIME_CONTRACT_PATH.relative_to(ROOT)),
        str(CANDIDATE_SOURCE_RELATIVE),
    }
    if not tracked_required.issubset(git_tracked_files):
        raise RuntimeError(
            "lock does not require all source/protocol files from Git"
        )
    if not set(git_tracked_files).issubset(entries):
        raise RuntimeError("git-tracked lock file is absent from files map")
    for relative in git_tracked_files:
        verify_git_tracked_at_head(relative)
    # The manifest cannot contain its own digest without a hash cycle, so
    # its bytes are authenticated by the required external SHA argument and
    # its Git blob is checked separately.
    verify_git_tracked_at_head(str(LOCK_PATH.relative_to(ROOT)))
    return lock, actual_lock_sha


def provenance_context(
    expected_lock_sha256: str,
    locked_commit: str,
) -> dict[str, Any]:
    protocol, protocol_sha = load_and_verify_protocol()
    runtime_contract, runtime_contract_sha = (
        load_and_verify_runtime_contract()
    )
    lock, lock_sha = load_and_verify_lock(expected_lock_sha256)
    commit = verify_locked_commit(locked_commit)
    return {
        "campaign": CAMPAIGN,
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "runtime_contract": runtime_contract,
        "runtime_contract_sha256": runtime_contract_sha,
        "lock": lock,
        "lock_sha256": lock_sha,
        "git_commit": commit,
    }


def cpu_topology(cpu: int) -> dict[str, int]:
    base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    return {
        "logical_cpu": cpu,
        "physical_package": int(
            (base / "physical_package_id").read_text().strip()
        ),
        "core_id": int((base / "core_id").read_text().strip()),
    }


def task_affinities() -> dict[str, list[int]]:
    observed: dict[str, list[int]] = {}
    for task in sorted(Path("/proc/self/task").iterdir()):
        try:
            observed[task.name] = sorted(
                os.sched_getaffinity(int(task.name))
            )
        except (FileNotFoundError, ProcessLookupError):
            continue
    return observed


def parse_parallel_value(report: str, label: str) -> int:
    match = re.search(rf"{re.escape(label)}\s*:\s*(\d+)", report)
    if match is None:
        raise RuntimeError(f"PyTorch thread report omitted {label!r}")
    return int(match.group(1))


def cpu_model_name() -> str:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return platform.processor()
    for line in path.read_text(errors="replace").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def cpu_flags() -> list[str]:
    for line in Path("/proc/cpuinfo").read_text(
        errors="replace"
    ).splitlines():
        if line.lower().startswith("flags"):
            return sorted(set(line.split(":", 1)[1].split()))
    raise RuntimeError("/proc/cpuinfo omitted CPU flags")


def avx512_binary_evidence(binary: Path) -> dict[str, Any]:
    """Prove that AVX-512 occurs in this kernel, not an unrelated wrapper."""

    nm = subprocess.run(
        ("nm", "-C", "--defined-only", str(binary)),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    symbols = [
        line
        for line in nm.splitlines()
        if "positive_same_vector_core" in line
        and "[clone .cold]" not in line
    ]
    if not symbols:
        raise RuntimeError("candidate core symbols are absent")
    disassembly = subprocess.run(
        ("objdump", "-d", "-C", str(binary)),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    selected: list[str] = []
    inside = False
    for line in disassembly.splitlines():
        header = re.match(r"^[0-9a-f]+ <(.+)>:$", line)
        if header:
            inside = (
                "positive_same_vector_core" in header.group(1)
                and "[clone .cold]" not in header.group(1)
            )
        if inside:
            selected.append(line)
    zmm_operands = re.findall(
        r"\bzmm(?:[0-9]|[12][0-9]|3[01])\b",
        "\n".join(selected),
    )
    if not zmm_operands:
        raise RuntimeError(
            "candidate core disassembly contains no AVX-512 zmm operands"
        )
    return {
        "symbols": symbols,
        "selected_disassembly_sha256": hashlib.sha256(
            "\n".join(selected).encode("utf-8")
        ).hexdigest(),
        "zmm_operand_count": len(zmm_operands),
        "zmm_registers": sorted(set(zmm_operands)),
    }


def verify_runtime_files(
    contract: dict[str, Any],
    modules: dict[str, Any],
) -> dict[str, Any]:
    expected_python = contract["python"]
    executable = Path(sys.executable).resolve()
    if (
        sys.version != expected_python["version"]
        or sha256_file(executable) != expected_python["executable_sha256"]
    ):
        raise RuntimeError("Python executable/version differs from lock")
    roots = {
        "torch": Path(os.environ["E2_TORCH_SITE"]).resolve(),
        "numpy": Path(os.environ["E2_NUMERIC_SITE"]).resolve(),
        "scipy": Path(os.environ["E2_NUMERIC_SITE"]).resolve(),
    }
    result: dict[str, Any] = {
        "python_executable": str(executable),
        "python_executable_sha256": sha256_file(executable),
        "packages": {},
    }
    for name, module in modules.items():
        expected = contract["packages"][name]
        module_path = Path(module.__file__).resolve()
        try:
            module_path.relative_to(roots[name])
        except ValueError as error:
            raise RuntimeError(
                f"{name} loaded from {module_path}, outside {roots[name]}"
            ) from error
        if module.__version__ != expected["version"]:
            raise RuntimeError(
                f"{name} version mismatch: {module.__version__!r}"
            )
        observed_probes: dict[str, str] = {}
        package_root = module_path.parent
        for relative, expected_hash in expected["file_sha256"].items():
            probe = package_root / relative
            observed_hash = sha256_file(probe)
            if observed_hash != expected_hash:
                raise RuntimeError(
                    f"{name} runtime file changed: {relative}"
                )
            observed_probes[relative] = observed_hash
        result["packages"][name] = {
            "version": module.__version__,
            "module_path": str(module_path),
            "site_root": str(roots[name]),
            "file_sha256": observed_probes,
        }
    return result


def configure_runtime() -> dict[str, Any]:
    global torch, candidate_extension

    if any(name in sys.modules for name in ("torch", "numpy", "scipy")):
        raise RuntimeError(
            "torch/numpy/scipy was imported before E2 runtime verification"
        )
    runtime_contract, runtime_contract_sha = (
        load_and_verify_runtime_contract()
    )
    required_environment = {
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": str(THREADS),
        "MKL_NUM_THREADS": str(THREADS),
        "OPENBLAS_NUM_THREADS": str(THREADS),
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
        "E2_PROCESS_CONCURRENCY": "1",
        "E2_EXCLUSIVE_NODE": "1",
        "E2_ONE_ATTEMPT": "1",
    }
    for name, expected in required_environment.items():
        actual = os.environ.get(name)
        if actual is None or actual.lower() != expected.lower():
            raise RuntimeError(
                f"{name} must be {expected!r} before Python starts, "
                f"got {actual!r}"
            )
    slurm_expected = {
        "SLURM_CPUS_PER_TASK": str(THREADS),
        "SLURM_NTASKS": "1",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_LOCALID": "0",
        "SLURM_PROCID": "0",
        "SLURM_JOB_ACCOUNT": EXPECTED_SLURM_ACCOUNT,
        "SLURM_JOB_PARTITION": EXPECTED_SLURM_PARTITION,
    }
    for name, expected in slurm_expected.items():
        actual = os.environ.get(name)
        if actual != expected:
            raise RuntimeError(
                f"{name} must be {expected!r}, got {actual!r}"
            )
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("compute actions require a Slurm job")
    restart_count = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    if restart_count != 0:
        raise RuntimeError(
            f"requeued/restarted execution forbidden: {restart_count}"
        )
    for name in ("E2_TORCH_SITE", "E2_NUMERIC_SITE"):
        path = Path(os.environ.get(name, ""))
        if not path.is_absolute() or not path.is_dir():
            raise RuntimeError(f"{name} must name an explicit runtime root")

    affinity_before = sorted(os.sched_getaffinity(0))
    if len(affinity_before) != THREADS:
        raise RuntimeError(
            f"expected exactly {THREADS} logical CPUs, got {affinity_before}"
        )
    topology = [cpu_topology(cpu) for cpu in affinity_before]
    packages = {item["physical_package"] for item in topology}
    cores = {
        (item["physical_package"], item["core_id"]) for item in topology
    }
    if len(packages) != 1 or len(cores) != THREADS:
        raise RuntimeError(
            "affinity must contain ten distinct physical cores on one socket"
        )
    for task in list(Path("/proc/self/task").iterdir()):
        try:
            os.sched_setaffinity(int(task.name), set(affinity_before))
        except (FileNotFoundError, ProcessLookupError):
            continue

    import importlib
    import site

    torch = importlib.import_module("torch")
    numpy = importlib.import_module("numpy")
    scipy = importlib.import_module("scipy")
    if site.ENABLE_USER_SITE is not False:
        raise RuntimeError("Python user-site loading is not disabled")
    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(INTEROP_THREADS)
    except RuntimeError as error:
        raise RuntimeError(
            "PyTorch interop pool initialized before E2 configuration"
        ) from error
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(HERE))
    candidate_extension = importlib.import_module(
        "e1_positive_candidate"
    )
    expected_binary = (ROOT / CANDIDATE_BINARY_RELATIVE).resolve()
    imported_binary = Path(candidate_extension.__file__).resolve()
    if imported_binary != expected_binary:
        raise RuntimeError(
            f"loaded candidate binary {imported_binary}, "
            f"expected {expected_binary}"
        )
    runtime_files = verify_runtime_files(
        runtime_contract,
        {"torch": torch, "numpy": numpy, "scipy": scipy},
    )
    flags = cpu_flags()
    missing_flags = sorted(set(REQUIRED_AVX512_FLAGS) - set(flags))
    if missing_flags:
        raise RuntimeError(
            f"target node lacks required CPU flags: {missing_flags}"
        )
    binary_evidence = avx512_binary_evidence(imported_binary)

    report = torch.__config__.parallel_info()
    runtime_values = {
        "aten_intraop": parse_parallel_value(
            report, "at::get_num_threads()"
        ),
        "aten_interop": parse_parallel_value(
            report, "at::get_num_interop_threads()"
        ),
        "openmp_max": parse_parallel_value(
            report, "omp_get_max_threads()"
        ),
        "mkl_max": parse_parallel_value(
            report, "mkl_get_max_threads()"
        ),
    }
    expected_runtime = {
        "aten_intraop": THREADS,
        "aten_interop": INTEROP_THREADS,
        "openmp_max": THREADS,
        "mkl_max": THREADS,
    }
    if runtime_values != expected_runtime:
        raise RuntimeError(
            f"thread runtime mismatch: expected {expected_runtime}, "
            f"got {runtime_values}"
        )
    return {
        "host": socket.gethostname(),
        "cpu_model": cpu_model_name(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "sys_path": list(sys.path),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pid": os.getpid(),
        "runtime_contract_path": str(RUNTIME_CONTRACT_PATH.resolve()),
        "runtime_contract_sha256": runtime_contract_sha,
        "runtime_files": runtime_files,
        "required_environment": required_environment,
        "runtime_roots": {
            "torch": str(Path(os.environ["E2_TORCH_SITE"]).resolve()),
            "numeric": str(
                Path(os.environ["E2_NUMERIC_SITE"]).resolve()
            ),
        },
        "slurm": {
            name: os.environ.get(name)
            for name in (
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_STEP_ID",
                "SLURM_CPUS_PER_TASK",
                "SLURM_NTASKS",
                "SLURM_JOB_NUM_NODES",
                "SLURM_LOCALID",
                "SLURM_PROCID",
                "SLURM_JOB_ACCOUNT",
                "SLURM_JOB_PARTITION",
                "SLURM_RESTART_COUNT",
            )
        },
        "affinity_before_torch_import": affinity_before,
        "cpu_topology": topology,
        "runtime_values": runtime_values,
        "torch_parallel_info": report,
        "candidate_binary": str(imported_binary),
        "candidate_binary_sha256": sha256_file(imported_binary),
        "required_cpu_flags": list(REQUIRED_AVX512_FLAGS),
        "cpu_flags": flags,
        "candidate_avx512_evidence": binary_evidence,
    }


def finalize_runtime(runtime: dict[str, Any]) -> None:
    affinities = task_affinities()
    selected = set(runtime["affinity_before_torch_import"])
    if not affinities:
        raise RuntimeError("no process tasks found for affinity audit")
    if any(
        not set(task_set).issubset(selected)
        for task_set in affinities.values()
    ):
        raise RuntimeError("a runtime thread escaped the selected CPU set")
    union = set().union(*(set(value) for value in affinities.values()))
    if union != selected:
        raise RuntimeError(
            f"worker affinity union {sorted(union)} does not equal "
            f"selected CPUs {sorted(selected)}"
        )
    runtime["tasks_after_pair"] = affinities
    runtime["worker_affinity_union"] = sorted(union)


def verify_live_array_task(
    runtime: dict[str, Any],
    task_id: int,
) -> None:
    """Fail before data/model work if Slurm and preregistered tasks differ."""

    slurm = runtime.get("slurm", {})
    if (
        slurm.get("SLURM_ARRAY_TASK_ID") != str(task_id)
        or not slurm.get("SLURM_ARRAY_JOB_ID")
        or not slurm.get("SLURM_JOB_ID")
    ):
        raise RuntimeError(
            "live Slurm array task does not match the frozen task mapping"
        )


def tensor_sha256(value: Any) -> str:
    contiguous = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def hash_named_tensors(items: list[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, value in items:
        digest.update(name.encode("utf-8"))
        digest.update(tensor_sha256(value).encode("ascii"))
    return digest.hexdigest()


def split_indices(n_examples: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    permutation = torch.randperm(n_examples, generator=generator)
    n_development = int(DEVELOPMENT_FRACTION * n_examples)
    development = permutation[:n_development].contiguous()
    test = permutation[n_development:].contiguous()

    inner_generator = torch.Generator().manual_seed(INNER_SPLIT_SEED)
    inner_order = torch.randperm(
        n_development, generator=inner_generator
    )
    n_inner_train = int(INNER_TRAIN_FRACTION * n_development)
    pilot_train = development[
        inner_order[:n_inner_train]
    ].contiguous()
    pilot_validation = development[
        inner_order[n_inner_train:]
    ].contiguous()
    return {
        "development": development,
        "test": test,
        "pilot_train": pilot_train,
        "pilot_validation": pilot_validation,
    }


def load_protein(stage: str) -> dict[str, Any]:
    """Load only the stage-permitted split tensors.

    scipy necessarily reads the source matrix, but pilot and throughput
    never index the fixed test rows into feature/label tensors and never
    call a held-out model evaluation.
    """

    from scipy.io import loadmat

    path = (ROOT / DATA_RELATIVE).resolve()
    raw = loadmat(path)
    if "data" not in raw:
        raise RuntimeError(f"{path} has no 'data' matrix")
    matrix = torch.as_tensor(raw["data"], dtype=torch.float32)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise RuntimeError("Protein data matrix has invalid shape")
    indices = split_indices(matrix.shape[0])

    if stage == "pilot-pair":
        train_index = indices["pilot_train"]
        evaluation_index = indices["pilot_validation"]
        evaluation_kind = "validation"
    elif stage in ("accuracy-pair", "throughput-pair"):
        train_index = indices["development"]
        evaluation_index = (
            indices["test"] if stage == "accuracy-pair" else None
        )
        evaluation_kind = (
            "test" if stage == "accuracy-pair" else "none"
        )
    else:
        raise ValueError(f"unknown stage {stage!r}")

    # Fit every preprocessing statistic only on the rows used for training
    # in this stage.
    train_rows = matrix[train_index]
    X_train_raw = train_rows[:, :-1]
    y_train_raw = train_rows[:, -1]
    x_min = X_train_raw.amin(dim=0)
    x_range = (
        X_train_raw.amax(dim=0) - x_min
    ).clamp_min(1e-8)
    y_mean = y_train_raw.mean()
    y_std = y_train_raw.std(unbiased=False).clamp_min(1e-8)
    X_train = ((X_train_raw - x_min) / x_range).contiguous()
    y_train = ((y_train_raw - y_mean) / y_std).contiguous()

    X_evaluation = None
    y_evaluation = None
    if evaluation_index is not None:
        evaluation_rows = matrix[evaluation_index]
        X_evaluation = (
            (evaluation_rows[:, :-1] - x_min) / x_range
        ).contiguous()
        y_evaluation = (
            (evaluation_rows[:, -1] - y_mean) / y_std
        ).contiguous()

    development_steps_per_epoch = (
        indices["development"].numel() // BATCH_SIZE
    )
    metadata = {
        "path": str(path),
        "sha256": sha256_file(path),
        "raw_shape": list(matrix.shape),
        "split_seed": SPLIT_SEED,
        "split_seed_derivation": SPLIT_DERIVATION,
        "split_seed_derivation_sha256": SPLIT_DERIVATION_SHA256,
        "inner_split_seed": INNER_SPLIT_SEED,
        "development_fraction": DEVELOPMENT_FRACTION,
        "inner_train_fraction": INNER_TRAIN_FRACTION,
        "train_examples": int(train_index.numel()),
        "evaluation_examples": (
            int(evaluation_index.numel())
            if evaluation_index is not None
            else 0
        ),
        "evaluation_kind": evaluation_kind,
        "test_tensor_constructed": stage == "accuracy-pair",
        "index_sha256": {
            name: tensor_sha256(value)
            for name, value in indices.items()
        },
        "normalizer_sha256": hash_named_tensors(
            [
                ("x_min", x_min),
                ("x_range", x_range),
                ("y_mean", y_mean.reshape(1)),
                ("y_std", y_std.reshape(1)),
            ]
        ),
        "steps_per_epoch": X_train.shape[0] // BATCH_SIZE,
        "development_steps_per_epoch": development_steps_per_epoch,
    }
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_evaluation": X_evaluation,
        "y_evaluation": y_evaluation,
        "metadata": metadata,
    }


def build_model(d_in: int) -> Any:
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
    )


def flatten_first_layer_gradient(model: Any) -> Any:
    return torch.cat(
        (
            model[0].weight.grad.reshape(-1),
            model[0].bias.grad.reshape(-1),
        )
    ).detach()


def apply_first_layer_step(
    model: Any,
    direction: Any,
    learning_rate: float,
) -> None:
    first = model[0]
    weight_count = first.weight.numel()
    first.weight.add_(
        direction[:weight_count].reshape_as(first.weight),
        alpha=-learning_rate,
    )
    first.bias.add_(
        direction[weight_count:],
        alpha=-learning_rate,
    )


def whiten(L: Any, M: Any) -> Any:
    left = torch.linalg.solve_triangular(L, M, upper=False)
    return torch.linalg.solve_triangular(
        L, left.mT, upper=False
    ).mT.contiguous()


def make_initial_state(d_in: int, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = build_model(d_in)
    d_param = (
        model[0].weight.numel() + model[0].bias.numel()
    )
    if RANK > d_param:
        raise RuntimeError(
            f"rank {RANK} exceeds first-layer dimension {d_param}"
        )
    U, _ = torch.linalg.qr(
        torch.randn(d_param, RANK), mode="reduced"
    )
    identity = torch.eye(RANK, dtype=torch.float32)
    G = (DAMPING * identity).contiguous()
    L = (DAMPING**0.5 * identity).contiguous()
    Aw = torch.zeros(RANK, RANK, dtype=torch.float32).contiguous()
    return {
        "model": model,
        "d_param": d_param,
        "U": U.contiguous(),
        "identity": identity,
        "G": G,
        "L": L,
        "Aw": Aw,
        "Aw_spare": torch.empty_like(Aw),
        "vz": torch.empty(RANK, dtype=torch.float32),
        "workspace": torch.empty(4 * RANK, dtype=torch.float32),
        "first_velocity": torch.zeros(d_param, dtype=torch.float32),
        "other_velocities": {
            layer_index: {
                "weight": torch.zeros_like(
                    model[layer_index].weight
                ),
                "bias": torch.zeros_like(model[layer_index].bias),
            }
            for layer_index in (2, 4, 6)
        },
        "steps": 0,
        "refresh_count": 0,
        "core_seconds": 0.0,
        "refresh_seconds": 0.0,
        "reset_seconds": 0.0,
        "reset_drift": [],
        "unexplained_sum": 0.0,
    }


def semantic_state_fingerprint(state: dict[str, Any]) -> str:
    tensors: list[tuple[str, Any]] = []
    for name, value in sorted(
        state["model"].state_dict().items()
    ):
        tensors.append((f"model.{name}", value))
    for name in (
        "U",
        "identity",
        "G",
        "L",
        "Aw",
        "first_velocity",
    ):
        tensors.append((name, state[name]))
    for layer_index in (2, 4, 6):
        for name in ("weight", "bias"):
            tensors.append(
                (
                    f"velocity.{layer_index}.{name}",
                    state["other_velocities"][layer_index][name],
                )
            )
    return hash_named_tensors(tensors)


def clone_initial_state(base: dict[str, Any]) -> dict[str, Any]:
    state = {
        "model": copy.deepcopy(base["model"]),
        "d_param": base["d_param"],
        "U": base["U"].clone(),
        "identity": base["identity"].clone(),
        "G": base["G"].clone(),
        "L": base["L"].clone(),
        "Aw": base["Aw"].clone(),
        "Aw_spare": torch.empty_like(base["Aw"]),
        "vz": torch.empty_like(base["vz"]),
        "workspace": torch.empty_like(base["workspace"]),
        "first_velocity": base["first_velocity"].clone(),
        "other_velocities": {
            layer_index: {
                name: value.clone()
                for name, value in velocities.items()
            }
            for layer_index, velocities in base[
                "other_velocities"
            ].items()
        },
        "steps": 0,
        "refresh_count": 0,
        "core_seconds": 0.0,
        "refresh_seconds": 0.0,
        "reset_seconds": 0.0,
        "reset_drift": [],
        "unexplained_sum": 0.0,
    }
    if semantic_state_fingerprint(state) != (
        semantic_state_fingerprint(base)
    ):
        raise RuntimeError("state clone changed the shared initialization")
    return state


def make_minibatch_schedule(
    n_train: int,
    seed: int,
    epochs: int,
) -> tuple[list[Any], str]:
    permutations: list[Any] = []
    digest = hashlib.sha256()
    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(
            seed * 100_003 + epoch
        )
        permutation = torch.randperm(
            n_train, generator=generator
        ).contiguous()
        permutations.append(permutation)
        digest.update(epoch.to_bytes(8, "little"))
        digest.update(
            permutation.numpy().tobytes(order="C")
        )
    return permutations, digest.hexdigest()


def relative_error(value: Any, reference: Any) -> float:
    numerator = torch.linalg.norm(value - reference)
    denominator = torch.linalg.norm(reference).clamp_min(1e-30)
    return float((numerator / denominator).item())


def train_step(
    state: dict[str, Any],
    xb: Any,
    yb: Any,
    method: str,
    *,
    force_nonrefresh: bool = False,
) -> dict[str, Any]:
    model = state["model"]
    prediction = model(xb).squeeze(-1)
    loss = torch.nn.functional.mse_loss(prediction, yb)
    model.zero_grad(set_to_none=True)
    loss.backward()
    g = flatten_first_layer_gradient(model)

    U = state["U"]
    a = U.mT @ g
    residual = g - U @ a
    g_energy = float(torch.dot(g, g).item())
    residual_energy = float(torch.dot(residual, residual).item())
    unexplained = residual_energy / max(g_energy, 1e-30)
    refreshed = (
        not force_nonrefresh
        and unexplained > REFRESH_THRESHOLD
        and residual_energy > 1e-24
    )

    if refreshed:
        refresh_start = time.perf_counter()
        rho = residual_energy**0.5
        U_bar = torch.cat(
            (U, (residual / rho).unsqueeze(1)), dim=1
        )
        c_bar = torch.cat(
            (a, torch.tensor([rho], dtype=a.dtype))
        )
        fisher_bar = torch.zeros(
            RANK + 1, RANK + 1, dtype=torch.float32
        )
        fisher_bar[:RANK, :RANK] = (
            state["G"] - DAMPING * state["identity"]
        )
        fisher_bar.add_(torch.outer(c_bar, c_bar))
        eigenvalues, eigenvectors = torch.linalg.eigh(
            fisher_bar
        )
        top = torch.arange(RANK, 0, -1)
        rotation = eigenvectors[:, top]
        fisher = torch.diag(
            eigenvalues[top].clamp_min(0.0)
        ).contiguous()
        state["U"] = (U_bar @ rotation).contiguous()
        state["G"] = (
            DAMPING * state["identity"] + fisher
        ).contiguous()
        state["L"] = torch.linalg.cholesky(
            state["G"]
        ).contiguous()
        state["Aw"] = whiten(state["L"], fisher)
        state["Aw_spare"] = torch.empty_like(state["Aw"])
        state["refresh_seconds"] += (
            time.perf_counter() - refresh_start
        )
        state["refresh_count"] += 1
        U = state["U"]
        a = U.mT @ g
        residual = g - U @ a
        y_solve = torch.linalg.solve_triangular(
            state["L"], a.unsqueeze(1), upper=False
        ).squeeze(1)
        x = torch.linalg.solve_triangular(
            state["L"].mT,
            y_solve.unsqueeze(1),
            upper=True,
        ).squeeze(1)
    else:
        core_start = time.perf_counter()
        if method == "ours":
            candidate_extension.positive_metric_operator_same_vector_out(
                state["L"],
                state["Aw"],
                a,
                state["Aw_spare"],
                state["vz"],
                state["workspace"],
            )
            state["Aw"], state["Aw_spare"] = (
                state["Aw_spare"],
                state["Aw"],
            )
            state["G"].add_(torch.outer(a, a))
            x = torch.linalg.solve_triangular(
                state["L"].mT,
                state["vz"].unsqueeze(1),
                upper=True,
            ).squeeze(1)
        elif method == "refact":
            state["G"].add_(torch.outer(a, a))
            state["L"] = torch.linalg.cholesky(
                state["G"]
            ).contiguous()
            state["Aw"] = whiten(
                state["L"],
                state["G"] - DAMPING * state["identity"],
            )
            y_solve = torch.linalg.solve_triangular(
                state["L"], a.unsqueeze(1), upper=False
            ).squeeze(1)
            x = torch.linalg.solve_triangular(
                state["L"].mT,
                y_solve.unsqueeze(1),
                upper=True,
            ).squeeze(1)
        else:
            raise ValueError(f"unknown method {method!r}")
        state["core_seconds"] += (
            time.perf_counter() - core_start
        )

    state["steps"] += 1
    if (
        method == "ours"
        and REFACTOR_EVERY > 0
        and state["steps"] % REFACTOR_EVERY == 0
    ):
        reset_start = time.perf_counter()
        L_exact = torch.linalg.cholesky(
            state["G"]
        ).contiguous()
        Aw_exact = whiten(
            L_exact,
            state["G"] - DAMPING * state["identity"],
        )
        drift = relative_error(state["Aw"], Aw_exact)
        state["reset_drift"].append(drift)
        state["L"] = L_exact
        state["Aw"] = Aw_exact
        state["Aw_spare"] = torch.empty_like(Aw_exact)
        state["reset_seconds"] += (
            time.perf_counter() - reset_start
        )

    direction = residual / DAMPING + state["U"] @ x
    if not bool(torch.isfinite(direction).all()):
        raise RuntimeError("non-finite preconditioned direction")
    with torch.no_grad():
        state["first_velocity"].mul_(MOMENTUM).add_(
            direction
        )
        apply_first_layer_step(
            model, state["first_velocity"], LEARNING_RATE
        )
        for layer_index in (2, 4, 6):
            layer = model[layer_index]
            velocity = state["other_velocities"][layer_index]
            velocity["weight"].mul_(MOMENTUM).add_(
                layer.weight.grad
            )
            velocity["bias"].mul_(MOMENTUM).add_(
                layer.bias.grad
            )
            layer.weight.add_(
                velocity["weight"], alpha=-LEARNING_RATE
            )
            layer.bias.add_(
                velocity["bias"], alpha=-LEARNING_RATE
            )
    loss_value = float(loss.detach().item())
    if not math.isfinite(loss_value):
        raise RuntimeError("non-finite training loss")
    state["unexplained_sum"] += unexplained
    return {
        "batch_mse": loss_value,
        "unexplained_energy": unexplained,
        "refreshed": refreshed,
    }


def warm_backends(
    base: dict[str, Any],
    X_train: Any,
    y_train: Any,
    first_index: Any,
) -> None:
    index = first_index[:BATCH_SIZE]
    xb, yb = X_train[index], y_train[index]
    # Both methods occupy both positions once.  A forced non-refresh on
    # disposable state ensures the two compared core backends are actually
    # initialized before application timing starts.
    for order in (("ours", "refact"), ("refact", "ours")):
        for method in order:
            disposable = clone_initial_state(base)
            train_step(
                disposable,
                xb,
                yb,
                method,
                force_nonrefresh=True,
            )


def selected_counters(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "steps": state["steps"],
        "refresh_count": state["refresh_count"],
        "core_seconds": state["core_seconds"],
        "refresh_seconds": state["refresh_seconds"],
        "reset_seconds": state["reset_seconds"],
        "max_reset_drift": max(
            state["reset_drift"], default=None
        ),
        "mean_unexplained_energy": (
            state["unexplained_sum"] / state["steps"]
            if state["steps"]
            else None
        ),
    }


def run_method(
    method: str,
    base: dict[str, Any],
    data: dict[str, Any],
    schedule: list[Any],
    *,
    budget_seconds: float | None,
    evaluation_kind: str,
) -> dict[str, Any]:
    state = clone_initial_state(base)
    initial_sha = semantic_state_fingerprint(state)
    X_train = data["X_train"]
    y_train = data["y_train"]
    selected = selected_counters(state)
    selected_training_seconds = 0.0
    attempted_steps = 0
    budget_crossed = False
    selected_batch_mse: float | None = None
    started = time.perf_counter()

    for permutation in schedule:
        for offset in range(
            0,
            X_train.shape[0] - BATCH_SIZE + 1,
            BATCH_SIZE,
        ):
            index = permutation[offset : offset + BATCH_SIZE]
            outcome = train_step(
                state,
                X_train[index],
                y_train[index],
                method,
            )
            attempted_steps += 1
            elapsed = time.perf_counter() - started
            if (
                budget_seconds is not None
                and elapsed > budget_seconds
            ):
                budget_crossed = True
                break
            selected = selected_counters(state)
            selected_training_seconds = elapsed
            selected_batch_mse = outcome["batch_mse"]
        if budget_crossed:
            break

    attempted_training_seconds = time.perf_counter() - started
    if budget_seconds is None:
        expected_steps = (
            len(schedule) * data["metadata"]["steps_per_epoch"]
        )
        if state["steps"] != expected_steps:
            raise RuntimeError(
                f"fixed-epoch run completed {state['steps']} steps, "
                f"expected {expected_steps}"
            )
        selected = selected_counters(state)
        selected_training_seconds = attempted_training_seconds
    else:
        if not budget_crossed:
            raise RuntimeError(
                "throughput run hit its 100-epoch safety cap "
                "before crossing the frozen budget"
            )
        if selected["steps"] < 1:
            raise RuntimeError("budget expired before one full batch")
        if selected_training_seconds > budget_seconds:
            raise RuntimeError("selected throughput state exceeds budget")

    evaluation_value = None
    evaluation_calls = 0
    test_evaluations = 0
    if evaluation_kind != "none":
        if data["X_evaluation"] is None:
            raise RuntimeError("requested evaluation tensor is absent")
        with torch.no_grad():
            evaluation_value = float(
                torch.nn.functional.mse_loss(
                    state["model"](
                        data["X_evaluation"]
                    ).squeeze(-1),
                    data["y_evaluation"],
                ).item()
            )
        evaluation_calls = 1
        test_evaluations = 1 if evaluation_kind == "test" else 0
        if not math.isfinite(evaluation_value):
            raise RuntimeError("non-finite final evaluation MSE")

    return {
        "method": method,
        "rank": RANK,
        "initial_state_sha256": initial_sha,
        "fixed_epoch_run": budget_seconds is None,
        "budget_seconds": budget_seconds,
        "budget_crossed": budget_crossed,
        "selected_steps": selected["steps"],
        "attempted_steps": attempted_steps,
        "selected_training_seconds": selected_training_seconds,
        "attempted_training_seconds": attempted_training_seconds,
        "selected_batch_mse": selected_batch_mse,
        "drop_last_incomplete_batch": True,
        "over_budget_attempt_excluded_from_reported_counters": (
            budget_seconds is not None
        ),
        "selected_model_state_materialized": budget_seconds is None,
        "refresh_count": selected["refresh_count"],
        "core_seconds": selected["core_seconds"],
        "refresh_seconds": selected["refresh_seconds"],
        "reset_seconds": selected["reset_seconds"],
        "max_reset_drift": selected["max_reset_drift"],
        "mean_unexplained_energy": selected[
            "mean_unexplained_energy"
        ],
        "evaluation_kind": evaluation_kind,
        "evaluation_calls": evaluation_calls,
        "final_evaluation_mse": evaluation_value,
        "test_evaluations": test_evaluations,
        "final_model_sha256": (
            semantic_state_fingerprint(state)
            if budget_seconds is None
            else None
        ),
    }


def artifact_path(
    results_root: Path,
    stage: str,
    seed: int,
) -> Path:
    prefixes = {
        "pilot-pair": "pilot",
        "accuracy-pair": "accuracy",
        "throughput-pair": "throughput",
    }
    return (
        results_root
        / CAMPAIGN
        / f"{prefixes[stage]}_seed{seed}.json"
    )


def common_artifact_provenance(
    context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "git_commit": context["git_commit"],
        "protocol_path": str(PROTOCOL_PATH.resolve()),
        "protocol_sha256": context["protocol_sha256"],
        "runtime_contract_path": str(RUNTIME_CONTRACT_PATH.resolve()),
        "runtime_contract_sha256": context[
            "runtime_contract_sha256"
        ],
        "lock_path": str(LOCK_PATH.resolve()),
        "lock_sha256": context["lock_sha256"],
        "driver_path": str(Path(__file__).resolve()),
        "driver_sha256": sha256_file(Path(__file__).resolve()),
        "data_path": str((ROOT / DATA_RELATIVE).resolve()),
        "data_sha256": sha256_file(ROOT / DATA_RELATIVE),
        "candidate_source_path": str(
            (ROOT / CANDIDATE_SOURCE_RELATIVE).resolve()
        ),
        "candidate_source_sha256": sha256_file(
            ROOT / CANDIDATE_SOURCE_RELATIVE
        ),
        "candidate_binary_path": str(
            (ROOT / CANDIDATE_BINARY_RELATIVE).resolve()
        ),
        "candidate_binary_sha256": sha256_file(
            ROOT / CANDIDATE_BINARY_RELATIVE
        ),
    }


def run_pair(
    stage: str,
    task_id: int,
    results_root: Path,
    context: dict[str, Any],
    expected_gate_sha256: str | None = None,
) -> Path:
    seeds = STAGE_SEEDS[stage]
    if not 0 <= task_id < len(seeds):
        raise RuntimeError(
            f"{stage} task ID must be in [0,{len(seeds)-1}]"
        )
    seed = seeds[task_id]
    gate = None
    if stage in ("accuracy-pair", "throughput-pair"):
        gate = verify_pilot_gate(
            results_root, context, expected_gate_sha256
        )
    claim_path = claim_compute_attempt(
        results_root, stage, task_id, context
    )
    runtime = configure_runtime()
    verify_live_array_task(runtime, task_id)
    preflight = synthetic_self_test()
    data = load_protein(stage)
    epochs = (
        THROUGHPUT_MAX_EPOCHS
        if stage == "throughput-pair"
        else EPOCHS
    )
    schedule, schedule_sha = make_minibatch_schedule(
        data["X_train"].shape[0],
        seed,
        epochs,
    )
    base = make_initial_state(
        data["X_train"].shape[1], seed
    )
    shared_initial_sha = semantic_state_fingerprint(base)
    warm_backends(
        base,
        data["X_train"],
        data["y_train"],
        schedule[0],
    )
    order = (
        ("ours", "refact")
        if task_id % 2 == 0
        else ("refact", "ours")
    )

    budget_seconds = None
    budget_provenance = None
    if stage == "throughput-pair":
        budget_path = results_root / CAMPAIGN / "budget.json"
        budget_payload = read_json(budget_path)
        audit_budget_against_pilots(
            results_root, budget_payload, context
        )
        budget_seconds = float(budget_payload["budget_seconds"])
        budget_provenance = {
            "path": str(budget_path.resolve()),
            "sha256": sha256_file(budget_path),
            "calibration_rule": budget_payload[
                "calibration_rule"
            ],
            "pilot_seeds": budget_payload["pilot_seeds"],
        }

    evaluation_kind = {
        "pilot-pair": "validation",
        "accuracy-pair": "test",
        "throughput-pair": "none",
    }[stage]
    methods: dict[str, Any] = {}
    for method in order:
        methods[method] = run_method(
            method,
            base,
            data,
            schedule,
            budget_seconds=budget_seconds,
            evaluation_kind=evaluation_kind,
        )
    if {
        methods["ours"]["initial_state_sha256"],
        methods["refact"]["initial_state_sha256"],
        shared_initial_sha,
    } != {shared_initial_sha}:
        raise RuntimeError("paired methods did not share initialization")
    if stage == "accuracy-pair":
        if any(
            methods[name]["test_evaluations"] != 1
            for name in ("ours", "refact")
        ):
            raise RuntimeError(
                "accuracy pair did not evaluate test exactly once per method"
            )
    elif any(
        methods[name]["test_evaluations"] != 0
        for name in ("ours", "refact")
    ):
        raise RuntimeError(
            f"{stage} unexpectedly evaluated the held-out test set"
        )
    finalize_runtime(runtime)

    payload = {
        "schema_version": 1,
        "experiment": "e2_preregistered_paired_protein",
        "campaign": CAMPAIGN,
        "status": "complete",
        "stage": stage,
        "task_id": task_id,
        "seed": seed,
        "method_order": list(order),
        "method_process_concurrency": 1,
        "methods_executed_sequentially": True,
        "shared_initial_state_sha256": shared_initial_sha,
        "minibatch_schedule_sha256": schedule_sha,
        "schedule_epochs": epochs,
        "dataset": data["metadata"],
        "training": {
            "rank": RANK,
            "batch_size": BATCH_SIZE,
            "drop_last_incomplete_batch": True,
            "learning_rate": LEARNING_RATE,
            "damping": DAMPING,
            "momentum": MOMENTUM,
            "refresh_threshold": REFRESH_THRESHOLD,
            "ours_refactor_every": REFACTOR_EVERY,
        },
        "budget": budget_provenance,
        "manual_pilot_gate": (
            {
                "path": str(pilot_gate_path(results_root).resolve()),
                "sha256": expected_gate_sha256,
                "created_unix_ns": gate["created_unix_ns"],
            }
            if gate is not None
            else None
        ),
        "methods": methods,
        "provenance": common_artifact_provenance(context),
        "runtime": runtime,
        "attempt_claim": {
            "path": str(claim_path.resolve()),
            "sha256": sha256_file(claim_path),
            "attempt_ordinal": 1,
        },
        "exact_node_preflight": preflight,
        "scientific_guardrails": {
            "historical_seed42_outcomes_previously_viewed": True,
            "v2_test_outcomes_read_before_preregistration": False,
            "v2_split_selected_mechanically": True,
            "test_evaluations_total": sum(
                methods[name]["test_evaluations"]
                for name in ("ours", "refact")
            ),
            "checkpoint_selection": False,
            "pair_complete_before_write": True,
        },
    }
    destination = artifact_path(
        results_root, stage, seed
    )
    write_once(payload, destination)
    return destination


def validate_common_artifact(
    payload: dict[str, Any],
    *,
    stage: str,
    task_id: int,
    seed: int,
    context: dict[str, Any],
) -> None:
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment")
        != "e2_preregistered_paired_protein"
        or payload.get("campaign") != CAMPAIGN
        or payload.get("status") != "complete"
        or payload.get("stage") != stage
        or payload.get("task_id") != task_id
        or payload.get("seed") != seed
    ):
        raise RuntimeError(
            f"invalid {stage} artifact for task {task_id}, seed {seed}"
        )
    expected_order = (
        ["ours", "refact"]
        if task_id % 2 == 0
        else ["refact", "ours"]
    )
    if payload.get("method_order") != expected_order:
        raise RuntimeError("method order violates frozen balance")
    if (
        payload.get("method_process_concurrency") != 1
        or payload.get("methods_executed_sequentially") is not True
    ):
        raise RuntimeError("pair concurrency contract violated")
    methods = payload.get("methods", {})
    if set(methods) != {"ours", "refact"}:
        raise RuntimeError("artifact does not contain exactly both methods")
    shared = payload.get("shared_initial_state_sha256")
    if not isinstance(shared, str) or re.fullmatch(
        r"[0-9a-f]{64}", shared
    ) is None:
        raise RuntimeError("shared initialization hash is invalid")
    if any(
        methods[name].get("initial_state_sha256") != shared
        for name in ("ours", "refact")
    ):
        raise RuntimeError("pair has non-identical initialization hashes")
    expected_fixed = stage != "throughput-pair"
    expected_evaluation_kind = {
        "pilot-pair": "validation",
        "accuracy-pair": "test",
        "throughput-pair": "none",
    }[stage]
    expected_evaluation_calls = (
        0 if stage == "throughput-pair" else 1
    )
    expected_test_count = 1 if stage == "accuracy-pair" else 0
    for name in ("ours", "refact"):
        method = methods[name]
        if method.get("method") != name:
            raise RuntimeError("method payload label mismatch")
        if method.get("rank") != RANK:
            raise RuntimeError("method payload rank mismatch")
        if (
            method.get("fixed_epoch_run") is not expected_fixed
            or method.get("drop_last_incomplete_batch") is not True
            or method.get("selected_model_state_materialized")
            is not expected_fixed
            or method.get(
                "over_budget_attempt_excluded_from_reported_counters"
            )
            is not (not expected_fixed)
        ):
            raise RuntimeError("method selection-mode contract mismatch")
        for field in (
            "selected_training_seconds",
            "attempted_training_seconds",
        ):
            value = method.get(field)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise RuntimeError(
                    f"{name} has invalid {field}: {value}"
                )
        if (
            method["attempted_training_seconds"]
            < method["selected_training_seconds"]
        ):
            raise RuntimeError(
                "attempted time precedes selected training time"
            )
        selected_loss = method.get("selected_batch_mse")
        if not isinstance(selected_loss, (int, float)) or not (
            math.isfinite(float(selected_loss))
        ):
            raise RuntimeError("selected batch loss is invalid")
        for field in ("selected_steps", "attempted_steps"):
            value = method.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or (
                value < 1
            ):
                raise RuntimeError(
                    f"{name} has invalid {field}: {value}"
                )
        if (
            method.get("evaluation_kind")
            != expected_evaluation_kind
            or method.get("evaluation_calls")
            != expected_evaluation_calls
            or method.get("test_evaluations")
            != expected_test_count
        ):
            raise RuntimeError("method evaluation contract is invalid")
        evaluation = method.get("final_evaluation_mse")
        if stage == "throughput-pair":
            if evaluation is not None:
                raise RuntimeError(
                    "throughput artifact contains an evaluation outcome"
                )
        elif not isinstance(evaluation, (int, float)) or not (
            math.isfinite(float(evaluation))
        ):
            raise RuntimeError("fixed-epoch evaluation is non-finite")
        final_state_sha = method.get("final_model_sha256")
        if expected_fixed:
            if not isinstance(final_state_sha, str) or re.fullmatch(
                r"[0-9a-f]{64}", final_state_sha
            ) is None:
                raise RuntimeError(
                    "fixed-epoch final state hash is invalid"
                )
        elif final_state_sha is not None:
            raise RuntimeError(
                "throughput artifact materialized a final state hash"
            )
    provenance = payload.get("provenance", {})
    required_provenance = {
        "git_commit": context["git_commit"],
        "protocol_sha256": context["protocol_sha256"],
        "runtime_contract_sha256": context[
            "runtime_contract_sha256"
        ],
        "lock_sha256": context["lock_sha256"],
    }
    if any(
        provenance.get(name) != value
        for name, value in required_provenance.items()
    ):
        raise RuntimeError("artifact provenance lock/commit mismatch")
    locked_files = context["lock"]["files"]
    expected_hashes = {
        "driver_sha256": locked_files[
            str(Path(__file__).resolve().relative_to(ROOT))
        ],
        "data_sha256": locked_files[str(DATA_RELATIVE)],
        "candidate_source_sha256": locked_files[
            str(CANDIDATE_SOURCE_RELATIVE)
        ],
        "candidate_binary_sha256": locked_files[
            str(CANDIDATE_BINARY_RELATIVE)
        ],
    }
    if any(
        provenance.get(name) != value
        for name, value in expected_hashes.items()
    ):
        raise RuntimeError("artifact contains a non-locked file hash")
    runtime = payload.get("runtime", {})
    if (
        runtime.get("runtime_values")
        != {
            "aten_intraop": THREADS,
            "aten_interop": INTEROP_THREADS,
            "openmp_max": THREADS,
            "mkl_max": THREADS,
        }
        or len(runtime.get("affinity_before_torch_import", []))
        != THREADS
        or runtime.get("worker_affinity_union")
        != runtime.get("affinity_before_torch_import")
    ):
        raise RuntimeError("artifact thread/affinity verification failed")
    for field in (
        "cpu_model",
        "platform",
        "python",
        "python_executable",
        "torch",
        "numpy",
        "scipy",
        "torch_parallel_info",
    ):
        if not isinstance(runtime.get(field), str) or not runtime[field]:
            raise RuntimeError(
                f"artifact runtime omitted {field}"
            )
    required_runtime_environment = {
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": str(THREADS),
        "MKL_NUM_THREADS": str(THREADS),
        "OPENBLAS_NUM_THREADS": str(THREADS),
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
        "E2_PROCESS_CONCURRENCY": "1",
        "E2_EXCLUSIVE_NODE": "1",
        "E2_ONE_ATTEMPT": "1",
    }
    if runtime.get("required_environment") != required_runtime_environment:
        raise RuntimeError("artifact runtime environment is not frozen")
    slurm = runtime.get("slurm", {})
    if (
        slurm.get("SLURM_ARRAY_TASK_ID") != str(task_id)
        or not slurm.get("SLURM_ARRAY_JOB_ID")
        or not slurm.get("SLURM_JOB_ID")
        or slurm.get("SLURM_CPUS_PER_TASK") != str(THREADS)
        or slurm.get("SLURM_NTASKS") != "1"
        or slurm.get("SLURM_JOB_NUM_NODES") != "1"
        or slurm.get("SLURM_LOCALID") != "0"
        or slurm.get("SLURM_PROCID") != "0"
        or slurm.get("SLURM_JOB_ACCOUNT") != EXPECTED_SLURM_ACCOUNT
        or slurm.get("SLURM_JOB_PARTITION")
        != EXPECTED_SLURM_PARTITION
        or slurm.get("SLURM_RESTART_COUNT") not in (None, "0")
    ):
        raise RuntimeError("artifact Slurm task mapping/resources mismatch")
    if (
        runtime.get("runtime_contract_sha256")
        != context["runtime_contract_sha256"]
        or runtime.get("runtime_files", {}).get(
            "python_executable_sha256"
        )
        != context["runtime_contract"]["python"]["executable_sha256"]
    ):
        raise RuntimeError("artifact runtime-file lock mismatch")
    evidence = runtime.get("candidate_avx512_evidence", {})
    if (
        not isinstance(evidence.get("zmm_operand_count"), int)
        or evidence["zmm_operand_count"] < 1
        or not set(REQUIRED_AVX512_FLAGS).issubset(
            runtime.get("cpu_flags", [])
        )
    ):
        raise RuntimeError("artifact lacks exact-node AVX-512 evidence")
    selected_cpus = runtime["affinity_before_torch_import"]
    if len(set(selected_cpus)) != THREADS:
        raise RuntimeError("artifact affinity contains duplicate CPUs")
    topology = runtime.get("cpu_topology", [])
    if len(topology) != THREADS:
        raise RuntimeError("artifact CPU topology is incomplete")
    packages = {entry.get("physical_package") for entry in topology}
    cores = {
        (entry.get("physical_package"), entry.get("core_id"))
        for entry in topology
    }
    if len(packages) != 1 or len(cores) != THREADS:
        raise RuntimeError("artifact did not use ten one-socket physical cores")

    dataset = payload.get("dataset", {})
    if dataset.get("sha256") != locked_files[str(DATA_RELATIVE)]:
        raise RuntimeError("artifact data hash mismatch")
    if dataset.get("evaluation_kind") != expected_evaluation_kind:
        raise RuntimeError("dataset evaluation kind is invalid")
    if (
        dataset.get("split_seed") != SPLIT_SEED
        or dataset.get("split_seed_derivation") != SPLIT_DERIVATION
        or dataset.get("split_seed_derivation_sha256")
        != SPLIT_DERIVATION_SHA256
    ):
        raise RuntimeError("artifact used the wrong held-out split")
    index_hashes = dataset.get("index_sha256")
    if (
        not isinstance(index_hashes, dict)
        or set(index_hashes)
        != {
            "development",
            "test",
            "pilot_train",
            "pilot_validation",
        }
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in index_hashes.values()
        )
    ):
        raise RuntimeError("dataset split-index hashes are invalid")
    normalizer_sha = dataset.get("normalizer_sha256")
    if not isinstance(normalizer_sha, str) or re.fullmatch(
        r"[0-9a-f]{64}", normalizer_sha
    ) is None:
        raise RuntimeError("dataset normalizer hash is invalid")
    expected_test_tensor = stage == "accuracy-pair"
    if dataset.get("test_tensor_constructed") is not (
        expected_test_tensor
    ):
        raise RuntimeError("stage constructed an unauthorized test tensor")
    steps_per_epoch = dataset.get("steps_per_epoch")
    train_examples = dataset.get("train_examples")
    if (
        not isinstance(steps_per_epoch, int)
        or steps_per_epoch < 1
        or not isinstance(train_examples, int)
        or steps_per_epoch != train_examples // BATCH_SIZE
    ):
        raise RuntimeError("dataset/drop-last step count is invalid")
    expected_schedule_epochs = (
        THROUGHPUT_MAX_EPOCHS
        if stage == "throughput-pair"
        else EPOCHS
    )
    if (
        payload.get("schedule_epochs") != expected_schedule_epochs
        or not isinstance(
            payload.get("minibatch_schedule_sha256"), str
        )
        or re.fullmatch(
            r"[0-9a-f]{64}",
            payload["minibatch_schedule_sha256"],
        )
        is None
    ):
        raise RuntimeError("minibatch schedule metadata is invalid")
    expected_training = {
        "rank": RANK,
        "batch_size": BATCH_SIZE,
        "drop_last_incomplete_batch": True,
        "learning_rate": LEARNING_RATE,
        "damping": DAMPING,
        "momentum": MOMENTUM,
        "refresh_threshold": REFRESH_THRESHOLD,
        "ours_refactor_every": REFACTOR_EVERY,
    }
    if payload.get("training") != expected_training:
        raise RuntimeError("artifact training configuration is not frozen")
    expected_guardrails = {
        "historical_seed42_outcomes_previously_viewed": True,
        "v2_test_outcomes_read_before_preregistration": False,
        "v2_split_selected_mechanically": True,
        "test_evaluations_total": 2 if stage == "accuracy-pair" else 0,
        "checkpoint_selection": False,
        "pair_complete_before_write": True,
    }
    if payload.get("scientific_guardrails") != expected_guardrails:
        raise RuntimeError("artifact scientific guardrails are invalid")
    preflight = payload.get("exact_node_preflight", {})
    cases = preflight.get("cases", [])
    rank_case = next(
        (case for case in cases if case.get("rank") == RANK),
        None,
    )
    if (
        preflight.get("status") != "PASS"
        or rank_case is None
        or rank_case.get("repeats") != PREFLIGHT_RANK_REPEATS
        or rank_case.get("production_reset", {}).get("after_update")
        != PREFLIGHT_RESET_AFTER
    ):
        raise RuntimeError("artifact exact-node preflight is incomplete")
    claim = payload.get("attempt_claim", {})
    claim_path = Path(claim.get("path", ""))
    if (
        claim.get("attempt_ordinal") != 1
        or not claim_path.is_file()
        or sha256_file(claim_path) != claim.get("sha256")
    ):
        raise RuntimeError("artifact one-attempt claim is invalid")
    claim_payload = read_json(claim_path)
    if (
        claim_payload.get("stage") != stage
        or claim_payload.get("task_id") != task_id
        or claim_payload.get("seed") != seed
        or claim_payload.get("attempt_ordinal") != 1
        or claim_payload.get("slurm_restart_count") != 0
    ):
        raise RuntimeError("artifact attempt claim does not match pair")
    gate = payload.get("manual_pilot_gate")
    if stage == "pilot-pair":
        if gate is not None:
            raise RuntimeError("pilot artifact unexpectedly cites a gate")
    elif (
        not isinstance(gate, dict)
        or re.fullmatch(r"[0-9a-f]{64}", gate.get("sha256", "")) is None
    ):
        raise RuntimeError("confirmation artifact omitted the manual gate")
    if stage in ("pilot-pair", "accuracy-pair"):
        expected_steps = (
            EPOCHS * dataset["steps_per_epoch"]
        )
        if any(
            methods[name].get("selected_steps") != expected_steps
            for name in ("ours", "refact")
        ):
            raise RuntimeError("fixed-epoch artifact has wrong step count")
        if any(
            methods[name].get("attempted_steps") != expected_steps
            for name in ("ours", "refact")
        ):
            raise RuntimeError(
                "fixed-epoch artifact attempted the wrong step count"
            )
        if any(
            methods[name].get("budget_seconds") is not None
            or methods[name].get("budget_crossed") is not False
            or methods[name]["attempted_training_seconds"]
            != methods[name]["selected_training_seconds"]
            for name in ("ours", "refact")
        ):
            raise RuntimeError(
                "fixed-epoch artifact contains budget selection"
            )
    else:
        budgets = {
            methods[name].get("budget_seconds")
            for name in ("ours", "refact")
        }
        if (
            len(budgets) != 1
            or None in budgets
            or payload.get("budget") is None
        ):
            raise RuntimeError("throughput pair has inconsistent budgets")
        if any(
            methods[name].get("budget_crossed") is not True
            or methods[name].get("selected_training_seconds")
            > methods[name].get("budget_seconds")
            for name in ("ours", "refact")
        ):
            raise RuntimeError("invalid budget-selected throughput state")
        for name in ("ours", "refact"):
            method = methods[name]
            if method.get("attempted_steps") != (
                method["selected_steps"] + 1
            ):
                raise RuntimeError(
                    "throughput must attempt exactly one crossing batch"
                )
            attempted_seconds = method.get(
                "attempted_training_seconds"
            )
            if attempted_seconds <= method["budget_seconds"]:
                raise RuntimeError(
                    "throughput crossing attempt did not exceed budget"
                )


def validate_budget_payload(
    payload: dict[str, Any],
    context: dict[str, Any],
) -> None:
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment")
        != "e2_preregistered_budget_calibration"
        or payload.get("campaign") != CAMPAIGN
        or payload.get("status") != "complete"
        or payload.get("pilot_seeds") != list(PILOT_SEEDS)
        or payload.get("test_values_used") is not False
        or payload.get("validation_values_used") is not False
    ):
        raise RuntimeError("invalid frozen budget artifact")
    provenance = payload.get("provenance", {})
    if (
        provenance.get("git_commit") != context["git_commit"]
        or provenance.get("protocol_sha256")
        != context["protocol_sha256"]
        or provenance.get("lock_sha256")
        != context["lock_sha256"]
    ):
        raise RuntimeError("budget provenance mismatch")
    budget = payload.get("budget_seconds")
    if not isinstance(budget, (int, float)) or not (
        math.isfinite(float(budget)) and budget > 0
    ):
        raise RuntimeError("budget must be finite and positive")


def audit_budget_against_pilots(
    results_root: Path,
    payload: dict[str, Any],
    context: dict[str, Any],
) -> None:
    """Recompute every budget input without reading validation outcomes."""

    validate_budget_payload(payload, context)
    rates: list[float] = []
    sources: list[dict[str, Any]] = []
    development_steps: set[int] = set()
    cpu_models: set[str] = set()
    for task_id, seed in enumerate(PILOT_SEEDS):
        path = artifact_path(
            results_root, "pilot-pair", seed
        )
        pilot = read_json(path)
        validate_common_artifact(
            pilot,
            stage="pilot-pair",
            task_id=task_id,
            seed=seed,
            context=context,
        )
        ours = pilot["methods"]["ours"]
        rate = (
            ours["selected_training_seconds"]
            / ours["selected_steps"]
        )
        rates.append(rate)
        development_steps.add(
            pilot["dataset"]["development_steps_per_epoch"]
        )
        cpu_models.add(pilot["runtime"]["cpu_model"])
        sources.append(
            {
                "seed": seed,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "ours_seconds": ours[
                    "selected_training_seconds"
                ],
                "ours_steps": ours["selected_steps"],
                "ours_seconds_per_step": rate,
            }
        )
    if len(development_steps) != 1 or len(cpu_models) != 1:
        raise RuntimeError(
            "pilot artifacts disagree on data size or CPU model"
        )
    target_steps = EPOCHS * next(iter(development_steps))
    median_rate = statistics.median(rates)
    expected_budget = median_rate * target_steps
    exact_fields = {
        "pilot_sources": sources,
        "pilot_ours_seconds_per_step": rates,
        "target_development_steps": target_steps,
        "cpu_model": next(iter(cpu_models)),
    }
    for field, expected in exact_fields.items():
        if payload.get(field) != expected:
            raise RuntimeError(
                f"budget field {field} does not recompute from pilots"
            )
    for field, expected in (
        ("median_ours_seconds_per_step", median_rate),
        ("budget_seconds", expected_budget),
    ):
        actual = payload.get(field)
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual),
            expected,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError(
                f"budget field {field} does not recompute from pilots"
            )


def calibrate_budget(
    results_root: Path,
    context: dict[str, Any],
) -> Path:
    directory = results_root / CAMPAIGN
    seconds_per_step: list[float] = []
    pilot_sources: list[dict[str, Any]] = []
    development_steps: set[int] = set()
    cpu_models: set[str] = set()
    for task_id, seed in enumerate(PILOT_SEEDS):
        path = artifact_path(
            results_root, "pilot-pair", seed
        )
        payload = read_json(path)
        validate_common_artifact(
            payload,
            stage="pilot-pair",
            task_id=task_id,
            seed=seed,
            context=context,
        )
        ours = payload["methods"]["ours"]
        seconds_per_step.append(
            ours["selected_training_seconds"]
            / ours["selected_steps"]
        )
        development_steps.add(
            payload["dataset"]["development_steps_per_epoch"]
        )
        cpu_models.add(payload["runtime"]["cpu_model"])
        pilot_sources.append(
            {
                "seed": seed,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "ours_seconds": ours[
                    "selected_training_seconds"
                ],
                "ours_steps": ours["selected_steps"],
                "ours_seconds_per_step": seconds_per_step[-1],
            }
        )
    if len(development_steps) != 1:
        raise RuntimeError("pilot artifacts disagree on development size")
    if len(cpu_models) != 1:
        raise RuntimeError("pilot artifacts used different CPU models")
    target_steps = EPOCHS * next(iter(development_steps))
    median_seconds_per_step = statistics.median(
        seconds_per_step
    )
    budget_seconds = median_seconds_per_step * target_steps
    payload = {
        "schema_version": 1,
        "experiment": "e2_preregistered_budget_calibration",
        "campaign": CAMPAIGN,
        "status": "complete",
        "calibration_rule": (
            "median pilot Ours application seconds per completed batch "
            "times batches in ten full-development epochs"
        ),
        "pilot_seeds": list(PILOT_SEEDS),
        "pilot_sources": pilot_sources,
        "pilot_ours_seconds_per_step": seconds_per_step,
        "median_ours_seconds_per_step": median_seconds_per_step,
        "target_development_steps": target_steps,
        "budget_seconds": budget_seconds,
        "test_values_used": False,
        "validation_values_used": False,
        "cpu_model": next(iter(cpu_models)),
        "provenance": common_artifact_provenance(context),
    }
    destination = directory / "budget.json"
    write_once(payload, destination)
    return destination


def pilot_gate_path(results_root: Path) -> Path:
    return results_root / CAMPAIGN / "pilot_gate.json"


def create_pilot_gate(
    results_root: Path,
    context: dict[str, Any],
    manual_attestation: str,
) -> Path:
    """Create the immutable human gate after all pilots have been audited."""

    if manual_attestation != MANUAL_GATE_ATTESTATION:
        raise RuntimeError(
            "manual gate attestation is absent or not the frozen sentence"
        )
    budget_path = results_root / CAMPAIGN / "budget.json"
    budget = read_json(budget_path)
    audit_budget_against_pilots(results_root, budget, context)
    pilot_sources: list[dict[str, Any]] = []
    split_signatures: set[str] = set()
    for task_id, seed in enumerate(PILOT_SEEDS):
        path = artifact_path(results_root, "pilot-pair", seed)
        payload = read_json(path)
        validate_common_artifact(
            payload,
            stage="pilot-pair",
            task_id=task_id,
            seed=seed,
            context=context,
        )
        split_signatures.add(
            sha256_json(payload["dataset"]["index_sha256"])
        )
        pilot_sources.append(
            {
                "task_id": task_id,
                "seed": seed,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
        )
    if len(split_signatures) != 1:
        raise RuntimeError("pilots disagree on the frozen split")
    payload = {
        "schema_version": 1,
        "experiment": "e2_preregistered_manual_pilot_gate",
        "campaign": CAMPAIGN,
        "status": "confirmation-authorized",
        "manual_attestation": manual_attestation,
        "manual_operator": os.environ.get("USER"),
        "created_unix_ns": time.time_ns(),
        "pilot_sources": pilot_sources,
        "budget": {
            "path": str(budget_path.resolve()),
            "sha256": sha256_file(budget_path),
            "budget_seconds": budget["budget_seconds"],
        },
        "split_index_signature_sha256": next(
            iter(split_signatures)
        ),
        "provenance": common_artifact_provenance(context),
        "confirmation_may_run": True,
        "outcome_dependent_protocol_change": False,
    }
    destination = pilot_gate_path(results_root)
    write_once(payload, destination)
    return destination


def verify_pilot_gate(
    results_root: Path,
    context: dict[str, Any],
    expected_sha256: str | None,
) -> dict[str, Any]:
    if (
        expected_sha256 is None
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise RuntimeError(
            "confirmation requires the exact manual pilot-gate SHA-256"
        )
    path = pilot_gate_path(results_root)
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("manual pilot-gate hash mismatch")
    payload = read_json(path)
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment")
        != "e2_preregistered_manual_pilot_gate"
        or payload.get("campaign") != CAMPAIGN
        or payload.get("status") != "confirmation-authorized"
        or payload.get("manual_attestation") != MANUAL_GATE_ATTESTATION
        or payload.get("confirmation_may_run") is not True
        or payload.get("outcome_dependent_protocol_change") is not False
    ):
        raise RuntimeError("manual pilot gate is invalid")
    provenance = payload.get("provenance", {})
    if any(
        provenance.get(field) != context[field]
        for field in (
            "git_commit",
            "protocol_sha256",
            "runtime_contract_sha256",
            "lock_sha256",
        )
    ):
        raise RuntimeError("manual pilot-gate provenance mismatch")
    budget_path = results_root / CAMPAIGN / "budget.json"
    budget = read_json(budget_path)
    audit_budget_against_pilots(results_root, budget, context)
    budget_entry = payload.get("budget", {})
    if (
        budget_entry.get("sha256") != sha256_file(budget_path)
        or budget_entry.get("budget_seconds")
        != budget["budget_seconds"]
    ):
        raise RuntimeError("manual gate does not freeze the audited budget")
    expected_sources = []
    for task_id, seed in enumerate(PILOT_SEEDS):
        pilot_path = artifact_path(
            results_root, "pilot-pair", seed
        )
        expected_sources.append(
            {
                "task_id": task_id,
                "seed": seed,
                "path": str(pilot_path.resolve()),
                "sha256": sha256_file(pilot_path),
            }
        )
    if payload.get("pilot_sources") != expected_sources:
        raise RuntimeError("manual gate does not freeze exact pilot bytes")
    return payload


def paired_t_summary(
    values: list[float],
    *,
    confidence: float,
) -> dict[str, Any]:
    if len(values) != 20:
        raise RuntimeError(
            f"paired confirmatory analysis requires 20 values, got {len(values)}"
        )
    if any(not math.isfinite(float(value)) for value in values):
        raise RuntimeError("paired analysis received a non-finite value")
    mean = statistics.mean(values)
    sample_sd = statistics.stdev(values)
    standard_error = sample_sd / math.sqrt(len(values))
    if confidence == 0.95:
        # Student-t quantile t_(0.975,19)
        critical = 2.093024054408263
    elif confidence == 0.90:
        # Student-t quantile t_(0.95,19)
        critical = 1.729132811521367
    else:
        raise ValueError("only frozen 90% and 95% intervals are supported")
    half_width = critical * standard_error
    return {
        "n_pairs": len(values),
        "mean": mean,
        "sample_sd": sample_sd,
        "standard_error": standard_error,
        "confidence": confidence,
        "student_t_df": len(values) - 1,
        "student_t_critical": critical,
        "ci_lower": mean - half_width,
        "ci_upper": mean + half_width,
        "values": values,
    }


def strict_expected_json_names() -> set[str]:
    names = {
        f"pilot_seed{seed}.json" for seed in PILOT_SEEDS
    }
    names.update(
        f"accuracy_seed{seed}.json"
        for seed in ACCURACY_SEEDS
    )
    names.update(
        f"throughput_seed{seed}.json"
        for seed in THROUGHPUT_SEEDS
    )
    names.add("budget.json")
    names.add("pilot_gate.json")
    names.add("preflight.json")
    return names


def aggregate(
    results_root: Path,
    context: dict[str, Any],
    expected_gate_sha256: str | None = None,
) -> Path:
    directory = results_root / CAMPAIGN
    destination = directory / "summary.json"
    if destination.exists():
        raise FileExistsError(
            f"write-once summary already exists: {destination}"
        )
    verify_pilot_gate(
        results_root, context, expected_gate_sha256
    )
    observed_names = {
        path.name for path in directory.glob("*.json")
    }
    expected_names = strict_expected_json_names()
    if observed_names != expected_names:
        raise RuntimeError(
            "strict aggregate requires exactly the preregistered artifacts; "
            f"missing={sorted(expected_names - observed_names)}, "
            f"extra={sorted(observed_names - expected_names)}"
        )
    budget_path = directory / "budget.json"
    budget = read_json(budget_path)
    audit_budget_against_pilots(
        results_root, budget, context
    )

    runtime_signatures: set[str] = set()
    split_index_signatures: set[str] = set()
    pilot_normalizers: set[str] = set()
    confirmation_normalizers: set[str] = set()

    def register_execution_signature(
        stage: str,
        artifact: dict[str, Any],
    ) -> None:
        runtime = artifact["runtime"]
        provenance = artifact["provenance"]
        runtime_signatures.add(
            sha256_json(
                {
                    "cpu_model": runtime["cpu_model"],
                    "platform": runtime["platform"],
                    "python": runtime["python"],
                    "torch": runtime["torch"],
                    "numpy": runtime["numpy"],
                    "scipy": runtime["scipy"],
                    "runtime_contract_sha256": runtime[
                        "runtime_contract_sha256"
                    ],
                    "runtime_files": runtime["runtime_files"],
                    "torch_parallel_info": runtime[
                        "torch_parallel_info"
                    ],
                    "runtime_values": runtime["runtime_values"],
                    "candidate_binary_sha256": provenance[
                        "candidate_binary_sha256"
                    ],
                    "slurm_account": runtime["slurm"][
                        "SLURM_JOB_ACCOUNT"
                    ],
                    "slurm_partition": runtime["slurm"][
                        "SLURM_JOB_PARTITION"
                    ],
                }
            )
        )
        split_index_signatures.add(
            sha256_json(artifact["dataset"]["index_sha256"])
        )
        destination = (
            pilot_normalizers
            if stage == "pilot-pair"
            else confirmation_normalizers
        )
        destination.add(
            artifact["dataset"]["normalizer_sha256"]
        )

    # Pilots are validated for provenance/completeness, but their training
    # and validation outcomes do not enter either confirmation estimate.
    pilot_artifacts = []
    for task_id, seed in enumerate(PILOT_SEEDS):
        path = artifact_path(
            results_root, "pilot-pair", seed
        )
        payload = read_json(path)
        validate_common_artifact(
            payload,
            stage="pilot-pair",
            task_id=task_id,
            seed=seed,
            context=context,
        )
        register_execution_signature("pilot-pair", payload)
        pilot_artifacts.append(
            {"seed": seed, "path": str(path), "sha256": sha256_file(path)}
        )

    accuracy_pairs: list[dict[str, Any]] = []
    accuracy_differences: list[float] = []
    accuracy_cpu_models: set[str] = set()
    for task_id, seed in enumerate(ACCURACY_SEEDS):
        path = artifact_path(
            results_root, "accuracy-pair", seed
        )
        payload = read_json(path)
        validate_common_artifact(
            payload,
            stage="accuracy-pair",
            task_id=task_id,
            seed=seed,
            context=context,
        )
        register_execution_signature("accuracy-pair", payload)
        ours = payload["methods"]["ours"][
            "final_evaluation_mse"
        ]
        refact = payload["methods"]["refact"][
            "final_evaluation_mse"
        ]
        difference = ours - refact
        accuracy_differences.append(difference)
        accuracy_cpu_models.add(payload["runtime"]["cpu_model"])
        accuracy_pairs.append(
            {
                "seed": seed,
                "ours_test_mse": ours,
                "refact_test_mse": refact,
                "ours_minus_refact": difference,
                "ours_refresh_count": payload["methods"]["ours"][
                    "refresh_count"
                ],
                "refact_refresh_count": payload["methods"]["refact"][
                    "refresh_count"
                ],
                "ours_max_reset_drift": payload["methods"]["ours"][
                    "max_reset_drift"
                ],
                "artifact_sha256": sha256_file(path),
            }
        )
    accuracy_95 = paired_t_summary(
        accuracy_differences, confidence=0.95
    )
    accuracy_90 = paired_t_summary(
        accuracy_differences, confidence=0.90
    )
    equivalent = (
        accuracy_90["ci_lower"] > -EQUIVALENCE_MARGIN
        and accuracy_90["ci_upper"] < EQUIVALENCE_MARGIN
    )

    throughput_pairs: list[dict[str, Any]] = []
    throughput_differences: list[float] = []
    throughput_log_ratios: list[float] = []
    throughput_cpu_models: set[str] = set()
    for task_id, seed in enumerate(THROUGHPUT_SEEDS):
        path = artifact_path(
            results_root, "throughput-pair", seed
        )
        payload = read_json(path)
        validate_common_artifact(
            payload,
            stage="throughput-pair",
            task_id=task_id,
            seed=seed,
            context=context,
        )
        register_execution_signature("throughput-pair", payload)
        budget_info = payload.get("budget", {})
        if (
            budget_info.get("sha256") != sha256_file(budget_path)
            or payload["methods"]["ours"]["budget_seconds"]
            != budget["budget_seconds"]
            or payload["methods"]["refact"]["budget_seconds"]
            != budget["budget_seconds"]
        ):
            raise RuntimeError(
                f"throughput seed {seed} used the wrong frozen budget"
            )
        ours_steps = payload["methods"]["ours"]["selected_steps"]
        refact_steps = payload["methods"]["refact"][
            "selected_steps"
        ]
        if ours_steps <= 0 or refact_steps <= 0:
            raise RuntimeError("throughput step ratio is undefined")
        difference = float(ours_steps - refact_steps)
        log_ratio = math.log(ours_steps / refact_steps)
        throughput_differences.append(difference)
        throughput_log_ratios.append(log_ratio)
        throughput_cpu_models.add(payload["runtime"]["cpu_model"])
        throughput_pairs.append(
            {
                "seed": seed,
                "ours_completed_batches": ours_steps,
                "refact_completed_batches": refact_steps,
                "ours_minus_refact_batches": difference,
                "ours_over_refact_batch_ratio": (
                    ours_steps / refact_steps
                ),
                "artifact_sha256": sha256_file(path),
            }
        )
    throughput_difference_summary = paired_t_summary(
        throughput_differences, confidence=0.95
    )
    log_ratio_summary = paired_t_summary(
        throughput_log_ratios, confidence=0.95
    )
    throughput_ratio_summary = {
        "geometric_mean_ours_over_refact": math.exp(
            log_ratio_summary["mean"]
        ),
        "ci_lower": math.exp(log_ratio_summary["ci_lower"]),
        "ci_upper": math.exp(log_ratio_summary["ci_upper"]),
        "log_scale_paired_summary": log_ratio_summary,
    }

    all_cpu_models = (
        accuracy_cpu_models | throughput_cpu_models
    )
    if len(all_cpu_models) != 1:
        raise RuntimeError(
            f"confirmation used mixed CPU models: {all_cpu_models}"
        )
    if budget["cpu_model"] not in all_cpu_models:
        raise RuntimeError(
            "pilot calibration and confirmation CPU model differ"
        )
    if len(runtime_signatures) != 1:
        raise RuntimeError(
            "campaign mixed Torch/Python/platform/parallel/binary runtimes"
        )
    if len(split_index_signatures) != 1:
        raise RuntimeError("campaign mixed dataset split-index hashes")
    if (
        len(pilot_normalizers) != 1
        or len(confirmation_normalizers) != 1
    ):
        raise RuntimeError(
            "campaign mixed preprocessing within a frozen stage"
        )
    payload = {
        "schema_version": 1,
        "experiment": "e2_preregistered_paired_protein_summary",
        "campaign": CAMPAIGN,
        "status": "complete",
        "analysis_frozen_before_confirmation": True,
        "no_optional_stopping": True,
        "pilot_artifacts": pilot_artifacts,
        "budget": {
            "path": str(budget_path),
            "sha256": sha256_file(budget_path),
            "budget_seconds": budget["budget_seconds"],
            "calibration_rule": budget["calibration_rule"],
            "test_values_used": False,
            "validation_values_used": False,
        },
        "accuracy_equivalence": {
            "estimand": "final test MSE(ours) - final test MSE(refact)",
            "rank": RANK,
            "epochs": EPOCHS,
            "pairs": accuracy_pairs,
            "paired_95_percent_ci": accuracy_95,
            "paired_90_percent_ci_for_tost": accuracy_90,
            "equivalence_margin": EQUIVALENCE_MARGIN,
            "equivalent_at_two_one_sided_alpha_0.05": equivalent,
            "test_evaluations_per_method_seed": 1,
            "interpretation": (
                "Equivalence is established only if the complete paired "
                "90% interval lies inside the preregistered margin; failure "
                "does not imply either superiority or inferiority."
            ),
        },
        "time_budget_throughput": {
            "estimand": (
                "completed batches under common frozen budget, "
                "ours minus refact"
            ),
            "rank": RANK,
            "pairs": throughput_pairs,
            "paired_batch_difference_95_percent_ci": (
                throughput_difference_summary
            ),
            "paired_batch_ratio_95_percent_ci": (
                throughput_ratio_summary
            ),
            "test_evaluations_per_method_seed": 0,
        },
        "resource_contract": {
            "cpu_model": next(iter(all_cpu_models)),
            "threads": THREADS,
            "interop_threads": INTEROP_THREADS,
            "experimental_processes_per_node": 1,
            "exclusive_node": True,
            "methods_sequential_within_pair": True,
            "runtime_signature_sha256": next(
                iter(runtime_signatures)
            ),
            "split_index_signature_sha256": next(
                iter(split_index_signatures)
            ),
            "pilot_normalizer_sha256": next(
                iter(pilot_normalizers)
            ),
            "confirmation_normalizer_sha256": next(
                iter(confirmation_normalizers)
            ),
        },
        "provenance": common_artifact_provenance(context),
    }
    write_once(payload, destination)
    return destination


def synthetic_self_test() -> dict[str, Any]:
    """Data-free checks, including 64 production-rank cumulative updates."""

    tolerances = {"L": 5e-6, "Aw": 5e-5, "vz": 5e-5}
    cases: list[dict[str, Any]] = []
    for rank, width, repeats, seed in (
        (17, 7, 1, 88001),
        (RANK, 32, PREFLIGHT_RANK_REPEATS, 88002),
    ):
        generator = torch.Generator().manual_seed(seed)
        B = (
            torch.randn(
                rank,
                width,
                generator=generator,
                dtype=torch.float32,
            )
            * 0.05
        )
        identity = torch.eye(rank, dtype=torch.float32)
        G = (identity + B @ B.mT).contiguous()
        M = (G - identity).contiguous()
        L_candidate = torch.linalg.cholesky(G).contiguous()
        Aw_current = whiten(L_candidate, M)
        Aw_spare = torch.empty_like(Aw_current)
        vz_candidate = torch.empty(rank, dtype=torch.float32)
        workspace = torch.empty(4 * rank, dtype=torch.float32)
        maximum_errors = {"L": 0.0, "Aw": 0.0, "vz": 0.0}
        reset_error = None
        for update_index in range(1, repeats + 1):
            z = (
                torch.randn(
                    rank,
                    generator=generator,
                    dtype=torch.float32,
                )
                * 0.01
            ).contiguous()
            candidate_extension.positive_metric_operator_same_vector_out(
                L_candidate,
                Aw_current,
                z,
                Aw_spare,
                vz_candidate,
                workspace,
            )
            G.add_(torch.outer(z, z))
            M.add_(torch.outer(z, z))
            L_reference = torch.linalg.cholesky(G).contiguous()
            Aw_reference = whiten(L_reference, M)
            vz_reference = torch.linalg.solve_triangular(
                L_reference, z.unsqueeze(1), upper=False
            ).squeeze(1)
            errors = {
                "L": relative_error(
                    L_candidate, L_reference
                ),
                "Aw": relative_error(Aw_spare, Aw_reference),
                "vz": relative_error(
                    vz_candidate, vz_reference
                ),
            }
            for name, value in errors.items():
                maximum_errors[name] = max(
                    maximum_errors[name], value
                )
            Aw_current, Aw_spare = Aw_spare, Aw_current
            if (
                rank == RANK
                and update_index == PREFLIGHT_RESET_AFTER
            ):
                reset_error = {
                    "after_update": update_index,
                    "L_relative_error": errors["L"],
                    "Aw_relative_error": errors["Aw"],
                }
                L_candidate.copy_(L_reference)
                Aw_current.copy_(Aw_reference)
        if any(
            maximum_errors[name] > tolerances[name]
            for name in maximum_errors
        ):
            raise RuntimeError(
                f"same-vector rank-{rank} repeated self-test failed: "
                f"{maximum_errors}"
            )
        cases.append(
            {
                "rank": rank,
                "width": width,
                "repeats": repeats,
                "seed": seed,
                "maximum_relative_errors": maximum_errors,
                "production_reset": reset_error,
            }
        )
    if cases[-1]["repeats"] < 50:
        raise RuntimeError("rank-1024 preflight performed fewer than 50 checks")
    return {
        "status": "PASS",
        "dtype": "float32",
        "tolerances": tolerances,
        "rank1024_required_repeats": PREFLIGHT_RANK_REPEATS,
        "rank1024_reset_after": PREFLIGHT_RESET_AFTER,
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        required=True,
        choices=(
            "verify-lock",
            "self-test",
            "pilot-pair",
            "calibrate-budget",
            "record-pilot-gate",
            "verify-pilot-gate",
            "accuracy-pair",
            "throughput-pair",
            "aggregate",
        ),
    )
    parser.add_argument("--task-id", type=int)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS,
    )
    parser.add_argument(
        "--expected-lock-sha256",
        required=True,
    )
    parser.add_argument("--locked-commit", required=True)
    parser.add_argument("--expected-gate-sha256")
    parser.add_argument("--manual-attestation")
    args = parser.parse_args()
    if args.action in STAGE_SEEDS and args.task_id is None:
        parser.error(
            f"--task-id is required for {args.action}"
        )
    if args.action not in STAGE_SEEDS and args.task_id is not None:
        parser.error(
            f"--task-id is not permitted for {args.action}"
        )
    gate_required = {
        "verify-pilot-gate",
        "accuracy-pair",
        "throughput-pair",
        "aggregate",
    }
    if (
        args.action in gate_required
        and args.expected_gate_sha256 is None
    ):
        parser.error(
            f"--expected-gate-sha256 is required for {args.action}"
        )
    if (
        args.action not in gate_required
        and args.expected_gate_sha256 is not None
    ):
        parser.error(
            f"--expected-gate-sha256 is not permitted for {args.action}"
        )
    if (
        args.action == "record-pilot-gate"
        and args.manual_attestation is None
    ):
        parser.error("--manual-attestation is required to record the gate")
    if (
        args.action != "record-pilot-gate"
        and args.manual_attestation is not None
    ):
        parser.error(
            f"--manual-attestation is not permitted for {args.action}"
        )
    return args


def main() -> None:
    args = parse_args()
    context = provenance_context(
        args.expected_lock_sha256,
        args.locked_commit,
    )
    results_root = args.results_root.resolve()
    if args.action == "verify-lock":
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "campaign": CAMPAIGN,
                    "git_commit": context["git_commit"],
                    "protocol_sha256": context[
                        "protocol_sha256"
                    ],
                    "lock_sha256": context["lock_sha256"],
                },
                sort_keys=True,
            )
        )
    elif args.action == "self-test":
        claim_path = claim_compute_attempt(
            results_root, "self-test", None, context
        )
        runtime = configure_runtime()
        preflight = synthetic_self_test()
        finalize_runtime(runtime)
        destination = (
            results_root / CAMPAIGN / "preflight.json"
        )
        write_once(
            {
                "schema_version": 1,
                "experiment": "e2_preregistered_exact_node_preflight",
                "campaign": CAMPAIGN,
                "status": "PASS",
                "preflight": preflight,
                "runtime": runtime,
                "attempt_claim": {
                    "path": str(claim_path.resolve()),
                    "sha256": sha256_file(claim_path),
                    "attempt_ordinal": 1,
                },
                "provenance": common_artifact_provenance(context),
            },
            destination,
        )
        print(destination)
    elif args.action in STAGE_SEEDS:
        assert args.task_id is not None
        print(
            run_pair(
                args.action,
                args.task_id,
                results_root,
                context,
                args.expected_gate_sha256,
            )
        )
    elif args.action == "calibrate-budget":
        print(calibrate_budget(results_root, context))
    elif args.action == "record-pilot-gate":
        assert args.manual_attestation is not None
        print(
            create_pilot_gate(
                results_root,
                context,
                args.manual_attestation,
            )
        )
    elif args.action == "verify-pilot-gate":
        gate = verify_pilot_gate(
            results_root,
            context,
            args.expected_gate_sha256,
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "gate_sha256": args.expected_gate_sha256,
                    "created_unix_ns": gate["created_unix_ns"],
                },
                sort_keys=True,
            )
        )
    else:
        print(
            aggregate(
                results_root,
                context,
                args.expected_gate_sha256,
            )
        )


if __name__ == "__main__":
    main()
