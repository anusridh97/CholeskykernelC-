"""Source-locked post-hoc E4 numerical-safety follow-up.

This campaign does not replace ``e4-direct-confirm-v1``.  It evaluates a
fixed randomized residual monitor and a conservative precision-promotion /
refresh policy on new confirmation seeds.  Dispatch receives only the probe
estimate; the exact residual is recorded for evaluation and is never an input
to the policy.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, TextIO

import torch

from .kernel_stability import make_near_duplicate_kernel, rbf


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = Path(__file__).with_name("e4_safety_followup_protocol.json")
DEVELOPMENT_INVENTORY_PATH = Path(__file__).with_name(
    "e4_safety_development_inventory.json"
)
CAMPAIGN_TAG = "e4-safety-followup-v1"
CORRECTED_DIR = ROOT / "Cholesky-main-2" / "corrected_incremental_cholesky"
IMPLEMENTATION_SOURCE = CORRECTED_DIR / "fast_update.cpp"
SETUP_SOURCE = CORRECTED_DIR / "setup_ext.py"
ORIGINAL_AUDIT = (
    ROOT
    / "rebuttal_results"
    / "e4_direct_confirmation"
    / "e4-direct-confirm-v1"
    / "audit_summary.json"
)

LOCKED_FILES = {
    "runner": "rebuttal_experiments/e4_safety_followup.py",
    "audit_wrapper": "rebuttal_experiments/audit_e4_safety_followup.py",
    "protocol": "rebuttal_experiments/e4_safety_followup_protocol.json",
    "development_inventory": (
        "rebuttal_experiments/e4_safety_development_inventory.json"
    ),
    "kernel_generator": "rebuttal_experiments/kernel_stability.py",
    "common_helpers": "rebuttal_experiments/common.py",
    "implementation_source": (
        "Cholesky-main-2/corrected_incremental_cholesky/fast_update.cpp"
    ),
    "extension_build": (
        "Cholesky-main-2/corrected_incremental_cholesky/setup_ext.py"
    ),
    "tests": "rebuttal_experiments/tests/test_e4_safety_followup.py",
    "freeze_sbatch": (
        "rebuttal_experiments/slurm/e4_safety_followup_freeze.sbatch"
    ),
    "preflight_sbatch": (
        "rebuttal_experiments/slurm/e4_safety_followup_preflight.sbatch"
    ),
    "confirmation_sbatch": (
        "rebuttal_experiments/slurm/e4_safety_followup_confirmation.sbatch"
    ),
    "original_e4_audit": (
        "rebuttal_results/e4_direct_confirmation/e4-direct-confirm-v1/"
        "audit_summary.json"
    ),
}

EXPECTED_ACCOUNT = "marlowe-m000151-pm06"
EXPECTED_PARTITION = "batch"
EXPECTED_QOS = "medium"
THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def tensor_sequence_sha256(tensors: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    digest.update(str(len(tensors)).encode("ascii"))
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        raw = value.numpy().tobytes(order="C")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(str(len(raw)).encode("ascii"))
        digest.update(raw)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def write_once_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite E4 artifact {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git",) + arguments,
        cwd=ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def require_tracked_clean(paths: list[str]) -> None:
    for item in paths:
        tracked = git("ls-files", "--error-unmatch", "--", item, check=False)
        if tracked.returncode != 0:
            raise RuntimeError(f"source-lock input is not tracked: {item}")
    changed = git("diff", "--quiet", "HEAD", "--", *paths, check=False)
    if changed.returncode != 0:
        raise RuntimeError("source-lock inputs have staged or unstaged changes")


def validate_static_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = read_json(PROTOCOL_PATH)
    development = read_json(DEVELOPMENT_INVENTORY_PATH)
    if protocol.get("campaign_tag") != CAMPAIGN_TAG:
        raise ValueError("E4 safety campaign tag differs from frozen value")
    if protocol.get("status") != "frozen_before_confirmation":
        raise ValueError("E4 safety protocol is not frozen")
    if protocol.get("label") != (
        "Post-hoc safety follow-up; does not replace or alter "
        "e4-direct-confirm-v1."
    ):
        raise ValueError("post-hoc scope label changed")
    problem = protocol["problem"]
    if problem["confirmation_seeds"] != list(range(46001, 46021)):
        raise ValueError("confirmation seeds changed")
    guard = protocol["guard"]
    expected_guard = {
        "probe_count": 8,
        "probe_distribution": "Rademacher {-1,+1}",
        "probe_seed_rule": "seed + 99001",
        "probe_reuse": (
            "One fixed rank-by-8 probe matrix per problem seed, paired across "
            "operators, deltas, and starting precisions."
        ),
        "estimated_residual": "||[L Aw L^T - M] Q||_F / ||M Q||_F",
        "evaluation_frequency": (
            "After initialization and after every update, before any guard action."
        ),
        "safety_factor": 2.0,
        "float32_limit": 0.0005,
        "float32_trigger": 0.00025,
        "float64_limit": 1e-10,
        "float64_trigger": 5e-11,
        "float32_action": (
            "Promote L and Aw to FP64 by exact Cholesky factorization and "
            "complete rewhitening from authoritative G and M; never demote."
        ),
        "float64_action": (
            "Exact FP64 Cholesky factorization and complete rewhitening from "
            "authoritative G and M."
        ),
        "exact_residual_dispatch_forbidden": True,
    }
    if guard != expected_guard:
        raise ValueError("guard differs from reviewed frozen policy")
    if protocol["original_e4_status_must_remain"] != "criteria_failed":
        raise ValueError("original E4 outcome protection changed")
    original = read_json(ORIGINAL_AUDIT)
    if original.get("scientific_status") != "criteria_failed":
        raise ValueError("original E4 criteria_failed status was altered")
    if development.get("status") != "closed_before_confirmation":
        raise ValueError("development inventory was not closed")
    if development["selection"]["selected_safety_factor"] != 2.0:
        raise ValueError("development-selected safety factor changed")
    future = development["future_confirmation"]
    if future["seeds"] != problem["confirmation_seeds"]:
        raise ValueError("development/confirmation seed declaration mismatch")
    if future["outcomes_inspected"] is not False:
        raise ValueError("inventory says confirmation outcomes were inspected")
    if future["artifacts_created"] is not False:
        raise ValueError("inventory says confirmation artifacts already existed")
    development_seeds = {
        *development["selection"]["seeds"],
        *development["final_policy_development_rerun"]["check_seeds"],
        *development["retrospective_nonconfirmatory_check"]["source_seeds"],
    }
    if development_seeds.intersection(problem["confirmation_seeds"]):
        raise ValueError("development and confirmation seed sets overlap")
    return protocol, development


def artifact_path(protocol: dict[str, Any], name: str) -> Path:
    return ROOT / protocol["paths"][name]


def confirmation_artifact_attestation(
    protocol: dict[str, Any],
) -> dict[str, Any]:
    names = ("preflight", "attempt", "journal", "result", "audit")
    existing = [
        relative(artifact_path(protocol, name))
        for name in names
        if artifact_path(protocol, name).exists()
    ]
    return {
        "checked_utc": utc_now(),
        "confirmation_seeds": protocol["problem"]["confirmation_seeds"],
        "checked_paths": [
            relative(artifact_path(protocol, name)) for name in names
        ],
        "existing_paths": existing,
        "no_confirmation_science_artifacts": not any(
            artifact_path(protocol, name).exists()
            for name in ("attempt", "journal", "result", "audit")
        ),
    }


def freeze_lock() -> tuple[dict[str, Any], Path]:
    protocol, development = validate_static_inputs()
    relative_paths = list(LOCKED_FILES.values())
    require_tracked_clean(relative_paths)
    environment = validate_runtime(require_slurm=True)
    attestation = confirmation_artifact_attestation(protocol)
    if attestation["existing_paths"]:
        raise RuntimeError(
            "freeze requires an untouched campaign directory; found "
            f"{attestation['existing_paths']}"
        )
    head = git("rev-parse", "HEAD").stdout.strip()
    manifest: dict[str, Any] = {}
    for key, item in LOCKED_FILES.items():
        path = ROOT / item
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest[key] = {
            "path": item,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    payload = {
        "schema_version": 1,
        "experiment": "e4_safety_followup_source_lock",
        "campaign_tag": CAMPAIGN_TAG,
        "status": "frozen_before_confirmation",
        "created_utc": utc_now(),
        "git_commit": head,
        "freeze_environment": environment,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "development_inventory_sha256": sha256(DEVELOPMENT_INVENTORY_PATH),
        "development_seed_count": (
            len(development["selection"]["seeds"])
            + len(
                development["final_policy_development_rerun"]["check_seeds"]
            )
        ),
        "confirmation_seed_attestation": attestation,
        "files": manifest,
    }
    path = artifact_path(protocol, "source_lock")
    write_once_json(payload, path)
    print(f"wrote immutable E4 safety source lock {path}", flush=True)
    return payload, path


def verify_source_lock() -> tuple[dict[str, Any], str, str]:
    protocol, _ = validate_static_inputs()
    path = artifact_path(protocol, "source_lock")
    if not path.is_file():
        raise FileNotFoundError("E4 safety source lock is missing")
    lock = read_json(path)
    if lock.get("campaign_tag") != CAMPAIGN_TAG:
        raise RuntimeError("source-lock campaign mismatch")
    if lock.get("protocol_sha256") != sha256(PROTOCOL_PATH):
        raise RuntimeError("protocol differs from source lock")
    if lock.get("development_inventory_sha256") != sha256(
        DEVELOPMENT_INVENTORY_PATH
    ):
        raise RuntimeError("development inventory differs from source lock")
    locked_commit = lock.get("git_commit")
    current_head = git("rev-parse", "HEAD").stdout.strip()
    if not isinstance(locked_commit, str):
        raise RuntimeError("source lock has no git commit")
    if (
        git(
            "merge-base",
            "--is-ancestor",
            locked_commit,
            current_head,
            check=False,
        ).returncode
        != 0
    ):
        raise RuntimeError("source-lock commit is not an ancestor of run HEAD")
    relative_lock = relative(path)
    require_tracked_clean([*LOCKED_FILES.values(), relative_lock])
    manifest = lock.get("files")
    if not isinstance(manifest, dict) or set(manifest) != set(LOCKED_FILES):
        raise RuntimeError("source-lock manifest keys differ")
    for key, item in LOCKED_FILES.items():
        record = manifest[key]
        locked_path = ROOT / item
        if record.get("path") != item:
            raise RuntimeError(f"source-lock path mismatch for {key}")
        if record.get("bytes") != locked_path.stat().st_size:
            raise RuntimeError(f"source-lock size mismatch for {item}")
        if record.get("sha256") != sha256(locked_path):
            raise RuntimeError(f"source-lock hash mismatch for {item}")
    attestation = lock.get("confirmation_seed_attestation", {})
    if attestation.get("no_confirmation_science_artifacts") is not True:
        raise RuntimeError("source lock lacks untouched-confirmation attestation")
    if attestation.get("confirmation_seeds") != list(range(46001, 46021)):
        raise RuntimeError("source-lock seed attestation mismatch")
    return lock, sha256(path), current_head


def configure_threads() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise


def validate_runtime(*, require_slurm: bool = True) -> dict[str, Any]:
    required = {
        **{name: "1" for name in THREAD_ENVIRONMENT},
        "PYTHONNOUSERSITE": "1",
    }
    invalid = {
        name: os.environ.get(name)
        for name, expected in required.items()
        if os.environ.get(name) != expected
    }
    if invalid:
        raise RuntimeError(f"non-hermetic E4 safety environment: {invalid}")
    configure_threads()
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("PyTorch is not single threaded")
    affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else []
    )
    if require_slurm:
        expected = {
            "SLURM_JOB_ACCOUNT": EXPECTED_ACCOUNT,
            "SLURM_JOB_PARTITION": EXPECTED_PARTITION,
            "SLURM_JOB_QOS": EXPECTED_QOS,
            "SLURM_CPUS_PER_TASK": "1",
        }
        bad_slurm = {
            name: os.environ.get(name)
            for name, value in expected.items()
            if os.environ.get(name) != value
        }
        if bad_slurm:
            raise RuntimeError(f"Slurm resource contract mismatch: {bad_slurm}")
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Slurm job ID is missing")
        if len(affinity) != 1:
            raise RuntimeError(f"expected one physical CPU, got affinity {affinity}")
    source_commit = os.environ.get("SOURCE_COMMIT")
    current_head = git("rev-parse", "HEAD").stdout.strip()
    if require_slurm and source_commit != current_head:
        raise RuntimeError(
            f"source commit mismatch: declared={source_commit!r}, "
            f"current={current_head!r}"
        )
    if git("diff", "--quiet", check=False).returncode != 0:
        raise RuntimeError("tracked worktree differs from committed source")
    if git("diff", "--cached", "--quiet", check=False).returncode != 0:
        raise RuntimeError("index differs from committed source")
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "process_affinity": affinity,
        "thread_environment": {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT
        },
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
        "source_commit": current_head,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_job_qos": os.environ.get("SLURM_JOB_QOS"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST"),
    }


def import_extension():
    configured = os.environ.get("E4_SAFETY_EXTENSION_PATH")
    if not configured:
        raise RuntimeError("E4_SAFETY_EXTENSION_PATH is missing")
    configured_path = Path(configured).resolve()
    if not configured_path.is_file():
        raise FileNotFoundError(configured_path)
    if str(configured_path.parent) not in sys.path:
        sys.path.insert(0, str(configured_path.parent))
    import fast_cholesky

    imported = Path(fast_cholesky.__file__).resolve()
    if imported != configured_path:
        raise RuntimeError(
            f"wrong extension imported: expected {configured_path}, got {imported}"
        )
    return fast_cholesky, configured_path


def whiten(L: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    left = torch.linalg.solve_triangular(L, M, upper=False)
    return torch.linalg.solve_triangular(
        L, left.mT, upper=False
    ).mT.contiguous()


def exact_residual(
    L: torch.Tensor, Aw: torch.Tensor, M_authoritative: torch.Tensor
) -> float:
    reconstructed = L.double() @ Aw.double() @ L.double().mT
    denominator = torch.linalg.matrix_norm(M_authoritative)
    if denominator.item() == 0:
        raise ValueError("operator norm is zero")
    return float(
        (torch.linalg.matrix_norm(reconstructed - M_authoritative) / denominator)
        .item()
    )


def factor_residual(
    L: torch.Tensor, G_authoritative: torch.Tensor
) -> float:
    denominator = torch.linalg.matrix_norm(G_authoritative)
    return float(
        (
            torch.linalg.matrix_norm(
                L.double() @ L.double().mT - G_authoritative
            )
            / denominator
        ).item()
    )


def probe_residual(
    L: torch.Tensor,
    Aw: torch.Tensor,
    M_authoritative: torch.Tensor,
    probes: torch.Tensor,
) -> float:
    right = L.double().mT @ probes
    residual_images = (
        L.double() @ (Aw.double() @ right) - M_authoritative @ probes
    )
    denominator = torch.linalg.vector_norm(M_authoritative @ probes)
    if denominator.item() == 0:
        raise ValueError("probe denominator is zero")
    return float((torch.linalg.vector_norm(residual_images) / denominator).item())


def precision_limit(active_precision: str) -> float:
    if active_precision == "float32":
        return 5e-4
    if active_precision == "float64":
        return 1e-10
    raise ValueError(active_precision)


def precision_trigger(active_precision: str) -> float:
    return precision_limit(active_precision) / 2.0


def decide_guard(active_precision: str, observed_probe_residual: float) -> str:
    """Return the frozen action using only precision and probe residual."""
    if not math.isfinite(observed_probe_residual):
        raise ValueError("nonfinite probe residual")
    if observed_probe_residual <= precision_trigger(active_precision):
        return "none"
    if active_precision == "float32":
        return "promote_fp64"
    if active_precision == "float64":
        return "refresh_fp64"
    raise ValueError(active_precision)


def apply_guard(
    *,
    action: str,
    L: torch.Tensor,
    Aw: torch.Tensor,
    G_authoritative: torch.Tensor,
    M_authoritative: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    if action == "none":
        return L, Aw, "float64" if L.dtype == torch.float64 else "float32"
    if action not in ("promote_fp64", "refresh_fp64"):
        raise ValueError(action)
    if action == "promote_fp64" and L.dtype != torch.float32:
        raise ValueError("promotion requires an FP32 state")
    if action == "refresh_fp64" and L.dtype != torch.float64:
        raise ValueError("FP64 refresh requires an FP64 state")
    L_new = torch.linalg.cholesky(G_authoritative).contiguous()
    Aw_new = whiten(L_new, M_authoritative)
    return L_new, Aw_new, "float64"


def make_probes(rank: int, seed: int, count: int = 8) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + 99001)
    return (
        2
        * torch.randint(
            0,
            2,
            (rank, count),
            dtype=torch.int64,
            generator=generator,
        )
        - 1
    ).double()


def self_test(fast_cholesky) -> dict[str, Any]:
    configure_threads()
    rank = 16
    generator = torch.Generator().manual_seed(9917)
    probes = make_probes(rank, 9917)
    outcomes: dict[str, Any] = {}
    for precision in ("float32", "float64"):
        dtype = torch.float32 if precision == "float32" else torch.float64
        G = torch.eye(rank, dtype=torch.float64)
        M = torch.randn(
            rank, rank, dtype=torch.float64, generator=generator
        ).to(dtype).double()
        L = torch.linalg.cholesky(G.to(dtype)).contiguous()
        Aw = whiten(L, M.to(dtype))
        baseline_probe = probe_residual(L, Aw, M, probes)
        if decide_guard(precision, baseline_probe) != "none":
            raise AssertionError("clean identity state unexpectedly triggered")
        Aw_bad = Aw.clone()
        Aw_bad[0, 0] += 1.0
        bad_probe = probe_residual(L, Aw_bad, M, probes)
        action = decide_guard(precision, bad_probe)
        expected = "promote_fp64" if precision == "float32" else "refresh_fp64"
        if action != expected:
            raise AssertionError(f"expected {expected}, got {action}")
        L_fixed, Aw_fixed, active = apply_guard(
            action=action,
            L=L,
            Aw=Aw_bad,
            G_authoritative=G,
            M_authoritative=M,
        )
        post = exact_residual(L_fixed, Aw_fixed, M)
        if active != "float64" or post > 1e-10:
            raise AssertionError("guard failed to restore FP64 state")
        z = torch.randn(rank, dtype=torch.float64, generator=generator) * 1e-2
        z_active = z.to(L_fixed.dtype)
        Aw_next, _ = fast_cholesky.fast_incremental_update(
            L_fixed,
            Aw_fixed,
            z_active.clone(),
            z_active,
            1,
            1,
            1e-8,
        )
        G_next = G + torch.outer(z, z)
        M_next = M + torch.outer(z, z)
        if exact_residual(L_fixed, Aw_next, M_next) > 1e-10:
            raise AssertionError("extension update failed self-test")
        outcomes[precision] = {
            "clean_probe_residual": baseline_probe,
            "injected_probe_residual": bad_probe,
            "action": action,
            "post_exact_residual": post,
        }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "once.json"
        write_once_json({"value": 1}, path)
        try:
            write_once_json({"value": 2}, path)
        except FileExistsError:
            pass
        else:
            raise AssertionError("write-once publication allowed overwrite")
    return outcomes


def _probe_and_exact(
    L: torch.Tensor,
    Aw: torch.Tensor,
    G_authoritative: torch.Tensor,
    M_authoritative: torch.Tensor,
    probes: torch.Tensor,
) -> dict[str, float]:
    values = {
        "probe_operator_residual": probe_residual(
            L, Aw, M_authoritative, probes
        ),
        "exact_operator_residual": exact_residual(L, Aw, M_authoritative),
        "exact_factor_residual": factor_residual(L, G_authoritative),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError(f"nonfinite diagnostic: {values}")
    return values


def run_trajectory(
    *,
    protocol: dict[str, Any],
    fast_cholesky,
    starting_precision: str,
    operator: str,
    delta: float,
    seed: int,
) -> dict[str, Any]:
    spec = protocol["problem"]
    dtype = (
        torch.float32 if starting_precision == "float32" else torch.float64
    )
    records: list[dict[str, Any]] = []
    failure = None
    failure_step = None
    fingerprints = None
    status = "ok"
    current_step = 0
    try:
        G_original, points = make_near_duplicate_kernel(
            spec["rank"], delta, spec["jitter"], seed
        )
        operator_generator = torch.Generator().manual_seed(seed + 7001)
        update_generator = torch.Generator().manual_seed(seed + 8001)
        if operator == "cumulative":
            M_original = G_original - spec["jitter"] * torch.eye(
                spec["rank"], dtype=torch.float64
            )
        elif operator == "random":
            raw = torch.randn(
                spec["rank"],
                spec["rank"],
                dtype=torch.float64,
                generator=operator_generator,
            )
            M_original = raw / torch.linalg.matrix_norm(raw, ord=2)
        else:
            raise ValueError(operator)
        updates_original: list[torch.Tensor] = []
        for _ in range(spec["steps"]):
            x_new = torch.rand(
                1, 3, dtype=torch.float64, generator=update_generator
            )
            updates_original.append(
                (
                    spec["update_scale"]
                    * rbf(points, x_new, lengthscale=0.4).squeeze(1)
                ).contiguous()
            )
        probes = make_probes(spec["rank"], seed, protocol["guard"]["probe_count"])
        fingerprints = {
            "base_metric_original_fp64_sha256": tensor_sha256(G_original),
            "base_operator_original_fp64_sha256": tensor_sha256(M_original),
            "update_sequence_original_fp64_sha256": tensor_sequence_sha256(
                updates_original
            ),
            "probe_sha256": tensor_sha256(probes),
        }

        # This cast is the information boundary.  All authoritative updates
        # below use the values represented by the starting-precision stream.
        G_authoritative = G_original.to(dtype).double()
        M_authoritative = M_original.to(dtype).double()
        updates_quantized = [value.to(dtype).double() for value in updates_original]
        fingerprints.update(
            {
                "base_metric_authoritative_sha256": tensor_sha256(
                    G_authoritative
                ),
                "base_operator_authoritative_sha256": tensor_sha256(
                    M_authoritative
                ),
                "update_sequence_quantized_sha256": tensor_sequence_sha256(
                    updates_quantized
                ),
            }
        )
        L = torch.linalg.cholesky(G_authoritative.to(dtype)).contiguous()
        Aw = whiten(L, M_authoritative.to(dtype))
        active_precision = starting_precision

        for step in range(spec["steps"] + 1):
            current_step = step
            pre = _probe_and_exact(
                L, Aw, G_authoritative, M_authoritative, probes
            )
            action = decide_guard(
                active_precision, pre["probe_operator_residual"]
            )
            limit_before = precision_limit(active_precision)
            trigger_before = precision_trigger(active_precision)
            false_negative = (
                pre["exact_operator_residual"] > limit_before
                and pre["probe_operator_residual"] <= trigger_before
            )
            L, Aw, precision_after = apply_guard(
                action=action,
                L=L,
                Aw=Aw,
                G_authoritative=G_authoritative,
                M_authoritative=M_authoritative,
            )
            post = (
                pre
                if action == "none"
                else _probe_and_exact(
                    L, Aw, G_authoritative, M_authoritative, probes
                )
            )
            records.append(
                {
                    "step": step,
                    "active_precision_before": active_precision,
                    "active_precision_after": precision_after,
                    "active_limit_before": limit_before,
                    "active_trigger_before": trigger_before,
                    "action": action,
                    "false_negative": false_negative,
                    "pre_guard": pre,
                    "post_guard": post,
                }
            )
            active_precision = precision_after
            if step == spec["steps"]:
                break
            z_quantized = updates_quantized[step]
            z_active = z_quantized.to(L.dtype)
            Aw, _ = fast_cholesky.fast_incremental_update(
                L,
                Aw,
                z_active.clone(),
                z_active,
                1,
                1,
                1e-8,
            )
            G_authoritative = G_authoritative + torch.outer(
                z_quantized, z_quantized
            )
            M_authoritative = M_authoritative + torch.outer(
                z_quantized, z_quantized
            )
        fingerprints.update(
            {
                "final_metric_authoritative_sha256": tensor_sha256(
                    G_authoritative
                ),
                "final_operator_authoritative_sha256": tensor_sha256(
                    M_authoritative
                ),
            }
        )
    except Exception as error:
        status = "failed"
        failure_step = current_step
        failure = f"{type(error).__name__}: {error}"
    return {
        "starting_precision": starting_precision,
        "operator": operator,
        "delta": delta,
        "seed": seed,
        "status": status,
        "failure_step": failure_step,
        "failure": failure,
        "input_fingerprints": fingerprints,
        "records": records,
    }


def append_journal(handle: TextIO, row: dict[str, Any]) -> None:
    handle.write(
        json.dumps(
            {"kind": "trajectory", "row": row},
            sort_keys=True,
            allow_nan=False,
        )
    )
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())


def verify_preflight(protocol: dict[str, Any], lock_sha: str) -> tuple[dict[str, Any], str]:
    path = artifact_path(protocol, "preflight")
    if not path.is_file():
        raise FileNotFoundError("preflight receipt is missing")
    tracked = git("ls-files", "--error-unmatch", "--", relative(path), check=False)
    if tracked.returncode != 0:
        raise RuntimeError("preflight receipt must be committed before confirmation")
    require_tracked_clean([relative(path)])
    payload = read_json(path)
    if payload.get("status") != "passed":
        raise RuntimeError("preflight did not pass")
    if payload.get("source_lock_sha256") != lock_sha:
        raise RuntimeError("preflight refers to a different source lock")
    return payload, sha256(path)


def run_preflight() -> tuple[dict[str, Any], Path]:
    protocol, _ = validate_static_inputs()
    _, lock_sha, head = verify_source_lock()
    environment = validate_runtime(require_slurm=True)
    fast_cholesky, extension = import_extension()
    tests = self_test(fast_cholesky)
    if artifact_path(protocol, "attempt").exists():
        raise RuntimeError("preflight found a confirmation attempt artifact")
    payload = {
        "schema_version": 1,
        "experiment": "e4_safety_followup_preflight",
        "campaign_tag": CAMPAIGN_TAG,
        "status": "passed",
        "created_utc": utc_now(),
        "source_lock_sha256": lock_sha,
        "source_commit": head,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "development_inventory_sha256": sha256(DEVELOPMENT_INVENTORY_PATH),
        "implementation_source_sha256": sha256(IMPLEMENTATION_SOURCE),
        "implementation_extension": str(extension),
        "implementation_extension_sha256": sha256(extension),
        "environment": environment,
        "self_tests": tests,
        "confirmation_artifacts_before_preflight": (
            confirmation_artifact_attestation(protocol)
        ),
    }
    path = artifact_path(protocol, "preflight")
    write_once_json(payload, path)
    print(f"wrote passing E4 safety preflight {path}", flush=True)
    return payload, path


def run_confirmation() -> tuple[dict[str, Any], Path]:
    protocol, _ = validate_static_inputs()
    source_lock, lock_sha, head = verify_source_lock()
    preflight, preflight_sha = verify_preflight(protocol, lock_sha)
    environment = validate_runtime(require_slurm=True)
    fast_cholesky, extension = import_extension()
    claim_path = artifact_path(protocol, "attempt")
    journal_path = artifact_path(protocol, "journal")
    result_path = artifact_path(protocol, "result")
    audit_path = artifact_path(protocol, "audit")
    for path in (claim_path, journal_path, result_path, audit_path):
        if path.exists():
            raise FileExistsError(
                f"confirmation is write-once; artifact already exists: {path}"
            )
    claim = {
        "schema_version": 1,
        "experiment": "e4_safety_followup_attempt",
        "campaign_tag": CAMPAIGN_TAG,
        "status": "claimed_before_science",
        "claimed_utc": utc_now(),
        "source_lock_sha256": lock_sha,
        "preflight_sha256": preflight_sha,
        "source_commit": head,
        "slurm_job_id": environment["slurm_job_id"],
        "implementation_source_sha256": sha256(IMPLEMENTATION_SOURCE),
        "implementation_extension": str(extension),
        "implementation_extension_sha256": sha256(extension),
        "confirmation_seeds": protocol["problem"]["confirmation_seeds"],
        "original_e4_scientific_status": read_json(ORIGINAL_AUDIT)[
            "scientific_status"
        ],
    }
    write_once_json(claim, claim_path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_fd = os.open(
        journal_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
    )
    started = utc_now()
    started_counter = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with os.fdopen(journal_fd, "w", encoding="utf-8") as journal:
        for starting_precision in protocol["problem"]["starting_precisions"]:
            for operator in protocol["problem"]["operators"]:
                for delta in protocol["problem"]["deltas"]:
                    for seed in protocol["problem"]["confirmation_seeds"]:
                        row = run_trajectory(
                            protocol=protocol,
                            fast_cholesky=fast_cholesky,
                            starting_precision=starting_precision,
                            operator=operator,
                            delta=delta,
                            seed=seed,
                        )
                        rows.append(row)
                        append_journal(journal, row)
                        print(
                            f"E4 safety {starting_precision} {operator} "
                            f"delta={delta:g} seed={seed} "
                            f"status={row['status']}",
                            flush=True,
                        )
    ended = utc_now()
    files = {
        key: {
            **record,
            "observed_sha256": sha256(ROOT / record["path"]),
        }
        for key, record in source_lock["files"].items()
    }
    payload = {
        "schema_version": 1,
        "experiment": "e4_safety_followup_confirmation",
        "campaign_tag": CAMPAIGN_TAG,
        "status": "complete",
        "protocol": protocol,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "development_inventory_sha256": sha256(DEVELOPMENT_INVENTORY_PATH),
        "source_lock": {
            "path": relative(artifact_path(protocol, "source_lock")),
            "sha256": lock_sha,
        },
        "preflight": {
            "path": relative(artifact_path(protocol, "preflight")),
            "sha256": preflight_sha,
            "slurm_job_id": preflight["environment"]["slurm_job_id"],
        },
        "attempt": {
            "path": relative(claim_path),
            "sha256": sha256(claim_path),
        },
        "journal": {
            "path": relative(journal_path),
            "sha256": sha256(journal_path),
            "line_count": len(rows),
        },
        "environment": {
            **environment,
            "started_utc": started,
            "ended_utc": ended,
            "elapsed_wall_seconds": time.perf_counter() - started_counter,
        },
        "provenance": {
            "locked_files": files,
            "implementation_extension": str(extension),
            "implementation_extension_sha256": sha256(extension),
        },
        "rows": rows,
    }
    write_once_json(payload, result_path)
    print(f"wrote E4 safety confirmation {result_path}", flush=True)
    return payload, result_path


def _expected_identities(protocol: dict[str, Any]) -> set[tuple[Any, ...]]:
    return {
        (precision, operator, delta, seed)
        for precision in protocol["problem"]["starting_precisions"]
        for operator in protocol["problem"]["operators"]
        for delta in protocol["problem"]["deltas"]
        for seed in protocol["problem"]["confirmation_seeds"]
    }


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "minimum": None, "maximum": None, "mean": None}
    return {
        "n": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": sum(values) / len(values),
    }


def audit_confirmation() -> tuple[dict[str, Any], Path]:
    protocol, development = validate_static_inputs()
    source_lock, lock_sha, _ = verify_source_lock()
    preflight, preflight_sha = verify_preflight(protocol, lock_sha)
    result_path = artifact_path(protocol, "result")
    claim_path = artifact_path(protocol, "attempt")
    journal_path = artifact_path(protocol, "journal")
    audit_path = artifact_path(protocol, "audit")
    if audit_path.exists():
        raise FileExistsError(f"refusing to overwrite audit {audit_path}")
    result = read_json(result_path)
    claim = read_json(claim_path)
    if result.get("status") != "complete":
        raise ValueError("result is not complete")
    if result.get("protocol") != protocol:
        raise ValueError("result embeds a different protocol")
    if result.get("protocol_sha256") != sha256(PROTOCOL_PATH):
        raise ValueError("result protocol hash mismatch")
    if result["source_lock"]["sha256"] != lock_sha:
        raise ValueError("result source-lock hash mismatch")
    if result["preflight"]["sha256"] != preflight_sha:
        raise ValueError("result preflight hash mismatch")
    if result["attempt"]["sha256"] != sha256(claim_path):
        raise ValueError("result attempt hash mismatch")
    if result["journal"]["sha256"] != sha256(journal_path):
        raise ValueError("result journal hash mismatch")
    if claim.get("status") != "claimed_before_science":
        raise ValueError("attempt was not claimed before science")
    if claim.get("source_lock_sha256") != lock_sha:
        raise ValueError("attempt source-lock mismatch")
    if claim.get("preflight_sha256") != preflight_sha:
        raise ValueError("attempt preflight mismatch")
    if claim.get("confirmation_seeds") != list(range(46001, 46021)):
        raise ValueError("attempt confirmation seeds changed")
    if claim.get("original_e4_scientific_status") != "criteria_failed":
        raise ValueError("attempt rewrote original E4 status")
    if preflight.get("status") != "passed":
        raise ValueError("preflight status changed")

    replay: list[dict[str, Any]] = []
    with journal_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            item = json.loads(line)
            if item.get("kind") != "trajectory" or not isinstance(
                item.get("row"), dict
            ):
                raise ValueError(f"bad journal line {line_number}")
            replay.append(item["row"])
    rows = result.get("rows")
    if replay != rows:
        raise ValueError("journal replay differs from result rows")
    if result["journal"]["line_count"] != len(replay):
        raise ValueError("journal line-count mismatch")

    expected = _expected_identities(protocol)
    observed: set[tuple[Any, ...]] = set()
    duplicate_count = 0
    failure_count = 0
    false_negative_count = 0
    post_by_start: dict[str, list[float]] = {"float32": [], "float64": []}
    post_by_cell: dict[tuple[str, str], list[float]] = {}
    action_counts: dict[str, int] = {
        "none": 0,
        "promote_fp64": 0,
        "refresh_fp64": 0,
    }
    promoted_trajectories = 0
    pairing: dict[tuple[str, float, int], dict[str, dict[str, Any]]] = {}
    expected_steps = protocol["problem"]["steps"]
    for row in rows:
        identity = (
            row["starting_precision"],
            row["operator"],
            row["delta"],
            row["seed"],
        )
        if identity in observed:
            duplicate_count += 1
        observed.add(identity)
        if identity not in expected:
            raise ValueError(f"undeclared row {identity}")
        fingerprints = row.get("input_fingerprints")
        if not isinstance(fingerprints, dict):
            raise ValueError(f"missing fingerprints for {identity}")
        pair_key = (row["operator"], row["delta"], row["seed"])
        pairing.setdefault(pair_key, {})[row["starting_precision"]] = fingerprints
        if row["status"] != "ok":
            failure_count += 1
            continue
        records = row["records"]
        if len(records) != expected_steps + 1:
            raise ValueError(f"record count mismatch for {identity}")
        if [record["step"] for record in records] != list(
            range(expected_steps + 1)
        ):
            raise ValueError(f"step sequence mismatch for {identity}")
        active = row["starting_precision"]
        trajectory_promoted = False
        for record in records:
            if record["active_precision_before"] != active:
                raise ValueError(f"active-precision sequence mismatch {identity}")
            probe = float(record["pre_guard"]["probe_operator_residual"])
            exact = float(record["pre_guard"]["exact_operator_residual"])
            if not math.isfinite(probe) or not math.isfinite(exact):
                raise ValueError(f"nonfinite pre-guard diagnostic {identity}")
            expected_action = decide_guard(active, probe)
            if record["action"] != expected_action:
                raise ValueError(f"dispatch mismatch for {identity}")
            limit = precision_limit(active)
            trigger = precision_trigger(active)
            expected_false_negative = exact > limit and probe <= trigger
            if record["false_negative"] is not expected_false_negative:
                raise ValueError(f"false-negative flag mismatch {identity}")
            false_negative_count += int(expected_false_negative)
            action_counts[expected_action] += 1
            if expected_action == "promote_fp64":
                trajectory_promoted = True
                expected_after = "float64"
            elif expected_action == "refresh_fp64":
                expected_after = "float64"
            else:
                expected_after = active
            if record["active_precision_after"] != expected_after:
                raise ValueError(f"post-action precision mismatch {identity}")
            for phase in ("pre_guard", "post_guard"):
                for name, value in record[phase].items():
                    if not math.isfinite(float(value)):
                        raise ValueError(
                            f"nonfinite {phase}/{name} for {identity}"
                        )
            post = float(record["post_guard"]["exact_operator_residual"])
            post_by_start[row["starting_precision"]].append(post)
            post_by_cell.setdefault(
                (row["starting_precision"], row["operator"]), []
            ).append(post)
            if expected_action == "none" and record["post_guard"] != record["pre_guard"]:
                raise ValueError(f"no-op diagnostics changed for {identity}")
            active = expected_after
        promoted_trajectories += int(trajectory_promoted)

    if observed != expected:
        raise ValueError(
            f"grid mismatch: missing={sorted(expected-observed)[:5]}, "
            f"extra={sorted(observed-expected)[:5]}"
        )
    if duplicate_count:
        raise ValueError(f"duplicate trajectory count {duplicate_count}")
    for pair_key, by_precision in pairing.items():
        if set(by_precision) != {"float32", "float64"}:
            raise ValueError(f"missing paired precision for {pair_key}")
        left = by_precision["float32"]
        right = by_precision["float64"]
        for key in (
            "base_metric_original_fp64_sha256",
            "base_operator_original_fp64_sha256",
            "update_sequence_original_fp64_sha256",
            "probe_sha256",
        ):
            if left[key] != right[key]:
                raise ValueError(f"original-input pairing failed for {pair_key}")

    criteria = protocol["predeclared_scientific_criteria"]
    checks = {
        "all_trajectories_complete": {
            "observed": failure_count == 0,
            "expected": True,
            "passed": failure_count == 0,
        },
        "maximum_post_policy_exact_residual_float32_start": {
            "observed": max(post_by_start["float32"], default=None),
            "maximum_allowed": criteria[
                "maximum_post_policy_exact_residual_float32_start"
            ],
        },
        "maximum_post_policy_exact_residual_float64_start": {
            "observed": max(post_by_start["float64"], default=None),
            "maximum_allowed": criteria[
                "maximum_post_policy_exact_residual_float64_start"
            ],
        },
        "false_negative_count": {
            "observed": false_negative_count,
            "expected": criteria["false_negative_count"],
            "passed": false_negative_count == criteria["false_negative_count"],
        },
    }
    for name in (
        "maximum_post_policy_exact_residual_float32_start",
        "maximum_post_policy_exact_residual_float64_start",
    ):
        observed_max = checks[name]["observed"]
        checks[name]["passed"] = (
            observed_max is not None
            and observed_max <= checks[name]["maximum_allowed"]
        )
    all_passed = all(check["passed"] for check in checks.values())
    original_audit = read_json(ORIGINAL_AUDIT)
    if original_audit["scientific_status"] != "criteria_failed":
        raise ValueError("original E4 scientific status changed during follow-up")
    summary = {
        "schema_version": 1,
        "experiment": "e4_safety_followup_audit",
        "campaign_tag": CAMPAIGN_TAG,
        "label": protocol["label"],
        "status": "integrity_passed",
        "scientific_status": (
            "criteria_passed" if all_passed else "criteria_failed"
        ),
        "created_utc": utc_now(),
        "source_result": relative(result_path),
        "source_result_sha256": sha256(result_path),
        "source_lock_sha256": lock_sha,
        "preflight_sha256": preflight_sha,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "development_inventory": {
            "path": relative(DEVELOPMENT_INVENTORY_PATH),
            "sha256": sha256(DEVELOPMENT_INVENTORY_PATH),
            "selection_seeds": development["selection"]["seeds"],
            "check_seeds": development[
                "final_policy_development_rerun"
            ]["check_seeds"],
            "confirmation_overlap": [],
        },
        "original_e4": {
            "path": relative(ORIGINAL_AUDIT),
            "sha256": sha256(ORIGINAL_AUDIT),
            "scientific_status_preserved": original_audit["scientific_status"],
        },
        "integrity": {
            "expected_trajectory_count": len(expected),
            "observed_trajectory_count": len(rows),
            "failure_count": failure_count,
            "duplicate_count": duplicate_count,
            "journal_replay_exact": True,
            "journal_line_count": len(replay),
            "paired_original_inputs_and_probes": True,
            "all_failures_retained": True,
            "dispatch_has_no_exact_residual_argument": True,
            "confirmation_seed_set": protocol["problem"]["confirmation_seeds"],
            "attempt_claimed_before_science": True,
            "source_files_match_lock": True,
        },
        "scientific_criteria": {
            "status": (
                "criteria_passed" if all_passed else "criteria_failed"
            ),
            "all_passed": all_passed,
            "checks": checks,
            "outcome_does_not_affect_integrity_status": True,
        },
        "results": {
            "post_policy_exact_residual_by_starting_precision": {
                key: _stats(values) for key, values in post_by_start.items()
            },
            "post_policy_exact_residual_by_cell": [
                {
                    "starting_precision": key[0],
                    "operator": key[1],
                    **_stats(values),
                }
                for key, values in sorted(post_by_cell.items())
            ],
            "action_counts": action_counts,
            "promoted_trajectory_count": promoted_trajectories,
            "false_negative_count": false_negative_count,
            "timing_claims": False,
        },
        "provenance": {
            "source_commit": result["environment"]["source_commit"],
            "slurm_job_id": result["environment"]["slurm_job_id"],
            "runtime_extension": result["provenance"][
                "implementation_extension"
            ],
            "runtime_extension_sha256": result["provenance"][
                "implementation_extension_sha256"
            ],
            "locked_file_count": len(source_lock["files"]),
        },
    }
    write_once_json(summary, audit_path)
    print(
        f"E4 safety audit integrity_passed, "
        f"scientific_status={summary['scientific_status']}; wrote {audit_path}",
        flush=True,
    )
    return summary, audit_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        required=True,
        choices=(
            "plan",
            "freeze-lock",
            "self-test",
            "preflight",
            "run-confirmation",
            "audit",
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "plan":
        protocol, development = validate_static_inputs()
        print(
            json.dumps(
                {
                    "protocol": protocol,
                    "development_inventory_sha256": sha256(
                        DEVELOPMENT_INVENTORY_PATH
                    ),
                    "expected_trajectory_count": len(
                        _expected_identities(protocol)
                    ),
                    "development_seed_count": len(
                        development["selection"]["seeds"]
                    )
                    + len(
                        development["final_policy_development_rerun"][
                            "check_seeds"
                        ]
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.action == "freeze-lock":
        freeze_lock()
    elif args.action == "self-test":
        validate_static_inputs()
        fast_cholesky, _ = import_extension()
        print(json.dumps(self_test(fast_cholesky), indent=2, sort_keys=True))
    elif args.action == "preflight":
        run_preflight()
    elif args.action == "run-confirmation":
        run_confirmation()
    elif args.action == "audit":
        audit_confirmation()
    else:
        raise AssertionError(args.action)


if __name__ == "__main__":
    main()
