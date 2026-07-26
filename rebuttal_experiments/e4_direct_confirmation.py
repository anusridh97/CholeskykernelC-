"""Run the frozen direct-error and signed-downdate E4 confirmation."""

from __future__ import annotations

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
from typing import Any, TextIO

import torch

from .common import ROOT, configure_threads, tensor_relative_error
from .kernel_stability import make_near_duplicate_kernel, rbf
from .stability import make_spd


PROTOCOL_PATH = Path(__file__).with_name("e4_direct_confirmation_protocol.json")
CAMPAIGN_TAG = "e4-direct-confirm-v1"
EXPECTED_PROTOCOL_SHA256 = (
    "10f2ad4bc27bb2bcaf39001f2f76fea8ed474f4de3988c7aff857786563987ee"
)
EXPECTED_ACCOUNT = "marlowe-m000151-pm06"
EXPECTED_PARTITION = "batch"
EXPECTED_QOS = "medium"
THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
IMPLEMENTATION_SOURCE = (
    ROOT
    / "Cholesky-main-2"
    / "corrected_incremental_cholesky"
    / "fast_update.cpp"
)
SETUP_SOURCE = IMPLEMENTATION_SOURCE.with_name("setup_ext.py")


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes(order="C")


def _tensor_sha256(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _tensor_sequence_sha256(tensors: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    digest.update(str(len(tensors)).encode("ascii"))
    for tensor in tensors:
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        raw = _tensor_bytes(tensor)
        digest.update(str(len(raw)).encode("ascii"))
        digest.update(raw)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def write_once_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite write-once E4 artifact: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(name)


def whiten(L: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    left = torch.linalg.solve_triangular(L, M, upper=False)
    return torch.linalg.solve_triangular(
        L,
        left.mT,
        upper=False,
    ).mT.contiguous()


def validate_protocol(protocol: dict[str, Any]) -> None:
    observed_hash = _sha256(PROTOCOL_PATH)
    if observed_hash != EXPECTED_PROTOCOL_SHA256:
        raise ValueError(
            "E4 protocol file differs from its frozen hash: "
            f"expected={EXPECTED_PROTOCOL_SHA256}, observed={observed_hash}"
        )
    if protocol != _read(PROTOCOL_PATH):
        raise ValueError("E4 protocol argument differs from the frozen file")
    if protocol.get("campaign_tag") != CAMPAIGN_TAG:
        raise ValueError("E4 campaign tag mismatch")
    if protocol.get("status") != "frozen_before_confirmation":
        raise ValueError("E4 protocol is not frozen")
    if protocol.get("threads") != 1:
        raise ValueError("E4 confirmation must be single threaded")
    if protocol.get("implementation_source") != _relative(IMPLEMENTATION_SOURCE):
        raise ValueError("E4 implementation source mismatch")
    if not protocol.get("retain_all_rows_and_failures"):
        raise ValueError("E4 protocol must retain every result")
    resource = protocol.get("resource_contract", {})
    expected_resource = {
        "account": EXPECTED_ACCOUNT,
        "partition": EXPECTED_PARTITION,
        "qos": EXPECTED_QOS,
        "processes": 1,
        "physical_cpus_per_process": 1,
        "omp_threads": 1,
        "mkl_threads": 1,
        "openblas_threads": 1,
        "timing_claims": False,
    }
    if resource != expected_resource:
        raise ValueError(
            f"E4 resource contract mismatch: expected {expected_resource}, "
            f"got {resource}"
        )


def validate_runtime() -> dict[str, Any]:
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
        raise RuntimeError(f"E4 environment is not hermetic: {invalid}")
    if os.environ.get("SLURM_JOB_ACCOUNT") != EXPECTED_ACCOUNT:
        raise RuntimeError("E4 is not on the frozen account")
    if os.environ.get("SLURM_JOB_PARTITION") != EXPECTED_PARTITION:
        raise RuntimeError("E4 is not on the frozen partition")
    if os.environ.get("SLURM_JOB_QOS") != EXPECTED_QOS:
        raise RuntimeError("E4 is not using the frozen QoS")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1":
        raise RuntimeError("E4 did not receive exactly one CPU per task")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("E4 requires a Slurm job ID")
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("E4 requires observable CPU affinity")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 1:
        raise RuntimeError(f"E4 requires one physical CPU, observed {affinity}")
    configure_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("E4 failed to establish one-thread PyTorch")
    source_commit = os.environ.get("SOURCE_COMMIT")
    actual_commit = _git_output("rev-parse", "HEAD")
    if not source_commit or source_commit != actual_commit:
        raise RuntimeError(
            f"E4 source-lock mismatch: declared={source_commit!r}, "
            f"actual={actual_commit!r}"
        )
    if subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0:
        raise RuntimeError("E4 tracked worktree differs from its locked commit")
    if (
        subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode
        != 0
    ):
        raise RuntimeError("E4 index differs from its locked commit")
    untracked = _git_output(
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    if untracked:
        raise RuntimeError(f"E4 worktree has untracked source inputs: {untracked}")
    return {
        "source_commit": actual_commit,
        "process_affinity": affinity,
        "thread_environment": {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT
        },
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def direct_diagnostics(
    *,
    step: int,
    phase: str,
    dtype: torch.dtype,
    G_quantized: torch.Tensor,
    M_quantized: torch.Tensor,
    G_original: torch.Tensor,
    M_original: torch.Tensor,
    L_incremental: torch.Tensor,
    Aw_incremental: torch.Tensor,
) -> dict[str, Any]:
    L_reference = torch.linalg.cholesky(G_quantized)
    Aw_reference = whiten(L_reference, M_quantized)
    L_fresh = torch.linalg.cholesky(G_quantized.to(dtype)).contiguous()
    Aw_fresh = whiten(L_fresh, M_quantized.to(dtype))
    L_inc64 = L_incremental.double()
    Aw_inc64 = Aw_incremental.double()
    L_fresh64 = L_fresh.double()
    Aw_fresh64 = Aw_fresh.double()
    L_original = torch.linalg.cholesky(G_original)
    Aw_original = whiten(L_original, M_original)
    return {
        "step": step,
        "phase": phase,
        "condition_quantized_reference": float(
            torch.linalg.cond(G_quantized).item()
        ),
        "incremental_vs_fresh_same_dtype": {
            "L_relative_error": tensor_relative_error(L_inc64, L_fresh64),
            "Aw_relative_error": tensor_relative_error(Aw_inc64, Aw_fresh64),
        },
        "incremental_vs_quantized_fp64_reference": {
            "L_relative_error": tensor_relative_error(L_inc64, L_reference),
            "Aw_relative_error": tensor_relative_error(Aw_inc64, Aw_reference),
        },
        "fresh_same_dtype_vs_quantized_fp64_reference": {
            "L_relative_error": tensor_relative_error(L_fresh64, L_reference),
            "Aw_relative_error": tensor_relative_error(Aw_fresh64, Aw_reference),
        },
        "incremental_vs_original_fp64_target": {
            "L_relative_error": tensor_relative_error(L_inc64, L_original),
            "Aw_relative_error": tensor_relative_error(Aw_inc64, Aw_original),
        },
        "backward_errors": {
            "incremental_factor": tensor_relative_error(
                L_inc64 @ L_inc64.mT,
                G_quantized,
            ),
            "fresh_factor": tensor_relative_error(
                L_fresh64 @ L_fresh64.mT,
                G_quantized,
            ),
            "incremental_operator": tensor_relative_error(
                L_inc64 @ Aw_inc64 @ L_inc64.mT,
                M_quantized,
            ),
            "fresh_operator": tensor_relative_error(
                L_fresh64 @ Aw_fresh64 @ L_fresh64.mT,
                M_quantized,
            ),
        },
    }


def append_journal(handle: TextIO, kind: str, row: dict[str, Any]) -> None:
    handle.write(
        json.dumps(
            {"kind": kind, "row": row},
            sort_keys=True,
            allow_nan=False,
        )
    )
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())


def positive_trajectories(
    protocol: dict[str, Any],
    fast_cholesky,
    journal: TextIO,
) -> list[dict[str, Any]]:
    spec = protocol["positive_trajectories"]
    rows = []
    checkpoints = set(int(value) for value in spec["checkpoints"])
    pre_reset_steps = set(int(value) for value in spec["pre_reset_steps"])
    for precision in spec["precisions"]:
        dtype = _dtype(precision)
        for condition in spec["conditions"]:
            for reset_interval in spec["reset_intervals"]:
                for seed in spec["seeds"]:
                    records = []
                    status = "ok"
                    failure = None
                    failure_step = None
                    current_step = 0
                    input_fingerprints = None
                    try:
                        G_original = make_spd(spec["rank"], condition, seed)
                        floor = float(torch.linalg.eigvalsh(G_original)[0].item())
                        M_original = G_original - floor * torch.eye(
                            spec["rank"],
                            dtype=torch.float64,
                        )
                        base_G_sha256 = _tensor_sha256(G_original)
                        base_M_sha256 = _tensor_sha256(M_original)
                        generator = torch.Generator().manual_seed(seed + 991)
                        G_for_updates = G_original.clone()
                        updates64 = []
                        for _ in range(spec["steps"]):
                            scale = (
                                spec["update_scale"]
                                * (
                                    torch.trace(G_for_updates) / spec["rank"]
                                ).sqrt()
                            )
                            z64 = (
                                torch.randn(
                                    spec["rank"],
                                    dtype=torch.float64,
                                    generator=generator,
                                )
                                * scale
                                / spec["rank"] ** 0.5
                            )
                            updates64.append(z64)
                            G_for_updates = G_for_updates + torch.outer(z64, z64)
                        update_sequence_sha256 = _tensor_sequence_sha256(updates64)
                        input_fingerprints = {
                            "base_metric_fp64_sha256": base_G_sha256,
                            "base_operator_fp64_sha256": base_M_sha256,
                            "update_sequence_fp64_sha256": (
                                update_sequence_sha256
                            ),
                        }
                        G_quantized = G_original.to(dtype).double()
                        M_quantized = M_original.to(dtype).double()
                        L = torch.linalg.cholesky(
                            G_quantized.to(dtype)
                        ).contiguous()
                        Aw = whiten(L, M_quantized.to(dtype))
                        for step in range(spec["steps"] + 1):
                            current_step = step
                            if step in checkpoints:
                                phase = (
                                    "post_reset"
                                    if reset_interval
                                    and step > 0
                                    and step % reset_interval == 0
                                    else "checkpoint"
                                )
                                records.append(
                                    direct_diagnostics(
                                        step=step,
                                        phase=phase,
                                        dtype=dtype,
                                        G_quantized=G_quantized,
                                        M_quantized=M_quantized,
                                        G_original=G_original,
                                        M_original=M_original,
                                        L_incremental=L,
                                        Aw_incremental=Aw,
                                    )
                                )
                            if step == spec["steps"]:
                                break
                            z64 = updates64[step]
                            z = z64.to(dtype)
                            Aw, _ = fast_cholesky.fast_incremental_update(
                                L,
                                Aw,
                                z.clone(),
                                z,
                                1,
                                1,
                                1e-8,
                            )
                            G_original = G_original + torch.outer(z64, z64)
                            M_original = M_original + torch.outer(z64, z64)
                            z_quantized = z.double()
                            G_quantized = G_quantized + torch.outer(
                                z_quantized,
                                z_quantized,
                            )
                            M_quantized = M_quantized + torch.outer(
                                z_quantized,
                                z_quantized,
                            )
                            next_step = step + 1
                            if (
                                reset_interval
                                and next_step % reset_interval == 0
                            ):
                                if next_step in pre_reset_steps:
                                    records.append(
                                        direct_diagnostics(
                                            step=next_step,
                                            phase="pre_reset",
                                            dtype=dtype,
                                            G_quantized=G_quantized,
                                            M_quantized=M_quantized,
                                            G_original=G_original,
                                            M_original=M_original,
                                            L_incremental=L,
                                            Aw_incremental=Aw,
                                        )
                                    )
                                L = torch.linalg.cholesky(
                                    G_quantized.to(dtype)
                                ).contiguous()
                                Aw = whiten(L, M_quantized.to(dtype))
                    except Exception as error:
                        status = "failed"
                        failure_step = current_step
                        failure = f"{type(error).__name__}: {error}"
                    row = {
                        "precision": precision,
                        "rank": spec["rank"],
                        "target_condition": condition,
                        "reset_interval": reset_interval,
                        "seed": seed,
                        "status": status,
                        "failure_stage": (
                            None if status == "ok" else "setup_or_trajectory"
                        ),
                        "failure_step": failure_step,
                        "failure": failure,
                        "input_fingerprints": input_fingerprints,
                        "records": records,
                    }
                    rows.append(row)
                    append_journal(journal, "positive", row)
                    print(
                        f"positive {precision} kappa={condition:g} "
                        f"reset={reset_interval} seed={seed} status={status}",
                        flush=True,
                    )
    return rows


def kernel_trajectories(
    protocol: dict[str, Any],
    fast_cholesky,
    journal: TextIO,
) -> list[dict[str, Any]]:
    spec = protocol["near_duplicate_kernels"]
    rows = []
    checkpoints = set(int(value) for value in spec["checkpoints"])
    pre_reset_steps = set(int(value) for value in spec["pre_reset_steps"])
    for precision in spec["precisions"]:
        dtype = _dtype(precision)
        for operator in spec["operators"]:
            for delta in spec["deltas"]:
                for reset_interval in spec["reset_intervals"]:
                    for seed in spec["seeds"]:
                        records = []
                        status = "ok"
                        failure = None
                        failure_step = None
                        current_step = 0
                        input_fingerprints = None
                        try:
                            G_original, points = make_near_duplicate_kernel(
                                spec["rank"],
                                delta,
                                spec["jitter"],
                                seed,
                            )
                            operator_generator = torch.Generator().manual_seed(
                                seed + 7001
                            )
                            update_generator = torch.Generator().manual_seed(
                                seed + 8001
                            )
                            if operator == "cumulative":
                                M_original = (
                                    G_original
                                    - spec["jitter"]
                                    * torch.eye(
                                        spec["rank"],
                                        dtype=torch.float64,
                                    )
                                )
                            else:
                                raw = torch.randn(
                                    spec["rank"],
                                    spec["rank"],
                                    dtype=torch.float64,
                                    generator=operator_generator,
                                )
                                M_original = raw / torch.linalg.matrix_norm(
                                    raw,
                                    ord=2,
                                )
                            base_G_sha256 = _tensor_sha256(G_original)
                            base_M_sha256 = _tensor_sha256(M_original)
                            updates64 = []
                            for _ in range(spec["steps"]):
                                x_new = torch.rand(
                                    1,
                                    3,
                                    dtype=torch.float64,
                                    generator=update_generator,
                                )
                                updates64.append(
                                    (
                                        spec["update_scale"]
                                        * rbf(
                                            points,
                                            x_new,
                                            lengthscale=0.4,
                                        ).squeeze(1)
                                    ).contiguous()
                                )
                            update_sequence_sha256 = _tensor_sequence_sha256(
                                updates64
                            )
                            input_fingerprints = {
                                "base_metric_fp64_sha256": base_G_sha256,
                                "base_operator_fp64_sha256": base_M_sha256,
                                "update_sequence_fp64_sha256": (
                                    update_sequence_sha256
                                ),
                            }
                            G_quantized = G_original.to(dtype).double()
                            M_quantized = M_original.to(dtype).double()
                            L = torch.linalg.cholesky(
                                G_quantized.to(dtype)
                            ).contiguous()
                            Aw = whiten(L, M_quantized.to(dtype))
                            for step in range(spec["steps"] + 1):
                                current_step = step
                                if step in checkpoints:
                                    phase = (
                                        "post_reset"
                                        if reset_interval
                                        and step > 0
                                        and step % reset_interval == 0
                                        else "checkpoint"
                                    )
                                    records.append(
                                        direct_diagnostics(
                                            step=step,
                                            phase=phase,
                                            dtype=dtype,
                                            G_quantized=G_quantized,
                                            M_quantized=M_quantized,
                                            G_original=G_original,
                                            M_original=M_original,
                                            L_incremental=L,
                                            Aw_incremental=Aw,
                                        )
                                    )
                                if step == spec["steps"]:
                                    break
                                z64 = updates64[step]
                                z = z64.to(dtype)
                                Aw, _ = fast_cholesky.fast_incremental_update(
                                    L,
                                    Aw,
                                    z.clone(),
                                    z,
                                    1,
                                    1,
                                    1e-8,
                                )
                                G_original = G_original + torch.outer(z64, z64)
                                M_original = M_original + torch.outer(z64, z64)
                                z_quantized = z.double()
                                G_quantized = G_quantized + torch.outer(
                                    z_quantized,
                                    z_quantized,
                                )
                                M_quantized = M_quantized + torch.outer(
                                    z_quantized,
                                    z_quantized,
                                )
                                next_step = step + 1
                                if (
                                    reset_interval
                                    and next_step % reset_interval == 0
                                ):
                                    if next_step in pre_reset_steps:
                                        records.append(
                                            direct_diagnostics(
                                                step=next_step,
                                                phase="pre_reset",
                                                dtype=dtype,
                                                G_quantized=G_quantized,
                                                M_quantized=M_quantized,
                                                G_original=G_original,
                                                M_original=M_original,
                                                L_incremental=L,
                                                Aw_incremental=Aw,
                                            )
                                        )
                                    L = torch.linalg.cholesky(
                                        G_quantized.to(dtype)
                                    ).contiguous()
                                    Aw = whiten(L, M_quantized.to(dtype))
                        except Exception as error:
                            status = "failed"
                            failure_step = current_step
                            failure = f"{type(error).__name__}: {error}"
                        row = {
                            "precision": precision,
                            "rank": spec["rank"],
                            "operator": operator,
                            "delta": delta,
                            "reset_interval": reset_interval,
                            "seed": seed,
                            "status": status,
                            "failure_stage": (
                                None if status == "ok" else "setup_or_trajectory"
                            ),
                            "failure_step": failure_step,
                            "failure": failure,
                            "input_fingerprints": input_fingerprints,
                            "records": records,
                        }
                        rows.append(row)
                        append_journal(journal, "kernel", row)
                        print(
                            f"kernel {precision} {operator} delta={delta:g} "
                            f"reset={reset_interval} seed={seed} status={status}",
                            flush=True,
                        )
    return rows


def _run_one_downdate(
    *,
    dtype: torch.dtype,
    G_quantized: torch.Tensor,
    M_quantized: torch.Tensor,
    L: torch.Tensor,
    Aw: torch.Tensor,
    z: torch.Tensor,
    w: torch.Tensor,
    operator_sign: int,
    tolerance: float,
    fast_cholesky,
) -> dict[str, Any]:
    z64 = z.double()
    w64 = w.double()
    G_new = G_quantized - torch.outer(z64, z64)
    M_new = M_quantized + operator_sign * torch.outer(z64, w64)
    L_work = L.clone()
    try:
        Aw_work, _ = fast_cholesky.fast_incremental_update(
            L_work,
            Aw,
            z.clone(),
            w,
            -1,
            operator_sign,
            tolerance,
        )
    except Exception as error:
        return {
            "status": "failed",
            "failure_stage": "signed_incremental_implementation",
            "failure": f"{type(error).__name__}: {error}",
            "diagnostics": None,
        }
    try:
        diagnostics = direct_diagnostics(
            step=1,
            phase="requested_or_guarded_downdate",
            dtype=dtype,
            G_quantized=G_new,
            M_quantized=M_new,
            G_original=G_new,
            M_original=M_new,
            L_incremental=L_work,
            Aw_incremental=Aw_work,
        )
    except Exception as error:
        return {
            "status": "failed",
            "failure_stage": "fresh_reference_or_diagnostics",
            "failure": f"{type(error).__name__}: {error}",
            "diagnostics": None,
        }
    return {
        "status": "ok",
        "failure_stage": None,
        "failure": None,
        "diagnostics": diagnostics,
    }


def _downdate_leverages(
    *,
    L_implementation: torch.Tensor,
    G_mathematical: torch.Tensor,
    z: torch.Tensor,
) -> tuple[float, float]:
    implementation_solve = torch.linalg.solve_triangular(
        L_implementation,
        z[:, None],
        upper=False,
    ).squeeze(1)
    implementation = float(
        torch.dot(implementation_solve, implementation_solve).item()
    )
    L_mathematical = torch.linalg.cholesky(G_mathematical)
    mathematical_solve = torch.linalg.solve_triangular(
        L_mathematical,
        z.double()[:, None],
        upper=False,
    ).squeeze(1)
    mathematical = float(
        torch.dot(mathematical_solve, mathematical_solve).item()
    )
    return implementation, mathematical


def signed_downdates(
    protocol: dict[str, Any],
    fast_cholesky,
    journal: TextIO,
) -> list[dict[str, Any]]:
    spec = protocol["signed_downdates"]
    rows = []
    guard = float(spec["guard_leverage"])
    tolerance = float(spec["implementation_downdate_tolerance"])
    for precision in spec["precisions"]:
        dtype = _dtype(precision)
        for rank in spec["ranks"]:
            for operator_case in spec["operator_cases"]:
                case_name = operator_case["name"]
                operator_sign = int(operator_case["operator_sign"])
                w_mode = operator_case["w_mode"]
                for leverage in spec["leverages"]:
                    for seed in spec["seeds"]:
                        row: dict[str, Any] = {
                            "precision": precision,
                            "rank": rank,
                            "operator_case": case_name,
                            "operator_sign": operator_sign,
                            "w_mode": w_mode,
                            "target_leverage": leverage,
                            "seed": seed,
                            "status": "ok",
                            "failure_stage": None,
                            "failure": None,
                        }
                        input_fingerprints = None
                        try:
                            generator = torch.Generator().manual_seed(seed)
                            raw = torch.randn(
                                rank,
                                rank,
                                dtype=torch.float64,
                                generator=generator,
                            )
                            G_original = raw @ raw.mT + rank * torch.eye(
                                rank,
                                dtype=torch.float64,
                            )
                            raw_m = torch.randn(
                                rank,
                                rank,
                                dtype=torch.float64,
                                generator=generator,
                            )
                            M_original = raw_m / torch.linalg.matrix_norm(
                                raw_m,
                                ord=2,
                            )
                            G_quantized = G_original.to(dtype).double()
                            M_quantized = M_original.to(dtype).double()
                            L = torch.linalg.cholesky(
                                G_quantized.to(dtype)
                            ).contiguous()
                            Aw = whiten(L, M_quantized.to(dtype))
                            direction64 = torch.randn(
                                rank,
                                dtype=torch.float64,
                                generator=generator,
                            )
                            direction64 = direction64 / torch.linalg.vector_norm(
                                direction64
                            )
                            L_original = torch.linalg.cholesky(G_original)
                            z64 = (
                                math.sqrt(float(leverage))
                                * (L_original @ direction64)
                            ).contiguous()
                            z = z64.to(dtype).contiguous()
                            random_w64 = (
                                torch.randn(
                                    rank,
                                    dtype=torch.float64,
                                    generator=generator,
                                )
                                / rank**0.5
                            )
                            input_fingerprints = {
                                "base_metric_fp64_sha256": _tensor_sha256(
                                    G_original
                                ),
                                "base_operator_fp64_sha256": _tensor_sha256(
                                    M_original
                                ),
                                "z_fp64_sha256": _tensor_sha256(z64),
                                "random_w_fp64_sha256": _tensor_sha256(
                                    random_w64
                                ),
                            }
                            if w_mode == "zero":
                                w = torch.zeros_like(z)
                            elif w_mode == "random":
                                w = random_w64.to(dtype).contiguous()
                            elif w_mode == "matched_z":
                                w = z.clone()
                            else:
                                raise ValueError(f"unknown w mode: {w_mode}")
                            original_solve = torch.linalg.solve_triangular(
                                L_original,
                                z64[:, None],
                                upper=False,
                            ).squeeze(1)
                            original_fp64_leverage = float(
                                torch.dot(original_solve, original_solve).item()
                            )
                            (
                                implementation_leverage,
                                mathematical_leverage,
                            ) = _downdate_leverages(
                                L_implementation=L,
                                G_mathematical=G_quantized,
                                z=z,
                            )
                            requested = _run_one_downdate(
                                dtype=dtype,
                                G_quantized=G_quantized,
                                M_quantized=M_quantized,
                                L=L,
                                Aw=Aw,
                                z=z,
                                w=w,
                                operator_sign=operator_sign,
                                tolerance=tolerance,
                                fast_cholesky=fast_cholesky,
                            )
                            guard_target = float(
                                torch.nextafter(
                                    torch.tensor(guard, dtype=dtype),
                                    torch.tensor(0.0, dtype=dtype),
                                ).item()
                            )
                            maximum_leverage = max(
                                implementation_leverage,
                                mathematical_leverage,
                            )
                            guard_scale = (
                                math.sqrt(guard_target / maximum_leverage)
                                if maximum_leverage > guard_target
                                else 1.0
                            )
                            z_guarded = (z * guard_scale).contiguous()
                            (
                                guarded_implementation_leverage,
                                guarded_mathematical_leverage,
                            ) = _downdate_leverages(
                                L_implementation=L,
                                G_mathematical=G_quantized,
                                z=z_guarded,
                            )
                            for _ in range(4):
                                guarded_maximum = max(
                                    guarded_implementation_leverage,
                                    guarded_mathematical_leverage,
                                )
                                if guarded_maximum <= guard:
                                    break
                                correction = math.sqrt(
                                    guard_target / guarded_maximum
                                )
                                guard_scale *= correction
                                z_guarded = (
                                    z_guarded * correction
                                ).contiguous()
                                (
                                    guarded_implementation_leverage,
                                    guarded_mathematical_leverage,
                                ) = _downdate_leverages(
                                    L_implementation=L,
                                    G_mathematical=G_quantized,
                                    z=z_guarded,
                                )
                            if max(
                                guarded_implementation_leverage,
                                guarded_mathematical_leverage,
                            ) > guard:
                                raise RuntimeError(
                                    "failed to enforce the E4 downdate guard"
                                )
                            if w_mode == "zero":
                                w_guarded = torch.zeros_like(z_guarded)
                                w_guard_scale = 1.0
                            elif w_mode == "random":
                                w_guarded = (w * guard_scale).contiguous()
                                w_guard_scale = guard_scale
                            else:
                                w_guarded = z_guarded.clone()
                                w_guard_scale = guard_scale
                            guarded = _run_one_downdate(
                                dtype=dtype,
                                G_quantized=G_quantized,
                                M_quantized=M_quantized,
                                L=L,
                                Aw=Aw,
                                z=z_guarded,
                                w=w_guarded,
                                operator_sign=operator_sign,
                                tolerance=tolerance,
                                fast_cholesky=fast_cholesky,
                            )
                            row.update(
                                {
                                    "original_fp64_leverage": (
                                        original_fp64_leverage
                                    ),
                                    "implementation_leverage": (
                                        implementation_leverage
                                    ),
                                    "mathematical_leverage": (
                                        mathematical_leverage
                                    ),
                                    "theoretical_transport_bound": (
                                        1.0 / (1.0 - mathematical_leverage)
                                    ),
                                    "input_fingerprints": input_fingerprints,
                                    "requested": requested,
                                    "guard": {
                                        "threshold": guard,
                                        "dtype_safe_target": guard_target,
                                        "scale_applied_to_z": guard_scale,
                                        "scale_applied_to_w": w_guard_scale,
                                        "event_weight_beta": guard_scale**2,
                                        "implementation_leverage": (
                                            guarded_implementation_leverage
                                        ),
                                        "mathematical_leverage": (
                                            guarded_mathematical_leverage
                                        ),
                                        "result": guarded,
                                    },
                                }
                            )
                        except Exception as error:
                            row.update(
                                {
                                    "status": "failed",
                                    "failure_stage": "signed_cell_setup_or_guard",
                                    "failure": f"{type(error).__name__}: {error}",
                                    "original_fp64_leverage": None,
                                    "implementation_leverage": None,
                                    "mathematical_leverage": None,
                                    "theoretical_transport_bound": None,
                                    "input_fingerprints": (
                                        input_fingerprints
                                    ),
                                    "requested": None,
                                    "guard": None,
                                }
                            )
                        rows.append(row)
                        append_journal(journal, "downdate", row)
                        if row["status"] == "ok":
                            requested_status = row["requested"]["status"]
                            guarded_status = row["guard"]["result"]["status"]
                            tau_text = (
                                f"tau_impl={row['implementation_leverage']:.6f} "
                                f"tau_math={row['mathematical_leverage']:.6f}"
                            )
                        else:
                            requested_status = "not_run"
                            guarded_status = "not_run"
                            tau_text = "tau=unavailable"
                        print(
                            f"downdate {precision} r={rank} "
                            f"case={case_name} {tau_text} "
                            f"cell={row['status']} "
                            f"requested={requested_status} "
                            f"guarded={guarded_status}",
                            flush=True,
                        )
    return rows


def run() -> dict[str, Any]:
    protocol = _read(PROTOCOL_PATH)
    validate_protocol(protocol)
    output = ROOT / protocol["result_path"]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    runtime = validate_runtime()

    import fast_cholesky

    extension_path = Path(fast_cholesky.__file__).resolve()
    if not extension_path.is_file():
        raise FileNotFoundError(extension_path)
    expected_extension_text = os.environ.get("E4_EXTENSION_PATH")
    if not expected_extension_text:
        raise RuntimeError("E4_EXTENSION_PATH is not set")
    expected_extension = Path(expected_extension_text).resolve()
    if extension_path != expected_extension:
        raise RuntimeError(
            f"E4 imported the wrong extension: expected {expected_extension}, "
            f"got {extension_path}"
        )
    expected_build_root = (
        Path("/scratch/m000151-pm06/arjo/job-builds")
        / str(os.environ.get("SLURM_JOB_ID"))
        / "e4-direct"
        / "lib"
    ).resolve()
    if extension_path.parent != expected_build_root:
        raise RuntimeError(
            f"E4 extension is outside the locked build root: {extension_path}"
        )
    claim_path = ROOT / protocol["attempt_claim_path"]
    journal_path = ROOT / protocol["journal_path"]
    if journal_path.exists():
        raise FileExistsError(f"refusing to overwrite {journal_path}")
    claim = {
        "schema_version": 1,
        "campaign_tag": CAMPAIGN_TAG,
        "status": "claimed_before_science",
        "source_commit": runtime["source_commit"],
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "implementation_source_sha256": _sha256(IMPLEMENTATION_SOURCE),
        "implementation_extension": str(extension_path),
        "implementation_extension_sha256": _sha256(extension_path),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "claimed_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_once_json(claim, claim_path)
    started = datetime.now(timezone.utc)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("x", encoding="utf-8") as journal:
        positive = positive_trajectories(protocol, fast_cholesky, journal)
        kernels = kernel_trajectories(protocol, fast_cholesky, journal)
        downdates = signed_downdates(protocol, fast_cholesky, journal)
        journal.flush()
        os.fsync(journal.fileno())
    ended = datetime.now(timezone.utc)
    provenance_paths = [
        PROTOCOL_PATH,
        Path(__file__).resolve(),
        Path(__file__).with_name("audit_e4_direct_confirmation.py"),
        Path(__file__).with_name("common.py"),
        IMPLEMENTATION_SOURCE,
        SETUP_SOURCE,
        Path(__file__).with_name("stability.py"),
        Path(__file__).with_name("kernel_stability.py"),
        Path(__file__).with_name("tests")
        / "test_e4_direct_confirmation.py",
        Path(__file__).with_name("slurm")
        / "e4_direct_confirmation.sbatch",
        extension_path,
        Path(sys.executable).resolve(),
        Path(torch.__file__).resolve(),
    ]
    torch_cpu_library = (
        Path(torch.__file__).resolve().parent / "lib" / "libtorch_cpu.so"
    )
    if torch_cpu_library.is_file():
        provenance_paths.append(torch_cpu_library)
    payload = {
        "schema_version": 1,
        "experiment": "e4_direct_confirmation",
        "campaign_tag": CAMPAIGN_TAG,
        "status": "complete",
        "protocol": protocol,
        "protocol_path": _relative(PROTOCOL_PATH),
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "positive_trajectories": positive,
        "near_duplicate_kernel_trajectories": kernels,
        "signed_downdates": downdates,
        "integrity": {
            "positive_row_count": len(positive),
            "kernel_row_count": len(kernels),
            "downdate_row_count": len(downdates),
            "all_rows_and_failures_retained": True,
            "confirmation_tuned_from_outcomes": False,
            "timing_claims": False,
            "positive_failure_count": sum(
                row["status"] != "ok" for row in positive
            ),
            "kernel_failure_count": sum(
                row["status"] != "ok" for row in kernels
            ),
            "downdate_cell_failure_count": sum(
                row["status"] != "ok" for row in downdates
            ),
        },
        "environment": {
            **runtime,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_config": torch.__config__.show(),
            "python_executable": str(Path(sys.executable).resolve()),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_account": os.environ.get("SLURM_JOB_ACCOUNT"),
            "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_job_qos": os.environ.get("SLURM_JOB_QOS"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "started_utc": started.isoformat(),
            "ended_utc": ended.isoformat(),
            "elapsed_wall_seconds": (ended - started).total_seconds(),
        },
        "provenance": {
            "files": {
                _relative(path): {
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in provenance_paths
            },
            "attempt_claim": {
                "path": _relative(claim_path),
                "sha256": _sha256(claim_path),
            },
            "row_journal": {
                "path": _relative(journal_path),
                "sha256": _sha256(journal_path),
                "line_count": len(positive) + len(kernels) + len(downdates),
            },
            "implementation_extension": str(extension_path),
            "build_flags": ["-O3", "-std=c++17", "-march=native"],
        },
    }
    write_once_json(payload, output)
    print(output, flush=True)
    return payload


if __name__ == "__main__":
    run()
