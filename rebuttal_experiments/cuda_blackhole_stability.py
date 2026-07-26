"""Real-trace repeated-update stability for the frozen black-hole protocol.

The input is the deterministic feature trace produced by
``prepare_blackhole_features.py``.  This driver deliberately keeps diagnostic
factorizations and probes outside the timed regions.  For the q=50 method,
however, the actual refresh (Cholesky, inverse solve, whitening, and state
copies) is included in the corresponding per-transition timing sample.
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
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import triton

from .common import add_student_pkg


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "rebuttal_results" / "blackhole"
DRIVER_SOURCE = Path(__file__).resolve()
STUDENT_SOURCE = ROOT / "student_pkg" / "doublesided_cholesky.py"
FEATURE_SOURCE = ROOT / "rebuttal_experiments" / "blackhole_features.py"

SCHEMA_VERSION = "blackhole_stability_v1"
FEATURE_SCHEMA_VERSION = "blackhole_features_v1"
CAMPAIGN_TAG = "blackhole-s4-v1"
HF_REPOSITORY = "polymathic-ai/post_neutron_star_merger"
HF_REVISION = "1fd582f3e29749ac4d0c14bc816bddb2394258cc"
RAW_RELATIVE_PATH = (
    "data/train/post_neutron_star_merger_scenario_4.hdf5"
)
RAW_BYTES = 14_109_638_656
RAW_SHA256 = "21e587c90c006c4b7d37e2b04899bd7e928967ae506e9ceb5afd05dfca6909fd"
TRACE_SHAPE = (181, 65_536)
LADDER_RANKS = (2_048, 4_096, 8_192, 16_384, 32_768, 49_152, 65_536)
STABILITY_RANKS = (8_192, 32_768)
INITIAL_TRANSITIONS = tuple(range(64))
WARMUP_TRANSITION = 64
EVALUATION_TRANSITIONS = (80, 91, 102, 113, 124, 135, 146, 157, 168, 179)
STABILITY_TRANSITIONS = tuple(range(64, 164))
CHECKPOINT_STEPS = (1, 10, 25, 50, 75, 100)
METHOD_ORDER = (
    "fresh_refact",
    "inverse_no_refresh",
    "inverse_refresh_q50",
)
RIDGE = 1.0
REFRESH_INTERVAL = 50
EXPECTED_GPU_PREFIX = "NVIDIA H100"
EXPECTED_GPU_MEMORY = 85_045_477_376
PROBE_SEED = 29_421
PROBE_COUNT = 16
CORRECTNESS_THRESHOLD = 1e-3
CHANNEL_ORDER = (
    "density",
    "internal_energy",
    "pressure",
    "temperature",
    "electron_fraction",
    "entropy",
    "magnetic_field/log_r",
    "magnetic_field/theta",
    "magnetic_field/phi",
    "velocity/log_r",
    "velocity/theta",
    "velocity/phi",
)
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ProtocolError(RuntimeError):
    """Raised when an input or runtime violates the frozen protocol."""


def _atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ProtocolError(f"invalid git commit returned by git: {value!r}")
    return value


def _source_metadata() -> dict[str, Any]:
    commit = _git_commit()
    sources = (DRIVER_SOURCE, STUDENT_SOURCE, FEATURE_SOURCE)
    relative_sources = [str(path.relative_to(ROOT)) for path in sources]
    tracked_dirty = (
        subprocess.run(
            [
                "git",
                "diff-index",
                "--quiet",
                "HEAD",
                "--",
                *relative_sources,
            ],
            cwd=ROOT,
            check=False,
        ).returncode
        != 0
    )
    sources_tracked = all(
        subprocess.run(
            [
                "git",
                "ls-files",
                "--error-unmatch",
                str(path.relative_to(ROOT)),
            ],
            cwd=ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
        for path in sources
    )
    git_dirty = tracked_dirty or not sources_tracked
    if git_dirty:
        raise ProtocolError(
            "the frozen stability campaign requires its driver, kernel, and "
            "feature producer sources to be tracked and clean"
        )
    return {
        "git_commit": commit,
        "git_dirty": False,
        "source_hashes": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in sources
        },
    }


def _assert_sources_unchanged(source: dict[str, Any]) -> None:
    hashes = source["source_hashes"]
    for relative_path, expected in hashes.items():
        path = ROOT / relative_path
        actual = _sha256(path)
        if actual != expected:
            raise ProtocolError(
                f"source changed during stability run: {relative_path}; "
                f"before={expected}, after={actual}"
            )


def _slurm_metadata(rank: int) -> dict[str, str]:
    values = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "job_partition": os.environ.get("SLURM_JOB_PARTITION"),
    }
    for key in ("job_id", "array_job_id", "array_task_id"):
        value = values[key]
        if value is None or not value.isdigit():
            raise ProtocolError(f"SLURM_{key.upper()} must be numeric")
    task_id = int(values["array_task_id"])
    expected_rank = STABILITY_RANKS[task_id] if task_id in (0, 1) else None
    if expected_rank != rank:
        raise ProtocolError(
            f"Slurm array task {task_id} maps to rank {expected_rank}, "
            f"not requested rank {rank}"
        )
    _require_equal(values["cpus_per_task"], "4", "SLURM_CPUS_PER_TASK")
    _require_equal(values["job_partition"], "preempt", "SLURM_JOB_PARTITION")
    return {key: str(value) for key, value in values.items()}


def _require_equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise ProtocolError(
            f"{context}={actual!r}; expected frozen value {expected!r}"
        )


def _resolve_trace_path(manifest_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ProtocolError("feature manifest trace.path must be a nonempty string")
    path = Path(value)
    if path.is_absolute():
        return path
    root_candidate = ROOT / path
    if root_candidate.exists():
        return root_candidate
    return manifest_path.parent / path


def _load_and_validate_manifest(
    manifest_path: Path,
    rank: int,
) -> tuple[dict[str, Any], Path, str, float]:
    if not manifest_path.is_file():
        raise ProtocolError(f"feature manifest does not exist: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise ProtocolError(f"invalid feature manifest JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise ProtocolError("feature manifest must be a JSON object")

    _require_equal(
        manifest.get("schema_version"),
        FEATURE_SCHEMA_VERSION,
        "manifest.schema_version",
    )
    _require_equal(
        manifest.get("campaign_tag"),
        CAMPAIGN_TAG,
        "manifest.campaign_tag",
    )
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ProtocolError("manifest.source must be an object")
    source_path_value = source.get("path")
    if not isinstance(source_path_value, str) or not source_path_value:
        raise ProtocolError("manifest.source.path must be a nonempty string")
    source_path = Path(source_path_value)
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    _require_equal(
        source_path.resolve(),
        FEATURE_SOURCE.resolve(),
        "manifest.source.path",
    )
    source_sha256 = source.get("sha256")
    if not isinstance(source_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", source_sha256
    ):
        raise ProtocolError("manifest.source.sha256 must be a lowercase SHA-256")
    _require_equal(
        _sha256(FEATURE_SOURCE),
        source_sha256,
        "feature producer source SHA-256",
    )

    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise ProtocolError("manifest.dataset must be an object")
    frozen_dataset = {
        "repository": HF_REPOSITORY,
        "revision": HF_REVISION,
        "relative_path": RAW_RELATIVE_PATH,
        "expected_bytes": RAW_BYTES,
        "actual_bytes": RAW_BYTES,
        "expected_sha256": RAW_SHA256,
        "sha256": RAW_SHA256,
    }
    for key, expected in frozen_dataset.items():
        _require_equal(dataset.get(key), expected, f"manifest.dataset.{key}")

    trace = manifest.get("trace")
    if not isinstance(trace, dict):
        raise ProtocolError("manifest.trace must be an object")
    frozen_trace = {
        "shape": list(TRACE_SHAPE),
        "dtype": "float32",
        "format": "npy",
        "fortran_order": False,
        "contiguous_order": "C",
        "chunks": None,
        "compression": None,
        "scaled": False,
        "max_rank": TRACE_SHAPE[1],
    }
    for key, expected in frozen_trace.items():
        _require_equal(trace.get(key), expected, f"manifest.trace.{key}")
    trace_sha256 = trace.get("sha256")
    if not isinstance(trace_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", trace_sha256
    ):
        raise ProtocolError("manifest.trace.sha256 must be a lowercase SHA-256")
    trace_path = _resolve_trace_path(manifest_path, trace.get("path"))
    if not trace_path.is_file():
        raise ProtocolError(f"feature trace does not exist: {trace_path}")
    actual_trace_sha256 = _sha256(trace_path)
    _require_equal(
        actual_trace_sha256,
        trace_sha256,
        "feature trace SHA-256",
    )

    transitions = manifest.get("transitions")
    if not isinstance(transitions, dict):
        raise ProtocolError("manifest.transitions must be an object")
    frozen_transitions = {
        "initial": list(INITIAL_TRANSITIONS),
        "warmup": WARMUP_TRANSITION,
        "evaluation": list(EVALUATION_TRANSITIONS),
        "stability": list(STABILITY_TRANSITIONS),
    }
    for key, expected in frozen_transitions.items():
        _require_equal(
            transitions.get(key),
            expected,
            f"manifest.transitions.{key}",
        )

    scaling = manifest.get("rank_scaling")
    if not isinstance(scaling, dict):
        raise ProtocolError("manifest.rank_scaling must be an object")
    _require_equal(
        scaling.get("ranks"),
        list(LADDER_RANKS),
        "manifest.rank_scaling.ranks",
    )
    scales = scaling.get("scales")
    if not isinstance(scales, dict):
        raise ProtocolError("manifest.rank_scaling.scales must be an object")
    if set(scales) != {str(value) for value in LADDER_RANKS}:
        raise ProtocolError(
            "manifest.rank_scaling.scales keys do not match the frozen ranks"
        )
    frozen_scaling = {
        "definition": (
            "s_r^2 = (1/64) * sum_{t=0}^{63} "
            "||trace[t, :r]||_2^2"
        ),
        "computed_from": "persisted unscaled float32 trace",
        "accumulation_dtype": "float64",
        "consumer_operation": "trace[t, :r] / s_r",
        "calibration_snapshots": list(INITIAL_TRANSITIONS),
    }
    for key, expected in frozen_scaling.items():
        _require_equal(
            scaling.get(key),
            expected,
            f"manifest.rank_scaling.{key}",
        )
    try:
        rank_scale = float(scales[str(rank)])
    except (TypeError, ValueError) as error:
        raise ProtocolError(f"invalid scale for rank {rank}") from error
    if not math.isfinite(rank_scale) or rank_scale <= 0:
        raise ProtocolError(f"rank scale must be positive and finite: {rank_scale}")

    normalization = manifest.get("normalization")
    if not isinstance(normalization, dict):
        raise ProtocolError("manifest.normalization must be an object")
    _require_equal(
        normalization.get("snapshots"),
        list(INITIAL_TRANSITIONS),
        "manifest.normalization.snapshots",
    )
    channels = normalization.get("channels")
    if not isinstance(channels, list) or len(channels) != 12:
        raise ProtocolError(
            "manifest.normalization.channels must contain twelve channels"
        )
    _require_equal(
        [
            channel.get("name") if isinstance(channel, dict) else None
            for channel in channels
        ],
        list(CHANNEL_ORDER),
        "manifest.normalization channel order",
    )
    frequency_order = manifest.get("frequency_order")
    if not isinstance(frequency_order, dict):
        raise ProtocolError("manifest.frequency_order must be an object")
    frozen_frequency = {
        "transform": "separable 3D DCT-II",
        "normalization": "ortho",
        "channel_count": 12,
        "channel_interleave": "frequency-major, then channel order",
        "channel_order": list(CHANNEL_ORDER),
        "prefix_rule": "rank r uses trace[:, :r], so all tested ranks are nested",
    }
    for key, expected in frozen_frequency.items():
        _require_equal(
            frequency_order.get(key),
            expected,
            f"manifest.frequency_order.{key}",
        )
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise ProtocolError("manifest.protocol must be an object")
    frozen_protocol = {
        "ridge": RIDGE,
        "scenario": 4,
        "scenario_name": "torus_MBH_10",
        "dynamic_channels": 12,
        "gcon_included": False,
        "raw_trace_scaled": False,
        "tested_ranks": list(LADDER_RANKS),
    }
    for key, expected in frozen_protocol.items():
        _require_equal(
            protocol.get(key),
            expected,
            f"manifest.protocol.{key}",
        )

    return manifest, trace_path, manifest_sha256, rank_scale


def _materialize_campaign_manifest(
    source_path: Path,
    destination: Path,
) -> None:
    source_bytes = source_path.read_bytes()
    if destination.exists():
        if destination.read_bytes() != source_bytes:
            raise ProtocolError(
                f"campaign manifest differs from supplied manifest: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(
        destination.suffix + f".{os.getpid()}.tmp"
    )
    temporary.write_bytes(source_bytes)
    os.replace(temporary, destination)


def _load_scaled_features(
    trace_path: Path,
    rank: int,
    manifest_scale: float,
) -> tuple[torch.Tensor, float]:
    trace = np.load(trace_path, mmap_mode="r", allow_pickle=False)
    if trace.shape != TRACE_SHAPE:
        raise ProtocolError(
            f"trace shape={trace.shape!r}; expected {TRACE_SHAPE!r}"
        )
    if trace.dtype != np.dtype(np.float32):
        raise ProtocolError(f"trace dtype={trace.dtype}; expected float32")
    if not trace.flags.c_contiguous:
        raise ProtocolError("trace must be C contiguous")

    calibration = np.asarray(trace[:64, :rank], dtype=np.float32)
    scale_squared = float(
        np.sum(
            np.square(calibration, dtype=np.float64),
            dtype=np.float64,
        )
        / 64.0
    )
    recomputed_scale = math.sqrt(scale_squared)
    if not math.isclose(
        recomputed_scale,
        manifest_scale,
        rel_tol=2e-12,
        abs_tol=0.0,
    ):
        raise ProtocolError(
            "manifest rank scale does not match the frozen definition: "
            f"manifest={manifest_scale:.17g}, recomputed={recomputed_scale:.17g}"
        )

    # Stability transition 163 consumes snapshot 164.  Make an owned,
    # contiguous host array before transferring it to the GPU.
    host = np.array(trace[:165, :rank], dtype=np.float32, order="C", copy=True)
    host /= np.float32(manifest_scale)
    features = torch.from_numpy(host).to(device="cuda", non_blocking=False)
    return features.contiguous(), recomputed_scale


def _fresh_whiten(
    L: torch.Tensor,
    M: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    torch.linalg.solve_triangular(
        L,
        M,
        upper=False,
        left=True,
        out=output,
    )
    torch.linalg.solve_triangular(
        L.mT,
        output,
        upper=True,
        left=False,
        out=output,
    )
    return output


def _make_initial_state(
    features: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z_initial = features[:64]
    w_initial = features[1:65]
    G0 = torch.mm(z_initial.mT, z_initial)
    G0.diagonal().add_(RIDGE)
    M0 = torch.mm(z_initial.mT, w_initial)
    L0 = torch.linalg.cholesky(G0)
    identity = torch.eye(rank, device="cuda", dtype=torch.float32)
    Linv0 = torch.empty_like(G0)
    torch.linalg.solve_triangular(
        L0,
        identity,
        upper=False,
        left=True,
        out=Linv0,
    )
    Linv0.tril_()
    Aw0 = torch.empty_like(M0)
    _fresh_whiten(L0, M0, Aw0)
    del identity
    torch.cuda.synchronize()
    return G0, M0, Linv0.contiguous(), Aw0.contiguous()


def _time_cuda(function: Callable[[], None]) -> tuple[float, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start.record()
    function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0, time.perf_counter() - wall_start


def _finite_full(tensor: torch.Tensor, chunk_elements: int = 1 << 24) -> bool:
    rows = tensor.shape[-2]
    elements_per_row = tensor.numel() // rows
    rows_per_chunk = max(1, chunk_elements // elements_per_row)
    for start in range(0, rows, rows_per_chunk):
        if not bool(
            torch.isfinite(tensor[start : start + rows_per_chunk]).all().item()
        ):
            return False
    return True


def _relative_error_chunked(
    actual: torch.Tensor,
    expected: torch.Tensor,
    chunk_elements: int = 1 << 24,
) -> float:
    if actual.shape != expected.shape:
        raise ProtocolError(
            f"relative-error shape mismatch: {actual.shape} != {expected.shape}"
        )
    rows = actual.shape[-2]
    elements_per_row = actual.numel() // rows
    rows_per_chunk = max(1, chunk_elements // elements_per_row)
    numerator = 0.0
    denominator = 0.0
    for start in range(0, rows, rows_per_chunk):
        stop = min(rows, start + rows_per_chunk)
        actual_chunk = actual[start:stop]
        expected_chunk = expected[start:stop]
        difference = actual_chunk - expected_chunk
        numerator += float(torch.sum(difference.double().square()).item())
        denominator += float(torch.sum(expected_chunk.double().square()).item())
        del difference
    return math.sqrt(numerator / max(denominator, np.finfo(np.float64).tiny))


def _probe_matrix(rank: int) -> torch.Tensor:
    generator = np.random.Generator(np.random.PCG64(PROBE_SEED))
    values = generator.integers(
        0,
        2,
        size=(rank, PROBE_COUNT),
        dtype=np.int8,
    )
    probes = values.astype(np.float32)
    probes *= 2.0
    probes -= 1.0
    probes /= math.sqrt(rank)
    return torch.from_numpy(probes).to(device="cuda")


def _probe_relative_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
    probes: torch.Tensor,
) -> float:
    actual_probe = actual @ probes
    expected_probe = expected @ probes
    denominator = torch.linalg.vector_norm(expected_probe).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    return float(
        (
            torch.linalg.vector_norm(actual_probe - expected_probe)
            / denominator
        ).item()
    )


def _fresh_probe_row(
    G: torch.Tensor,
    M: torch.Tensor,
    L: torch.Tensor,
    Aw: torch.Tensor,
    probes: torch.Tensor,
) -> dict[str, float]:
    gp = G @ probes
    reconstructed_gp = L @ (L.mT @ probes)
    mp = M @ probes
    reconstructed_mp = L @ (Aw @ (L.mT @ probes))
    tiny = torch.finfo(torch.float32).tiny
    return {
        "factor_probe_residual": float(
            (
                torch.linalg.vector_norm(reconstructed_gp - gp)
                / torch.linalg.vector_norm(gp).clamp_min(tiny)
            ).item()
        ),
        "operator_probe_residual": float(
            (
                torch.linalg.vector_norm(reconstructed_mp - mp)
                / torch.linalg.vector_norm(mp).clamp_min(tiny)
            ).item()
        ),
    }


def _summary(samples: list[dict[str, Any]]) -> dict[str, float]:
    event = [float(row["event_seconds"]) for row in samples]
    wall = [float(row["wall_seconds"]) for row in samples]
    return {
        "event_seconds_mean": statistics.fmean(event),
        "event_seconds_sample_sd": statistics.stdev(event),
        "event_seconds_median": statistics.median(event),
        "wall_seconds_mean": statistics.fmean(wall),
        "wall_seconds_sample_sd": statistics.stdev(wall),
        "wall_seconds_median": statistics.median(wall),
        "event_seconds_total": math.fsum(event),
        "wall_seconds_total": math.fsum(wall),
    }


def _run_fresh(
    G0: torch.Tensor,
    M0: torch.Tensor,
    features: torch.Tensor,
    probes: torch.Tensor,
) -> dict[str, Any]:
    rank = G0.shape[0]
    G = G0.clone()
    M = M0.clone()
    L = torch.empty_strided(
        (rank, rank),
        (1, rank),
        device="cuda",
        dtype=torch.float32,
    )
    Aw = torch.empty_like(M)
    info = torch.empty((), device="cuda", dtype=torch.int32)
    samples: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []

    for step, transition in enumerate(STABILITY_TRANSITIONS, start=1):
        z = features[transition]
        w = features[transition + 1]

        def update() -> None:
            torch.addr(G, z, z, out=G)
            torch.addr(M, z, w, out=M)
            torch.linalg.cholesky_ex(
                G,
                check_errors=False,
                out=(L, info),
            )
            _fresh_whiten(L, M, Aw)

        event_seconds, wall_seconds = _time_cuda(update)
        info_value = int(info.item())
        if info_value != 0:
            raise ProtocolError(
                f"fresh Cholesky failed at transition {transition}: info={info_value}"
            )
        samples.append(
            {
                "sample_index": step - 1,
                "step": step,
                "transition": transition,
                "event_seconds": event_seconds,
                "wall_seconds": wall_seconds,
                "update_event_seconds": event_seconds,
                "update_wall_seconds": wall_seconds,
                "refresh_event_seconds": 0.0,
                "refresh_wall_seconds": 0.0,
                "refreshed": False,
            }
        )
        if step in CHECKPOINT_STEPS:
            checkpoint_finite = all(
                _finite_full(value) for value in (G, M, L, Aw)
            )
            if not checkpoint_finite:
                raise ProtocolError(
                    f"fresh state is non-finite at transition {transition}"
                )
            checkpoints.append(
                {
                    "step": step,
                    "transition": transition,
                    "measurement_phase": "post_step",
                    "finite_full": checkpoint_finite,
                    **_fresh_probe_row(G, M, L, Aw, probes),
                }
            )

    finite = all(_finite_full(value) for value in (G, M, L, Aw))
    if not finite:
        raise ProtocolError("fresh-refactor state contains non-finite values")
    return {
        "method": "fresh_refact",
        "refresh_interval": 1,
        "refresh_count": len(STABILITY_TRANSITIONS),
        "includes_G_M_outer_updates": True,
        "refresh_in_timed_step": True,
        "samples": samples,
        "summary": _summary(samples),
        "checkpoints": checkpoints,
        "finite_full": finite,
    }


def _allocate_exact_buffers(
    rank: int,
    like: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    L = torch.empty_strided(
        (rank, rank),
        (1, rank),
        device="cuda",
        dtype=torch.float32,
    )
    Linv = torch.empty_like(like)
    Aw = torch.empty_like(like)
    identity = torch.eye(rank, device="cuda", dtype=torch.float32)
    info = torch.empty((), device="cuda", dtype=torch.int32)
    return L, Linv, Aw, identity, info


def _exact_state(
    G: torch.Tensor,
    M: torch.Tensor,
    L: torch.Tensor,
    Linv: torch.Tensor,
    Aw: torch.Tensor,
    identity: torch.Tensor,
    info: torch.Tensor,
) -> None:
    torch.linalg.cholesky_ex(
        G,
        check_errors=False,
        out=(L, info),
    )
    torch.linalg.solve_triangular(
        L,
        identity,
        upper=False,
        left=True,
        out=Linv,
    )
    Linv.tril_()
    _fresh_whiten(L, M, Aw)


def _inverse_checkpoint(
    *,
    step: int,
    transition: int,
    age: int,
    measurement_phase: str,
    G: torch.Tensor,
    M: torch.Tensor,
    Linv: torch.Tensor,
    Aw: torch.Tensor,
    Linv_exact: torch.Tensor,
    Aw_exact: torch.Tensor,
    probes: torch.Tensor,
) -> dict[str, Any]:
    finite = all(_finite_full(value) for value in (G, M, Linv, Aw))
    if not finite:
        raise ProtocolError(
            f"inverse state is non-finite at transition {transition}"
        )
    transformed = Linv.mT @ probes
    factor_expected = probes
    factor_actual = Linv @ (G @ transformed)
    operator_expected = Linv @ (M @ transformed)
    operator_actual = Aw @ probes
    tiny = torch.finfo(torch.float32).tiny
    return {
        "step": step,
        "transition": transition,
        "age_since_refresh": age,
        "measurement_phase": measurement_phase,
        "finite_full": finite,
        "Linv_relative_error": _relative_error_chunked(Linv, Linv_exact),
        "Aw_relative_error": _relative_error_chunked(Aw, Aw_exact),
        "Linv_probe_relative_error": _probe_relative_error(
            Linv,
            Linv_exact,
            probes,
        ),
        "Aw_probe_relative_error": _probe_relative_error(
            Aw,
            Aw_exact,
            probes,
        ),
        "factor_probe_residual": float(
            (
                torch.linalg.vector_norm(factor_actual - factor_expected)
                / torch.linalg.vector_norm(factor_expected).clamp_min(tiny)
            ).item()
        ),
        "operator_probe_residual": float(
            (
                torch.linalg.vector_norm(operator_actual - operator_expected)
                / torch.linalg.vector_norm(operator_expected).clamp_min(tiny)
            ).item()
        ),
    }


def _run_inverse(
    *,
    G0: torch.Tensor,
    M0: torch.Tensor,
    Linv0: torch.Tensor,
    Aw0: torch.Tensor,
    features: torch.Tensor,
    probes: torch.Tensor,
    refresh_interval: int,
    tile: int,
) -> dict[str, Any]:
    add_student_pkg()
    from doublesided_cholesky import (
        TritonInverseInplaceWorkspace,
        incremental_update_triton_inverse_factor_inplace,
    )

    rank = G0.shape[0]
    G = G0.clone()
    M = M0.clone()
    Linv = Linv0.clone()
    Aw = Aw0.clone()
    workspace = TritonInverseInplaceWorkspace(Aw)
    (
        L_exact,
        Linv_exact,
        Aw_exact,
        identity,
        info,
    ) = _allocate_exact_buffers(rank, G)
    samples: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    refresh_count = 0
    last_refresh_step = 0

    for step, transition in enumerate(STABILITY_TRANSITIONS, start=1):
        z = features[transition]
        w = features[transition + 1]

        def update() -> None:
            torch.addr(G, z, z, out=G)
            torch.addr(M, z, w, out=M)
            incremental_update_triton_inverse_factor_inplace(
                Linv,
                Aw,
                z,
                w,
                tile_rows=tile,
                tile_cols=tile,
                setup_num_warps=4,
                sweep_num_warps=1,
                workspace=workspace,
            )

        update_event, update_wall = _time_cuda(update)
        at_refresh = (
            refresh_interval > 0 and step % refresh_interval == 0
        )
        pre_refresh_Linv = None
        pre_refresh_Aw = None
        refresh_event = 0.0
        refresh_wall = 0.0

        if at_refresh:
            # These diagnostic snapshots are explicitly outside timing.
            pre_refresh_Linv = Linv.clone()
            pre_refresh_Aw = Aw.clone()
            torch.cuda.synchronize()

            def refresh() -> None:
                _exact_state(
                    G,
                    M,
                    L_exact,
                    Linv_exact,
                    Aw_exact,
                    identity,
                    info,
                )
                Linv.copy_(Linv_exact)
                Aw.copy_(Aw_exact)

            refresh_event, refresh_wall = _time_cuda(refresh)
            info_value = int(info.item())
            if info_value != 0:
                raise ProtocolError(
                    "refresh Cholesky failed at transition "
                    f"{transition}: info={info_value}"
                )
            refresh_count += 1

        event_seconds = update_event + refresh_event
        wall_seconds = update_wall + refresh_wall
        samples.append(
            {
                "sample_index": step - 1,
                "step": step,
                "transition": transition,
                "event_seconds": event_seconds,
                "wall_seconds": wall_seconds,
                "update_event_seconds": update_event,
                "update_wall_seconds": update_wall,
                "refresh_event_seconds": refresh_event,
                "refresh_wall_seconds": refresh_wall,
                "refreshed": at_refresh,
            }
        )

        if step in CHECKPOINT_STEPS:
            if at_refresh:
                assert pre_refresh_Linv is not None
                assert pre_refresh_Aw is not None
                row = _inverse_checkpoint(
                    step=step,
                    transition=transition,
                    age=step - last_refresh_step,
                    measurement_phase="pre_refresh",
                    G=G,
                    M=M,
                    Linv=pre_refresh_Linv,
                    Aw=pre_refresh_Aw,
                    Linv_exact=Linv_exact,
                    Aw_exact=Aw_exact,
                    probes=probes,
                )
                row.update(
                    {
                        "post_refresh_Linv_relative_error": (
                            _relative_error_chunked(Linv, Linv_exact)
                        ),
                        "post_refresh_Aw_relative_error": (
                            _relative_error_chunked(Aw, Aw_exact)
                        ),
                        "post_refresh_finite_full": all(
                            _finite_full(value)
                            for value in (G, M, Linv, Aw)
                        ),
                    }
                )
                checkpoints.append(row)
                del pre_refresh_Linv, pre_refresh_Aw
                last_refresh_step = step
            else:
                # This exact diagnostic is outside the timed step.
                _exact_state(
                    G,
                    M,
                    L_exact,
                    Linv_exact,
                    Aw_exact,
                    identity,
                    info,
                )
                torch.cuda.synchronize()
                info_value = int(info.item())
                if info_value != 0:
                    raise ProtocolError(
                        "diagnostic Cholesky failed at transition "
                        f"{transition}: info={info_value}"
                    )
                checkpoints.append(
                    _inverse_checkpoint(
                        step=step,
                        transition=transition,
                        age=step - last_refresh_step,
                        measurement_phase="post_step",
                        G=G,
                        M=M,
                        Linv=Linv,
                        Aw=Aw,
                        Linv_exact=Linv_exact,
                        Aw_exact=Aw_exact,
                        probes=probes,
                    )
                )

    finite = all(_finite_full(value) for value in (G, M, Linv, Aw))
    if not finite:
        raise ProtocolError("inverse-factor state contains non-finite values")
    method = (
        "inverse_no_refresh"
        if refresh_interval == 0
        else "inverse_refresh_q50"
    )
    return {
        "method": method,
        "refresh_interval": refresh_interval,
        "refresh_count": refresh_count,
        "includes_G_M_outer_updates": True,
        "refresh_in_timed_step": True,
        "tile_rows": tile,
        "tile_cols": tile,
        "setup_num_warps": 4,
        "sweep_num_warps": 1,
        "workspace_bytes": workspace.nbytes,
        "samples": samples,
        "summary": _summary(samples),
        "checkpoints": checkpoints,
        "finite_full": finite,
    }


def _warm_up(
    G0: torch.Tensor,
    M0: torch.Tensor,
    Linv0: torch.Tensor,
    Aw0: torch.Tensor,
    features: torch.Tensor,
    tile: int,
) -> None:
    add_student_pkg()
    from doublesided_cholesky import (
        TritonInverseInplaceWorkspace,
        incremental_update_triton_inverse_factor_inplace,
    )

    rank = G0.shape[0]
    z = features[WARMUP_TRANSITION]
    w = features[WARMUP_TRANSITION + 1]

    warm_G = G0.clone()
    warm_M = M0.clone()
    warm_L = torch.empty_strided(
        (rank, rank),
        (1, rank),
        device="cuda",
        dtype=torch.float32,
    )
    warm_Aw = torch.empty_like(warm_M)
    warm_info = torch.empty((), device="cuda", dtype=torch.int32)
    torch.addr(warm_G, z, z, out=warm_G)
    torch.addr(warm_M, z, w, out=warm_M)
    torch.linalg.cholesky_ex(
        warm_G,
        check_errors=False,
        out=(warm_L, warm_info),
    )
    _fresh_whiten(warm_L, warm_M, warm_Aw)

    warm_Linv = Linv0.clone()
    warm_inverse_aw = Aw0.clone()
    warm_workspace = TritonInverseInplaceWorkspace(warm_inverse_aw)
    incremental_update_triton_inverse_factor_inplace(
        warm_Linv,
        warm_inverse_aw,
        z,
        w,
        tile_rows=tile,
        tile_cols=tile,
        setup_num_warps=4,
        sweep_num_warps=1,
        workspace=warm_workspace,
    )
    torch.cuda.synchronize()
    info_value = int(warm_info.item())
    if info_value != 0:
        raise ProtocolError(f"warm-up Cholesky failed with info={info_value}")
    del (
        warm_G,
        warm_M,
        warm_L,
        warm_Aw,
        warm_info,
        warm_Linv,
        warm_inverse_aw,
        warm_workspace,
    )
    torch.cuda.empty_cache()


def _validate_method_payload(payload: dict[str, Any]) -> None:
    method = payload["method"]
    samples = payload["samples"]
    checkpoints = payload["checkpoints"]
    if len(samples) != len(STABILITY_TRANSITIONS):
        raise ProtocolError(f"{method}: unexpected timing sample count")
    if [row["transition"] for row in samples] != list(STABILITY_TRANSITIONS):
        raise ProtocolError(f"{method}: timing transition order changed")
    if [row["step"] for row in checkpoints] != list(CHECKPOINT_STEPS):
        raise ProtocolError(f"{method}: checkpoint schedule changed")

    expected_refreshes = (
        len(STABILITY_TRANSITIONS)
        if method == "fresh_refact"
        else len(STABILITY_TRANSITIONS) // REFRESH_INTERVAL
        if method == "inverse_refresh_q50"
        else 0
    )
    if payload["refresh_count"] != expected_refreshes:
        raise ProtocolError(
            f"{method}: refresh_count={payload['refresh_count']}, "
            f"expected {expected_refreshes}"
        )
    expected_refreshed_steps = (
        [50, 100] if method == "inverse_refresh_q50" else []
    )
    actual_refreshed_steps = [
        row["step"] for row in samples if row["refreshed"]
    ]
    if actual_refreshed_steps != expected_refreshed_steps:
        raise ProtocolError(
            f"{method}: refreshed steps={actual_refreshed_steps}, "
            f"expected {expected_refreshed_steps}"
        )
    for row in samples:
        if not math.isclose(
            row["event_seconds"],
            row["update_event_seconds"] + row["refresh_event_seconds"],
            rel_tol=2e-9,
            abs_tol=1e-12,
        ):
            raise ProtocolError(f"{method}: event timing decomposition failed")
        if not math.isclose(
            row["wall_seconds"],
            row["update_wall_seconds"] + row["refresh_wall_seconds"],
            rel_tol=2e-9,
            abs_tol=1e-12,
        ):
            raise ProtocolError(f"{method}: wall timing decomposition failed")

    diagnostic_keys = ("factor_probe_residual", "operator_probe_residual")
    if method != "fresh_refact":
        diagnostic_keys += (
            "Linv_relative_error",
            "Aw_relative_error",
            "Linv_probe_relative_error",
            "Aw_probe_relative_error",
        )
    diagnostic_values = []
    for row in checkpoints:
        if row.get("finite_full") is not True:
            raise ProtocolError(f"{method}: non-finite checkpoint state")
        if (
            "post_refresh_finite_full" in row
            and row["post_refresh_finite_full"] is not True
        ):
            raise ProtocolError(f"{method}: non-finite post-refresh state")
        for key in diagnostic_keys:
            value = float(row[key])
            if not math.isfinite(value) or value < 0:
                raise ProtocolError(f"{method}: invalid checkpoint {key}={value}")
            diagnostic_values.append(value)
        for key in (
            "post_refresh_Linv_relative_error",
            "post_refresh_Aw_relative_error",
        ):
            if key in row:
                value = float(row[key])
                if not math.isfinite(value) or value < 0:
                    raise ProtocolError(
                        f"{method}: invalid checkpoint {key}={value}"
                    )
                diagnostic_values.append(value)
    headline_valid = (
        bool(payload["finite_full"])
        and bool(diagnostic_values)
        and max(diagnostic_values) <= CORRECTNESS_THRESHOLD
    )
    payload["correctness_threshold"] = CORRECTNESS_THRESHOLD
    payload["headline_valid"] = headline_valid


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.campaign_tag != CAMPAIGN_TAG:
        raise ProtocolError(
            f"campaign tag {args.campaign_tag!r} != frozen tag {CAMPAIGN_TAG!r}"
        )
    if not TAG_RE.fullmatch(args.campaign_tag):
        raise ProtocolError(f"unsafe campaign tag: {args.campaign_tag!r}")
    if args.rank not in STABILITY_RANKS:
        raise ProtocolError(
            f"rank {args.rank} is not a frozen stability rank {STABILITY_RANKS}"
        )
    campaign_dir = args.results_root.resolve() / args.campaign_tag
    output = campaign_dir / f"stability_r{args.rank}.json"
    if output.exists():
        raise ProtocolError(f"refusing to overwrite existing artifact: {output}")
    source_metadata = _source_metadata()
    slurm_metadata = _slurm_metadata(args.rank)
    if not torch.cuda.is_available():
        raise ProtocolError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise ProtocolError(
            "the frozen protocol requires exactly one visible GPU, got "
            f"{torch.cuda.device_count()}"
        )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gpu = torch.cuda.get_device_name(0)
    gpu_properties = torch.cuda.get_device_properties(0)
    gpu_memory = gpu_properties.total_memory
    if not gpu.startswith(EXPECTED_GPU_PREFIX):
        raise ProtocolError(
            f"GPU {gpu!r} does not satisfy prefix {EXPECTED_GPU_PREFIX!r}"
        )
    _require_equal(gpu_memory, EXPECTED_GPU_MEMORY, "GPU total memory")

    manifest_path = args.feature_manifest.resolve()
    (
        manifest,
        trace_path,
        manifest_sha256,
        manifest_scale,
    ) = _load_and_validate_manifest(manifest_path, args.rank)
    _materialize_campaign_manifest(
        manifest_path,
        campaign_dir / "feature_manifest.json",
    )

    features, recomputed_scale = _load_scaled_features(
        trace_path,
        args.rank,
        manifest_scale,
    )
    G0, M0, Linv0, Aw0 = _make_initial_state(features, args.rank)
    probes = _probe_matrix(args.rank)
    tile = 8 if args.rank <= 8_192 else 32
    _warm_up(G0, M0, Linv0, Aw0, features, tile)

    torch.cuda.reset_peak_memory_stats()
    methods = [
        _run_fresh(G0, M0, features, probes),
        _run_inverse(
            G0=G0,
            M0=M0,
            Linv0=Linv0,
            Aw0=Aw0,
            features=features,
            probes=probes,
            refresh_interval=0,
            tile=tile,
        ),
        _run_inverse(
            G0=G0,
            M0=M0,
            Linv0=Linv0,
            Aw0=Aw0,
            features=features,
            probes=probes,
            refresh_interval=REFRESH_INTERVAL,
            tile=tile,
        ),
    ]
    _require_equal(
        [row["method"] for row in methods],
        list(METHOD_ORDER),
        "method order",
    )
    for method_payload in methods:
        _validate_method_payload(method_payload)
    peak_memory = torch.cuda.max_memory_allocated()

    _assert_sources_unchanged(source_metadata)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "cuda_blackhole_stability",
        "campaign_tag": args.campaign_tag,
        "rank": args.rank,
        "dtype": "float32",
        "ridge": RIDGE,
        "initial_transitions": list(INITIAL_TRANSITIONS),
        "warmup_transition": WARMUP_TRANSITION,
        "stability_transitions": list(STABILITY_TRANSITIONS),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "method_order": list(METHOD_ORDER),
        "rank_scale": manifest_scale,
        "rank_scale_recomputed": recomputed_scale,
        "feature_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
            "campaign_copy": str(campaign_dir / "feature_manifest.json"),
        },
        "trace": {
            "path": str(trace_path.resolve()),
            "sha256": manifest["trace"]["sha256"],
            "shape": list(TRACE_SHAPE),
            "dtype": "float32",
        },
        "dataset": {
            "repository": HF_REPOSITORY,
            "revision": HF_REVISION,
            "relative_path": RAW_RELATIVE_PATH,
            "bytes": RAW_BYTES,
            "sha256": RAW_SHA256,
        },
        "implementation": source_metadata,
        "hardware": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "gpu": gpu,
            "gpu_total_memory": gpu_memory,
            "compute_capability": [
                int(gpu_properties.major),
                int(gpu_properties.minor),
            ],
            "visible_device_count": torch.cuda.device_count(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "python": sys.version,
            "numpy": np.__version__,
            "triton": triton.__version__,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "slurm": slurm_metadata,
        },
        "timing_contract": {
            "unit": "seconds",
            "samples_are_per_transition": True,
            "G_M_outer_updates_included": True,
            "q50_refresh_included": True,
            "diagnostics_excluded": True,
            "cuda_event_and_synchronized_wall_samples_preserved": True,
        },
        "probe_contract": {
            "seed": PROBE_SEED,
            "count": probes.shape[1],
            "distribution": "NumPy PCG64 Rademacher (+/-1/sqrt(rank))",
        },
        "peak_allocated_bytes_including_harness": peak_memory,
        "methods": methods,
    }

    _atomic_json_dump(payload, output)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"wrote {output}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True, choices=STABILITY_RANKS)
    parser.add_argument("--campaign-tag", required=True)
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=RESULTS_ROOT / CAMPAIGN_TAG / "feature_manifest.json",
    )
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
