from __future__ import annotations

import argparse
import hashlib
import os
import platform
import socket
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .common import (
    RESULTS,
    add_student_pkg,
    atomic_json_dump,
    result_path,
    tensor_relative_error,
)
from .cuda_scaling import batched_whiten


def _time_cuda(function, prepare, repeats: int) -> tuple[list[float], list[float]]:
    event_samples = []
    wall_samples = []
    for _ in range(repeats):
        prepare()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        function()
        end.record()
        torch.cuda.synchronize()
        event_samples.append(start.elapsed_time(end) / 1000.0)
        wall_samples.append(time.perf_counter() - wall_start)
    return event_samples, wall_samples


def _vectors(rank: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    z = torch.randn(rank, device="cuda", generator=generator) * 0.05
    w = torch.randn(rank, device="cuda", generator=generator) * 0.05
    return z.contiguous(), w.contiguous()


def _finite(tensor: torch.Tensor, chunk_elements: int = 1 << 26) -> bool:
    """Check every element without materializing a tensor-sized predicate.

    A dense FP32 rank-65536 matrix occupies 16 GiB.  The eager expression
    ``torch.isfinite(tensor).all()`` can request another matrix-sized buffer,
    which does not fit beside the four recurrent state matrices on an 80 GB
    H100.  Row chunks work for both row-major state and column-major Cholesky
    output while bounding the predicate buffer to roughly 64 MiB.
    """
    rows = tensor.shape[-2]
    elements_per_row = tensor.numel() // rows
    rows_per_chunk = max(1, chunk_elements // elements_per_row)
    for start in range(0, rows, rows_per_chunk):
        stop = start + rows_per_chunk
        finite_chunk = torch.isfinite(tensor[..., start:stop, :])
        if not bool(finite_chunk.all().item()):
            return False
    return True


def _probe_matrix(rank: int, width: int, seed: int) -> torch.Tensor:
    """Return the campaign's deterministic, cross-process sketch matrix."""
    generator = np.random.Generator(np.random.PCG64(seed + rank))
    values = generator.integers(
        0,
        2,
        size=(rank, width),
        dtype=np.int8,
    )
    values = values.astype(np.float32)
    values *= 2.0
    values -= 1.0
    values *= rank**-0.5
    return torch.from_numpy(values).to(device="cuda")


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(expected).clamp_min(
        torch.finfo(expected.dtype).tiny
    )
    return float(
        (torch.linalg.vector_norm(actual - expected) / denominator).item()
    )


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(item) for item in contiguous.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _probe_path(args: argparse.Namespace, mode: str) -> Path:
    tuning_suffix = (
        f"_tr{args.tile_rows}_tc{args.tile_cols}"
        f"_setupw{args.setup_num_warps}_sweepw{args.sweep_num_warps}"
        if mode == "inverse_inplace_recurrent"
        else ""
    )
    stem = (
        f"cuda_large_compare_{mode}_r{args.rank}_seed{args.seed}"
        f"{tuning_suffix}_{os.environ.get('SLURM_JOB_ID', 'local')}"
    )
    if args.output_suffix:
        stem += f"_{args.output_suffix}"
    return RESULTS / "cuda_tuning" / f"{stem}_probes.npz"


def _write_probe_actions(
    args: argparse.Namespace,
    mode: str,
    probe: torch.Tensor,
    inverse_factor_action: torch.Tensor,
    whitened_operator_action: torch.Tensor,
    residuals: dict[str, float],
) -> dict:
    """Write small O(rq) cross-method actions after the timed region."""
    destination = _probe_path(args, mode)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".npz",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    probe_array = probe.detach().cpu().numpy()
    inverse_factor_array = inverse_factor_action.detach().cpu().numpy()
    whitened_operator_array = whitened_operator_action.detach().cpu().numpy()
    try:
        np.savez(
            temporary,
            probe=probe_array,
            inverse_factor_action=inverse_factor_array,
            whitened_operator_action=whitened_operator_array,
        )
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "probe_validation": True,
        "probe_generator": "numpy.PCG64 Rademacher",
        "probe_seed": args.probe_seed + args.rank,
        "probe_width": args.probe_width,
        "probe_tolerance": args.probe_tolerance,
        "probe_identity_residuals": residuals,
        "probe_artifact": str(destination.relative_to(RESULTS.parent)),
        "probe_artifact_bytes": destination.stat().st_size,
        "probe_artifact_sha256": _sha256_file(destination),
        "probe_array_sha256": _sha256_array(probe_array),
        "inverse_factor_action_sha256": _sha256_array(inverse_factor_array),
        "whitened_operator_action_sha256": _sha256_array(
            whitened_operator_array
        ),
        "probe_asymptotic_extra_storage": "O(r*q)",
        "probe_excluded_from_timing": True,
    }


def _inverse_probe_validation(
    args: argparse.Namespace,
    G: torch.Tensor,
    M: torch.Tensor,
    Linv: torch.Tensor,
    Aw: torch.Tensor,
) -> dict:
    probe = _probe_matrix(args.rank, args.probe_width, args.probe_seed)
    inverse_factor_action = Linv @ probe
    whitened_operator_action = Aw @ probe
    inverse_transpose_action = Linv.mT @ probe
    metric_identity_action = Linv @ (G @ inverse_transpose_action)
    operator_definition_action = Linv @ (M @ inverse_transpose_action)
    residuals = {
        "inverse_metric_identity": _relative_error(
            metric_identity_action,
            probe,
        ),
        "whitened_operator_definition": _relative_error(
            whitened_operator_action,
            operator_definition_action,
        ),
    }
    if max(residuals.values()) > args.probe_tolerance:
        raise RuntimeError(
            "inverse-factor sketch validation failed: "
            f"{residuals!r}, tolerance={args.probe_tolerance}"
        )
    return _write_probe_actions(
        args,
        "inverse_inplace_recurrent",
        probe,
        inverse_factor_action,
        whitened_operator_action,
        residuals,
    )


def _refact_probe_validation(
    args: argparse.Namespace,
    G: torch.Tensor,
    M: torch.Tensor,
    L: torch.Tensor,
    Aw: torch.Tensor,
) -> dict:
    probe = _probe_matrix(args.rank, args.probe_width, args.probe_seed)
    inverse_factor_action = torch.linalg.solve_triangular(
        L,
        probe,
        upper=False,
    )
    whitened_operator_action = Aw @ probe
    factor_transpose_action = L.mT @ probe
    residuals = {
        "factor_metric_identity": _relative_error(
            L @ factor_transpose_action,
            G @ probe,
        ),
        "whitened_operator_definition": _relative_error(
            L @ (Aw @ factor_transpose_action),
            M @ probe,
        ),
    }
    if max(residuals.values()) > args.probe_tolerance:
        raise RuntimeError(
            "fresh-refactor sketch validation failed: "
            f"{residuals!r}, tolerance={args.probe_tolerance}"
        )
    return _write_probe_actions(
        args,
        "fresh_refact_recurrent",
        probe,
        inverse_factor_action,
        whitened_operator_action,
        residuals,
    )


def run_inverse(args: argparse.Namespace) -> dict:
    add_student_pkg()
    from doublesided_cholesky import (
        TritonInverseInplaceWorkspace,
        incremental_update_triton_inverse_factor_inplace,
    )

    rank = args.rank
    z, w = _vectors(rank, args.seed)
    inverse_diagonal = rank**-0.5
    operator_diagonal = 1.0 / rank
    G = torch.empty((rank, rank), device="cuda", dtype=torch.float32)
    M = torch.empty_like(G)
    Linv = torch.empty((rank, rank), device="cuda", dtype=torch.float32)
    Aw = torch.empty_like(Linv)
    workspace = TritonInverseInplaceWorkspace(Aw)

    def prepare():
        G.zero_()
        G.diagonal().fill_(float(rank))
        M.zero_()
        M.diagonal().fill_(1.0)
        Linv.zero_()
        Linv.diagonal().fill_(inverse_diagonal)
        Aw.zero_()
        Aw.diagonal().fill_(operator_diagonal)

    def update():
        torch.addr(G, z, z, out=G)
        torch.addr(M, z, w, out=M)
        return incremental_update_triton_inverse_factor_inplace(
            Linv,
            Aw,
            z,
            w,
            tile_rows=args.tile_rows,
            tile_cols=args.tile_cols,
            setup_num_warps=args.setup_num_warps,
            sweep_num_warps=args.sweep_num_warps,
            workspace=workspace,
        )

    for _ in range(args.warmup):
        prepare()
        update()
        torch.cuda.synchronize()
    prepare()
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    event_samples, wall_samples = _time_cuda(update, prepare, args.repeats)
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    finite = _finite(G) and _finite(M) and _finite(Linv) and _finite(Aw)
    if not finite:
        raise RuntimeError("inverse-factor output contains non-finite values")
    Linv_error = None
    Aw_error = None
    if rank <= args.correctness_max_rank:
        L_exact = torch.linalg.cholesky(G)
        identity = torch.eye(
            rank,
            device="cuda",
            dtype=torch.float32,
        )
        Linv_exact = torch.linalg.solve_triangular(
            L_exact,
            identity,
            upper=False,
        )
        Aw_exact = batched_whiten(
            L_exact.unsqueeze(0),
            M.unsqueeze(0),
        )[0]
        Linv_error = tensor_relative_error(Linv, Linv_exact)
        Aw_error = tensor_relative_error(Aw, Aw_exact)
        if Linv_error > 5e-6 or Aw_error > 5e-6:
            raise RuntimeError(
                "inverse-factor correctness failed: "
                f"Linv={Linv_error:.3e}, Aw={Aw_error:.3e}"
            )
    payload = {
        "mode": "inverse_inplace_recurrent",
        "rank": rank,
        "seed": args.seed,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "event_seconds_samples": event_samples,
        "wall_seconds_samples": wall_samples,
        "event_seconds_median": statistics.median(event_samples),
        "wall_seconds_median": statistics.median(wall_samples),
        "updates_per_second": 1.0 / statistics.median(event_samples),
        "finite_full": finite,
        "correctness_checked": rank <= args.correctness_max_rank,
        "Linv_relative_error": Linv_error,
        "Aw_relative_error": Aw_error,
        "dense_state_matrices": 4,
        "includes_G_M_outer_updates": True,
        "maintains_exact_G_M_for_refresh": True,
        "workspace_bytes": workspace.nbytes,
        "allocated_before_timing_bytes": allocated_before,
        "reserved_before_timing_bytes": reserved_before,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "tile_rows": args.tile_rows,
        "tile_cols": args.tile_cols,
        "setup_num_warps": args.setup_num_warps,
        "sweep_num_warps": args.sweep_num_warps,
    }
    if rank >= args.probe_min_rank:
        payload.update(_inverse_probe_validation(args, G, M, Linv, Aw))
    else:
        payload["probe_validation"] = False
    return payload


def run_refact(args: argparse.Namespace) -> dict:
    rank = args.rank
    z, w = _vectors(rank, args.seed)
    G = torch.empty((rank, rank), device="cuda", dtype=torch.float32)
    M = torch.empty_like(G)
    # PyTorch's CUDA Cholesky output is column-major.  Supplying that layout
    # directly avoids a hidden full-sized layout proxy and its copy-back.
    L = torch.empty_strided(
        (rank, rank),
        (1, rank),
        device="cuda",
        dtype=torch.float32,
    )
    Aw = torch.empty_like(G)
    info = torch.empty((), device="cuda", dtype=torch.int32)

    def prepare():
        G.zero_()
        G.diagonal().fill_(float(rank))
        M.zero_()
        M.diagonal().fill_(1.0)
        L.zero_()
        Aw.zero_()
        info.zero_()

    def refactor():
        torch.addr(G, z, z, out=G)
        torch.addr(M, z, w, out=M)
        # Preserve recurrent G and write directly to PyTorch's native
        # column-major Cholesky output layout.
        torch.linalg.cholesky_ex(
            G,
            check_errors=False,
            out=(L, info),
        )
        torch.linalg.solve_triangular(
            L,
            M,
            upper=False,
            left=True,
            out=Aw,
        )
        torch.linalg.solve_triangular(
            L.mT,
            Aw,
            upper=True,
            left=False,
            out=Aw,
        )
        return L, Aw

    for _ in range(args.warmup):
        prepare()
        refactor()
        torch.cuda.synchronize()
        if int(info.item()) != 0:
            raise RuntimeError(f"warmup Cholesky failed with info={int(info.item())}")
    prepare()
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    event_samples, wall_samples = _time_cuda(refactor, prepare, args.repeats)
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    info_value = int(info.item())
    finite = _finite(G) and _finite(M) and _finite(L) and _finite(Aw)
    if info_value != 0:
        raise RuntimeError(f"Cholesky failed with info={info_value}")
    if not finite:
        raise RuntimeError("fresh-refactor output contains non-finite values")
    payload = {
        "mode": "fresh_refact_recurrent",
        "rank": rank,
        "seed": args.seed,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "event_seconds_samples": event_samples,
        "wall_seconds_samples": wall_samples,
        "event_seconds_median": statistics.median(event_samples),
        "wall_seconds_median": statistics.median(wall_samples),
        "updates_per_second": 1.0 / statistics.median(event_samples),
        "finite_full": finite,
        "cholesky_info": info_value,
        "dense_state_matrices": 4,
        "cholesky_output_layout": "column-major",
        "includes_G_M_outer_updates": True,
        "destructive_inputs": False,
        "maintains_exact_G_M_for_next_update": True,
        "allocated_before_timing_bytes": allocated_before,
        "reserved_before_timing_bytes": reserved_before,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }
    if rank >= args.probe_min_rank:
        payload.update(_refact_probe_validation(args, G, M, L, Aw))
    else:
        payload["probe_validation"] = False
    return payload


def alias_check(rank: int, seed: int) -> dict:
    z, w = _vectors(rank, seed)
    G_plus = torch.eye(rank, device="cuda", dtype=torch.float32) * rank
    torch.addr(G_plus, z, z, out=G_plus)
    M_plus = torch.eye(rank, device="cuda", dtype=torch.float32)
    torch.addr(M_plus, z, w, out=M_plus)
    L_reference = torch.linalg.cholesky(G_plus)
    Aw_reference = torch.linalg.solve_triangular(
        L_reference,
        M_plus,
        upper=False,
    )
    Aw_reference = torch.linalg.solve_triangular(
        L_reference.mT,
        Aw_reference,
        upper=True,
        left=False,
    )

    G_test = torch.eye(rank, device="cuda", dtype=torch.float32) * rank
    M_test = torch.eye(rank, device="cuda", dtype=torch.float32)
    L_test = torch.empty_strided(
        (rank, rank),
        (1, rank),
        device="cuda",
        dtype=torch.float32,
    )
    Aw_test = torch.empty_like(M_test)
    G_pointer = G_test.data_ptr()
    M_pointer = M_test.data_ptr()
    L_pointer = L_test.data_ptr()
    Aw_pointer = Aw_test.data_ptr()
    info = torch.empty((), device="cuda", dtype=torch.int32)
    torch.addr(G_test, z, z, out=G_test)
    torch.addr(M_test, z, w, out=M_test)
    torch.linalg.cholesky_ex(
        G_test,
        check_errors=False,
        out=(L_test, info),
    )
    torch.linalg.solve_triangular(
        L_test,
        M_test,
        upper=False,
        out=Aw_test,
    )
    torch.linalg.solve_triangular(
        L_test.mT,
        Aw_test,
        upper=True,
        left=False,
        out=Aw_test,
    )
    torch.cuda.synchronize()
    payload = {
        "mode": "alias_check",
        "rank": rank,
        "seed": seed,
        "cholesky_info": int(info.item()),
        "factor_pointer_preserved": G_test.data_ptr() == G_pointer,
        "operator_pointer_preserved": M_test.data_ptr() == M_pointer,
        "factor_output_pointer_preserved": L_test.data_ptr() == L_pointer,
        "operator_output_pointer_preserved": Aw_test.data_ptr() == Aw_pointer,
        "L_relative_error": float(
            torch.linalg.norm(L_test - L_reference)
            / torch.linalg.norm(L_reference)
        ),
        "Aw_relative_error": float(
            torch.linalg.norm(Aw_test - Aw_reference)
            / torch.linalg.norm(Aw_reference)
        ),
    }
    if payload["cholesky_info"] != 0:
        raise RuntimeError(
            f"alias-check Cholesky failed with info={payload['cholesky_info']}"
        )
    pointer_fields = (
        "factor_pointer_preserved",
        "operator_pointer_preserved",
        "factor_output_pointer_preserved",
        "operator_output_pointer_preserved",
    )
    if not all(payload[field] for field in pointer_fields):
        raise RuntimeError("alias-check changed a preallocated tensor pointer")
    if payload["L_relative_error"] > 5e-6 or payload["Aw_relative_error"] > 5e-6:
        raise RuntimeError(
            "alias-check numerical mismatch: "
            f"L={payload['L_relative_error']:.3e}, "
            f"Aw={payload['Aw_relative_error']:.3e}"
        )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("inverse", "refact", "alias-check"),
        required=True,
    )
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tile-rows", type=int, default=8)
    parser.add_argument("--tile-cols", type=int, default=8)
    parser.add_argument("--setup-num-warps", type=int, default=4)
    parser.add_argument("--sweep-num-warps", type=int, default=4)
    parser.add_argument("--output-suffix")
    # Exact fresh-factor comparison remains comfortably within 80 GB through
    # r=32768. Larger rows retain full finite-output checks only.
    parser.add_argument("--correctness-max-rank", type=int, default=32768)
    parser.add_argument("--probe-min-rank", type=int, default=49152)
    parser.add_argument("--probe-width", type=int, default=16)
    parser.add_argument("--probe-seed", type=int, default=29421)
    parser.add_argument("--probe-tolerance", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.mode == "inverse":
        payload = run_inverse(args)
    elif args.mode == "refact":
        payload = run_refact(args)
    else:
        payload = alias_check(args.rank, args.seed)
    payload.update(
        {
            "experiment": "cuda_large_rank_compare",
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_memory": torch.cuda.get_device_properties(0).total_memory,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "output_suffix": args.output_suffix,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "matrix_bytes": args.rank * args.rank * 4,
        }
    )
    tuning_suffix = (
        f"_tr{args.tile_rows}_tc{args.tile_cols}"
        f"_setupw{args.setup_num_warps}_sweepw{args.sweep_num_warps}"
        if args.mode == "inverse"
        else ""
    )
    stem = (
        f"cuda_large_compare_{payload['mode']}_r{args.rank}_seed{args.seed}"
        f"{tuning_suffix}_"
        f"{os.environ.get('SLURM_JOB_ID', 'local')}"
    )
    if args.output_suffix:
        stem += f"_{args.output_suffix}"
    output = result_path("cuda_tuning", stem)
    atomic_json_dump(payload, output)
    print(payload, flush=True)
    print(f"wrote {output}", flush=True)
