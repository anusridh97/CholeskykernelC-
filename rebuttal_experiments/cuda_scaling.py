from __future__ import annotations

import argparse
import os
import platform
import socket
import statistics
import time

import torch

from .common import (
    add_student_pkg,
    atomic_json_dump,
    configure_threads,
    result_path,
    set_reproducible_seed,
    tensor_relative_error,
)


def batched_whiten(L: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    left = torch.linalg.solve_triangular(L, M, upper=False)
    return torch.linalg.solve_triangular(L, left.mT, upper=False).mT.contiguous()


def make_problem(batch: int, rank: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    raw = torch.randn(batch, rank, rank, device="cuda", generator=generator)
    G = raw @ raw.mT + rank * torch.eye(rank, device="cuda").unsqueeze(0)
    L = torch.linalg.cholesky(G).contiguous()
    M = torch.randn(batch, rank, rank, device="cuda", generator=generator)
    Aw = batched_whiten(L, M)
    z = torch.randn(batch, rank, device="cuda", generator=generator) * 0.05
    w = torch.randn(batch, rank, device="cuda", generator=generator) * 0.05
    return G, L, M, Aw, z, w


def time_cuda(function, repeats: int, prepare=None):
    event_samples = []
    wall_samples = []
    for _ in range(repeats):
        if prepare is not None:
            prepare()
            torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start_event.record()
        function()
        end_event.record()
        torch.cuda.synchronize()
        wall_samples.append(time.perf_counter() - wall_start)
        event_samples.append(start_event.elapsed_time(end_event) / 1000.0)
    return (
        statistics.median(event_samples),
        statistics.median(wall_samples),
        event_samples,
        wall_samples,
    )


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    configure_threads(args.threads)
    set_reproducible_seed(args.seed)
    add_student_pkg()
    from doublesided_cholesky import (
        TritonTiledWorkspace,
        incremental_update_triton_cumsum,
        incremental_update_triton_inverse_factor,
        incremental_update_triton_inverse_state,
        incremental_update_triton_parallel,
        incremental_update_triton_scan,
        incremental_update_triton_batched,
        incremental_update_triton_tiled_legacy,
        incremental_update_triton_tiled_v2,
    )

    G, L, M, Aw, z, w = make_problem(args.batch, args.rank, args.seed)
    L_work = L.clone()
    Linv = (
        torch.linalg.inv(L).tril().contiguous()
        if args.impl in ("inverse", "inverse_factor")
        else None
    )
    Linv_work = Linv.clone() if Linv is not None else None
    tile_rows = getattr(args, "tile_rows", None)
    tile_cols = getattr(args, "tile_cols", None)
    sweep_num_warps = getattr(args, "sweep_num_warps", None)
    solve_width = getattr(args, "solve_width", 2)
    solve_right = getattr(args, "solve_right", False)
    inverse_gemv_backend = getattr(args, "inverse_gemv_backend", "matmul")
    output_suffix = getattr(args, "output_suffix", None)
    if args.impl == "tiled_v2":
        tile_rows = tile_rows or max(1, min(8, args.rank // 128))
        tile_cols = tile_cols or max(1, min(8, args.rank // 128))
        sweep_num_warps = sweep_num_warps or 1
    elif args.impl in (
        "scan",
        "cumsum",
        "parallel",
        "inverse",
        "inverse_factor",
    ):
        tile_rows = tile_rows or max(1, min(8, 2048 // args.rank))
        tile_cols = tile_cols or max(1, min(8, 2048 // args.rank))
        sweep_num_warps = sweep_num_warps or 8
    workspace = None
    if args.impl in (
        "tiled_v2",
        "scan",
        "cumsum",
        "parallel",
        "inverse",
        "inverse_factor",
    ):
        workspace = TritonTiledWorkspace(
            Aw,
            solve_width=solve_width if args.impl == "parallel" else 2,
        )

    def triton_update():
        if args.impl == "fused":
            return incremental_update_triton_batched(
                L_work, Aw, z, w, num_warps=args.num_warps
            )
        if args.impl == "tiled":
            if args.batch != 1:
                raise ValueError("the tiled implementation currently supports batch=1")
            aw_out, vz = incremental_update_triton_tiled_legacy(
                L_work[0], Aw[0], z[0], w[0], num_warps=args.num_warps
            )
            return aw_out.unsqueeze(0), vz.unsqueeze(0)
        if args.impl == "tiled_v2":
            return incremental_update_triton_tiled_v2(
                L_work,
                Aw,
                z,
                w,
                tile_rows=tile_rows,
                tile_cols=tile_cols,
                setup_num_warps=args.num_warps,
                sweep_num_warps=sweep_num_warps or 1,
                workspace=workspace,
            )
        if args.impl == "scan":
            return incremental_update_triton_scan(
                L_work,
                Aw,
                z,
                w,
                tile_rows=tile_rows,
                tile_cols=tile_cols,
                setup_num_warps=args.num_warps,
                sweep_num_warps=sweep_num_warps or 8,
                workspace=workspace,
            )
        if args.impl == "cumsum":
            return incremental_update_triton_cumsum(
                L_work,
                Aw,
                z,
                w,
                tile_rows=tile_rows,
                tile_cols=tile_cols,
                setup_num_warps=args.num_warps,
                sweep_num_warps=sweep_num_warps or 8,
                workspace=workspace,
            )
        if args.impl == "parallel":
            return incremental_update_triton_parallel(
                L_work,
                Aw,
                z,
                w,
                tile_rows=tile_rows,
                tile_cols=tile_cols,
                setup_num_warps=args.num_warps,
                sweep_num_warps=sweep_num_warps or 8,
                solve_width=solve_width,
                solve_right=solve_right,
                workspace=workspace,
            )
        if args.impl == "inverse":
            return incremental_update_triton_inverse_state(
                L_work,
                Linv_work,
                Aw,
                z,
                w,
                tile_rows=tile_rows,
                tile_cols=tile_cols,
                setup_num_warps=args.num_warps,
                sweep_num_warps=sweep_num_warps or 8,
                gemv_backend=inverse_gemv_backend,
                workspace=workspace,
            )
        if args.impl == "inverse_factor":
            return incremental_update_triton_inverse_factor(
                Linv_work,
                Aw,
                z,
                w,
                tile_rows=tile_rows,
                tile_cols=tile_cols,
                setup_num_warps=args.num_warps,
                sweep_num_warps=sweep_num_warps or 8,
                gemv_backend=inverse_gemv_backend,
                workspace=workspace,
            )
        raise ValueError(f"unknown CUDA implementation: {args.impl}")

    def run_single_for_correctness(L_test, Linv_test=None):
        if args.impl == "fused":
            return incremental_update_triton_batched(
                L_test, Aw, z, w, num_warps=args.num_warps
            )
        if args.impl == "tiled":
            if args.batch != 1:
                raise ValueError("the original tiled implementation supports batch=1")
            aw_out, vz = incremental_update_triton_tiled_legacy(
                L_test[0], Aw[0], z[0], w[0], num_warps=args.num_warps
            )
            return aw_out.unsqueeze(0), vz.unsqueeze(0)
        elif args.impl == "tiled_v2":
            return incremental_update_triton_tiled_v2(
                L_test,
                Aw,
                z,
                w,
                tile_rows=tile_rows,
                tile_cols=tile_cols,
                setup_num_warps=args.num_warps,
                sweep_num_warps=sweep_num_warps or 1,
                workspace=workspace,
            )
        elif args.impl == "scan":
            return incremental_update_triton_scan(
                L_test,
                Aw,
                z,
                w,
                tile_rows=tile_rows,
                tile_cols=tile_cols,
                setup_num_warps=args.num_warps,
                sweep_num_warps=sweep_num_warps or 8,
                workspace=workspace,
            )
        elif args.impl == "cumsum":
            return incremental_update_triton_cumsum(
                L_test,
                Aw,
                z,
                w,
                tile_rows=tile_rows,
                tile_cols=tile_cols,
                setup_num_warps=args.num_warps,
                sweep_num_warps=sweep_num_warps or 8,
                workspace=workspace,
            )
        elif args.impl == "parallel":
            return incremental_update_triton_parallel(
                L_test,
                Aw,
                z,
                w,
                tile_rows=tile_rows,
                tile_cols=tile_cols,
                setup_num_warps=args.num_warps,
                sweep_num_warps=sweep_num_warps or 8,
                solve_width=solve_width,
                solve_right=solve_right,
                workspace=workspace,
            )
        elif args.impl == "inverse":
            if Linv_test is None:
                raise ValueError("inverse implementation requires Linv_test")
            return incremental_update_triton_inverse_state(
                L_test,
                Linv_test,
                Aw,
                z,
                w,
                tile_rows=tile_rows,
                tile_cols=tile_cols,
                setup_num_warps=args.num_warps,
                sweep_num_warps=sweep_num_warps or 8,
                gemv_backend=inverse_gemv_backend,
                workspace=workspace,
            )
        elif args.impl == "inverse_factor":
            if Linv_test is None:
                raise ValueError("inverse-factor implementation requires Linv_test")
            return incremental_update_triton_inverse_factor(
                Linv_test,
                Aw,
                z,
                w,
                tile_rows=tile_rows,
                tile_cols=tile_cols,
                setup_num_warps=args.num_warps,
                sweep_num_warps=sweep_num_warps or 8,
                gemv_backend=inverse_gemv_backend,
                workspace=workspace,
            )
        else:
            raise ValueError(f"unknown CUDA implementation: {args.impl}")

    def refactor():
        G_new = G + z.unsqueeze(2) @ z.unsqueeze(1)
        M_new = M + z.unsqueeze(2) @ w.unsqueeze(1)
        L_new = torch.linalg.cholesky(G_new)
        return L_new, batched_whiten(L_new, M_new)

    for _ in range(args.warmup):
        L_work.copy_(L)
        if Linv_work is not None:
            Linv_work.copy_(Linv)
        triton_update()
        refactor()
    torch.cuda.synchronize()

    L_exact, Aw_exact = refactor()
    L_test = L.clone()
    Linv_test = (
        torch.linalg.inv(L_test).tril().contiguous()
        if args.impl in ("inverse", "inverse_factor")
        else None
    )
    Aw_test, _ = run_single_for_correctness(L_test, Linv_test)
    torch.cuda.synchronize()
    L_error = (
        None
        if args.impl == "inverse_factor"
        else tensor_relative_error(L_test, L_exact)
    )
    Aw_error = tensor_relative_error(Aw_test, Aw_exact)
    Linv_error = (
        tensor_relative_error(Linv_test, torch.linalg.inv(L_exact))
        if Linv_test is not None
        else None
    )
    if (
        (L_error is not None and L_error > args.tolerance)
        or Aw_error > args.tolerance
        or (Linv_error is not None and Linv_error > args.tolerance)
    ):
        raise RuntimeError(
            f"CUDA correctness failed: L={L_error}, "
            f"Linv={Linv_error}, Aw={Aw_error:.3e}"
        )

    memory_baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    def prepare_incremental():
        L_work.copy_(L)
        if Linv_work is not None:
            Linv_work.copy_(Linv)

    (
        triton_event,
        triton_wall,
        triton_event_samples,
        triton_wall_samples,
    ) = time_cuda(
        triton_update,
        args.repeats,
        prepare=prepare_incremental,
    )
    triton_peak = torch.cuda.max_memory_allocated() - memory_baseline
    memory_baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    (
        refact_event,
        refact_wall,
        refact_event_samples,
        refact_wall_samples,
    ) = time_cuda(refactor, args.repeats)
    refact_peak = torch.cuda.max_memory_allocated() - memory_baseline

    rank = args.rank
    batch = args.batch
    incremental_flops = batch * (18 * rank * rank + 15 * rank)
    refactor_flops = batch * ((7.0 / 3.0) * rank**3 + 4 * rank**2)
    minimum_incremental_bytes = batch * 4 * (4 * rank * rank + 8 * rank)
    row = {
        "rank": rank,
        "batch": batch,
        "impl": args.impl,
        "num_warps": args.num_warps,
        "sweep_num_warps": sweep_num_warps,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "solve_width": solve_width if args.impl == "parallel" else None,
        "solve_right": solve_right if args.impl == "parallel" else None,
        "cached_inverse": args.impl in ("inverse", "inverse_factor"),
        "materializes_L": args.impl != "inverse_factor",
        "inverse_layout_contract": (
            "row-major contiguous lower triangle with zero upper triangle"
            if args.impl in ("inverse", "inverse_factor")
            else None
        ),
        "inverse_initial_stride": (
            list(Linv.stride()) if Linv is not None else None
        ),
        "inverse_initial_contiguous": (
            Linv.is_contiguous() if Linv is not None else None
        ),
        "inverse_gemv_backend": (
            inverse_gemv_backend
            if args.impl in ("inverse", "inverse_factor")
            else None
        ),
        "output_suffix": output_suffix,
        "seed": args.seed,
        "repeats": args.repeats,
        "triton_event_seconds": triton_event,
        "triton_wall_seconds": triton_wall,
        "triton_event_seconds_samples": triton_event_samples,
        "triton_wall_seconds_samples": triton_wall_samples,
        "refact_event_seconds": refact_event,
        "refact_wall_seconds": refact_wall,
        "refact_event_seconds_samples": refact_event_samples,
        "refact_wall_seconds_samples": refact_wall_samples,
        "event_speedup": refact_event / triton_event,
        "wall_speedup": refact_wall / triton_wall,
        "triton_updates_per_second": batch / triton_event,
        "refact_updates_per_second": batch / refact_event,
        "incremental_achieved_flops_per_second": incremental_flops / triton_event,
        "refact_achieved_flops_per_second": refactor_flops / refact_event,
        "minimum_effective_bytes": minimum_incremental_bytes,
        "minimum_effective_bandwidth_gb_s": minimum_incremental_bytes
        / triton_event
        / 1e9,
        "triton_peak_extra_allocated_bytes": triton_peak,
        "triton_workspace_bytes": workspace.nbytes if workspace is not None else 0,
        "refact_peak_extra_allocated_bytes": refact_peak,
        "L_relative_error": L_error,
        "Linv_relative_error": Linv_error,
        "Aw_relative_error": Aw_error,
    }
    payload = {
        "experiment": "cuda_scaling",
        **row,
        "environment": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_memory": torch.cuda.get_device_properties(0).total_memory,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    }
    stem = f"cuda_{args.impl}_r{rank}_b{batch}_w{args.num_warps}"
    if args.impl in (
        "tiled_v2",
        "scan",
        "cumsum",
        "parallel",
        "inverse",
        "inverse_factor",
    ):
        stem += (
            f"_sw{sweep_num_warps}"
            f"_tr{tile_rows}_tc{tile_cols}"
        )
    if args.impl == "parallel":
        stem += f"_rhs{solve_width}"
        if solve_right:
            stem += "_right"
    if args.impl in ("inverse", "inverse_factor"):
        stem += f"_gemv-{inverse_gemv_backend}"
    stem += f"_seed{args.seed}"
    if output_suffix:
        stem += f"_{output_suffix}"
    output = result_path("cuda", stem)
    atomic_json_dump(payload, output)
    print(
        f"gpu={payload['environment']['gpu']} impl={args.impl} r={rank} b={batch} "
        f"triton={1e3*triton_event:.4f}ms refact={1e3*refact_event:.4f}ms "
        f"speedup={row['event_speedup']:.3f}x error={Aw_error:.3e}",
        flush=True,
    )
    print(f"wrote {output}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument(
        "--impl",
        choices=(
            "fused",
            "tiled",
            "tiled_v2",
            "scan",
            "cumsum",
            "parallel",
            "inverse",
            "inverse_factor",
        ),
        default="fused",
    )
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--sweep-num-warps", type=int)
    parser.add_argument("--tile-rows", type=int)
    parser.add_argument("--tile-cols", type=int)
    parser.add_argument("--solve-width", type=int, default=2)
    parser.add_argument("--solve-right", action="store_true")
    parser.add_argument(
        "--inverse-gemv-backend",
        choices=("triton", "matmul"),
        default="matmul",
    )
    parser.add_argument("--output-suffix")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=5e-6)
    parser.add_argument(
        "--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
