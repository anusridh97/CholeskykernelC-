from __future__ import annotations

import argparse
import os
import platform
import socket
import time

import torch

from .common import (
    add_student_pkg,
    atomic_json_dump,
    result_path,
    tensor_relative_error,
)
from .cuda_scaling import batched_whiten, make_problem


def _time_sequence(function) -> tuple[float, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start.record()
    function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0, time.perf_counter() - wall_start


def _updates(rank: int, steps: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed + 2000)
    z = torch.randn(
        steps,
        1,
        rank,
        device="cuda",
        generator=generator,
    ) * 0.05
    w = torch.randn(
        steps,
        1,
        rank,
        device="cuda",
        generator=generator,
    ) * 0.05
    return z.contiguous(), w.contiguous()


def run(args: argparse.Namespace) -> dict:
    add_student_pkg()
    from doublesided_cholesky import (
        TritonTiledWorkspace,
        incremental_update_triton_inverse_factor,
    )

    G0, L0, M0, Aw0, _, _ = make_problem(1, args.rank, args.seed)
    z_steps, w_steps = _updates(args.rank, args.steps, args.seed)

    # Compile kernels and initialize solver/library plans outside every timed
    # sequence.
    warm_inverse = torch.linalg.inv(L0).tril().contiguous()
    warm_aw = Aw0.clone()
    warm_workspace = TritonTiledWorkspace(warm_aw, solve_width=2)
    incremental_update_triton_inverse_factor(
        warm_inverse,
        warm_aw,
        z_steps[0],
        w_steps[0],
        tile_rows=args.tile_rows,
        tile_cols=args.tile_cols,
        setup_num_warps=args.setup_num_warps,
        sweep_num_warps=args.sweep_num_warps,
        workspace=warm_workspace,
    )
    warm_G = G0 + z_steps[0].unsqueeze(2) @ z_steps[0].unsqueeze(1)
    warm_M = M0 + z_steps[0].unsqueeze(2) @ w_steps[0].unsqueeze(1)
    warm_L = torch.linalg.cholesky(warm_G)
    batched_whiten(warm_L, warm_M)
    torch.cuda.synchronize()

    G_refact = G0.clone()
    M_refact = M0.clone()
    final_refact = {}

    def refact_sequence():
        for index in range(args.steps):
            z = z_steps[index]
            w = w_steps[index]
            G_refact.add_(z.unsqueeze(2) @ z.unsqueeze(1))
            M_refact.add_(z.unsqueeze(2) @ w.unsqueeze(1))
            L_new = torch.linalg.cholesky(G_refact)
            Aw_new = batched_whiten(L_new, M_refact)
        final_refact["L"] = L_new
        final_refact["Aw"] = Aw_new

    torch.cuda.reset_peak_memory_stats()
    refact_event, refact_wall = _time_sequence(refact_sequence)
    refact_peak = torch.cuda.max_memory_allocated()

    rows = []
    for interval in args.refresh_intervals:
        G = G0.clone()
        M = M0.clone()
        Linv = torch.linalg.inv(L0).tril().contiguous()
        Aw_work = Aw0.clone()
        workspace = TritonTiledWorkspace(Aw_work, solve_width=2)
        identity = torch.eye(
            args.rank,
            device="cuda",
            dtype=torch.float32,
        ).unsqueeze(0)
        refresh_count = 0

        def inverse_sequence():
            nonlocal Aw_work, refresh_count
            for index in range(args.steps):
                z = z_steps[index]
                w = w_steps[index]
                # These exact-state updates are required for self-contained
                # refreshes and are deliberately inside the timed region.
                G.add_(z.unsqueeze(2) @ z.unsqueeze(1))
                M.add_(z.unsqueeze(2) @ w.unsqueeze(1))
                Aw_work, _ = incremental_update_triton_inverse_factor(
                    Linv,
                    Aw_work,
                    z,
                    w,
                    tile_rows=args.tile_rows,
                    tile_cols=args.tile_cols,
                    setup_num_warps=args.setup_num_warps,
                    sweep_num_warps=args.sweep_num_warps,
                    workspace=workspace,
                )
                if interval > 0 and (index + 1) % interval == 0:
                    L_refresh = torch.linalg.cholesky(G)
                    Linv_refresh = torch.linalg.solve_triangular(
                        L_refresh,
                        identity,
                        upper=False,
                    )
                    Linv.copy_(Linv_refresh.tril())
                    Aw_work.copy_(batched_whiten(L_refresh, M))
                    refresh_count += 1

        torch.cuda.reset_peak_memory_stats()
        event_seconds, wall_seconds = _time_sequence(inverse_sequence)
        peak_bytes = torch.cuda.max_memory_allocated()
        L_exact = torch.linalg.cholesky(G)
        Linv_exact = torch.linalg.solve_triangular(
            L_exact,
            identity,
            upper=False,
        )
        Aw_exact = batched_whiten(L_exact, M)

        # A separate untimed replay measures drift immediately before each
        # refresh. Measuring it inside the timed sequence would incorrectly
        # charge extra diagnostic factorizations to the method.
        validation_G = G0.clone()
        validation_M = M0.clone()
        validation_Linv = torch.linalg.inv(L0).tril().contiguous()
        validation_Aw = Aw0.clone()
        validation_workspace = TritonTiledWorkspace(
            validation_Aw,
            solve_width=2,
        )
        pre_refresh_rows = []
        last_recorded_step = 0
        for index in range(args.steps):
            z = z_steps[index]
            w = w_steps[index]
            validation_G.add_(z.unsqueeze(2) @ z.unsqueeze(1))
            validation_M.add_(z.unsqueeze(2) @ w.unsqueeze(1))
            validation_Aw, _ = incremental_update_triton_inverse_factor(
                validation_Linv,
                validation_Aw,
                z,
                w,
                tile_rows=args.tile_rows,
                tile_cols=args.tile_cols,
                setup_num_warps=args.setup_num_warps,
                sweep_num_warps=args.sweep_num_warps,
                workspace=validation_workspace,
            )
            at_refresh = interval > 0 and (index + 1) % interval == 0
            at_final_tail = index + 1 == args.steps and not at_refresh
            if at_refresh or at_final_tail:
                validation_L = torch.linalg.cholesky(validation_G)
                validation_Linv_exact = torch.linalg.solve_triangular(
                    validation_L,
                    identity,
                    upper=False,
                )
                validation_Aw_exact = batched_whiten(
                    validation_L,
                    validation_M,
                )
                pre_refresh_rows.append(
                    {
                        "step": index + 1,
                        "age_since_refresh": index + 1 - last_recorded_step,
                        "Linv_relative_error": tensor_relative_error(
                            validation_Linv,
                            validation_Linv_exact,
                        ),
                        "Aw_relative_error": tensor_relative_error(
                            validation_Aw,
                            validation_Aw_exact,
                        ),
                    }
                )
                if at_refresh:
                    validation_Linv.copy_(validation_Linv_exact.tril())
                    validation_Aw.copy_(validation_Aw_exact)
                    last_recorded_step = index + 1

        rows.append(
            {
                "refresh_interval": interval,
                "refresh_count": refresh_count,
                "event_seconds_total": event_seconds,
                "wall_seconds_total": wall_seconds,
                "event_seconds_per_update": event_seconds / args.steps,
                "wall_seconds_per_update": wall_seconds / args.steps,
                "event_speedup_vs_fresh_every_step": (
                    refact_event / event_seconds
                ),
                "wall_speedup_vs_fresh_every_step": (
                    refact_wall / wall_seconds
                ),
                "Linv_relative_error": tensor_relative_error(
                    Linv,
                    Linv_exact,
                ),
                "Aw_relative_error": tensor_relative_error(
                    Aw_work,
                    Aw_exact,
                ),
                "pre_refresh_errors": pre_refresh_rows,
                "max_pre_refresh_Linv_relative_error": max(
                    item["Linv_relative_error"] for item in pre_refresh_rows
                ),
                "max_pre_refresh_Aw_relative_error": max(
                    item["Aw_relative_error"] for item in pre_refresh_rows
                ),
                "final_tail_age": args.steps - last_recorded_step,
                "peak_allocated_bytes_including_harness": peak_bytes,
                "algorithm_persistent_dense_matrices": 5,
                "benchmark_additional_identity_matrix": True,
            }
        )

    return {
        "experiment": "cuda_inverse_refresh",
        "rank": args.rank,
        "steps": args.steps,
        "seed": args.seed,
        "refresh_intervals": args.refresh_intervals,
        "tile_rows": args.tile_rows,
        "tile_cols": args.tile_cols,
        "setup_num_warps": args.setup_num_warps,
        "sweep_num_warps": args.sweep_num_warps,
        "fresh_event_seconds_total": refact_event,
        "fresh_wall_seconds_total": refact_wall,
        "fresh_event_seconds_per_update": refact_event / args.steps,
        "fresh_wall_seconds_per_update": refact_wall / args.steps,
        "fresh_peak_allocated_bytes_including_harness": refact_peak,
        "rows": rows,
        "environment": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_memory": torch.cuda.get_device_properties(0).total_memory,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--refresh-intervals",
        type=int,
        nargs="+",
        default=[0, 10, 50, 100, 200],
    )
    parser.add_argument("--tile-rows", type=int, required=True)
    parser.add_argument("--tile-cols", type=int, required=True)
    parser.add_argument("--setup-num-warps", type=int, default=4)
    parser.add_argument("--sweep-num-warps", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    payload = run(arguments)
    output = result_path(
        "cuda_tuning",
        f"cuda_inverse_refresh_r{arguments.rank}_s{arguments.steps}_"
        f"seed{arguments.seed}_"
        f"{os.environ.get('SLURM_JOB_ID', 'local')}",
    )
    atomic_json_dump(payload, output)
    print(payload, flush=True)
    print(f"wrote {output}", flush=True)
