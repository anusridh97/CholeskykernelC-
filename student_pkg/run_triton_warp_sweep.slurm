#!/bin/bash
#SBATCH --job-name=triton-warps
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=triton-warps-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python - <<'PY'
import statistics
import time

import torch

from doublesided_cholesky import incremental_update_triton, incremental_update_triton_tiled


def make_problem(r, device):
    torch.manual_seed(42)
    A = torch.randn(r, r, device=device)
    G = A @ A.T + r * torch.eye(r, device=device)
    L = torch.linalg.cholesky(G).contiguous()
    M = torch.randn(r, r, device=device)
    Linv = torch.linalg.inv(L)
    Aw = Linv @ M @ Linv.T
    z = torch.randn(r, device=device) * 0.1
    w = torch.randn(r, device=device) * 0.1
    return G, L, M, Aw, z, w


def time_cuda(fn, repeats):
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def time_wall(fn, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def run_update(impl, L, Aw, z, w, num_warps):
    if impl == "fused":
        return incremental_update_triton(L, Aw, z, w, num_warps=num_warps)
    if impl == "tiled":
        return incremental_update_triton_tiled(L, Aw, z, w, num_warps=num_warps)
    raise ValueError(f"unknown implementation: {impl}")


def bench_gpu_resident(r, num_warps, repeats, impl):
    _, L, _, Aw, z, w = make_problem(r, "cuda")

    def triton_update():
        run_update(impl, L.clone(), Aw, z, w, num_warps)

    for _ in range(10):
        triton_update()
    torch.cuda.synchronize()

    return time_cuda(triton_update, repeats), time_wall(triton_update, repeats)


def bench_with_transfers(r, num_warps, repeats, impl):
    _, L_cpu, _, Aw_cpu, z_cpu, w_cpu = make_problem(r, "cpu")
    L_cpu = L_cpu.pin_memory()
    Aw_cpu = Aw_cpu.pin_memory()
    z_cpu = z_cpu.pin_memory()
    w_cpu = w_cpu.pin_memory()

    def triton_roundtrip():
        L = L_cpu.to("cuda", non_blocking=True)
        Aw = Aw_cpu.to("cuda", non_blocking=True)
        z = z_cpu.to("cuda", non_blocking=True)
        w = w_cpu.to("cuda", non_blocking=True)
        Aw_out, vz = run_update(impl, L, Aw, z, w, num_warps)
        Aw_out.cpu()
        vz.cpu()

    for _ in range(5):
        triton_roundtrip()
    torch.cuda.synchronize()

    return time_wall(triton_roundtrip, repeats)


print("device:", torch.cuda.get_device_name(0))
print("CUDA event timing excludes CPU launch overhead.")
print("Roundtrip wall timing includes H2D inputs, D2H outputs, CPU launch overhead, and GPU execution.")
print()
print(f"{'impl':>6}  {'r':>6}  {'warps':>5}  {'gpu event ms':>12}  {'gpu wall ms':>11}  {'roundtrip ms':>12}")
print("-" * 66)

for r in [16, 32, 64, 128, 256, 512]:
    repeats = 100 if r <= 128 else 40 if r <= 256 else 20
    for num_warps in [1, 2, 4, 8]:
        for impl in ["fused", "tiled"]:
            event_ms, wall_ms = bench_gpu_resident(r, num_warps, repeats, impl)
            roundtrip_ms = bench_with_transfers(r, num_warps, max(10, repeats // 2), impl)
            print(f"{impl:>6}  {r:>6}  {num_warps:>5}  {event_ms:>12.4f}  {wall_ms:>11.4f}  {roundtrip_ms:>12.4f}")
    print()
PY
