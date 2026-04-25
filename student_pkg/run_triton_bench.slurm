#!/bin/bash
#SBATCH --job-name=triton-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=triton-bench-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python - <<'PY'
import statistics
import torch

from doublesided_cholesky import incremental_update_triton


def make_problem(r):
    torch.manual_seed(42)
    A = torch.randn(r, r, device="cuda")
    G = A @ A.T + r * torch.eye(r, device="cuda")
    L = torch.linalg.cholesky(G).contiguous()
    M = torch.randn(r, r, device="cuda")
    Linv = torch.linalg.inv(L)
    Aw = Linv @ M @ Linv.T
    z = torch.randn(r, device="cuda") * 0.1
    w = torch.randn(r, device="cuda") * 0.1
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


def bench_one(r):
    G, L, M, Aw, z, w = make_problem(r)

    def triton_update():
        incremental_update_triton(L.clone(), Aw, z, w)

    def refactor():
        G_new = G + torch.outer(z, z)
        M_new = M + torch.outer(z, w)
        L_new = torch.linalg.cholesky(G_new)
        Linv_new = torch.linalg.inv(L_new)
        Linv_new @ M_new @ Linv_new.T

    for _ in range(10):
        triton_update()
        refactor()
    torch.cuda.synchronize()

    repeats = 100 if r <= 128 else 40 if r <= 256 else 20
    triton_ms = time_cuda(triton_update, repeats)
    refactor_ms = time_cuda(refactor, repeats)
    return triton_ms, refactor_ms, refactor_ms / triton_ms


print("device:", torch.cuda.get_device_name(0))
print(f"{'r':>6}  {'Triton (ms)':>12}  {'Refact (ms)':>12}  {'Speedup':>9}")
print("-" * 48)
for r in [16, 32, 64, 128, 256, 512]:
    triton_ms, refactor_ms, speedup = bench_one(r)
    print(f"{r:>6}  {triton_ms:>12.4f}  {refactor_ms:>12.4f}  {speedup:>8.2f}x")
PY
