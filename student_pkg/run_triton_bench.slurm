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

from doublesided_cholesky import incremental_update_triton, incremental_update_triton_tiled


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


def refactor_update(G, M, z, w):
    G_new = G + torch.outer(z, z)
    M_new = M + torch.outer(z, w)
    L_new = torch.linalg.cholesky(G_new)
    Linv_new = torch.linalg.inv(L_new)
    return L_new, Linv_new @ M_new @ Linv_new.T


def bench_one(r):
    G, L, M, Aw, z, w = make_problem(r)

    L_exact, Aw_exact = refactor_update(G, M, z, w)

    L_fused = L.clone()
    Aw_fused, _ = incremental_update_triton(L_fused, Aw, z, w)
    L_tiled = L.clone()
    Aw_tiled, _ = incremental_update_triton_tiled(L_tiled, Aw, z, w)
    torch.cuda.synchronize()

    fused_err = (torch.norm(Aw_fused - Aw_exact) / torch.norm(Aw_exact)).item()
    tiled_err = (torch.norm(Aw_tiled - Aw_exact) / torch.norm(Aw_exact)).item()
    tiled_vs_fused = (torch.norm(Aw_tiled - Aw_fused) / torch.norm(Aw_fused)).item()
    L_tiled_err = (torch.norm(L_tiled - L_exact) / torch.norm(L_exact)).item()
    assert fused_err < 1e-4, f"fused Aw error too high for r={r}: {fused_err:.2e}"
    assert tiled_err < 1e-4, f"tiled Aw error too high for r={r}: {tiled_err:.2e}"
    assert tiled_vs_fused < 1e-5, f"tiled/fused mismatch for r={r}: {tiled_vs_fused:.2e}"
    assert L_tiled_err < 1e-4, f"tiled L error too high for r={r}: {L_tiled_err:.2e}"

    def fused_update():
        incremental_update_triton(L.clone(), Aw, z, w)

    def tiled_update():
        incremental_update_triton_tiled(L.clone(), Aw, z, w)

    def refactor():
        refactor_update(G, M, z, w)

    for _ in range(10):
        fused_update()
        tiled_update()
        refactor()
    torch.cuda.synchronize()

    repeats = 100 if r <= 128 else 40 if r <= 256 else 20
    fused_ms = time_cuda(fused_update, repeats)
    tiled_ms = time_cuda(tiled_update, repeats)
    refactor_ms = time_cuda(refactor, repeats)
    return fused_ms, tiled_ms, refactor_ms, fused_err, tiled_err, tiled_vs_fused


device_name = torch.cuda.get_device_name(0)
print("device:", device_name)
if "L40S" not in device_name:
    raise RuntimeError(f"expected an L40S GPU, got {device_name}")

print(
    f"{'r':>6}  {'fused ms':>10}  {'tiled ms':>10}  {'refact ms':>10}  "
    f"{'fused speedup':>13}  {'tiled speedup':>13}  {'tiled/fused':>12}  "
    f"{'fused err':>10}  {'tiled err':>10}"
)
print("-" * 110)
for r in [16, 32, 64, 128, 256, 512]:
    fused_ms, tiled_ms, refactor_ms, fused_err, tiled_err, tiled_vs_fused = bench_one(r)
    print(
        f"{r:>6}  {fused_ms:>10.4f}  {tiled_ms:>10.4f}  {refactor_ms:>10.4f}  "
        f"{refactor_ms / fused_ms:>12.2f}x  {refactor_ms / tiled_ms:>12.2f}x  "
        f"{fused_ms / tiled_ms:>11.2f}x  {fused_err:>10.2e}  {tiled_err:>10.2e}"
    )
PY
