#!/bin/bash
#SBATCH --job-name=triton-nsys
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=triton-nsys-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

R="${R:-512}"
NUM_WARPS="${NUM_WARPS:-4}"
REPEATS="${REPEATS:-25}"
REPORT="triton-nsys-r${R}-w${NUM_WARPS}-${SLURM_JOB_ID:-manual}"

nsys profile \
  --trace=cuda,nvtx,osrt \
  --stats=true \
  --force-overwrite=true \
  -o "${REPORT}" \
  python - "$R" "$NUM_WARPS" "$REPEATS" <<'PY'
import sys

import torch

from doublesided_cholesky import incremental_update_triton


r = int(sys.argv[1])
num_warps = int(sys.argv[2])
repeats = int(sys.argv[3])

torch.manual_seed(42)
A = torch.randn(r, r, device="cuda")
G = A @ A.T + r * torch.eye(r, device="cuda")
L = torch.linalg.cholesky(G).contiguous()
M = torch.randn(r, r, device="cuda")
Linv = torch.linalg.inv(L)
Aw = Linv @ M @ Linv.T
z = torch.randn(r, device="cuda") * 0.1
w = torch.randn(r, device="cuda") * 0.1

for _ in range(10):
    incremental_update_triton(L.clone(), Aw, z, w, num_warps=num_warps)
torch.cuda.synchronize()

torch.cuda.nvtx.range_push(f"triton_incremental_r{r}_w{num_warps}")
for _ in range(repeats):
    incremental_update_triton(L.clone(), Aw, z, w, num_warps=num_warps)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

print(f"profiled r={r}, num_warps={num_warps}, repeats={repeats}")
PY
