#!/bin/bash
#SBATCH --job-name=triton-chol
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=triton-chol-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "host: $(hostname)"
echo "job: ${SLURM_JOB_ID:-unknown}"
echo "date: $(date)"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

python - <<'PY'
import sys
import torch
import triton

print("python:", sys.executable)
print("torch:", torch.__version__)
print("triton:", triton.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

python -m pytest tests/test_correctness.py::test_triton_matches_ground_truth -v
