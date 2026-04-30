#!/bin/bash
#SBATCH --job-name=ska-microbench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ska-microbench-%j.out
#SBATCH --error=logs/ska-microbench-%j.err

# Per-token Triton kernel throughput sweep on a single L40S.
# Run this BEFORE main training to confirm the kernel is firing on GPU
# and to get a baseline tokens/sec number for sanity-checking train throughput.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs results figures

REPO_ROOT="$(pwd)"
for _ in 1 2 3 4 5; do
    [ -d "$REPO_ROOT/student_pkg" ] && break
    REPO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
done
[ -d "$REPO_ROOT/student_pkg" ] || { echo "Cannot find student_pkg/ above the submit dir."; exit 1; }

module load cuda 2>/dev/null || true
if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
fi

# shellcheck disable=SC1091
source "$REPO_ROOT/student_pkg/.venv/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "$REPO_ROOT/experiments/backend"

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "GPU: $(nvidia-smi -L)"
echo

python microbench.py \
    --r 64 128 256 \
    --bh 8 32 128 \
    --T 512 2048 8192 \
    --csv "$REPO_ROOT/results/microbench.csv" \
    --plot "$REPO_ROOT/figures/microbench.pdf"

echo
echo "Done. CSV: $REPO_ROOT/results/microbench.csv"
echo "      PDF: $REPO_ROOT/figures/microbench.pdf"
