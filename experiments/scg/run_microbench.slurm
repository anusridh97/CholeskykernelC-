#!/bin/bash
#SBATCH --job-name=ska-microbench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/ska-microbench-%j.out
#SBATCH --error=logs/ska-microbench-%j.err

# Per-token Triton kernel throughput sweep on a single GPU.
# Run this BEFORE main training to confirm the kernel is firing on the
# H200/H100/A100 you've been allocated and to baseline tokens/sec.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs results figures

REPO_ROOT="$(pwd)"
for _ in 1 2 3 4 5; do
    [ -d "$REPO_ROOT/student_pkg" ] && break
    REPO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
done
[ -d "$REPO_ROOT/student_pkg" ] || { echo "Cannot find student_pkg/ above the submit dir."; exit 1; }

if [ -f "$REPO_ROOT/.scg_env" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.scg_env"
fi
VENV_DIR="${SCG_VENV_DIR:-$REPO_ROOT/student_pkg/.venv}"
CUDA_MOD="${SCG_CUDA_MOD:-cuda}"

module load "$CUDA_MOD" 2>/dev/null || true
if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "$REPO_ROOT/experiments/backend"

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "GPU: $(nvidia-smi -L)"
echo

python microbench.py \
    --r 64 128 256 \
    --bh 8 32 128 \
    --T 512 2048 8192 \
    --csv "$REPO_ROOT/results/microbench_scg.csv" \
    --plot "$REPO_ROOT/figures/microbench_scg.pdf"

echo
echo "Done. CSV: $REPO_ROOT/results/microbench_scg.csv"
echo "      PDF: $REPO_ROOT/figures/microbench_scg.pdf"
