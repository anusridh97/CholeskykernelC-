#!/bin/bash
#SBATCH --job-name=ska-sanity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ska-sanity-%j.out
#SBATCH --error=logs/ska-sanity-%j.err

# Sanity-check training: 165M model, 100M SlimPajama tokens, single L40S.
# Goal: confirm the training pipeline works end-to-end and the loss is
# trending downward. Should finish in ~1-2h.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

REPO_ROOT="$(pwd)"
if [ -d "$REPO_ROOT/../student_pkg" ]; then
    REPO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
fi

module load cuda 2>/dev/null || true
if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
fi

# shellcheck disable=SC1091
source "$REPO_ROOT/student_pkg/.venv/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
mkdir -p "$HF_HOME"

CKPT_DIR="${SKA_CKPT_DIR:-$REPO_ROOT/checkpoints/sanity}"
mkdir -p "$CKPT_DIR"

cd "$REPO_ROOT/experiments"

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "GPU: $(nvidia-smi -L)"
echo

python train.py \
    --config sanity \
    --out_dir "$CKPT_DIR" \
    --micro_batch 2

echo "Sanity training done. Final checkpoint:"
ls -lt "$CKPT_DIR"/ckpt_*.pt | head -n 1
