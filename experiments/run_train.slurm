#!/bin/bash
#SBATCH --job-name=ska-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ska-train-%j.out
#SBATCH --error=logs/ska-train-%j.err

# Production training of 340M SKA model on SlimPajama.
#
# This script auto-resumes from the most recent checkpoint in CKPT_DIR if
# present, so the same script can be submitted multiple times to run beyond
# the 24h time cap. Use submit_chain.sh to fire a chain of dependent jobs.
#
# Override the config via the SKA_CONFIG env var when submitting:
#   sbatch --export=ALL,SKA_CONFIG=prod   experiments/run_train.slurm
#   sbatch --export=ALL,SKA_CONFIG=sanity experiments/run_train.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

# Activate venv (lives in student_pkg/.venv per the existing kernel slurm scripts).
REPO_ROOT="$(pwd)"
# If submitted from experiments/, walk up.
if [ -d "$REPO_ROOT/student_pkg" ]; then
    :
elif [ -d "$REPO_ROOT/../student_pkg" ]; then
    REPO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
fi

# Load CUDA so Triton and torch can find nvcc + the runtime.
module load cuda 2>/dev/null || true
if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
fi

# shellcheck disable=SC1091
source "$REPO_ROOT/student_pkg/.venv/bin/activate"

# Threading: leave 4 CPU cores per GPU for dataloader/tokenizer.
N_GPUS="${SLURM_GPUS_ON_NODE:-4}"
THREADS_PER_GPU=$(( ${SLURM_CPUS_PER_TASK:-16} / N_GPUS ))
export OMP_NUM_THREADS=$THREADS_PER_GPU
export MKL_NUM_THREADS=$THREADS_PER_GPU
export OPENBLAS_NUM_THREADS=$THREADS_PER_GPU
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

# HF cache: keep on Farmshare's scratch or home, not /tmp (per-job ephemeral).
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
mkdir -p "$HF_HOME"

# Config and checkpoint dir.
CONFIG="${SKA_CONFIG:-prod}"
CKPT_DIR="${SKA_CKPT_DIR:-$REPO_ROOT/checkpoints/${CONFIG}}"
mkdir -p "$CKPT_DIR"

# Auto-resume: find the latest checkpoint in CKPT_DIR.
RESUME_FLAG=""
LATEST_CKPT=$(ls -t "$CKPT_DIR"/ckpt_*.pt 2>/dev/null | head -n 1 || true)
if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming from $LATEST_CKPT"
    RESUME_FLAG="--resume $LATEST_CKPT"
else
    echo "No checkpoint found in $CKPT_DIR; starting from scratch."
fi

# Per-GPU micro batch.
case "$CONFIG" in
    sanity) MICRO_BATCH=2 ;;
    prod)   MICRO_BATCH=4 ;;
    *) echo "Unknown SKA_CONFIG=$CONFIG"; exit 1 ;;
esac

cd "$REPO_ROOT/experiments"

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "GPUs: $(nvidia-smi -L | tr '\n' '|')"
echo "Config: $CONFIG | micro batch: $MICRO_BATCH | n_gpus: $N_GPUS"
echo "Checkpoint dir: $CKPT_DIR"
echo

# Pick a free port on this node so multiple chained jobs do not collide.
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

torchrun \
    --standalone \
    --nproc_per_node "$N_GPUS" \
    --master_port "$MASTER_PORT" \
    train.py \
        --config "$CONFIG" \
        --out_dir "$CKPT_DIR" \
        --micro_batch "$MICRO_BATCH" \
        $RESUME_FLAG

echo "Training segment finished. Latest checkpoint:"
ls -lt "$CKPT_DIR"/ckpt_*.pt 2>/dev/null | head -n 1 || echo "  (none)"
