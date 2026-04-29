#!/bin/bash
#SBATCH --job-name=ska-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ska-train-%j.out
#SBATCH --error=logs/ska-train-%j.err

# Production training of the SKA model on SlimPajama.
# Configured for 4 L40S on a single node (Farmshare oat-XX), 48h time cap.
#
# Auto-resumes from the most recent checkpoint in CKPT_DIR, so the same
# script can be submitted multiple times to span longer runs. Use
# submit_chain.sh to fire a chain of dependent jobs.
#
# Override the config via the SKA_CONFIG env var when submitting:
#   sbatch --export=ALL,SKA_CONFIG=prod        experiments/run_train.slurm
#   sbatch --export=ALL,SKA_CONFIG=prod_1gpu   experiments/run_train.slurm  (fallback)
#   sbatch --export=ALL,SKA_CONFIG=sanity      experiments/run_train.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

# Locate the repo root (handles submission from either repo root or experiments/).
REPO_ROOT="$(pwd)"
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

# Threading: 32 CPUs across 4 GPUs = 8 CPU threads per GPU for dataloading.
N_GPUS="${SLURM_GPUS_ON_NODE:-4}"
THREADS_PER_GPU=$(( ${SLURM_CPUS_PER_TASK:-32} / N_GPUS ))
export OMP_NUM_THREADS=$THREADS_PER_GPU
export MKL_NUM_THREADS=$THREADS_PER_GPU
export OPENBLAS_NUM_THREADS=$THREADS_PER_GPU
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

# HF cache: keep on home or scratch, not /tmp (per-job ephemeral).
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

# Per-GPU micro batch (one L40S has 48 GB; conservative defaults for FP32 ctx 2048).
case "$CONFIG" in
    sanity)    MICRO_BATCH=2 ;;
    prod_1gpu) MICRO_BATCH=2 ;;
    prod)      MICRO_BATCH=2 ;;
    *) echo "Unknown SKA_CONFIG=$CONFIG"; exit 1 ;;
esac

cd "$REPO_ROOT/experiments"

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "GPUs: $(nvidia-smi -L | tr '\n' ' | ')"
echo "Config: $CONFIG | micro batch: $MICRO_BATCH | n_gpus: $N_GPUS"
echo "Checkpoint dir: $CKPT_DIR"
echo

# Pick a free port on this node so multiple jobs do not collide.
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

if [ "$N_GPUS" -gt 1 ]; then
    torchrun \
        --standalone \
        --nproc_per_node "$N_GPUS" \
        --master_port "$MASTER_PORT" \
        train.py \
            --config "$CONFIG" \
            --out_dir "$CKPT_DIR" \
            --micro_batch "$MICRO_BATCH" \
            $RESUME_FLAG
else
    python train.py \
        --config "$CONFIG" \
        --out_dir "$CKPT_DIR" \
        --micro_batch "$MICRO_BATCH" \
        $RESUME_FLAG
fi

echo "Training segment finished. Latest checkpoint:"
ls -lt "$CKPT_DIR"/ckpt_*.pt 2>/dev/null | head -n 1 || echo "  (none)"
