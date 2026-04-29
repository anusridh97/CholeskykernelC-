#!/usr/bin/env bash
# Launch training. Usage:
#   bash launch_training.sh sanity    # 1 GPU, 165M, 100M tokens
#   bash launch_training.sh prod      # 4 GPUs, 340M, 5B tokens

set -euo pipefail

CONFIG="${1:-prod}"
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
EXP_DIR="$REPO_ROOT/experiments"
CKPT_DIR="$REPO_ROOT/checkpoints/${CONFIG}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/logs"

mkdir -p "$CKPT_DIR" "$LOG_DIR"

# Activate env.
source "$REPO_ROOT/student_pkg/.venv/bin/activate"
cd "$EXP_DIR"

# HF cache to pod disk (avoid blowing the home volume).
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
mkdir -p "$HF_HOME"

# Pin numerics.
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN

# How many GPUs? RunPod provides this via env or nvidia-smi.
N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
echo "Detected $N_GPUS GPU(s); config=$CONFIG"

# Per-GPU micro batch.
case "$CONFIG" in
    sanity) MICRO_BATCH=2 ;;
    prod)   MICRO_BATCH=4 ;;
    *) echo "Unknown config $CONFIG"; exit 1 ;;
esac

LOG_FILE="$LOG_DIR/train_${CONFIG}_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG_FILE"

if [ "$N_GPUS" -eq 1 ]; then
    python train.py \
        --config "$CONFIG" \
        --out_dir "$CKPT_DIR" \
        --micro_batch "$MICRO_BATCH" \
        2>&1 | tee "$LOG_FILE"
else
    torchrun \
        --nproc_per_node "$N_GPUS" \
        --master_port 29500 \
        train.py \
        --config "$CONFIG" \
        --out_dir "$CKPT_DIR" \
        --micro_batch "$MICRO_BATCH" \
        2>&1 | tee "$LOG_FILE"
fi

echo "Training done. Checkpoints in $CKPT_DIR"
