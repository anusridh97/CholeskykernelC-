#!/bin/bash
#SBATCH --job-name=ska-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/ska-train-%j.out
#SBATCH --error=logs/ska-train-%j.err

# Production training on SCG. SLURM directives that depend on cluster
# specifics (partition, account, gres, time, qos) are NOT hard-coded.
# Set them via env vars and pass through with `sbatch --export=ALL`:
#
#   export SCG_PARTITION=<partition>          # e.g. batch, sphinx, nih_s10
#   export SCG_ACCOUNT=<PI_SUNetID>           # required for batch/nih_s10
#   export SCG_GRES='gpu:h200:8'              # or gpu:a100:8, gpu:1, etc.
#   export SCG_TIME=48:00:00                  # wall time
#   export SCG_QOS=                           # optional; e.g. scavenger
#   export SCG_CPUS=32                        # CPU cores (8/GPU is reasonable)
#   export SCG_MEM=200G                       # node memory
#   sbatch --export=ALL experiments/run_train_scg.slurm
#
# Override the SKA training config via SKA_CONFIG (see experiments/train.py).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

REPO_ROOT="$(pwd)"
for _ in 1 2 3 4 5; do
    [ -d "$REPO_ROOT/student_pkg" ] && break
    REPO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
done
[ -d "$REPO_ROOT/student_pkg" ] || { echo "Cannot find student_pkg/ above the submit dir."; exit 1; }

# Source the env file written by setup_scg.sh (gives us SCG_VENV_DIR, SCG_CUDA_MOD).
if [ -f "$REPO_ROOT/.scg_env" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.scg_env"
fi
VENV_DIR="${SCG_VENV_DIR:-$REPO_ROOT/student_pkg/.venv}"
CUDA_MOD="${SCG_CUDA_MOD:-cuda}"

# Re-apply SLURM constraints from env vars via scontrol if SLURM didn't see them
# at submission time (defensive; the proper place is `sbatch --export=ALL ...`).
echo "Submitting from: ${SLURM_SUBMIT_DIR:-.}"
echo "Partition: ${SLURM_JOB_PARTITION:-?}  GRES: ${SLURM_JOB_GRES:-?}  Time: ${SLURM_JOB_TIME_LIMIT:-?}"

# Load CUDA so Triton + torch can resolve the runtime.
module load "$CUDA_MOD" 2>/dev/null || true
if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
fi

# Activate venv.
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Threading.
N_GPUS="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
N_GPUS=${N_GPUS:-1}
THREADS_PER_GPU=$(( ${SLURM_CPUS_PER_TASK:-32} / N_GPUS ))
[ "$THREADS_PER_GPU" -lt 1 ] && THREADS_PER_GPU=1
export OMP_NUM_THREADS=$THREADS_PER_GPU
export MKL_NUM_THREADS=$THREADS_PER_GPU
export OPENBLAS_NUM_THREADS=$THREADS_PER_GPU
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

# HF cache: lab storage if available, otherwise home.
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
mkdir -p "$HF_HOME"

# SKA config and checkpoint dir.
CONFIG="${SKA_CONFIG:-prod_scg}"
CKPT_DIR="${SKA_CKPT_DIR:-$REPO_ROOT/checkpoints/${CONFIG}}"
mkdir -p "$CKPT_DIR"

# Auto-resume.
RESUME_FLAG=""
LATEST_CKPT=$(ls -t "$CKPT_DIR"/ckpt_*.pt 2>/dev/null | head -n 1 || true)
if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming from $LATEST_CKPT"
    RESUME_FLAG="--resume $LATEST_CKPT"
else
    echo "No checkpoint found in $CKPT_DIR; starting from scratch."
fi

# Per-GPU micro batch.
# NOTE: SKA's chunkwise autograd retains per-token Cholesky intermediates
# (L, Aw, z, w, L_new, vz, vw, k_tilde, T_mat) for the closed-form backward.
# Memory scales as B * H * r^2 * ctx_len * n_layers. For ctx_len=2048,
# r=128, the per-batch overhead is ~50 GB. Keep micro_batch conservative.
case "$CONFIG" in
    sanity)    MICRO_BATCH=1 ;;        # 152M, ctx 2048, ~70 GB on H200
    prod_1gpu) MICRO_BATCH=1 ;;        # 340M, ctx 2048, OOMs at 2 on L40S
    prod)      MICRO_BATCH=2 ;;        # 340M, ctx 2048, ~80 GB on L40S 4x
    prod_scg)  MICRO_BATCH=4 ;;        # 340M, ctx 2048, ~120 GB on H200 8x
    *) echo "Unknown SKA_CONFIG=$CONFIG"; exit 1 ;;
esac
MICRO_BATCH="${SKA_MICRO_BATCH:-$MICRO_BATCH}"

cd "$REPO_ROOT/experiments/backend"

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "GPUs: $(nvidia-smi -L | tr '\n' ' | ')"
echo "Config: $CONFIG | micro batch: $MICRO_BATCH | n_gpus: $N_GPUS"
echo "Checkpoint dir: $CKPT_DIR"
echo

MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

# Optional extra flags (e.g. --retrieval_ratio 0.15 --wandb_project ska-pretrain).
# Passed verbatim to train.py. See experiments/backend/train.py argparse for options.
EXTRA_ARGS="${SKA_EXTRA_ARGS:-}"

if [ "$N_GPUS" -gt 1 ]; then
    torchrun \
        --standalone \
        --nproc_per_node "$N_GPUS" \
        --master_port "$MASTER_PORT" \
        train.py \
            --config "$CONFIG" \
            --out_dir "$CKPT_DIR" \
            --micro_batch "$MICRO_BATCH" \
            $RESUME_FLAG $EXTRA_ARGS
else
    python train.py \
        --config "$CONFIG" \
        --out_dir "$CKPT_DIR" \
        --micro_batch "$MICRO_BATCH" \
        $RESUME_FLAG $EXTRA_ARGS
fi

echo "Training segment finished. Latest checkpoint:"
ls -lt "$CKPT_DIR"/ckpt_*.pt 2>/dev/null | head -n 1 || echo "  (none)"
