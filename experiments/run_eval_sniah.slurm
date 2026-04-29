#!/bin/bash
#SBATCH --job-name=ska-eval-sniah
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ska-eval-sniah-%j.out
#SBATCH --error=logs/ska-eval-sniah-%j.err

# S-NIAH evaluation at 5 context lengths x 3 variants on a trained checkpoint.
#
# Specify the checkpoint and config via env vars when submitting:
#   sbatch --export=ALL,SKA_CKPT=path/to/ckpt_final.pt,SKA_CONFIG=prod \
#       experiments/run_eval_sniah.slurm
#
# Defaults: latest "prod" checkpoint, all 5 lengths, all 3 variants, 100 ex.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs results

REPO_ROOT="$(pwd)"
if [ -d "$REPO_ROOT/../student_pkg" ]; then
    REPO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
fi
# shellcheck disable=SC1091
source "$REPO_ROOT/student_pkg/.venv/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"

CONFIG="${SKA_CONFIG:-prod}"
CKPT_DEFAULT="$REPO_ROOT/checkpoints/${CONFIG}/ckpt_final.pt"
CKPT="${SKA_CKPT:-$CKPT_DEFAULT}"

if [ ! -f "$CKPT" ]; then
    # Fall back to latest in dir.
    CKPT_DIR="$(dirname "$CKPT")"
    CKPT=$(ls -t "$CKPT_DIR"/ckpt_*.pt 2>/dev/null | head -n 1 || true)
    if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
        echo "No checkpoint found. Set SKA_CKPT=path/to/ckpt.pt"
        exit 1
    fi
fi

LENGTHS="${SKA_LENGTHS:-2048 4096 8192 16384 32768}"
VARIANTS="${SKA_VARIANTS:-1 2 3}"
N_EX="${SKA_N_EX:-100}"
SEED="${SKA_SEED:-0}"

OUT="$REPO_ROOT/results/sniah_$(basename "$CKPT" .pt)_seed${SEED}.csv"

cd "$REPO_ROOT/experiments"

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "GPU:        $(nvidia-smi -L)"
echo "Checkpoint: $CKPT"
echo "Lengths:    $LENGTHS"
echo "Variants:   $VARIANTS"
echo "N_examples: $N_EX"
echo "Seed:       $SEED"
echo "Output:     $OUT"
echo

python eval_sniah.py \
    --ckpt "$CKPT" \
    --config "$CONFIG" \
    --tokenizer "meta-llama/Llama-2-7b-hf" \
    --lengths $LENGTHS \
    --variants $VARIANTS \
    --n_examples "$N_EX" \
    --out "$OUT" \
    --seed "$SEED"

echo "Eval done. Plot with:"
echo "  python experiments/plot_sniah.py --ours $OUT --out figures/"
