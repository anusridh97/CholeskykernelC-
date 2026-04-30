#!/bin/bash
#SBATCH --job-name=ska-eval-sniah
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/ska-eval-sniah-%j.out
#SBATCH --error=logs/ska-eval-sniah-%j.err

# S-NIAH evaluation on SCG. Single GPU is sufficient for inference.
# Submit via the wrapper: see submit_chain_scg.sh / scg.md.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs results

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

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"

CONFIG="${SKA_CONFIG:-prod_scg}"
CKPT_DEFAULT="$REPO_ROOT/checkpoints/${CONFIG}/ckpt_final.pt"
CKPT="${SKA_CKPT:-$CKPT_DEFAULT}"

if [ ! -f "$CKPT" ]; then
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

cd "$REPO_ROOT/experiments/backend"

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "GPU:        $(nvidia-smi -L)"
echo "Checkpoint: $CKPT"
echo "Output:     $OUT"
echo

python eval_sniah.py \
    --ckpt "$CKPT" \
    --config "$CONFIG" \
    --tokenizer "gpt2" \
    --lengths $LENGTHS \
    --variants $VARIANTS \
    --n_examples "$N_EX" \
    --out "$OUT" \
    --seed "$SEED"

echo "Eval done."
