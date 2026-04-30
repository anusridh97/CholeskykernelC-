#!/bin/bash
#SBATCH --job-name=ska-eval-lm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/ska-eval-lm-%j.out
#SBATCH --error=logs/ska-eval-lm-%j.err

# Commonsense + ICR evaluation on SCG via lm-evaluation-harness.

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
[ -f "$CKPT" ] || { echo "No ckpt at $CKPT"; exit 1; }

OUT_DIR="$REPO_ROOT/results/lm_eval_$(basename "$CKPT" .pt)"
mkdir -p "$OUT_DIR"

COMMONSENSE_TASKS="arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq,sciq,lambada_openai,wikitext"
ICR_TASKS="fda,swde,squad_completion,triviaqa,drop"

cd "$REPO_ROOT/experiments/backend"

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "GPU:    $(nvidia-smi -L)"
echo "Ckpt:   $CKPT"
echo "Output: $OUT_DIR"

python eval_lm_harness.py \
    --ckpt "$CKPT" \
    --config "$CONFIG" \
    --tokenizer "gpt2" \
    --tasks "${COMMONSENSE_TASKS},${ICR_TASKS}" \
    --batch_size 8 \
    --max_length 2048 \
    --out_dir "$OUT_DIR"

echo "Eval done. Results in $OUT_DIR"
