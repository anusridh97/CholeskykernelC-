#!/bin/bash
#SBATCH --job-name=ska-eval-lm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/ska-eval-lm-%j.out
#SBATCH --error=logs/ska-eval-lm-%j.err

# Commonsense reasoning + ICR evaluation via lm-evaluation-harness.
# Reproduces the columns of PDN's Table 1 (commonsense + ICR avg) at 2K ctx.
#
# Specify the checkpoint via env vars when submitting:
#   sbatch --export=ALL,SKA_CKPT=path/to/ckpt_final.pt,SKA_CONFIG=prod \
#       experiments/run_eval_lm.slurm

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
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"

CONFIG="${SKA_CONFIG:-prod}"
CKPT_DEFAULT="$REPO_ROOT/checkpoints/${CONFIG}/ckpt_final.pt"
CKPT="${SKA_CKPT:-$CKPT_DEFAULT}"
[ -f "$CKPT" ] || { echo "No ckpt at $CKPT"; exit 1; }

OUT_DIR="$REPO_ROOT/results/lm_eval_$(basename "$CKPT" .pt)"
mkdir -p "$OUT_DIR"

# PDN's eval suite: commonsense + ICR at 2K context.
# The harness loads the model via a thin wrapper (eval_lm_harness.py) so the
# SKA layer's autograd.Function is bypassed for inference.
COMMONSENSE_TASKS="arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq,sciq,lambada_openai,wikitext"
ICR_TASKS="fda,swde,squad_completion,triviaqa,drop"

cd "$REPO_ROOT/experiments"

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "GPU:    $(nvidia-smi -L)"
echo "Ckpt:   $CKPT"
echo "Output: $OUT_DIR"
echo

# Wrapper around the SKA model that exposes the lm-eval-harness HF interface.
# See eval_lm_harness.py for details.
python eval_lm_harness.py \
    --ckpt "$CKPT" \
    --config "$CONFIG" \
    --tokenizer "meta-llama/Llama-2-7b-hf" \
    --tasks "${COMMONSENSE_TASKS},${ICR_TASKS}" \
    --batch_size 8 \
    --max_length 2048 \
    --out_dir "$OUT_DIR"

echo "Eval done. Results in $OUT_DIR"
