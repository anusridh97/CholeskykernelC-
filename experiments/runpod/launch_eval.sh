#!/usr/bin/env bash
# Run S-NIAH evaluation on a trained checkpoint.
# Usage:
#   bash launch_eval.sh path/to/ckpt_final.pt [config=prod]

set -euo pipefail

CKPT="${1:?usage: launch_eval.sh <ckpt_path> [config]}"
CONFIG="${2:-prod}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." &> /dev/null && pwd)
EXP_DIR="$REPO_ROOT/experiments/backend"
RESULTS_DIR="$REPO_ROOT/results"

mkdir -p "$RESULTS_DIR"

source "$REPO_ROOT/student_pkg/.venv/bin/activate"
cd "$EXP_DIR"

OUT="$RESULTS_DIR/sniah_$(basename "$CKPT" .pt).csv"
echo "Output: $OUT"

# Lengths trimmed for budget: 32K is the most expensive.
# If running tight on time, drop 16384 and 32768.
python eval_sniah.py \
    --ckpt "$CKPT" \
    --config "$CONFIG" \
    --tokenizer "gpt2" \
    --lengths 2048 4096 8192 16384 32768 \
    --variants 1 2 3 \
    --n_examples 100 \
    --out "$OUT" \
    --seed 0

echo "Done. Plot with experiments/backend/plot_sniah.py"
