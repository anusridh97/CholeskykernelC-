#!/usr/bin/env bash
# Optional: install lm-evaluation-harness for the commonsense + ICR eval
# phase only. Run this AFTER training completes, before submitting
# run_eval_lm.slurm.
#
# Kept separate from the main setup because lm_eval drags in libcst and
# pyarrow >= 20 as build deps which fight with SCG's old system toolchain.
# By installing them in a second pass (after the core training stack is
# already in the venv), pip can find compatible wheels more reliably.
#
# Usage:
#   bash experiments/scg/setup_lm_eval.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." &> /dev/null && pwd)

if [ -f "$REPO_ROOT/.scg_env" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.scg_env"
fi
VENV_DIR="${SCG_VENV_DIR:-$REPO_ROOT/student_pkg/.venv}"
CUDA_MOD="${SCG_CUDA_MOD:-cuda}"
GCC_MOD="${SCG_GCC_MOD:-}"

# Load CUDA + a recent GCC for any from-source builds.
if command -v module >/dev/null 2>&1; then
    module load "$CUDA_MOD" 2>/dev/null || true
    if [ -n "$GCC_MOD" ]; then
        module load "$GCC_MOD" 2>/dev/null || true
    else
        for cand in gcc/14 gcc/13 gcc/12 gcc/11 gcc/10 gcc/9.3.0 gcc/9 gcc; do
            if module load "$cand" 2>/dev/null; then
                echo "Loaded $cand"
                break
            fi
        done
    fi
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Install lm_eval, preferring wheels. Pin pyarrow / libcst / pandas to
# wheel-friendly versions just in case.
uv pip install \
    "pandas<2.3" \
    "pyarrow<17" \
    "lm_eval==0.4.4"

echo
echo "lm_eval installed. Verify:"
echo "  python -c 'import lm_eval; print(lm_eval.__version__)'"
echo
echo "You can now submit:"
echo "  sbatch experiments/scg/run_eval_lm.slurm"
