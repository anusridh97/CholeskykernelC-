#!/usr/bin/env bash
# One-shot environment setup on a Farmshare login node.
# Run this BEFORE submitting any SLURM jobs. No GPU required.
#
# Sets up the venv at student_pkg/.venv (matching the existing kernel slurm
# scripts), installs all training/eval dependencies, and builds the C++
# extension.

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
VENV_DIR="$REPO_ROOT/student_pkg/.venv"
echo "Repo root: $REPO_ROOT"
echo "Venv:      $VENV_DIR"

cd "$REPO_ROOT/student_pkg"

# uv: fast python env manager. Install if not present.
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installs to ~/.cargo/bin or ~/.local/bin depending on system.
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi

# Create venv (Python 3.11 recommended; 3.10/3.12 also fine).
if [ ! -d "$VENV_DIR" ]; then
    uv venv --python 3.11
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Load CUDA so the C++ extension and Triton see nvcc/CUDA libs.
# Skip silently if `module` isn't on PATH (e.g., running outside Farmshare).
if command -v module >/dev/null 2>&1; then
    module load cuda || true
    if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
        export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
    fi
fi

# Core deps. Pin torch to a CUDA-12.x build (Farmshare GPU nodes run CUDA 12.x).
uv pip install --upgrade pip
uv pip install \
    "torch==2.4.0" \
    "triton>=3.0.0" \
    "transformers>=4.43" \
    "datasets>=2.20" \
    "accelerate>=0.33" \
    "huggingface_hub>=0.24" \
    "numpy<2" \
    "pytest" \
    "pybind11" \
    "matplotlib" \
    "lm_eval"

# Build the C++ extension (CPU correctness path, kernel debugging).
python setup_ext.py build_ext --inplace

# Sanity-check the kernel + manual backward on CPU (no GPU needed).
echo
echo "--- CPU correctness ---"
python -m pytest tests/test_correctness.py -v -k "reference" || true

echo
echo "--- SKA recurrence + manual backward (CPU) ---"
cd "$REPO_ROOT/experiments"
python ska_recurrence.py

echo
echo "--- Model sizing ---"
python ska_model.py

echo
echo "Setup complete."
echo
echo "Next steps:"
echo "  1. (Optional) login to HuggingFace for SlimPajama:"
echo "       huggingface-cli login"
echo "  2. Submit the sanity training to verify GPU path:"
echo "       sbatch experiments/run_train_sanity.slurm"
echo "  3. Once sanity passes, kick off the main run:"
echo "       bash experiments/submit_chain.sh prod 2"
echo "       (the '2' submits two chained 24h jobs; adjust for your token target)"
