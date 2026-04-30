#!/usr/bin/env bash
# Run this once on a fresh RunPod pod (any Linux + CUDA 12.1+ image).
# Uses uv for fast python env setup.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." &> /dev/null && pwd)
echo "Repo root: $REPO_ROOT"

# 1. System deps (RunPod images vary; this is idempotent).
apt-get update -y && apt-get install -y git build-essential curl ninja-build || true

# 2. Python env via uv (fast).
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env" 2>/dev/null || true
fi

cd "$REPO_ROOT/student_pkg"
if [ ! -d ".venv" ]; then
    uv venv --python 3.11
fi
source .venv/bin/activate

# 3. Core deps. Triton + CUDA-matching torch.
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
    "pybind11"

# 4. Build the C++ extension (used for CPU correctness tests, not training).
python setup_ext.py build_ext --inplace

# 5. Sanity-check the kernel.
echo "--- Kernel correctness ---"
python -m pytest tests/test_correctness.py -v -k "triton or reference" || true

echo "--- SKA wrapper sanity ---"
cd "$REPO_ROOT/experiments/backend"
python ska_recurrence.py

echo "--- Model sizing ---"
python ska_model.py

echo
echo "Setup complete. To start training:"
echo "  bash experiments/runpod/launch_training.sh sanity   # 1 GPU, 1h, ~\$3"
echo "  bash experiments/runpod/launch_training.sh prod     # 4 GPUs, ~14h, ~\$170"
