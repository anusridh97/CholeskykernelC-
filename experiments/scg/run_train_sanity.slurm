#!/bin/bash
#SBATCH --job-name=ska-sanity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/ska-sanity-%j.out
#SBATCH --error=logs/ska-sanity-%j.err

# Single-GPU sanity training on SCG. Cluster-specific directives passed
# via `sbatch --partition=... --account=... --gres=... --time=...`.
# See submit_chain_scg.sh for the canonical wrapper.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
mkdir -p "$HF_HOME"

CKPT_DIR="${SKA_CKPT_DIR:-$REPO_ROOT/checkpoints/sanity_scg}"
mkdir -p "$CKPT_DIR"

cd "$REPO_ROOT/experiments/backend"

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "GPU: $(nvidia-smi -L)"
echo

# Optional extra flags (e.g. --retrieval_ratio 0.15 --wandb_project ska-pretrain).
# Passed verbatim to train.py. See experiments/backend/train.py argparse for options.
EXTRA_ARGS="${SKA_EXTRA_ARGS:-}"

python train.py \
    --config sanity \
    --out_dir "$CKPT_DIR" \
    --micro_batch "${SKA_MICRO_BATCH:-2}" \
    $EXTRA_ARGS

echo "Sanity training done. Final checkpoint:"
ls -lt "$CKPT_DIR"/ckpt_*.pt | head -n 1
