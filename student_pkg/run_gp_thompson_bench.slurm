#!/bin/bash
#SBATCH --job-name=gp-thompson
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=gp-thompson-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "host: $(hostname)"
echo "job: ${SLURM_JOB_ID:-unknown}"
echo "date: $(date)"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

python -m tests.bench_gp_thompson
