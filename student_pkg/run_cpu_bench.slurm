#!/bin/bash
#SBATCH --job-name=cpu-bench
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=cpu-bench-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "host: $(hostname)"
echo "job: ${SLURM_JOB_ID:-unknown}"
echo "date: $(date)"
echo "cpu:  $(grep -m1 'model name' /proc/cpuinfo | sed 's/^[^:]*: //')"

python -m tests.bench_timing
