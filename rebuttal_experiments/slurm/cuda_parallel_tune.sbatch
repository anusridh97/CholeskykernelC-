#!/bin/bash
#SBATCH --job-name=cuda-parallel-tune
#SBATCH --account=marlowe-m000151
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=rebuttal_logs/cuda-parallel-tune-%j.out
#SBATCH --error=rebuttal_logs/cuda-parallel-tune-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PWD}/rebuttal_deps:${PWD}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

/usr/bin/python3 -m rebuttal_experiments.cuda_parallel_tune
