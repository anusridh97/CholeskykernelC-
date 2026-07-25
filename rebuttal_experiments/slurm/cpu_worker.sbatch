#!/bin/bash
#SBATCH --job-name=rebuttal-cpu-worker
#SBATCH --account=marlowe-m000151
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=rebuttal_logs/cpu-worker-%j.out
#SBATCH --error=rebuttal_logs/cpu-worker-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PWD}/rebuttal_deps:${PWD}"
python3 -m rebuttal_experiments.worker --mode cpu --workers 10 --retries 2
