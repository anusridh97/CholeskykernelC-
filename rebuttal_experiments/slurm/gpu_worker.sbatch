#!/bin/bash
#SBATCH --job-name=rebuttal-gpu-worker
#SBATCH --account=marlowe-m000151
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=rebuttal_logs/gpu-worker-%j.out
#SBATCH --error=rebuttal_logs/gpu-worker-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PWD}/rebuttal_deps:${PWD}"
python3 -m rebuttal_experiments.worker --mode gpu --gpus 8 --retries 2
