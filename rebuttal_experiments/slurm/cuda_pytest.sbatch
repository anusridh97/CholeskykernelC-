#!/bin/bash
#SBATCH --job-name=cuda-pytest
#SBATCH --account=marlowe-m000151
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=rebuttal_logs/cuda-pytest-%j.out
#SBATCH --error=rebuttal_logs/cuda-pytest-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PWD}/rebuttal_deps:${PWD}/student_pkg"

/usr/bin/python3 -m pytest student_pkg/tests/test_correctness.py -q
