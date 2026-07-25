#!/bin/bash
#SBATCH --job-name=cuda-parallel-profile
#SBATCH --account=marlowe-m000151
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=rebuttal_logs/cuda-parallel-profile-%j.out
#SBATCH --error=rebuttal_logs/cuda-parallel-profile-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PWD}/rebuttal_deps:${PWD}"

/usr/bin/python3 -m rebuttal_experiments.cuda_parallel_profile \
  --rank 256 --tile-rows 8 --tile-cols 8 --repeats 30
/usr/bin/python3 -m rebuttal_experiments.cuda_parallel_profile \
  --rank 512 --tile-rows 4 --tile-cols 8 --repeats 30
/usr/bin/python3 -m rebuttal_experiments.cuda_parallel_profile \
  --rank 1024 --tile-rows 4 --tile-cols 8 --repeats 30
