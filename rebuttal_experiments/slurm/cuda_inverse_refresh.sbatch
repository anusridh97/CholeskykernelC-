#!/bin/bash
#SBATCH --job-name=cuda-inverse-refresh
#SBATCH --account=marlowe-m000151
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=rebuttal_logs/cuda-inverse-refresh-%j.out
#SBATCH --error=rebuttal_logs/cuda-inverse-refresh-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PWD}/rebuttal_deps:${PWD}"

for seed in 0 1 2; do
  /usr/bin/python3 -m rebuttal_experiments.cuda_inverse_refresh \
    --rank 256 --steps 200 --seed "${seed}" \
    --tile-rows 2 --tile-cols 8 \
    --setup-num-warps 8 --sweep-num-warps 8
  /usr/bin/python3 -m rebuttal_experiments.cuda_inverse_refresh \
    --rank 512 --steps 200 --seed "${seed}" \
    --tile-rows 1 --tile-cols 8 \
    --setup-num-warps 4 --sweep-num-warps 8
  /usr/bin/python3 -m rebuttal_experiments.cuda_inverse_refresh \
    --rank 1024 --steps 200 --seed "${seed}" \
    --tile-rows 4 --tile-cols 8 \
    --setup-num-warps 4 --sweep-num-warps 8
done
