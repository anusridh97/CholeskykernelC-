#!/bin/bash
#SBATCH --job-name=cuda-inverse-condition
#SBATCH --account=marlowe-m000151
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=rebuttal_logs/cuda-inverse-condition-%j.out
#SBATCH --error=rebuttal_logs/cuda-inverse-condition-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PWD}/rebuttal_deps:${PWD}"

for condition in 1e2 1e4 1e6; do
  for seed in 0 1 2; do
    /usr/bin/python3 -m rebuttal_experiments.cuda_inverse_stability \
      --rank 256 --steps 200 --seed "${seed}" \
      --condition "${condition}" \
      --tile-rows 2 --tile-cols 8 \
      --setup-num-warps 8 --sweep-num-warps 8
  done
done
