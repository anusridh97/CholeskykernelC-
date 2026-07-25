#!/bin/bash
#SBATCH --job-name=cuda-large-inplace-tune
#SBATCH --account=marlowe-m000151
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=rebuttal_logs/cuda-large-inplace-tune-%j.out
#SBATCH --error=rebuttal_logs/cuda-large-inplace-tune-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PWD}/rebuttal_deps:${PWD}"

for rank in 8192 32768; do
  for tile in 8 16 32 64; do
    for sweep_warps in 1 2 4; do
      /usr/bin/python3 -m rebuttal_experiments.cuda_large_rank_compare \
        --mode inverse --rank "${rank}" --warmup 1 --repeats 5 \
        --correctness-max-rank 0 \
        --tile-rows "${tile}" --tile-cols "${tile}" \
        --setup-num-warps 4 --sweep-num-warps "${sweep_warps}"
    done
  done
done
