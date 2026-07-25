#!/bin/bash
#SBATCH --job-name=cuda-inverse-corrected-tune
#SBATCH --account=marlowe-m000151
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=rebuttal_logs/cuda-inverse-corrected-tune-%j.out
#SBATCH --error=rebuttal_logs/cuda-inverse-corrected-tune-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PWD}/rebuttal_deps:${PWD}"

for setup_warps in 1 2 4 8; do
  /usr/bin/python3 -m rebuttal_experiments.cuda_scaling \
    --rank 256 --batch 1 --impl inverse_factor \
    --num-warps "${setup_warps}" --sweep-num-warps 8 \
    --tile-rows 2 --tile-cols 8 --seed 0 \
    --warmup 10 --repeats 60 \
    --output-suffix "corrected-tune-${SLURM_JOB_ID}"
  /usr/bin/python3 -m rebuttal_experiments.cuda_scaling \
    --rank 512 --batch 1 --impl inverse_factor \
    --num-warps "${setup_warps}" --sweep-num-warps 8 \
    --tile-rows 1 --tile-cols 8 --seed 0 \
    --warmup 10 --repeats 60 \
    --output-suffix "corrected-tune-${SLURM_JOB_ID}"
  /usr/bin/python3 -m rebuttal_experiments.cuda_scaling \
    --rank 1024 --batch 1 --impl inverse_factor \
    --num-warps "${setup_warps}" --sweep-num-warps 8 \
    --tile-rows 4 --tile-cols 8 --seed 0 \
    --warmup 10 --repeats 60 \
    --output-suffix "corrected-tune-${SLURM_JOB_ID}"
done
