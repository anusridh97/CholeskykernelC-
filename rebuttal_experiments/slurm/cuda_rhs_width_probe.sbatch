#!/bin/bash
#SBATCH --job-name=cuda-rhs-width
#SBATCH --account=marlowe-m000151
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=rebuttal_logs/cuda-rhs-width-%j.out
#SBATCH --error=rebuttal_logs/cuda-rhs-width-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PWD}/rebuttal_deps:${PWD}"

for width in 2 4 8 16 32 64 128; do
  /usr/bin/python3 -m rebuttal_experiments.cuda_scaling \
    --rank 1024 --batch 1 --impl parallel \
    --num-warps 4 --sweep-num-warps 8 --tile-rows 4 --tile-cols 8 \
    --solve-width "${width}" --warmup 5 --repeats 30 \
    --output-suffix "rhs-width${SLURM_JOB_ID}"
done

for width in 2 8 16 32 64; do
  /usr/bin/python3 -m rebuttal_experiments.cuda_parallel_profile \
    --rank 1024 --tile-rows 4 --tile-cols 8 \
    --solve-width "${width}" --repeats 20
done
