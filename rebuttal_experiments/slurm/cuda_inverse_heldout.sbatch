#!/bin/bash
#SBATCH --job-name=cuda-inverse-heldout
#SBATCH --account=marlowe-m000151
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=rebuttal_logs/cuda-inverse-heldout-%j.out
#SBATCH --error=rebuttal_logs/cuda-inverse-heldout-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PWD}/rebuttal_deps:${PWD}"

# Configurations were selected using seed 0 in job 393427. These three seeds
# are held out from that selection.
for seed in 1 2 3; do
  /usr/bin/python3 -m rebuttal_experiments.cuda_scaling \
    --rank 256 --batch 1 --impl inverse_factor \
    --num-warps 8 --sweep-num-warps 8 \
    --tile-rows 2 --tile-cols 8 --seed "${seed}" \
    --warmup 10 --repeats 60 \
    --output-suffix "corrected-heldout-${SLURM_JOB_ID}"
  /usr/bin/python3 -m rebuttal_experiments.cuda_scaling \
    --rank 512 --batch 1 --impl inverse_factor \
    --num-warps 1 --sweep-num-warps 8 \
    --tile-rows 1 --tile-cols 8 --seed "${seed}" \
    --warmup 10 --repeats 60 \
    --output-suffix "corrected-heldout-${SLURM_JOB_ID}"
  /usr/bin/python3 -m rebuttal_experiments.cuda_scaling \
    --rank 1024 --batch 1 --impl inverse_factor \
    --num-warps 2 --sweep-num-warps 8 \
    --tile-rows 4 --tile-cols 8 --seed "${seed}" \
    --warmup 10 --repeats 60 \
    --output-suffix "corrected-heldout-${SLURM_JOB_ID}"
done
