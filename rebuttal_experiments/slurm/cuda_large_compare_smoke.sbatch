#!/bin/bash
#SBATCH --job-name=cuda-large-compare-smoke
#SBATCH --account=marlowe-m000151
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=rebuttal_logs/cuda-large-compare-smoke-%j.out
#SBATCH --error=rebuttal_logs/cuda-large-compare-smoke-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PWD}/rebuttal_deps:${PWD}"

/usr/bin/python3 -m rebuttal_experiments.cuda_large_rank_compare \
  --mode alias-check --rank 256
/usr/bin/python3 -m rebuttal_experiments.cuda_large_rank_compare \
  --mode inverse --rank 256 --warmup 1 --repeats 10
/usr/bin/python3 -m rebuttal_experiments.cuda_large_rank_compare \
  --mode refact --rank 256 --warmup 1 --repeats 10
/usr/bin/python3 -m rebuttal_experiments.cuda_large_rank_compare \
  --mode inverse --rank 2048 --warmup 1 --repeats 5
/usr/bin/python3 -m rebuttal_experiments.cuda_large_rank_compare \
  --mode refact --rank 2048 --warmup 1 --repeats 5
