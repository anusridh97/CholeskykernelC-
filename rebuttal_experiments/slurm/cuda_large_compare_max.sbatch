#!/bin/bash
#SBATCH --job-name=cuda-large-max
#SBATCH --account=marlowe-m000151
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=rebuttal_logs/cuda-large-max-%j.out
#SBATCH --error=rebuttal_logs/cuda-large-max-%j.err

set -u
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PWD}/rebuttal_deps:${PWD}"
failures=0

run_case() {
  if ! /usr/bin/python3 -m rebuttal_experiments.cuda_large_rank_compare "$@"; then
    echo "FAILED $*" >&2
    failures=$((failures + 1))
  fi
}

# The two configurations were essentially tied at r=32768 in selection job
# 393466 and expose the same 2048 sweep warps at r=65536.
run_case --mode inverse --rank 65536 --warmup 1 --repeats 3 \
  --correctness-max-rank 0 \
  --tile-rows 32 --tile-cols 32 \
  --setup-num-warps 4 --sweep-num-warps 1
run_case --mode inverse --rank 65536 --warmup 1 --repeats 3 \
  --correctness-max-rank 0 \
  --tile-rows 64 --tile-cols 64 \
  --setup-num-warps 4 --sweep-num-warps 2
run_case --mode refact --rank 65536 --warmup 1 --repeats 3

if (( failures > 0 )); then
  echo "${failures} rank-65536 cases failed" >&2
  exit 1
fi
