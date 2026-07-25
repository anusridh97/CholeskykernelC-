#!/bin/bash
#SBATCH --job-name=cuda-large-compare
#SBATCH --account=marlowe-m000151
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=rebuttal_logs/cuda-large-compare-%j.out
#SBATCH --error=rebuttal_logs/cuda-large-compare-%j.err

set -u
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PWD}/rebuttal_deps:${PWD}"
failures=0

run_case() {
  mode="$1"
  rank="$2"
  repeats="$3"
  warmup="$4"
  extra_args=()
  if [[ "${mode}" == "inverse" ]]; then
    # Selected on job 393466, which kept the selection ranks separate from
    # this final measurement ladder.  The optimum moves to wider tiles as r
    # grows; one sweep warp won decisively at both selection ranks.
    if (( rank <= 8192 )); then
      tile=8
    elif (( rank <= 16384 )); then
      tile=16
    else
      tile=32
    fi
    extra_args=(
      --tile-rows "${tile}" --tile-cols "${tile}"
      --setup-num-warps 4 --sweep-num-warps 1
      --correctness-max-rank 32768
    )
  fi
  if ! /usr/bin/python3 -m rebuttal_experiments.cuda_large_rank_compare \
    --mode "${mode}" --rank "${rank}" \
    --warmup "${warmup}" --repeats "${repeats}" \
    "${extra_args[@]}"; then
    echo "FAILED mode=${mode} rank=${rank}" >&2
    failures=$((failures + 1))
  fi
}

for rank in 2048 4096 8192 16384 32768 49152 65536; do
  if (( rank <= 8192 )); then
    repeats=10
  elif (( rank <= 32768 )); then
    repeats=5
  else
    repeats=3
  fi
  run_case inverse "${rank}" "${repeats}" 1
  run_case refact "${rank}" "${repeats}" 1
done

if (( failures > 0 )); then
  echo "${failures} large-rank comparison cases failed" >&2
  exit 1
fi
