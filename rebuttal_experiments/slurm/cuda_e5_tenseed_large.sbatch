#!/bin/bash
#SBATCH --job-name=e5-n10-large
#SBATCH --account=marlowe-m000151
#SBATCH --partition=preempt
#SBATCH --array=1-10%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=rebuttal_logs/e5-n10-large-%A_%a.out
#SBATCH --error=rebuttal_logs/e5-n10-large-%A_%a.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PWD}/rebuttal_deps:${PWD}"

# Seed 0 selected the frozen configurations, so the measurement campaign uses
# ten held-out seeds. Each array task owns one seed and one collision-safe tag.
seed="${SLURM_ARRAY_TASK_ID}"
if (( seed < 1 || seed > 10 )); then
  echo "expected an array task in 1..10, got ${seed}" >&2
  exit 2
fi
campaign_tag="${E5_RUN_TAG:-e5-n10-${SLURM_ARRAY_JOB_ID}}"
output_suffix="${campaign_tag}-task${SLURM_ARRAY_TASK_ID}"
failures=0

run_case() {
  local mode="$1"
  local rank="$2"
  local repeats="$3"
  local warmup="$4"
  local tile
  local -a extra_args=()

  if [[ "${mode}" == "inverse" ]]; then
    # Frozen selection: 8x8 through 8192, 16x16 at 16384, and 32x32 above.
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
    --mode "${mode}" --rank "${rank}" --seed "${seed}" \
    --warmup "${warmup}" --repeats "${repeats}" \
    --output-suffix "${output_suffix}" \
    "${extra_args[@]}"; then
    echo "FAILED seed=${seed} mode=${mode} rank=${rank}" >&2
    failures=$((failures + 1))
  fi
}

run_pair() {
  local rank="$1"
  local repeats="$2"

  # Counterbalance thermal/order effects: odd seeds run incremental first,
  # while even seeds run fresh refactorization first.
  if (( seed % 2 == 1 )); then
    run_case inverse "${rank}" "${repeats}" 1
    run_case refact "${rank}" "${repeats}" 1
  else
    run_case refact "${rank}" "${repeats}" 1
    run_case inverse "${rank}" "${repeats}" 1
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
  run_pair "${rank}" "${repeats}"
done

if (( failures > 0 )); then
  echo "${failures} large-rank comparison cases failed" >&2
  exit 1
fi
