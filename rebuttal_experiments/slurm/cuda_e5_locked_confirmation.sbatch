#!/bin/bash
#SBATCH --job-name=e5-locked-confirm
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --array=5001-5010%10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=rebuttal_logs/e5-locked-%A_%a.out
#SBATCH --error=rebuttal_logs/e5-locked-%A_%a.err

set -euo pipefail

readonly repository="${E5_LOCKED_REPOSITORY:?set E5_LOCKED_REPOSITORY}"
readonly campaign_dir="${E5_CAMPAIGN_DIR:?set E5_CAMPAIGN_DIR}"
readonly lock="${campaign_dir}/campaign_lock.json"
readonly dependency_root="/users/ravid/.local/lib/python3.10/site-packages"
readonly seed="${SLURM_ARRAY_TASK_ID}"
readonly campaign_tag="e5-locked-confirm-v2"
readonly output_suffix="${campaign_tag}-task${seed}"
readonly attestation="${campaign_dir}/attestation_seed${seed}.json"
readonly claim="${campaign_dir}/.claims/seed${seed}.claim"

if (( seed < 5001 || seed > 5010 )); then
  echo "expected an array task in 5001..5010, got ${seed}" >&2
  exit 2
fi
if [[ ! -r "${campaign_dir}/preflight.json" ]]; then
  echo "the immutable data-free preflight receipt is missing" >&2
  exit 2
fi
if [[ -e "${attestation}" ]]; then
  echo "refusing to replace ${attestation}" >&2
  exit 3
fi
mkdir -p "${campaign_dir}/.claims"
if ! mkdir "${claim}"; then
  echo "seed ${seed} is already claimed: ${claim}" >&2
  exit 3
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${dependency_root}:${repository}"
export TRITON_CACHE_DIR="${campaign_dir}/triton-cache-seed${seed}"
mkdir -p "${TRITON_CACHE_DIR}"
cd "${repository}"

/usr/bin/python3 -m rebuttal_experiments.e5_locked_confirmation verify-task \
  --repository "${repository}" \
  --lock "${lock}" \
  --seed "${seed}"

run_scaling() {
  /usr/bin/python3 -m rebuttal_experiments.cuda_scaling \
    "$@" \
    --seed "${seed}" --warmup 10 --repeats 60 \
    --threads "${SLURM_CPUS_PER_TASK}" \
    --output-suffix "${output_suffix}"
}

for rank in 64 128; do
  for batch in 1 4 16 64; do
    run_scaling \
      --impl fused --rank "${rank}" --batch "${batch}" --num-warps 4
  done
done

run_scaling \
  --impl parallel --rank 256 --batch 1 \
  --num-warps 8 --sweep-num-warps 8 \
  --tile-rows 2 --tile-cols 8 --solve-width 2
run_scaling \
  --impl parallel --rank 512 --batch 1 \
  --num-warps 4 --sweep-num-warps 8 \
  --tile-rows 1 --tile-cols 8 --solve-width 2
run_scaling \
  --impl parallel --rank 1024 --batch 1 \
  --num-warps 4 --sweep-num-warps 8 \
  --tile-rows 4 --tile-cols 8 --solve-width 2

run_scaling \
  --impl inverse_factor --rank 256 --batch 1 \
  --num-warps 8 --sweep-num-warps 8 \
  --tile-rows 2 --tile-cols 8 --inverse-gemv-backend matmul
run_scaling \
  --impl inverse_factor --rank 512 --batch 1 \
  --num-warps 1 --sweep-num-warps 8 \
  --tile-rows 1 --tile-cols 8 --inverse-gemv-backend matmul
run_scaling \
  --impl inverse_factor --rank 1024 --batch 1 \
  --num-warps 2 --sweep-num-warps 8 \
  --tile-rows 4 --tile-cols 8 --inverse-gemv-backend matmul

run_stability() {
  /usr/bin/python3 -m rebuttal_experiments.cuda_inverse_stability \
    "$@" --seed "${seed}" --output-suffix "${output_suffix}"
}

run_refresh() {
  /usr/bin/python3 -m rebuttal_experiments.cuda_inverse_refresh \
    "$@" --seed "${seed}" --output-suffix "${output_suffix}"
}

run_stability \
  --rank 256 --steps 500 \
  --tile-rows 2 --tile-cols 8 \
  --setup-num-warps 8 --sweep-num-warps 8
run_stability \
  --rank 512 --steps 500 \
  --tile-rows 1 --tile-cols 8 \
  --setup-num-warps 4 --sweep-num-warps 8
run_stability \
  --rank 1024 --steps 500 \
  --tile-rows 4 --tile-cols 8 \
  --setup-num-warps 4 --sweep-num-warps 8
run_stability \
  --inplace --rank 1024 --steps 500 \
  --tile-rows 8 --tile-cols 8 \
  --setup-num-warps 4 --sweep-num-warps 4

for condition in 1e2 1e4 1e6; do
  run_stability \
    --rank 256 --steps 200 --condition "${condition}" \
    --tile-rows 2 --tile-cols 8 \
    --setup-num-warps 8 --sweep-num-warps 8
done

run_refresh \
  --rank 256 --steps 200 \
  --tile-rows 2 --tile-cols 8 \
  --setup-num-warps 8 --sweep-num-warps 8
run_refresh \
  --rank 512 --steps 200 \
  --tile-rows 1 --tile-cols 8 \
  --setup-num-warps 4 --sweep-num-warps 8
run_refresh \
  --rank 1024 --steps 200 \
  --tile-rows 4 --tile-cols 8 \
  --setup-num-warps 4 --sweep-num-warps 8

run_large() {
  local mode="$1"
  local rank="$2"
  local repeats="$3"
  local tile
  local -a extra_args=()
  if [[ "${mode}" == "inverse" ]]; then
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
  /usr/bin/python3 -m rebuttal_experiments.cuda_large_rank_compare \
    --mode "${mode}" --rank "${rank}" --seed "${seed}" \
    --warmup 1 --repeats "${repeats}" \
    --probe-min-rank 49152 --probe-width 16 \
    --probe-seed 29421 --probe-tolerance 1e-3 \
    --output-suffix "${output_suffix}" \
    "${extra_args[@]}"
}

for rank in 2048 4096 8192 16384 32768 49152 65536; do
  if (( rank <= 8192 )); then
    repeats=10
  elif (( rank <= 32768 )); then
    repeats=5
  else
    repeats=3
  fi
  if (( seed % 2 == 1 )); then
    run_large inverse "${rank}" "${repeats}"
    run_large refact "${rank}" "${repeats}"
  else
    run_large refact "${rank}" "${repeats}"
    run_large inverse "${rank}" "${repeats}"
  fi
done

/usr/bin/python3 -m rebuttal_experiments.e5_locked_confirmation attest-task \
  --repository "${repository}" \
  --lock "${lock}" \
  --seed "${seed}" \
  --output "${attestation}"

chmod a-w "${attestation}" "${claim}"
