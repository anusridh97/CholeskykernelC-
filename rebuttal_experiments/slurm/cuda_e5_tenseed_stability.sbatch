#!/bin/bash
#SBATCH --job-name=e5-n10-stability
#SBATCH --account=marlowe-m000151
#SBATCH --partition=preempt
#SBATCH --array=1-10%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=rebuttal_logs/e5-n10-stability-%A_%a.out
#SBATCH --error=rebuttal_logs/e5-n10-stability-%A_%a.err

set -euo pipefail
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

run_stability() {
  /usr/bin/python3 -m rebuttal_experiments.cuda_inverse_stability \
    "$@" --seed "${seed}" --output-suffix "${output_suffix}"
}

run_refresh() {
  /usr/bin/python3 -m rebuttal_experiments.cuda_inverse_refresh \
    "$@" --seed "${seed}" --output-suffix "${output_suffix}"
}

# Corrected 500-update stability rows.
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

# Condition-number stress at fixed r=256.
for condition in 1e2 1e4 1e6; do
  run_stability \
    --rank 256 --steps 200 --condition "${condition}" \
    --tile-rows 2 --tile-cols 8 \
    --setup-num-warps 8 --sweep-num-warps 8
done

# Self-contained refresh sweep; the driver's frozen default intervals are
# q in {none, 10, 50, 100, 200}.
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
