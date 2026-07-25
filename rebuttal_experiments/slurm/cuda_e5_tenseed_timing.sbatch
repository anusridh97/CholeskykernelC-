#!/bin/bash
#SBATCH --job-name=e5-n10-timing
#SBATCH --account=marlowe-m000151
#SBATCH --partition=preempt
#SBATCH --array=1-10%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=rebuttal_logs/e5-n10-timing-%A_%a.out
#SBATCH --error=rebuttal_logs/e5-n10-timing-%A_%a.err

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

run_scaling() {
  /usr/bin/python3 -m rebuttal_experiments.cuda_scaling \
    "$@" \
    --seed "${seed}" --warmup 10 --repeats 60 \
    --threads "${SLURM_CPUS_PER_TASK}" \
    --output-suffix "${output_suffix}"
}

# Original E5 table: frozen fused rows.
for rank in 64 128; do
  for batch in 1 4 16 64; do
    run_scaling \
      --impl fused --rank "${rank}" --batch "${batch}" --num-warps 4
  done
done

# Original E5 table: frozen baseline-compatible parallel rows.
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

# Frozen corrected inverse-factor rows selected on seed 0.
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
