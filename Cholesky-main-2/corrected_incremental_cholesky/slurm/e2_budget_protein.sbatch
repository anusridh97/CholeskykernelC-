#!/bin/bash
#SBATCH --job-name=e2-protein-budget
#SBATCH --account=marlowe-m000151
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=rebuttal_logs/e2-protein-budget-%j.out
#SBATCH --error=rebuttal_logs/e2-protein-budget-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

package_dir="${PWD}/Cholesky-main-2/corrected_incremental_cholesky"
runner="${package_dir}/e2_budget_protein.py"
results_dir="${package_dir}/results/e2_budget_protein"
data_path="${PWD}/Cholesky-main-2/data/protein.mat"
campaign_tag="${E2_RUN_TAG:-e2proper-${SLURM_JOB_ID}}"
python_path="${package_dir}:${PWD}:${PWD}/rebuttal_deps"

export PYTHONPATH="${python_path}"
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10

run_python() {
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=10 \
    /usr/bin/python3 "${runner}" "$@"
}

wait_group() {
  local status=0
  local pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  return "${status}"
}

# Three disjoint pilot seeds calibrate the 10-epoch Ours-r1024 budget on the
# same node and concurrency level used for evaluation. Pilots never evaluate
# the held-out test partition.
pilot_pids=()
for seed in 1000 1001 1002; do
  run_python \
    --action run --phase pilot --method ours --rank 1024 --seed "${seed}" \
    --campaign-tag "${campaign_tag}" --results-dir "${results_dir}" \
    --data-path "${data_path}" --max-epochs 10 --threads 10 &
  pilot_pids+=("$!")
done
wait_group "${pilot_pids[@]}"

/usr/bin/python3 "${runner}" \
  --action calibrate-budget \
  --campaign-tag "${campaign_tag}" --results-dir "${results_dir}"
budget_file="${results_dir}/${campaign_tag}/budget.json"

# Match pilot concurrency: for each frozen method/rank row, run its three
# evaluation seeds concurrently on disjoint ten-core Slurm job steps.
for specification in "ours 1024" "refact 256" "refact 512" "refact 1024"; do
  read -r method rank <<< "${specification}"
  eval_pids=()
  for seed in 0 1 2; do
    run_python \
      --action run --phase eval --method "${method}" --rank "${rank}" \
      --seed "${seed}" --campaign-tag "${campaign_tag}" \
      --results-dir "${results_dir}" --data-path "${data_path}" \
      --budget-file "${budget_file}" --max-epochs 10 --threads 10 &
    eval_pids+=("$!")
  done
  wait_group "${eval_pids[@]}"
done

/usr/bin/python3 "${runner}" \
  --action aggregate \
  --campaign-tag "${campaign_tag}" --results-dir "${results_dir}"
