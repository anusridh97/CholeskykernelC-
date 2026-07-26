#!/bin/bash
#SBATCH --job-name=e1-paired-confirm
#SBATCH --account=marlowe-m000151
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-59%10
#SBATCH --requeue
#SBATCH --output=rebuttal_logs/e1-paired-confirm-%A_%a.out
#SBATCH --error=rebuttal_logs/e1-paired-confirm-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

task_index="${SLURM_ARRAY_TASK_ID}"
if (( task_index % 2 == 0 )); then
  thread_count=1
else
  thread_count=10
fi

package_dir="${PWD}/Cholesky-main-2/corrected_incremental_cholesky"
export PYTHONPATH="${package_dir}:${PWD}:${PWD}/rebuttal_deps"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${thread_count}"
export MKL_NUM_THREADS="${thread_count}"
export OPENBLAS_NUM_THREADS="${thread_count}"
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores

srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=10 \
  --hint=nomultithread --distribution=block:block --cpu-bind=cores \
  /usr/bin/python3 rebuttal_experiments/e1_confirmation.py \
  --action run-pair --phase confirmation --task-index "${task_index}" \
  --campaign-tag e1-confirm-fastpositive-v1
