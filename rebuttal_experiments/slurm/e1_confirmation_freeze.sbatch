#!/bin/bash
#SBATCH --job-name=e1-v5-freeze
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=rebuttal_logs/e1-v5-freeze-%j.out
#SBATCH --error=rebuttal_logs/e1-v5-freeze-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

candidate_dir="${PWD}/rebuttal_experiments"
package_dir="${PWD}/Cholesky-main-2/corrected_incremental_cholesky"
export E1_TORCH_SITE=/users/arjo/.local/lib/python3.10/site-packages
export PYTHONPATH="${candidate_dir}:${package_dir}:${PWD}:${PWD}/rebuttal_deps:${E1_TORCH_SITE}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export E1_PROCESS_CONCURRENCY=1
export E1_EXCLUSIVE_NODE=1

/usr/bin/python3 rebuttal_experiments/e1_confirmation.py \
  --action self-test --campaign-tag e1-confirm-samevector-out-v5

(
  cd "${candidate_dir}"
  /usr/bin/python3 setup_e1_positive_candidate.py build_ext --inplace --force
)

exec /cm/shared/apps/slurm/current/bin/srun \
  --exclusive --nodes=1 --ntasks=1 --cpus-per-task=10 \
  --hint=nomultithread --cpu-bind=cores --kill-on-bad-exit=1 \
  /usr/bin/python3 rebuttal_experiments/e1_confirmation.py \
  --action freeze-lock --campaign-tag e1-confirm-samevector-out-v5
