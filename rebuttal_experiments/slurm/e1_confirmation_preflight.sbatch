#!/bin/bash
#SBATCH --job-name=e1-confirm-preflight
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=rebuttal_logs/e1-confirm-preflight-%j.out
#SBATCH --error=rebuttal_logs/e1-confirm-preflight-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

package_dir="${PWD}/Cholesky-main-2/corrected_incremental_cholesky"
export E1_TORCH_SITE=/users/arjo/.local/lib/python3.10/site-packages
export PYTHONPATH="${package_dir}:${PWD}:${PWD}/rebuttal_deps:${E1_TORCH_SITE}"
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores

(
  cd "${package_dir}"
  /usr/bin/python3 setup_ext.py build_ext --inplace --force
)

srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=10 \
  --hint=nomultithread --distribution=block:block --cpu-bind=cores \
  /usr/bin/python3 rebuttal_experiments/e1_confirmation.py \
  --action freeze-lock --campaign-tag e1-confirm-fastpositive-v2

srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=10 \
  --hint=nomultithread --distribution=block:block --cpu-bind=cores \
  /usr/bin/python3 rebuttal_experiments/e1_confirmation.py \
  --action preflight --campaign-tag e1-confirm-fastpositive-v2
