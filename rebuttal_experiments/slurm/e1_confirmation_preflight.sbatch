#!/bin/bash
#SBATCH --job-name=e1-v5-preflight
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
#SBATCH --output=rebuttal_logs/e1-v5-preflight-%j.out
#SBATCH --error=rebuttal_logs/e1-v5-preflight-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH=/cm/shared/apps/slurm/current/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

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

exec /cm/shared/apps/slurm/current/bin/srun \
  --exclusive --nodes=1 --ntasks=1 --cpus-per-task=10 \
  --hint=nomultithread --cpu-bind=cores --kill-on-bad-exit=1 \
  /usr/bin/python3 rebuttal_experiments/e1_confirmation.py \
  --action preflight --campaign-tag e1-confirm-samevector-out-v5
