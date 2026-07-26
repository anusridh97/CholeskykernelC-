#!/bin/bash
#SBATCH --job-name=e4-direct-confirm
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=rebuttal_logs/e4-direct-confirm-%j.out
#SBATCH --error=rebuttal_logs/e4-direct-confirm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

: "${SOURCE_COMMIT:?submit with --export=ALL,SOURCE_COMMIT=<exact commit>}"
test "$(git rev-parse HEAD)" = "${SOURCE_COMMIT}"
git diff --quiet
git diff --cached --quiet

readonly TORCH_SITE="/scratch/m000151-pm06/arjo/blackhole-0977166-final/torch_site"
readonly REBUTTAL_DEPS="/scratch/m000151-pm06/arjo/blackhole-0977166-final/rebuttal_deps"
readonly SOURCE_DIR="${PWD}/Cholesky-main-2/corrected_incremental_cholesky"
readonly BUILD_ROOT="/scratch/m000151-pm06/arjo/job-builds/${SLURM_JOB_ID}/e4-direct"
mkdir -p "${BUILD_ROOT}/temp" "${BUILD_ROOT}/lib" rebuttal_logs

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export MAX_JOBS=1
export PYTHONPATH="${BUILD_ROOT}/lib:${PWD}:${REBUTTAL_DEPS}:${TORCH_SITE}"
export E4_EXTENSION_PATH="${BUILD_ROOT}/lib/fast_cholesky.so"

shadow_extension="$(find "${PWD}" -type f -name 'fast_cholesky*.so' -print -quit)"
if [[ -n "${shadow_extension}" ]]; then
  echo "refusing ignored/shadow extension: ${shadow_extension}" >&2
  exit 1
fi

(
  cd "${SOURCE_DIR}"
  /usr/bin/python3 setup_ext.py build_ext \
    --build-temp "${BUILD_ROOT}/temp" \
    --build-lib "${BUILD_ROOT}/lib"
)

test -f "${E4_EXTENSION_PATH}"

/usr/bin/python3 -m pytest -q \
  rebuttal_experiments/tests/test_e4_direct_confirmation.py

/cm/shared/apps/slurm/current/bin/srun \
  --exclusive \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --hint=nomultithread \
  --cpu-bind=cores \
  /usr/bin/python3 -m rebuttal_experiments.e4_direct_confirmation

/usr/bin/python3 -m rebuttal_experiments.audit_e4_direct_confirmation
