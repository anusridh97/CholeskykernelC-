#!/bin/bash
#SBATCH --job-name=e3-metric-confirm
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=rebuttal_logs/e3-metric-confirm-%j.out
#SBATCH --error=rebuttal_logs/e3-metric-confirm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

: "${SOURCE_COMMIT:?submit with --export=ALL,SOURCE_COMMIT=<exact commit>}"
test "$(git rev-parse HEAD)" = "${SOURCE_COMMIT}"
git diff --quiet
git diff --cached --quiet

readonly TORCH_SITE="/scratch/m000151-pm06/arjo/blackhole-0977166-final/torch_site"
readonly REBUTTAL_DEPS="/scratch/m000151-pm06/arjo/blackhole-0977166-final/rebuttal_deps"
readonly BUILD_ROOT="${SLURM_TMPDIR}/e3-metric-build"
mkdir -p "${BUILD_ROOT}/temp-metric" "${BUILD_ROOT}/temp-one-sided" \
  "${BUILD_ROOT}/lib" rebuttal_logs

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

(
  cd rebuttal_experiments
  /usr/bin/python3 setup_e3_metric_only.py build_ext \
    --build-temp "${BUILD_ROOT}/temp-metric" \
    --build-lib "${BUILD_ROOT}/lib"
  /usr/bin/python3 setup_one_sided.py build_ext \
    --build-temp "${BUILD_ROOT}/temp-one-sided" \
    --build-lib "${BUILD_ROOT}/lib"
)

/usr/bin/python3 -m pytest -q \
  rebuttal_experiments/tests/test_e3_metric_only.py

srun \
  --exclusive \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --hint=nomultithread \
  --cpu-bind=cores \
  /usr/bin/python3 -m rebuttal_experiments.e3_metric_only_confirmation

/usr/bin/python3 -m rebuttal_experiments.audit_e3_metric_only_confirmation
