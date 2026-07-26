#!/bin/bash
#SBATCH --job-name=e6-compact-confirm
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
#SBATCH --output=rebuttal_logs/e6-compact-confirm-%j.out
#SBATCH --error=rebuttal_logs/e6-compact-confirm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

: "${SOURCE_COMMIT:?submit with --export=ALL,SOURCE_COMMIT=<exact commit>}"
test "$(git rev-parse HEAD)" = "${SOURCE_COMMIT}"
git diff --quiet
git diff --cached --quiet

readonly TORCH_SITE="/scratch/m000151-pm06/arjo/blackhole-0977166-final/torch_site"
readonly REBUTTAL_DEPS="/scratch/m000151-pm06/arjo/blackhole-0977166-final/rebuttal_deps"
readonly BUILD_ROOT="${SLURM_TMPDIR}/e6-compact-build"
mkdir -p "${BUILD_ROOT}/temp" "${BUILD_ROOT}/lib" rebuttal_logs

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export MAX_JOBS=8
export PYTHONPATH="${BUILD_ROOT}/lib:${PWD}/student_pkg:${PWD}:${REBUTTAL_DEPS}:${TORCH_SITE}"

(
  cd student_pkg
  /usr/bin/python3 setup_ext.py build_ext \
    --build-temp "${BUILD_ROOT}/temp" \
    --build-lib "${BUILD_ROOT}/lib"
)

/usr/bin/python3 -m pytest -q student_pkg/tests/test_correctness.py

srun \
  --exclusive \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --hint=nomultithread \
  --cpu-bind=cores \
  bash -lc '
    set -euo pipefail
    for rank in 256 1024; do
      for seed in 61001 61002 61003 61004 61005 61006 61007 61008 61009 61010; do
        /usr/bin/python3 -m rebuttal_experiments.rank_k \
          --confirmation \
          --rank "${rank}" \
          --ks 1 2 4 8 16 32 \
          --seed "${seed}" \
          --repeats 30 \
          --implementation compact \
          --threads 1
      done
    done
    /usr/bin/python3 -m rebuttal_experiments.aggregate_e6_compact_confirmation
  '
