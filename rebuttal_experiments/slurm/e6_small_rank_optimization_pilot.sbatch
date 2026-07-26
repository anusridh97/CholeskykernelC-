#!/bin/bash
#SBATCH --job-name=e6-small-rank-pilot
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=96G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=rebuttal_logs/e6-small-rank-pilot-%j.out
#SBATCH --error=rebuttal_logs/e6-small-rank-pilot-%j.err

# Development-only implementation-selection pilot.  These five seeds are
# excluded from any confirmation estimate or paper table.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

: "${SOURCE_COMMIT:?submit with --export=ALL,SOURCE_COMMIT=<exact commit>}"
test "$(git rev-parse HEAD)" = "${SOURCE_COMMIT}"
git diff --quiet
git diff --cached --quiet

readonly TORCH_SITE="/scratch/m000151-pm06/arjo/blackhole-0977166-final/torch_site"
readonly REBUTTAL_DEPS="/scratch/m000151-pm06/arjo/blackhole-0977166-final/rebuttal_deps"
readonly BUILD_ROOT="/scratch/m000151-pm06/arjo/job-builds/${SLURM_JOB_ID}/e6-small-rank-pilot"
readonly OUTPUT_ROOT="rebuttal_results/e6_small_rank_optimization_pilot"
mkdir -p "${BUILD_ROOT}/temp" "${BUILD_ROOT}/lib" "${OUTPUT_ROOT}" rebuttal_logs

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export MAX_JOBS=1
export PYTHONPATH="${BUILD_ROOT}/lib:${PWD}/student_pkg:${PWD}:${REBUTTAL_DEPS}:${TORCH_SITE}"

(
  cd student_pkg
  /usr/bin/python3 setup_ext.py build_ext \
    --build-temp "${BUILD_ROOT}/temp" \
    --build-lib "${BUILD_ROOT}/lib"
)

/usr/bin/python3 -m pytest -q student_pkg/tests/test_correctness.py

/cm/shared/apps/slurm/current/bin/srun \
  --exclusive \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --hint=nomultithread \
  --cpu-bind=cores \
  bash -lc '
    set -euo pipefail
    for seed in 87001 87002 87003 87004 87005; do
      /usr/bin/python3 -m rebuttal_experiments.e6_small_rank_optimization_pilot \
        --seed "${seed}" \
        --repeats 60 \
        --output "rebuttal_results/e6_small_rank_optimization_pilot/seed_${seed}.json"
    done
  '
