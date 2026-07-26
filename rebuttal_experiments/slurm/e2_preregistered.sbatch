#!/bin/bash
#SBATCH --job-name=e2-preregistered
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=rebuttal_logs/e2-preregistered-%A_%a.out
#SBATCH --error=rebuttal_logs/e2-preregistered-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

: "${E2_STAGE:?set E2_STAGE to self-test, pilot-pair, accuracy-pair, throughput-pair, or aggregate}"
: "${E2_LOCKED_COMMIT:?set E2_LOCKED_COMMIT to the full detached commit SHA}"
: "${E2_LOCK_SHA256:?set E2_LOCK_SHA256 to the frozen lock-manifest SHA-256}"
: "${E2_TORCH_SITE:?set E2_TORCH_SITE to the frozen explicit Torch site}"
: "${E2_NUMERIC_SITE:?set E2_NUMERIC_SITE to the frozen explicit NumPy/SciPy site}"

if [[ "${SLURM_JOB_ACCOUNT:-}" != "marlowe-m000151-pm06" ]]; then
  echo "wrong Slurm account: ${SLURM_JOB_ACCOUNT:-unset}" >&2
  exit 2
fi
if [[ "${SLURM_JOB_PARTITION:-}" != "batch" ]]; then
  echo "wrong Slurm partition: ${SLURM_JOB_PARTITION:-unset}" >&2
  exit 2
fi
if [[ "${SLURM_RESTART_COUNT:-0}" != "0" ]]; then
  echo "requeued/restarted executions are forbidden" >&2
  exit 2
fi

driver="${PWD}/rebuttal_experiments/e2_preregistered_campaign.py"
results_root="${E2_RESULTS_ROOT:-${PWD}/rebuttal_results/e2_preregistered}"

unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONPATH="${PWD}:${E2_NUMERIC_SITE}:${E2_TORCH_SITE}"
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export E2_PROCESS_CONCURRENCY=1
export E2_EXCLUSIVE_NODE=1
export E2_ONE_ATTEMPT=1
export SLURM_EXPORT_ENV=ALL

arguments=(
  --action "${E2_STAGE}"
  --results-root "${results_root}"
  --expected-lock-sha256 "${E2_LOCK_SHA256}"
  --locked-commit "${E2_LOCKED_COMMIT}"
)

case "${E2_STAGE}" in
  accuracy-pair|throughput-pair|aggregate)
    : "${E2_GATE_SHA256:?confirmation stages require the exact manual gate SHA-256}"
    arguments+=(--expected-gate-sha256 "${E2_GATE_SHA256}")
    ;;
esac

case "${E2_STAGE}" in
  pilot-pair)
    : "${SLURM_ARRAY_TASK_ID:?pilot-pair requires --array=0-4}"
    if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 4 )); then
      echo "pilot-pair task ID must be in [0,4]" >&2
      exit 2
    fi
    arguments+=(--task-id "${SLURM_ARRAY_TASK_ID}")
    ;;
  accuracy-pair|throughput-pair)
    : "${SLURM_ARRAY_TASK_ID:?confirmation pair requires --array=0-19}"
    if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 19 )); then
      echo "confirmation task ID must be in [0,19]" >&2
      exit 2
    fi
    arguments+=(--task-id "${SLURM_ARRAY_TASK_ID}")
    ;;
  self-test|aggregate)
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
      echo "${E2_STAGE} must not be submitted as an array" >&2
      exit 2
    fi
    ;;
  *)
    echo "unknown E2_STAGE=${E2_STAGE}" >&2
    exit 2
    ;;
esac

exec srun \
  --exclusive \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=10 \
  --hint=nomultithread \
  --cpu-bind=cores \
  --kill-on-bad-exit=1 \
  /usr/bin/python3 "${driver}" "${arguments[@]}"
