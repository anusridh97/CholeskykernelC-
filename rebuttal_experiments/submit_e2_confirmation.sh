#!/bin/bash
#
# Submit confirmation only after record_e2_pilot_gate.sh and external review.

set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"
if git symbolic-ref -q HEAD >/dev/null; then
  echo "Refusing a movable branch: checkout the locked commit detached." >&2
  exit 2
fi
: "${E2_GATE_SHA256:?set the exact audited write-once gate SHA-256}"
: "${E2_TORCH_SITE:?set the explicit frozen Torch site}"
: "${E2_NUMERIC_SITE:?set the explicit frozen NumPy/SciPy site}"

locked_commit="$(git rev-parse HEAD)"
lock_path="${root}/rebuttal_experiments/e2_preregistered_lock.json"
lock_sha256="$(sha256sum "${lock_path}" | awk '{print $1}')"
results_root="${E2_RESULTS_ROOT:-${root}/rebuttal_results/e2_preregistered}"
sbatch_script="${root}/rebuttal_experiments/slurm/e2_preregistered.sbatch"
driver="${root}/rebuttal_experiments/e2_preregistered_campaign.py"

/usr/bin/python3 "${driver}" \
  --action verify-pilot-gate \
  --expected-gate-sha256 "${E2_GATE_SHA256}" \
  --results-root "${results_root}" \
  --expected-lock-sha256 "${lock_sha256}" \
  --locked-commit "${locked_commit}"

common_export="E2_LOCKED_COMMIT=${locked_commit},E2_LOCK_SHA256=${lock_sha256},E2_GATE_SHA256=${E2_GATE_SHA256},E2_RESULTS_ROOT=${results_root},E2_TORCH_SITE=${E2_TORCH_SITE},E2_NUMERIC_SITE=${E2_NUMERIC_SITE}"

accuracy_job="$(
  sbatch --parsable \
    --array=0-19%5 \
    --export="${common_export},E2_STAGE=accuracy-pair" \
    "${sbatch_script}"
)"
throughput_job="$(
  sbatch --parsable \
    --array=0-19%5 \
    --export="${common_export},E2_STAGE=throughput-pair" \
    "${sbatch_script}"
)"
aggregate_job="$(
  sbatch --parsable \
    --dependency="afterok:${accuracy_job}:${throughput_job}" \
    --export="${common_export},E2_STAGE=aggregate" \
    "${sbatch_script}"
)"

printf 'accuracy=%s\n' "${accuracy_job}"
printf 'throughput=%s\n' "${throughput_job}"
printf 'aggregate=%s\n' "${aggregate_job}"
