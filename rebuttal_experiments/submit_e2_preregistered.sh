#!/bin/bash
#
# Submit the immutable preflight and five PILOT PAIRS ONLY.
# Confirmation has a separate script and cannot run without a manually
# recorded, write-once pilot gate.

set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"

if git symbolic-ref -q HEAD >/dev/null; then
  echo "Refusing a movable branch: checkout the locked commit detached." >&2
  exit 2
fi

: "${E2_TORCH_SITE:?set the explicit frozen Torch site}"
: "${E2_NUMERIC_SITE:?set the explicit frozen NumPy/SciPy site}"
if [[ ! -d "${E2_TORCH_SITE}" || ! -d "${E2_NUMERIC_SITE}" ]]; then
  echo "Both explicit runtime roots must exist." >&2
  exit 2
fi

locked_commit="$(git rev-parse HEAD)"
lock_path="${root}/rebuttal_experiments/e2_preregistered_lock.json"
lock_sha256="$(sha256sum "${lock_path}" | awk '{print $1}')"
results_root="${E2_RESULTS_ROOT:-${root}/rebuttal_results/e2_preregistered}"
sbatch_script="${root}/rebuttal_experiments/slurm/e2_preregistered.sbatch"
driver="${root}/rebuttal_experiments/e2_preregistered_campaign.py"

/usr/bin/python3 "${driver}" \
  --action verify-lock \
  --results-root "${results_root}" \
  --expected-lock-sha256 "${lock_sha256}" \
  --locked-commit "${locked_commit}"

campaign_dir="${results_root}/e2-positive-preregistered-v2"
if [[ -e "${campaign_dir}" ]]; then
  echo "Refusing to reuse an existing one-attempt campaign directory: ${campaign_dir}" >&2
  exit 2
fi
mkdir -p "${results_root}" "${root}/rebuttal_logs"

common_export="E2_LOCKED_COMMIT=${locked_commit},E2_LOCK_SHA256=${lock_sha256},E2_RESULTS_ROOT=${results_root},E2_TORCH_SITE=${E2_TORCH_SITE},E2_NUMERIC_SITE=${E2_NUMERIC_SITE}"

self_test_job="$(
  sbatch --parsable \
    --export="${common_export},E2_STAGE=self-test" \
    "${sbatch_script}"
)"
pilot_job="$(
  sbatch --parsable \
    --dependency="afterok:${self_test_job}" \
    --array=0-4%5 \
    --export="${common_export},E2_STAGE=pilot-pair" \
    "${sbatch_script}"
)"

printf 'preflight=%s\n' "${self_test_job}"
printf 'pilot=%s\n' "${pilot_job}"
printf 'confirmation_submitted=no\n'
