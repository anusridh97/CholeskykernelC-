#!/bin/bash
#
# Run only after a human has audited all five completed pilot artifacts.

set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"
if git symbolic-ref -q HEAD >/dev/null; then
  echo "Refusing a movable branch: checkout the locked commit detached." >&2
  exit 2
fi
: "${E2_MANUAL_GATE_APPROVED:?set to YES_AFTER_AUDITING_ALL_FIVE_PILOTS}"
if [[ "${E2_MANUAL_GATE_APPROVED}" != "YES_AFTER_AUDITING_ALL_FIVE_PILOTS" ]]; then
  echo "The exact manual pilot-audit approval token is required." >&2
  exit 2
fi

locked_commit="$(git rev-parse HEAD)"
lock_path="${root}/rebuttal_experiments/e2_preregistered_lock.json"
lock_sha256="$(sha256sum "${lock_path}" | awk '{print $1}')"
results_root="${E2_RESULTS_ROOT:-${root}/rebuttal_results/e2_preregistered}"
driver="${root}/rebuttal_experiments/e2_preregistered_campaign.py"
attestation="I audited all five E2 v2 pilots and freeze confirmation without outcome-dependent changes"

/usr/bin/python3 "${driver}" \
  --action calibrate-budget \
  --results-root "${results_root}" \
  --expected-lock-sha256 "${lock_sha256}" \
  --locked-commit "${locked_commit}"

gate_path="$(
  /usr/bin/python3 "${driver}" \
    --action record-pilot-gate \
    --manual-attestation "${attestation}" \
    --results-root "${results_root}" \
    --expected-lock-sha256 "${lock_sha256}" \
    --locked-commit "${locked_commit}"
)"
printf 'gate_path=%s\n' "${gate_path}"
printf 'gate_sha256=%s\n' "$(sha256sum "${gate_path}" | awk '{print $1}')"
