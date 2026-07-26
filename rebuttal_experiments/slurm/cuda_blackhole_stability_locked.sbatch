#!/bin/bash
#SBATCH --job-name=bh-s4-stability-locked
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --array=0-1%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=rebuttal_logs/blackhole-stability-locked-%A_%a.out
#SBATCH --error=rebuttal_logs/blackhole-stability-locked-%A_%a.err

set -euo pipefail

readonly outer_tag="blackhole-s4-v1-stability-locked-rerun2"
readonly inner_tag="blackhole-s4-v1"
readonly control_repo="${BLACKHOLE_STABILITY_CONTROL_REPO:?set BLACKHOLE_STABILITY_CONTROL_REPO}"
readonly locked_worktree="${BLACKHOLE_STABILITY_LOCKED_WORKTREE:?set BLACKHOLE_STABILITY_LOCKED_WORKTREE}"
readonly campaign_dir="${BLACKHOLE_STABILITY_CAMPAIGN_DIR:?set BLACKHOLE_STABILITY_CAMPAIGN_DIR}"
readonly torch_site="${BLACKHOLE_STABILITY_TORCH_SITE:?set BLACKHOLE_STABILITY_TORCH_SITE}"
readonly dep_root="${BLACKHOLE_STABILITY_DEP_ROOT:?set BLACKHOLE_STABILITY_DEP_ROOT}"
readonly helper="${control_repo}/rebuttal_experiments/blackhole_stability_locked.py"
readonly manifest="${campaign_dir}/feature_manifest.json"
ranks=(8192 32768)

if [[ "${BLACKHOLE_STABILITY_CAMPAIGN_TAG:-${outer_tag}}" != "${outer_tag}" ]]; then
  echo "outer campaign tag is frozen to ${outer_tag}" >&2
  exit 2
fi
if [[ ! "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-1]$ ]]; then
  echo "expected SLURM_ARRAY_TASK_ID in 0..1" >&2
  exit 2
fi
readonly task_id="${SLURM_ARRAY_TASK_ID}"
readonly rank="${ranks[task_id]}"
readonly inner_dir="${campaign_dir}/inner/${inner_tag}"
readonly result="${inner_dir}/stability_r${rank}.json"
readonly attestation="${campaign_dir}/attestation_stability_r${rank}.json"
readonly claim="${campaign_dir}/.claims/stability_r${rank}.claim"

if [[ -e "${result}" || -e "${attestation}" ]]; then
  echo "refusing to replace an existing rank-${rank} artifact" >&2
  exit 3
fi
mkdir -p "${campaign_dir}/.claims"
if ! mkdir "${claim}"; then
  echo "rank ${rank} is already claimed; preserve and inspect ${claim}" >&2
  exit 3
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${torch_site}:${dep_root}:${locked_worktree}"

/usr/bin/python3 "${helper}" verify-task \
  --control-repo "${control_repo}" \
  --source-repo "${locked_worktree}" \
  --campaign-dir "${campaign_dir}" \
  --rank "${rank}"

cd "${locked_worktree}"
# The unmodified v1 driver checks the legacy partition label.  This one
# variable is changed only for the inner process.  The outer verifier and
# attestation continue to see and require the real batch allocation.
env SLURM_JOB_PARTITION=preempt \
  /usr/bin/python3 -m rebuttal_experiments.cuda_blackhole_stability \
    --rank "${rank}" \
    --campaign-tag "${inner_tag}" \
    --feature-manifest "${manifest}" \
    --results-root "${campaign_dir}/inner"

/usr/bin/python3 "${helper}" attest-task \
  --control-repo "${control_repo}" \
  --source-repo "${locked_worktree}" \
  --campaign-dir "${campaign_dir}" \
  --rank "${rank}"

chmod a-w "${result}" "${attestation}" "${claim}"
