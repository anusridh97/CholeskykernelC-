#!/bin/bash
#SBATCH --job-name=bh-s4-locked-r2
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --array=0-6%7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=rebuttal_logs/blackhole-locked-%A_%a.out
#SBATCH --error=rebuttal_logs/blackhole-locked-%A_%a.err

set -euo pipefail

readonly outer_tag="blackhole-s4-v1-locked-rerun2"
readonly inner_tag="blackhole-s4-v1"
readonly control_repo="${BLACKHOLE_CONTROL_REPO:?set BLACKHOLE_CONTROL_REPO}"
readonly locked_worktree="${BLACKHOLE_LOCKED_WORKTREE:?set BLACKHOLE_LOCKED_WORKTREE}"
readonly campaign_dir="${BLACKHOLE_CAMPAIGN_DIR:?set BLACKHOLE_CAMPAIGN_DIR}"
readonly dep_root="${BLACKHOLE_DEP_ROOT:-${control_repo}/rebuttal_deps}"
readonly helper="${control_repo}/rebuttal_experiments/blackhole_locked_rerun.py"
readonly manifest="${campaign_dir}/feature_manifest.json"
ranks=(2048 4096 8192 16384 32768 49152 65536)

if [[ "${BLACKHOLE_CAMPAIGN_TAG:-${outer_tag}}" != "${outer_tag}" ]]; then
  echo "outer campaign tag is frozen to ${outer_tag}" >&2
  exit 2
fi
if [[ ! "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-6]$ ]]; then
  echo "expected SLURM_ARRAY_TASK_ID in 0..6" >&2
  exit 2
fi
readonly task_id="${SLURM_ARRAY_TASK_ID}"
readonly rank="${ranks[task_id]}"
readonly result="${campaign_dir}/ladder_r${rank}.json"
readonly attestation="${campaign_dir}/attestation_r${rank}.json"
readonly claim="${campaign_dir}/.claims/ladder_r${rank}.claim"

if [[ -e "${result}" || -e "${attestation}" ]]; then
  echo "refusing to replace an existing rank-${rank} artifact" >&2
  exit 3
fi
mkdir -p "${campaign_dir}/.claims"
if ! mkdir "${claim}"; then
  echo "rank ${rank} is already claimed; inspect before any retry: ${claim}" >&2
  exit 3
fi

/usr/bin/python3 "${helper}" verify-task \
  --control-repo "${control_repo}" \
  --source-repo "${locked_worktree}" \
  --locked-worktree "${locked_worktree}" \
  --campaign-dir "${campaign_dir}" \
  --rank "${rank}"

cd "${locked_worktree}"
export PYTHONPATH="${dep_root}:${locked_worktree}"
/usr/bin/python3 -m rebuttal_experiments.cuda_blackhole_rank_scaling \
  --rank "${rank}" \
  --campaign-tag "${inner_tag}" \
  --manifest "${manifest}" \
  --output-dir "${campaign_dir}"

/usr/bin/python3 "${helper}" attest-task \
  --control-repo "${control_repo}" \
  --source-repo "${locked_worktree}" \
  --locked-worktree "${locked_worktree}" \
  --campaign-dir "${campaign_dir}" \
  --rank "${rank}"

chmod a-w "${result}" "${attestation}"
chmod a-w "${claim}"
