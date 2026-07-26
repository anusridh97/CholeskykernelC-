#!/bin/bash
#SBATCH --job-name=blackhole-ladder
#SBATCH --account=marlowe-m000151
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=rebuttal_logs/blackhole-ladder-%j.out
#SBATCH --error=rebuttal_logs/blackhole-ladder-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PWD}/rebuttal_deps:${PWD}"

readonly campaign_tag="blackhole-s4-v1"
readonly default_manifest="rebuttal_results/blackhole/${campaign_tag}/feature_manifest.json"
readonly manifest="${BLACKHOLE_FEATURE_MANIFEST:-${default_manifest}}"
readonly default_output_dir="rebuttal_results/blackhole/${campaign_tag}"
readonly output_dir="${BLACKHOLE_OUTPUT_DIR:-${default_output_dir}}"
ranks=(2048 4096 8192 16384 32768 49152 65536)

if [[ "${BLACKHOLE_CAMPAIGN_TAG:-${campaign_tag}}" != "${campaign_tag}" ]]; then
  echo "campaign tag is frozen to ${campaign_tag}" >&2
  exit 2
fi
if [[ ! -f "${manifest}" ]]; then
  echo "feature manifest is missing: ${manifest}" >&2
  exit 2
fi

for rank in "${ranks[@]}"; do
  result_path="${output_dir}/ladder_r${rank}.json"
  if [[ -f "${result_path}" ]]; then
    echo "resume: preserving completed rank ${rank}: ${result_path}"
    continue
  fi
  args=(
    --rank "${rank}"
    --campaign-tag "${campaign_tag}"
    --manifest "${manifest}"
  )
  if [[ -n "${BLACKHOLE_OUTPUT_DIR:-}" ]]; then
    args+=(--output-dir "${output_dir}")
  fi

  /usr/bin/python3 -m rebuttal_experiments.cuda_blackhole_rank_scaling \
    "${args[@]}"
done
