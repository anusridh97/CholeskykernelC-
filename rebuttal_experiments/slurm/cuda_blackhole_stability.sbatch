#!/bin/bash
#SBATCH --job-name=bh-s4-stability
#SBATCH --account=marlowe-m000151
#SBATCH --partition=preempt
#SBATCH --array=0-1%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=rebuttal_logs/blackhole-stability-%A_%a.out
#SBATCH --error=rebuttal_logs/blackhole-stability-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PWD}/rebuttal_deps:${PWD}"

campaign_tag="blackhole-s4-v1"
feature_manifest="${BLACKHOLE_FEATURE_MANIFEST:-${PWD}/rebuttal_results/blackhole/${campaign_tag}/feature_manifest.json}"
ranks=(8192 32768)

task_id="${SLURM_ARRAY_TASK_ID}"
if (( task_id < 0 || task_id >= ${#ranks[@]} )); then
  echo "expected an array task in 0..$((${#ranks[@]} - 1)), got ${task_id}" >&2
  exit 2
fi
if [[ ! -f "${feature_manifest}" ]]; then
  echo "missing frozen feature manifest: ${feature_manifest}" >&2
  exit 2
fi

/usr/bin/python3 -m rebuttal_experiments.cuda_blackhole_stability \
  --rank "${ranks[task_id]}" \
  --campaign-tag "${campaign_tag}" \
  --feature-manifest "${feature_manifest}"
