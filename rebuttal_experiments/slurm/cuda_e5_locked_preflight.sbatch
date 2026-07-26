#!/bin/bash
#SBATCH --job-name=e5-lock-preflight
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=rebuttal_logs/e5-lock-preflight-%j.out
#SBATCH --error=rebuttal_logs/e5-lock-preflight-%j.err

set -euo pipefail

readonly repository="${E5_LOCKED_REPOSITORY:?set E5_LOCKED_REPOSITORY}"
readonly campaign_dir="${E5_CAMPAIGN_DIR:?set E5_CAMPAIGN_DIR}"
readonly lock="${campaign_dir}/campaign_lock.json"
readonly receipt="${campaign_dir}/preflight.json"
readonly dependency_root="/users/ravid/.local/lib/python3.10/site-packages"

if [[ -e "${receipt}" ]]; then
  echo "refusing to replace preflight receipt ${receipt}" >&2
  exit 3
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${dependency_root}:${repository}"
cd "${repository}"

/usr/bin/python3 -m rebuttal_experiments.e5_locked_confirmation preflight \
  --repository "${repository}" \
  --lock "${lock}" \
  --output "${receipt}"

chmod a-w "${receipt}"
