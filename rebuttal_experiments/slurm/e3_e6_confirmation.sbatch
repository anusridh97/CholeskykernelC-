#!/bin/bash
#SBATCH --job-name=e3-e6-confirm
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=rebuttal_logs/e3-e6-confirm-%j.out
#SBATCH --error=rebuttal_logs/e3-e6-confirm-%j.err

# E3 and E6 are intentionally serialized on one exclusive node.  This avoids
# wasting a second eight-GPU node on CPU-only work and prevents either timing
# campaign from contending for the memory subsystem with the other.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

: "${SOURCE_COMMIT:?submit with --export=ALL,SOURCE_COMMIT=<exact commit>}"
test "$(git rev-parse HEAD)" = "${SOURCE_COMMIT}"
git diff --quiet
git diff --cached --quiet

bash rebuttal_experiments/slurm/e3_metric_only_confirmation.sbatch
bash rebuttal_experiments/slurm/e6_compact_confirmation.sbatch
