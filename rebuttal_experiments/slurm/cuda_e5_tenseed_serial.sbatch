#!/bin/bash
#SBATCH --job-name=e5-n10-all
#SBATCH --account=marlowe-m000151
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=rebuttal_logs/e5-n10-all-%j.out
#SBATCH --error=rebuttal_logs/e5-n10-all-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# The shared account has a strict submitted-job cap.  This fallback uses one
# allocation while retaining one fresh Python process per seed and benchmark
# row.  The child sbatch files are executed as ordinary shell scripts, so
# their resource directives are ignored and this allocation owns all work.
export SLURM_ARRAY_JOB_ID="${SLURM_JOB_ID}"
for seed in $(seq 1 10); do
  export SLURM_ARRAY_TASK_ID="${seed}"
  bash rebuttal_experiments/slurm/cuda_e5_tenseed_timing.sbatch
  bash rebuttal_experiments/slurm/cuda_e5_tenseed_stability.sbatch
  bash rebuttal_experiments/slurm/cuda_e5_tenseed_large.sbatch
done
