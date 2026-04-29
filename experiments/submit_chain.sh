#!/usr/bin/env bash
# Submit a chain of dependent training jobs to survive the 24h time cap.
#
# Usage:
#   bash submit_chain.sh prod 2        # 2 chained 24h jobs (~48h total wall)
#   bash submit_chain.sh prod 3        # 3 chained jobs (~72h total wall)
#   bash submit_chain.sh sanity 1      # one sanity run
#
# Each job auto-resumes from the latest checkpoint left by the previous job.
# After the final job, fire eval automatically by setting RUN_EVAL=1:
#   RUN_EVAL=1 bash submit_chain.sh prod 2

set -euo pipefail

CONFIG="${1:-prod}"
N_JOBS="${2:-2}"
RUN_EVAL="${RUN_EVAL:-0}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)"
EXP_DIR="$REPO_ROOT/experiments"
mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/checkpoints/${CONFIG}"

echo "Submitting chain: config=$CONFIG, n_jobs=$N_JOBS"

PREV_JOB=""
TRAIN_SCRIPT="$EXP_DIR/run_train.slurm"
if [ "$CONFIG" = "sanity" ]; then
    TRAIN_SCRIPT="$EXP_DIR/run_train_sanity.slurm"
fi

for ((i=1; i<=N_JOBS; i++)); do
    if [ -z "$PREV_JOB" ]; then
        JOB_ID=$(sbatch --parsable \
            --export=ALL,SKA_CONFIG="$CONFIG" \
            "$TRAIN_SCRIPT")
    else
        JOB_ID=$(sbatch --parsable \
            --dependency=afterok:"$PREV_JOB" \
            --export=ALL,SKA_CONFIG="$CONFIG" \
            "$TRAIN_SCRIPT")
    fi
    echo "  segment $i: job $JOB_ID (depends on ${PREV_JOB:-none})"
    PREV_JOB=$JOB_ID
done

if [ "$RUN_EVAL" = "1" ]; then
    EVAL_JOB=$(sbatch --parsable \
        --dependency=afterok:"$PREV_JOB" \
        --export=ALL,SKA_CONFIG="$CONFIG" \
        "$EXP_DIR/run_eval_sniah.slurm")
    echo "  eval (S-NIAH): job $EVAL_JOB"
    LM_JOB=$(sbatch --parsable \
        --dependency=afterok:"$PREV_JOB" \
        --export=ALL,SKA_CONFIG="$CONFIG" \
        "$EXP_DIR/run_eval_lm.slurm")
    echo "  eval (LM):     job $LM_JOB"
fi

echo
echo "Monitor with:  squeue -u \$USER"
echo "Tail logs:     tail -f logs/ska-train-<job_id>.out"
