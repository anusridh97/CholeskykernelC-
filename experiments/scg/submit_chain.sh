#!/usr/bin/env bash
# SCG-specific chained job submission. Reads cluster constraints from env
# vars and passes them to sbatch on the command line so the slurm scripts
# stay cluster-agnostic.
#
# Required env vars (export or pass on the cmd line):
#   SCG_PARTITION   e.g. batch, sphinx, nih_s10
#   SCG_ACCOUNT     PI SUNetID for accounting (use --account=<SUNetID>)
#   SCG_GRES        GPU GRES, e.g. gpu:h200:8 or gpu:a100:8 or gpu:1
#
# Optional env vars (with defaults shown):
#   SCG_TIME        wall time per segment (default 24:00:00)
#   SCG_QOS         optional QoS (default: empty / cluster default)
#   SCG_CPUS        CPUs per task (default scales with GPU count)
#   SCG_MEM         node memory request (default 200G)
#
# Usage:
#   bash submit_chain_scg.sh prod_scg 2     # 2 chained training segments
#   bash submit_chain_scg.sh sanity 1       # one sanity run
#   RUN_EVAL=1 bash submit_chain_scg.sh prod_scg 2

set -euo pipefail

CONFIG="${1:-prod_scg}"
N_JOBS="${2:-2}"
RUN_EVAL="${RUN_EVAL:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." &> /dev/null && pwd)"
EXP_DIR="$SCRIPT_DIR"
mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/checkpoints/${CONFIG}"

# Validate required env.
: "${SCG_PARTITION?Set SCG_PARTITION (e.g. 'sphinx' or 'batch')}"
: "${SCG_ACCOUNT?Set SCG_ACCOUNT (e.g. 'mysunet'; passed as --account)}"
: "${SCG_GRES?Set SCG_GRES (e.g. 'gpu:h200:8' or 'gpu:a100:8')}"

SCG_TIME="${SCG_TIME:-24:00:00}"
SCG_MEM="${SCG_MEM:-200G}"

# Default CPU count: scale with GPU count derived from SCG_GRES.
GPU_COUNT=$(echo "$SCG_GRES" | awk -F: '{print $NF}')
SCG_CPUS="${SCG_CPUS:-$(( GPU_COUNT * 8 ))}"

# Build the common sbatch options.
SBATCH_ARGS=(
    --partition="$SCG_PARTITION"
    --account="$SCG_ACCOUNT"
    --gres="$SCG_GRES"
    --time="$SCG_TIME"
    --cpus-per-task="$SCG_CPUS"
    --mem="$SCG_MEM"
)
if [ -n "${SCG_QOS:-}" ]; then
    SBATCH_ARGS+=(--qos="$SCG_QOS")
fi

# Pick the right slurm script (now with the cluster suffix dropped).
if [ "$CONFIG" = "sanity" ]; then
    TRAIN_SCRIPT="$EXP_DIR/run_train_sanity.slurm"
else
    TRAIN_SCRIPT="$EXP_DIR/run_train.slurm"
fi

echo "Submitting chain on SCG"
echo "  partition=$SCG_PARTITION  account=$SCG_ACCOUNT  gres=$SCG_GRES"
echo "  time=$SCG_TIME  cpus=$SCG_CPUS  mem=$SCG_MEM  qos=${SCG_QOS:-default}"
echo "  config=$CONFIG  n_jobs=$N_JOBS"
echo

PREV_JOB=""
for ((i=1; i<=N_JOBS; i++)); do
    if [ -z "$PREV_JOB" ]; then
        JOB_ID=$(sbatch --parsable \
            "${SBATCH_ARGS[@]}" \
            --export=ALL,SKA_CONFIG="$CONFIG" \
            "$TRAIN_SCRIPT")
    else
        JOB_ID=$(sbatch --parsable \
            "${SBATCH_ARGS[@]}" \
            --dependency=afterok:"$PREV_JOB" \
            --export=ALL,SKA_CONFIG="$CONFIG" \
            "$TRAIN_SCRIPT")
    fi
    echo "  segment $i: job $JOB_ID (depends on ${PREV_JOB:-none})"
    PREV_JOB=$JOB_ID
done

if [ "$RUN_EVAL" = "1" ]; then
    # Eval needs only one GPU; override SCG_GRES to one GPU of the same type
    # if you can; fall back to whatever the user set.
    EVAL_GRES="${SCG_EVAL_GRES:-${SCG_GRES%:*}:1}"
    EVAL_ARGS=(
        --partition="$SCG_PARTITION"
        --account="$SCG_ACCOUNT"
        --gres="$EVAL_GRES"
        --time="${SCG_EVAL_TIME:-06:00:00}"
        --cpus-per-task=4
        --mem=48G
    )
    if [ -n "${SCG_QOS:-}" ]; then
        EVAL_ARGS+=(--qos="$SCG_QOS")
    fi

    EVAL_JOB=$(sbatch --parsable \
        "${EVAL_ARGS[@]}" \
        --dependency=afterok:"$PREV_JOB" \
        --export=ALL,SKA_CONFIG="$CONFIG" \
        "$EXP_DIR/run_eval_sniah.slurm")
    echo "  eval (S-NIAH): job $EVAL_JOB"

    LM_JOB=$(sbatch --parsable \
        "${EVAL_ARGS[@]}" \
        --dependency=afterok:"$PREV_JOB" \
        --export=ALL,SKA_CONFIG="$CONFIG" \
        "$EXP_DIR/run_eval_lm.slurm")
    echo "  eval (LM):     job $LM_JOB"
fi

echo
echo "Monitor with:  squeue -u \$USER"
