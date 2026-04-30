# SCG runbook: SKA vs PDN end-to-end on Stanford Center for Genomics

This document mirrors `farmshare.md` but for the SCG cluster. It targets
the H100/H200 nodes if you have access, falling back to A100 nodes if
not. The goal is the same single hard-hitting figure: S-NIAH accuracy
vs context length, SKA line vs PDN published baselines.

If you also have a Farmshare run going, both can run in parallel; whichever
finishes first is what you put in the paper.

---

## 0. What you are about to do

Train a 340M-parameter SKA model on FineWeb-Edu for ~15B tokens (PDN's
full compute budget) on SCG's GPU partition, then evaluate on S-NIAH at
five context lengths (2K to 32K). Produce one figure that compares your
line to PDN's published numbers.

Compute envelope assuming 8 x H200 on a single sphinx-class node:

| Phase | Wall | Notes |
|---|---|---|
| Setup (login node) | 5 min | uv venv install, build C++ extension |
| Sanity training (152M, 100M tokens) | ~40 min | one H200 |
| Production training (340M, 15B tokens) | ~16-24 hours | 8 H200, 1 chained job is enough |
| S-NIAH eval | ~1 hour | one H200 (faster than L40S) |
| LM eval | ~1 hour | one H200 |
| **Total** | **~24-30 hours** | |

For comparison: same workload on Farmshare 4x L40S takes ~5 days.

If you only get A100 nodes, throughput is roughly half; figure on ~36-48h
total. Still inside a comfortable window.

---

## 1. Prerequisites

You need:

- An SCG account (https://login.scg.stanford.edu/accounts/).
  Confirms you can `ssh <sunet>@login.scg.stanford.edu`.
- A PI account / SUNetID for billing. Almost all SCG partitions require
  `--account=<PI_SUNetID>`. If you don't know your PI's sunet, ask them.
- The repo cloned somewhere. Either `$HOME` or `/labs/<lab_name>/<your_dir>/`
  if your home quota is tight (~6 GB needed for venv + checkpoints).
- A HuggingFace read-token for FineWeb-Edu (optional but recommended for
  rate limits).

---

## 2. SSH and clone

```bash
ssh <yoursunet>@login.scg.stanford.edu
cd $HOME    # or cd /labs/<lab_name>/<your_dir>/ if home quota is tight
git clone <your-repo-url> CholeskykernelC-
cd CholeskykernelC-
git checkout jack
ls experiments/run_*_scg.slurm   # confirm SCG variants exist
```

---

## 3. Discover your environment (5 min)

These four commands tell you what to put in the SCG_* env vars later.
Run them on the login node:

```bash
# What partitions exist and what GRES they offer:
sinfo -o "%.18P %.10l %.30G %.6D"

# Which partitions / accounts you can submit to:
sacctmgr show user $USER --associations -p

# Which CUDA module name to use:
module avail cuda 2>&1 | head -20

# Storage quotas (in case you need /labs/ for the venv):
df -h $HOME
quota -s 2>/dev/null || true
ls /labs/ 2>/dev/null | head
```

Save the outputs somewhere; you'll reference them when setting env vars.

---

## 4. Run the setup script (5 min, login node)

```bash
# If your home is tight, point the venv at lab storage:
# export SCG_VENV_DIR=/labs/<lab>/<your_dir>/ska_venv

# Pick the CUDA module name from `module avail cuda`:
# export SCG_CUDA_MOD=cuda/12.4    # or whatever yours is named

bash experiments/setup_scg.sh
```

End of output should print the SCG correctness checks:

```
== Manual backward vs autograd-through-reference ==
  grad_L  : rel err < 1e-15
  ...
== gradcheck ==
  gradcheck: True
== Multi-step BPTT ==
  multi-step grad_k rel err: < 1e-15
  ...
340M model: ~341M params
165M model: ~152M params
Setup complete. Wrote .scg_env.
```

The `.scg_env` file persists `SCG_VENV_DIR` and `SCG_CUDA_MOD` so the
slurm scripts find them automatically.

---

## 5. Set the SCG submission env vars (one-time, can go in ~/.bashrc)

```bash
# REQUIRED:
export SCG_PARTITION=<your_partition>     # e.g. sphinx, batch, nih_s10
export SCG_ACCOUNT=<PI_SUNetID>           # e.g. mypi
export SCG_GRES='gpu:h200:8'              # or gpu:a100:8 etc.

# OPTIONAL:
export SCG_TIME=24:00:00                  # wall time per segment
export SCG_QOS=                           # leave empty unless you have a special QoS
export SCG_CPUS=64                        # CPUs (default: 8 per GPU)
export SCG_MEM=400G                       # memory (default: 200G; bump for big nodes)

# For evals (smaller request):
export SCG_EVAL_GRES='gpu:h200:1'         # one GPU is enough
export SCG_EVAL_TIME=06:00:00
```

Add to `~/.bashrc` to persist. Re-source after editing:

```bash
source ~/.bashrc
```

### Filling these in based on what your `sinfo` showed

Look at the GRES column from step 3:

- If you see `gpu:h200:8` listed: set `SCG_GRES='gpu:h200:8'`.
- If you see `gpu:h100:8`: set `SCG_GRES='gpu:h100:8'`.
- If you see only `gpu:a100:N`: set `SCG_GRES='gpu:a100:N'`.
- If you see `gpu:N` with no type: set `SCG_GRES='gpu:N'`.

For partition: pick the one with the GPUs you want. If multiple options
work, prefer one labeled `sphinx` (newest hardware) or whatever the
documentation steers you toward.

---

## 6. Sanity training (~40 min wall, one H200)

```bash
sbatch \
    --partition="$SCG_PARTITION" \
    --account="$SCG_ACCOUNT" \
    --gres="${SCG_EVAL_GRES:-gpu:1}" \
    --time=04:00:00 \
    --cpus-per-task=4 \
    --mem=48G \
    ${SCG_QOS:+--qos=$SCG_QOS} \
    experiments/run_train_sanity_scg.slurm

squeue -u $USER
```

When it starts, tail the log:

```bash
tail -f logs/ska-sanity-$(squeue -u $USER -h -t RUNNING -o "%i" | head -1).out
```

Loss should drop from ~10 to ~6 over 100M tokens. Don't need to wait
for the full 100M to complete; once you see ~10 step lines with
descending loss, sanity has succeeded.

---

## 7. Microbench (optional but recommended, 30 min)

Confirms the Triton kernel actually fires on H200/H100/A100 and gives
you a baseline for tokens/sec.

```bash
sbatch \
    --partition="$SCG_PARTITION" \
    --account="$SCG_ACCOUNT" \
    --gres="${SCG_EVAL_GRES:-gpu:1}" \
    --time=00:30:00 \
    --cpus-per-task=4 \
    --mem=24G \
    ${SCG_QOS:+--qos=$SCG_QOS} \
    experiments/run_microbench_scg.slurm
```

When it finishes, the log prints `Kernel path: triton` and a
throughput table. On H200 expect tokens/sec roughly 4-5x the L40S
numbers.

---

## 8. Production training chain (1-2 days wall)

The whole 15B-token PDN-matched run on 8 H200 should fit in a single
24h job. Submit one or two as insurance:

```bash
RUN_EVAL=1 bash experiments/submit_chain_scg.sh prod_scg 2
```

The script reads SCG_PARTITION / SCG_ACCOUNT / SCG_GRES / SCG_TIME /
SCG_QOS from your env and passes them as `sbatch --partition=...` etc.
This means the slurm files themselves stay cluster-agnostic.

You'll see:

```
Submitting chain on SCG
  partition=sphinx  account=mypi  gres=gpu:h200:8
  time=24:00:00  cpus=64  mem=200G  qos=default
  config=prod_scg  n_jobs=2

  segment 1: job 12345 (depends on none)
  segment 2: job 12346 (depends on 12345)
  eval (S-NIAH): job 12347
  eval (LM):     job 12348
```

Monitor:

```bash
squeue -u $USER
tail -f logs/ska-train-$(squeue -u $USER -h -t RUNNING -o "%i" | head -1).out
```

---

## 9. Evaluation (auto-fires after segment 2 if RUN_EVAL=1 was set)

Outputs:

- `results/sniah_ckpt_final_seed0.csv`
- `results/lm_eval_ckpt_final/results.json`

If you didn't set RUN_EVAL=1, fire eval manually:

```bash
sbatch \
    --partition="$SCG_PARTITION" \
    --account="$SCG_ACCOUNT" \
    --gres="${SCG_EVAL_GRES:-gpu:1}" \
    --time="${SCG_EVAL_TIME:-06:00:00}" \
    --cpus-per-task=4 \
    --mem=48G \
    ${SCG_QOS:+--qos=$SCG_QOS} \
    experiments/run_eval_sniah_scg.slurm
```

---

## 10. Make the figure (login node, 1 minute)

```bash
source "$SCG_VENV_DIR/bin/activate"
python experiments/plot_sniah.py \
    --ours results/sniah_ckpt_final_seed0.csv \
    --out figures/

scp <yoursunet>@login.scg.stanford.edu:~/CholeskykernelC-/figures/*.pdf ./
```

Replace placeholder PDN numbers in `experiments/plot_sniah.py` with the
real ones from PDN Table 5 before locking the figure.

---

## 11. Common SCG-specific issues

### `--account` is required but you didn't set it

Error: `Invalid account or account/partition combination specified`.
Fix: confirm your PI SUNetID via `sacctmgr show user $USER --associations -p`,
then `export SCG_ACCOUNT=<sunet>` and resubmit.

### `gres/gpu count too high`

Some partitions cap GPUs per job. If `--gres=gpu:h200:8` is rejected,
try `--gres=gpu:h200:4` or `--gres=gpu:h200:1` to find what's allowed.

### Preemption (job killed mid-run)

If your QoS is preemptible, segments can be killed before completing.
Auto-resume picks up from the last checkpoint. We checkpoint every
250M tokens in `prod_scg`, so you lose at most ~1h of training.

To recover: just resubmit:

```bash
bash experiments/submit_chain_scg.sh prod_scg 1
```

### `Module load cuda failed`

Run `module avail cuda 2>&1 | head` to see the exact module names. SCG
might use `cuda/12.4` or `nvidia-cuda` or similar. Set:

```bash
export SCG_CUDA_MOD=<exact_name>
# Re-edit .scg_env or re-run setup_scg.sh.
```

### `quota exceeded` during venv install or training

Move to lab storage:

```bash
export SCG_VENV_DIR=/labs/<lab_name>/<your_dir>/ska_venv
bash experiments/setup_scg.sh
# Also point HF cache + checkpoints there:
echo 'export HF_HOME=/labs/<lab>/<your_dir>/hf_cache' >> ~/.bashrc
echo 'export SKA_CKPT_DIR=/labs/<lab>/<your_dir>/checkpoints/prod_scg' >> ~/.bashrc
source ~/.bashrc
```

---

## 12. Cheat sheet

```bash
# Setup (once):
ssh <sunet>@login.scg.stanford.edu
cd CholeskykernelC-
bash experiments/setup_scg.sh

# Set env (once, persist in ~/.bashrc):
export SCG_PARTITION=<partition>
export SCG_ACCOUNT=<PI_SUNetID>
export SCG_GRES='gpu:h200:8'
export SCG_QOS=          # if applicable
export HF_TOKEN=hf_XXX

# Sanity (one GPU, ~40 min):
sbatch --partition=$SCG_PARTITION --account=$SCG_ACCOUNT \
    --gres=gpu:1 --time=04:00:00 --cpus-per-task=4 --mem=48G \
    experiments/run_train_sanity_scg.slurm

# Main run (8 GPUs, ~16-24h, 15B tokens):
RUN_EVAL=1 bash experiments/submit_chain_scg.sh prod_scg 2

# Plot:
python experiments/plot_sniah.py \
    --ours results/sniah_ckpt_final_seed0.csv \
    --out figures/
```
