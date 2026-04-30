# SCG runbook: SKA from-scratch pretraining + S-NIAH eval

End-to-end instructions for training a 340M SKA (Structured Kernel Attention)
model on FineWeb-Edu and evaluating on S-NIAH long-context retrieval, on
Stanford's SCG cluster.

The deliverable is one figure: S-NIAH accuracy vs context length, our SKA
line vs PDN published baselines.

All commands assume you are on the SCG login node and inside the repo
root (the directory that contains `student_pkg/` and `experiments/`).

---

## 0. What this run produces

Train 340M SKA on FineWeb-Edu for ~15B tokens (PDN-matched compute) on
the SCG GPU partition, then evaluate on S-NIAH at five context lengths
(2k, 4k, 8k, 16k, 32k) and on a standard zero-shot LM suite.

**Compute envelope on 8xH200 single-node (`prod_scg` config):**

| Phase | Wall | GPUs |
|---|---|---|
| Setup (login node) | ~5 min | 0 |
| Sanity training (152M, 100M tokens) | ~40 min | 1 |
| Microbench (optional) | ~10 min | 1 |
| Production training (340M, 15B tokens) | 16 to 24 h | 8 |
| S-NIAH eval | ~1 h | 1 |
| LM-eval-harness | ~1 h | 1 |
| **Total** | ~24 to 30 h | mixed |

A100 nodes are roughly half throughput, so plan for 36 to 48 h total. The
4xL40S Farmshare equivalent takes ~5 days.

---

## 1. Prerequisites

You need:

- An SCG account (https://login.scg.stanford.edu/accounts/). Verify with `ssh <sunet>@login.scg.stanford.edu`.
- A PI SUNetID for billing. Almost all SCG partitions require `--account=<PI_SUNetID>`. If you don't know your PI's SUNet, ask them or run `sacctmgr show user $USER --associations -p`.
- The repo cloned somewhere (~6 GB free for venv + checkpoints). Use `/labs/<lab>/<your_dir>/` if `$HOME` is tight.
- A HuggingFace read token (optional but recommended for FineWeb-Edu rate limits).

---

## 2. SSH and clone

```bash
ssh <yoursunet>@login.scg.stanford.edu
cd $HOME    # or cd /labs/<lab>/<your_dir>/ if home quota is tight
git clone <your-repo-url> CholeskykernelC-
cd CholeskykernelC-
ls experiments/scg/run_*.slurm   # confirm slurm files are here
```

---

## 3. Discover your environment (one-time, ~2 min)

These commands tell you what to put in the `SCG_*` env vars later:

```bash
# Partitions and GRES they offer:
sinfo -o "%.18P %.10l %.30G %.6D"

# Partitions / accounts you can submit to:
sacctmgr show user $USER --associations -p

# Available CUDA module name:
module avail cuda 2>&1 | head -20

# Available GCC module (need >= 9 for C++17 compilation):
module avail gcc 2>&1 | head -20

# Storage quotas:
df -h $HOME
quota -s 2>/dev/null || true
ls /labs/ 2>/dev/null | head
```

Save the outputs; you reference them when setting env vars.

---

## 4. Run the setup script (~5 min, login node)

```bash
# If $HOME is tight, point the venv at lab storage:
# export SCG_VENV_DIR=/labs/<lab>/<your_dir>/ska_venv

# Pick the CUDA module name from `module avail cuda`:
# export SCG_CUDA_MOD=cuda/12.4    # or whatever yours is

# Optional: pick a specific GCC module (default: auto-detects gcc/14 down to gcc/9):
# export SCG_GCC_MOD=gcc/12

bash experiments/scg/setup.sh
```

End of output should print correctness checks:

```
== Manual backward vs autograd-through-reference ==
  grad_L  : rel err < 1e-15
  ...
== gradcheck ==
  gradcheck: True
== Multi-step BPTT ==
  ...
340M model: ~341M params
165M model: ~152M params
Setup complete. Wrote /path/to/CholeskykernelC-/.scg_env.
```

The `.scg_env` file persists `SCG_VENV_DIR` and `SCG_CUDA_MOD` so all
downstream slurm scripts find them automatically.

### What setup.sh does

- Loads CUDA + a recent GCC module (needs C++17 for `contourpy` etc.; the script tries `gcc/14` down to `gcc/9` automatically).
- Installs `uv` if missing.
- Builds a Python 3.11 venv with pinned wheel-friendly deps (torch 2.4.0, triton, transformers, datasets, numpy<2, pyarrow<17, contourpy<1.3, matplotlib).
- Builds the `student_pkg` C++ extension (CPU correctness path).
- Runs SKA recurrence + manual-backward correctness tests on CPU.
- Writes `.scg_env` at the repo root.

`lm_eval` is intentionally **not** installed here (its build deps fight
with SCG's old toolchain). Install it later in step 9 via a second pass.

---

## 5. Set submission env vars (one-time, persist in `~/.bashrc`)

```bash
# REQUIRED:
export SCG_PARTITION=<your_partition>     # e.g. sphinx, batch, nih_s10
export SCG_ACCOUNT=<PI_SUNetID>           # the SUNet that gets billed
export SCG_GRES='gpu:h200:8'              # or gpu:h100:8, gpu:a100:8, gpu:1, etc.

# OPTIONAL:
export SCG_TIME=24:00:00                  # wall time per training segment
export SCG_QOS=                           # leave empty unless you have a special QoS
export SCG_CPUS=64                        # default: 8 per GPU
export SCG_MEM=400G                       # default: 200G
export HF_TOKEN=hf_XXXXXXXXXXXX           # HuggingFace token (FineWeb-Edu)

# For evals (smaller request):
export SCG_EVAL_GRES='gpu:h200:1'         # one GPU is enough for inference
export SCG_EVAL_TIME=06:00:00
```

Add these to `~/.bashrc` and re-source:

```bash
source ~/.bashrc
```

### Choosing `SCG_GRES` from your `sinfo` output

| You see in GRES column | Set `SCG_GRES` to |
|---|---|
| `gpu:h200:8` | `gpu:h200:8` |
| `gpu:h100:8` | `gpu:h100:8` |
| `gpu:a100:N` (N=1,4,8) | `gpu:a100:N` |
| `gpu:N` with no type | `gpu:N` |

Prefer `sphinx` or whichever partition has the newest hardware available
to your account.

---

## 6. Sanity training (~40 min wall, single GPU)

Submit:

```bash
sbatch \
    --partition="$SCG_PARTITION" \
    --account="$SCG_ACCOUNT" \
    --gres="${SCG_EVAL_GRES:-gpu:1}" \
    --time=04:00:00 \
    --cpus-per-task=4 \
    --mem=48G \
    ${SCG_QOS:+--qos=$SCG_QOS} \
    experiments/scg/run_train_sanity.slurm

squeue -u $USER
```

Tail the log once it starts:

```bash
tail -f logs/ska-sanity-$(squeue -u $USER -h -t RUNNING -o "%i" | head -1).out
```

Loss should drop from ~10 to ~6 over 100M tokens. You don't need to wait
for completion. Once you see ~10 step lines with descending loss, sanity
has succeeded.

The sanity config trains the 152M model on 100M tokens of FineWeb-Edu at
context length 2048 (matches PDN's sanity setup).

---

## 7. Microbench (optional, ~10 min)

Confirms the Triton kernel actually fires on your allocated GPU and
gives you a baseline tokens/sec table:

```bash
sbatch \
    --partition="$SCG_PARTITION" \
    --account="$SCG_ACCOUNT" \
    --gres="${SCG_EVAL_GRES:-gpu:1}" \
    --time=00:30:00 \
    --cpus-per-task=4 \
    --mem=24G \
    ${SCG_QOS:+--qos=$SCG_QOS} \
    experiments/scg/run_microbench.slurm
```

When it finishes you get:

- `results/microbench_scg.csv` (table of us/token across r, BH, T sweeps)
- `figures/microbench_scg.pdf`

Look for `Kernel path: triton` in the log. On H200 expect roughly
4-5x the L40S throughput.

---

## 8. Production training chain (1-2 days wall)

The full 15B-token run on 8 H200 fits one 24h job comfortably. Submit
two as preemption insurance:

```bash
RUN_EVAL=1 bash experiments/scg/submit_chain.sh prod_scg 2
```

This wrapper reads your `SCG_*` env vars and submits two chained
training segments. The slurm scripts themselves are cluster-agnostic;
all cluster config is passed via `sbatch --partition=...` etc.

You should see:

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

`RUN_EVAL=1` chains both eval jobs after the last training segment so
the figure pipeline runs unattended.

Monitor:

```bash
squeue -u $USER
tail -f logs/ska-train-$(squeue -u $USER -h -t RUNNING -o "%i" | head -1).out
```

### Auto-resume

The slurm script auto-resumes from the most recent `ckpt_*.pt` in
`checkpoints/prod_scg/` if one exists. On a preempted job, just resubmit:

```bash
bash experiments/scg/submit_chain.sh prod_scg 1
```

It picks up from the last 250M-token checkpoint. Worst-case loss is
~1 hour of training.

---

## 9. Evaluation

If `RUN_EVAL=1` was set in step 8, both evals fire automatically after
training. Outputs:

- `results/sniah_ckpt_final_seed0.csv`
- `results/lm_eval_ckpt_final/results.json`

### Manual S-NIAH eval

```bash
sbatch \
    --partition="$SCG_PARTITION" \
    --account="$SCG_ACCOUNT" \
    --gres="${SCG_EVAL_GRES:-gpu:1}" \
    --time="${SCG_EVAL_TIME:-06:00:00}" \
    --cpus-per-task=4 \
    --mem=48G \
    ${SCG_QOS:+--qos=$SCG_QOS} \
    experiments/scg/run_eval_sniah.slurm
```

Override defaults via env vars:

```bash
SKA_CKPT=/path/to/ckpt.pt \
SKA_LENGTHS="2048 4096 8192 16384 32768" \
SKA_VARIANTS="1 2 3" \
SKA_N_EX=100 \
sbatch experiments/scg/run_eval_sniah.slurm
```

### Manual lm-eval-harness eval

`lm_eval` was intentionally skipped during initial setup (its build deps
conflict with SCG's old toolchain). Install it now in a second pass:

```bash
bash experiments/scg/setup_lm_eval.sh
python -c 'import lm_eval; print(lm_eval.__version__)'   # confirms install
```

Then submit:

```bash
sbatch \
    --partition="$SCG_PARTITION" \
    --account="$SCG_ACCOUNT" \
    --gres="${SCG_EVAL_GRES:-gpu:1}" \
    --time="${SCG_EVAL_TIME:-06:00:00}" \
    --cpus-per-task=4 \
    --mem=48G \
    ${SCG_QOS:+--qos=$SCG_QOS} \
    experiments/scg/run_eval_lm.slurm
```

Default tasks: `arc_easy, arc_challenge, hellaswag, piqa, winogrande, boolq, sciq, lambada_openai, wikitext` plus an in-context-recall set (`fda, swde, squad_completion, triviaqa, drop`).

---

## 10. Make the figure (login node, ~1 min)

```bash
source "$SCG_VENV_DIR/bin/activate"
python experiments/backend/plot_sniah.py \
    --ours results/sniah_ckpt_final_seed0.csv \
    --out figures/

# Pull to your laptop:
scp <yoursunet>@login.scg.stanford.edu:CholeskykernelC-/figures/*.pdf ./
```

Note: `experiments/backend/plot_sniah.py` ships placeholder PDN baseline
numbers. Replace them with PDN Table 5's real values before locking the
final figure.

---

## 11. Optional: enable retrieval data and WandB

`train.py` accepts three opt-in flags useful for the from-scratch run:

| Flag | Default | What it does |
|---|---|---|
| `--retrieval_ratio` | 0.0 (off) | Mix synthetic S-NIAH-style fact-distractor-question examples into training at this fraction. 0.10-0.20 is a reasonable starting point for boosting long-context recall. |
| `--wandb_project` | "" (off) | WandB project name for live logging. |
| `--wandb_run_name` | `{config}-seed{seed}` | WandB run name. |
| `--health_every_steps` | 200 | Sample SKA operator health metrics (max abs A_w, min/max diag L, Frobenius norms) every N grad steps. WandB only. Set 0 to disable. |

Pass them through the slurm scripts via the `SKA_EXTRA_ARGS` env var:

```bash
# WandB needs a one-time login. Stores token in ~/.netrc.
source "$SCG_VENV_DIR/bin/activate"
pip install wandb
wandb login

# Submit the prod chain with retrieval mixed in and WandB on:
export SKA_EXTRA_ARGS="--retrieval_ratio 0.15 --wandb_project ska-pretrain"
RUN_EVAL=1 bash experiments/scg/submit_chain.sh prod_scg 2
```

This passthrough is wired through `--export=ALL` in `submit_chain.sh`, so
`SKA_EXTRA_ARGS` reaches the compute node automatically.

For a sanity test of just retrieval mixing:

```bash
SKA_EXTRA_ARGS="--retrieval_ratio 0.15" \
sbatch \
    --partition="$SCG_PARTITION" --account="$SCG_ACCOUNT" \
    --gres=gpu:1 --time=04:00:00 --cpus-per-task=4 --mem=48G \
    --export=ALL \
    experiments/scg/run_train_sanity.slurm
```

`retrieval_ratio` is also persisted in each checkpoint, so eval scripts
can read it back from `ckpt["retrieval_ratio"]`.

---

## 12. Common SCG-specific issues

### `Invalid account or account/partition combination`

You forgot `--account` or `SCG_ACCOUNT`. Confirm your PI's SUNet via:

```bash
sacctmgr show user $USER --associations -p
```

Then `export SCG_ACCOUNT=<PI_SUNet>` and resubmit.

### `gres/gpu count too high`

Some partitions cap GPUs per job. If `gpu:h200:8` is rejected, try
`gpu:h200:4` or `gpu:h200:1`. The training scripts auto-detect GPU
count from `nvidia-smi` and adjust DDP accordingly.

### Job preempted mid-run

If your QoS is preemptible, segments can be killed before completing.
Auto-resume picks up from the last checkpoint. `prod_scg` checkpoints
every 250M tokens, so you lose at most ~1h. Just resubmit:

```bash
bash experiments/scg/submit_chain.sh prod_scg 1
```

### `Module load cuda failed` or `Module load gcc failed`

Run `module avail cuda` and `module avail gcc` to find exact module
names on your cluster. Then:

```bash
export SCG_CUDA_MOD=cuda/12.4
export SCG_GCC_MOD=gcc/12
bash experiments/scg/setup.sh   # re-run with corrected names
```

### `quota exceeded` during venv install or training

Move to lab storage:

```bash
export SCG_VENV_DIR=/labs/<lab>/<your_dir>/ska_venv
bash experiments/scg/setup.sh

# Also redirect HF cache and checkpoints:
echo 'export HF_HOME=/labs/<lab>/<your_dir>/hf_cache' >> ~/.bashrc
echo 'export SKA_CKPT_DIR=/labs/<lab>/<your_dir>/checkpoints/prod_scg' >> ~/.bashrc
source ~/.bashrc
```

### `No checkpoint found` from `run_eval_sniah.slurm`

Defaults to `checkpoints/$SKA_CONFIG/ckpt_final.pt` (config defaults to
`prod_scg`). Override:

```bash
SKA_CKPT=/path/to/ckpt.pt sbatch experiments/scg/run_eval_sniah.slurm
```

### `lm_eval` install fails with `libcst` or `pyarrow` build errors

Re-run with a newer GCC loaded:

```bash
module avail gcc
export SCG_GCC_MOD=gcc/12   # or 11, 13, 14
bash experiments/scg/setup_lm_eval.sh
```

If still failing, the simplest workaround is to skip lm-eval-harness on
SCG and run it on a different machine after copying the checkpoint:

```bash
scp <sunet>@login.scg.stanford.edu:CholeskykernelC-/checkpoints/prod_scg/ckpt_final.pt ./
# then run eval_lm_harness.py locally
```

---

## 13. Cheat sheet

```bash
# --- Setup (once) ---
ssh <sunet>@login.scg.stanford.edu
cd CholeskykernelC-
bash experiments/scg/setup.sh

# --- Env (once, in ~/.bashrc) ---
export SCG_PARTITION=<partition>
export SCG_ACCOUNT=<PI_SUNetID>
export SCG_GRES='gpu:h200:8'
export SCG_QOS=                                    # if applicable
export HF_TOKEN=hf_XXX

# --- Sanity (one GPU, ~40 min) ---
sbatch \
    --partition=$SCG_PARTITION --account=$SCG_ACCOUNT \
    --gres=gpu:1 --time=04:00:00 --cpus-per-task=4 --mem=48G \
    experiments/scg/run_train_sanity.slurm

# --- Optional: microbench (~10 min) ---
sbatch \
    --partition=$SCG_PARTITION --account=$SCG_ACCOUNT \
    --gres=gpu:1 --time=00:30:00 --cpus-per-task=4 --mem=24G \
    experiments/scg/run_microbench.slurm

# --- Main run (8 GPUs, ~16-24h, 15B tokens) ---
RUN_EVAL=1 bash experiments/scg/submit_chain.sh prod_scg 2

# --- With retrieval data + WandB ---
wandb login   # one-time
export SKA_EXTRA_ARGS="--retrieval_ratio 0.15 --wandb_project ska-pretrain"
RUN_EVAL=1 bash experiments/scg/submit_chain.sh prod_scg 2

# --- lm-eval (after training, two-step) ---
bash experiments/scg/setup_lm_eval.sh
sbatch \
    --partition=$SCG_PARTITION --account=$SCG_ACCOUNT \
    --gres=gpu:1 --time=06:00:00 --cpus-per-task=4 --mem=48G \
    experiments/scg/run_eval_lm.slurm

# --- Plot ---
source $SCG_VENV_DIR/bin/activate
python experiments/backend/plot_sniah.py \
    --ours results/sniah_ckpt_final_seed0.csv \
    --out figures/
```
