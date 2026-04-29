# Farmshare runbook: SKA vs PDN end-to-end

This document is the single source of truth for running the SKA-vs-PDN
S-NIAH experiment on Stanford Farmshare. It walks you from a fresh shell
to the final figure, with diagnostics for every common failure mode.

If anything in this document conflicts with `experiments/PLAN.md` or the
README, this document wins.

---

## 0. What you are about to do

Train a 340M-parameter language model with the SKA (exact double-sided
Cholesky preconditioning) recurrence in place of standard attention,
on SlimPajama, for ~8B tokens (about half PDN's 15B-token budget).
Then evaluate on S-NIAH at five context lengths (2K to 32K). Produce
one figure that compares your line to PDN's published numbers from
arXiv:2604.21100v1, Table 5.

Total compute: ~4-5 days on 4x L40S (one Farmshare oat-XX node), free.

Total monetary cost: $0 (Farmshare is free).

Farmshare's gpu partition has 6 nodes (oat-01 through oat-06), each
with 4 L40S, 64 CPUs, 256 GB memory. Time cap is 48h per job, so
the main run is split into 3 chained 48h jobs.

---

## 1. Prerequisites

You need:

- A Stanford SUNet ID with Farmshare access. New users:
  request via https://srcc.stanford.edu/farmshare-onboarding.
- The repository cloned somewhere accessible from Farmshare
  (your home directory is fine; ~25MB of source plus checkpoints
  and logs that grow to ~5-10 GB).
- A HuggingFace account with a read token, for the gated SlimPajama
  dataset. Generate one at https://huggingface.co/settings/tokens.

You do not need:

- A GPU on your local machine. Everything heavy runs on Farmshare.
- Sherlock or any other Stanford cluster. Farmshare is sufficient.

---

## 2. SSH and clone the repo (5 minutes)

```bash
ssh <yoursunet>@rice.stanford.edu
```

If this is your first time, accept the host key. If you get prompted
for a password, you may want to set up an SSH key
(https://srcc.stanford.edu/farmshare-storage-and-quotas#ssh-keys).

Once on rice:

```bash
cd $HOME
git clone <your-repo-url> CholeskykernelC-
cd CholeskykernelC-

# Make sure you're on the branch with the experiments/ folder.
git checkout jack
ls experiments/   # should list ska_recurrence.py, run_train.slurm, etc.
```

If `experiments/` is empty or missing, you cloned the wrong branch.

---

## 3. One-time environment setup on a login node (5 minutes)

The setup script installs `uv`, creates a Python venv at
`student_pkg/.venv`, installs PyTorch, Triton, transformers, datasets,
lm-evaluation-harness, builds the C++ kernel extension, and runs the
CPU correctness checks. None of this needs a GPU.

```bash
bash experiments/setup_farmshare.sh
```

Expected end-of-output:

```
== Manual backward vs autograd-through-reference ==
  grad_L  : rel err < 1e-15
  grad_Aw : rel err < 1e-15
  grad_z  : rel err < 1e-15
  grad_w  : rel err < 1e-15

== gradcheck ==
  gradcheck: True

== Multi-step BPTT ==
  multi-step grad_k rel err: < 1e-15
  multi-step grad_v rel err: < 1e-15

OK

340M model: ~341M params
165M model: ~152M params
```

If any rel err exceeds 1e-8, or gradcheck returns False, stop here.
The math is broken and no amount of compute will save you.

---

## 4. HuggingFace authentication (1 minute)

SlimPajama is a gated dataset. The most reliable approach is the
`HF_TOKEN` environment variable, which works regardless of which CLI
version is installed (`huggingface-cli` was renamed to `hf` in newer
`huggingface_hub` releases; both currently exist as aliases but only
the env var is guaranteed to be picked up by every code path).

Get a read-token from https://huggingface.co/settings/tokens. Then:

```bash
echo 'export HF_TOKEN=hf_XXXXXXXXXXXXXX' >> ~/.bashrc
source ~/.bashrc
```

Verify auth works programmatically (this is the path the training
job uses, so this is the canonical test):

```bash
source ~/git/CholeskykernelC-/student_pkg/.venv/bin/activate
python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"
# Should print {'name': '<your_username>', ...}
```

If you prefer the CLI:

```bash
hf auth login            # newer huggingface_hub (>= 0.32)
huggingface-cli login    # older versions, still works as alias
```

Both write to `~/.cache/huggingface/token`. If the env var is set,
SLURM jobs pick it up via `--export=ALL` (the default).

---

## 5. Interactive GPU sanity check (10 minutes)

Get an interactive GPU shell. This bypasses the SLURM batch queue
and gives instant feedback for tiny tests.

```bash
srun -p gpu --gres=gpu:1 --pty bash
```

If your `~/.bashrc` already loads CUDA per the SRCC docs:

```bash
echo 'module load cuda' >> ~/.bashrc
echo 'export CUDA_HOME=$(dirname $(dirname $(which nvcc)))' >> ~/.bashrc
source ~/.bashrc
```

then once you're on the GPU node:

```bash
source student_pkg/.venv/bin/activate

# Confirm CUDA + GPU.
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected:  True NVIDIA L40S

# Confirm the Triton path is firing.
python experiments/ska_recurrence.py
# Expected new output (compared to login-node):
#   == Forward consistency (Triton vs reference) ==
#     L       : rel err < 1e-6
#     Aw      : rel err < 1e-6
#     vz      : rel err < 1e-6
#     vw      : rel err < 1e-6
#     k_tilde : rel err < 1e-6

# 30-second microbench, single config.
python experiments/microbench.py --r 128 --bh 32 --T 2048
# Expected:  Kernel path: triton
#            tok/s in the 8-15K range
```

If all three pass, exit the interactive shell:

```bash
exit
```

You now know the kernel works on Farmshare's L40S.

If `python -c "import torch; ..."` returns False, CUDA is not loaded
on the GPU node. Re-source `~/.bashrc`.

If `microbench.py` reports `Kernel path: reference (CPU)`, Triton
failed to import (usually a CUDA mismatch between torch and the node).
See section 12 (Common issues).

---

## 6. SLURM-based microbench (optional, 30 min wall, 5 min run)

Same thing as section 5 but as a batch job that produces a CSV and
a PDF for the supplementary figure.

```bash
sbatch experiments/run_microbench.slurm
squeue -u $USER     # watch for the job to start and finish
```

When the job finishes:

```bash
cat logs/ska-microbench-*.out | tail -40
```

Outputs land in `results/microbench.csv` and `figures/microbench.pdf`.

---

## 7. Sanity training (4 hour wall, ~1.5h run)

Confirms the full training pipeline works end-to-end on a tiny model
before you commit to the multi-day production run.

```bash
sbatch experiments/run_train_sanity.slurm
```

You'll get a job ID. Watch the log:

```bash
tail -f logs/ska-sanity-<jobid>.out
```

What to expect in the first 5 minutes:

```

By 100M tokens (end of run) loss should be in the 6.0-6.5 range. If
it goes NaN or stays above 9, kill with `scancel <jobid>` and check
section 12.

---

## 8. Production training chain (4-5 days wall, FREE)

This is the main run. Three chained 48h jobs on a 4x L40S node,
each auto-resuming from the previous one's checkpoint, totaling
~8B tokens of 340M training on SlimPajama. About half of PDN's
15B-token compute, on free hardware.

```bash
RUN_EVAL=1 bash experiments/submit_chain.sh prod 3
```

You'll see:

```
Submitting chain: config=prod, n_jobs=3
  segment 1: job 12346 (depends on none)
  segment 2: job 12347 (depends on 12346)
  segment 3: job 12348 (depends on 12347)
  eval (S-NIAH): job 12349
  eval (LM):     job 12350
```

Each training segment fires only if the previous one exited zero.
The two eval jobs fire only after segment 3 succeeds.

Monitor every few hours:

```bash
squeue -u $USER

# Tail whichever segment is currently running:
tail -n 50 $(ls -t logs/ska-train-*.out | head -1)
```

Loss should drop from ~6 (where sanity ended) to ~3.0-3.5 by 8B
tokens. Checkpoints land in `checkpoints/prod/ckpt_<tokens>.pt`
every 1B tokens.

If you have buffer days, push further by submitting more segments:

```bash
bash experiments/submit_chain.sh prod 1     # +~3B tokens, +48h wall
```

If the gpu partition is too crowded for 4-GPU jobs, fall back to
single-GPU (slower):

```bash
RUN_EVAL=1 bash experiments/submit_chain.sh prod_1gpu 6
```

### If a segment fails

The downstream segments won't fire (`afterok` requires success). Find
the cause in `logs/ska-train-<failed_jobid>.err`. Fix it, then
re-submit any remaining segments:

```bash
bash experiments/submit_chain.sh prod_1gpu 1   # one fresh segment
```

Auto-resume picks up from the latest valid checkpoint.

### If the loss diverges or NaNs mid-segment

Kill it:

```bash
scancel <jobid>   # the active segment
scancel <jobid_dep1> <jobid_dep2> ...   # any pending ones
```

Lower the peak LR in `experiments/train.py` (`prod_1gpu` config:
`peak_lr` 2.5e-4 → 1.5e-4), delete the most recent (corrupted)
checkpoint, and resubmit:

```bash
rm checkpoints/prod_1gpu/ckpt_<latest_tokens>.pt
bash experiments/submit_chain.sh prod_1gpu 3
```

---

## 9. Evaluation (5 hours wall total)

If you launched with `RUN_EVAL=1` in section 8, evals fire automatically
after the last training segment. Otherwise:

```bash
sbatch --export=ALL,SKA_CONFIG=prod_1gpu experiments/run_eval_sniah.slurm
sbatch --export=ALL,SKA_CONFIG=prod_1gpu experiments/run_eval_lm.slurm
```

Outputs:

```
results/sniah_ckpt_final_seed0.csv
results/lm_eval_ckpt_final/results.json
```

The S-NIAH CSV has rows like:

```
variant,context_length,accuracy,n_examples
1,2048,0.78,100
1,4096,0.71,100
1,8192,0.62,100
1,16384,0.50,100
1,32768,0.32,100
2,2048,0.74,100
...
```

Solid (>0.5) at 2K-4K and degrading gracefully past 8K is the
expected shape. Near-zero everywhere means the model didn't train
enough; submit one more training segment.

---

## 10. Make the figure (1 minute, login node)

```bash
exit                       # leave any GPU shell
ssh <yoursunet>@rice.stanford.edu
cd CholeskykernelC-
source student_pkg/.venv/bin/activate

python experiments/plot_sniah.py \
    --ours results/sniah_ckpt_final_seed0.csv \
    --out figures/
```

Two files:

```
figures/sniah_extrapolation.pdf
figures/sniah_extrapolation.png
```

scp them to your local machine:

```bash
# From your laptop:
scp <yoursunet>@rice.stanford.edu:~/CholeskykernelC-/figures/*.pdf ./
```

### Before locking the figure

The `PDN_NUMBERS` dict at the top of `experiments/plot_sniah.py`
contains placeholder PDN baselines. Replace with the exact numbers
from the PDF (Table 5 of arXiv:2604.21100v1) before you put the
figure in the paper. Re-run `plot_sniah.py` after editing.

---

## 11. Optional polish (a few extra hours each, all free)

### Multiple-seed S-NIAH error bars

```bash
for seed in 1 2 3 4; do
    sbatch --export=ALL,SKA_SEED=$seed,SKA_CONFIG=prod_1gpu \
        experiments/run_eval_sniah.slurm
done
```

Then aggregate the CSVs in a small Python script.

### Commonsense + ICR table

Already produced if you used `RUN_EVAL=1`. Inspect:

```bash
python -c "import json; print(json.dumps(json.load(open('results/lm_eval_ckpt_final/results.json')), indent=2))" | head -60
```

### Throughput supplementary figure

Already produced by `run_microbench.slurm` in section 6.
Pulls into the appendix.

---

## 12. Common issues and fixes

### "Unable to contact slurm controller (connect failure)"

The Farmshare controller is down. Check
https://srcc.stanford.edu/farmshare-overview or email
research-computing-support@stanford.edu for an ETA. There is no
secondary controller within Farmshare to fail over to.

While waiting, you can run the CPU-side sanity checks (section 3)
or use your interactive GPU shell (section 5) if it's still alive.

### "Requested node configuration is not available"

The GRES specifier doesn't match Farmshare's actual config. The
SLURM scripts in this repo use `--gres=gpu:1` (no type), matching the
existing kernel scripts in `student_pkg/run_*.slurm`. If you see this
error, check that you pulled the latest commit; older versions
specified `--gres=gpu:l40s:1`.

If you also see "Requested partition does not exist," try `sinfo`
to list available partitions and replace `--partition=gpu` accordingly.

### Job hangs in PD (pending) state for hours

```bash
squeue -u $USER -o "%.10i %.9P %.8j %.8u %.2t %.10M %.6D %R"
```

The right column shows the reason. Common ones:

- `Resources` -> the cluster is full, wait it out.
- `Priority` -> other jobs are ahead; fine, just wait.
- `QOSGrpCpuLimit` -> you've hit a per-account quota; wait or
  cancel some jobs.
- `(launch failed requeued held)` -> the job hit some startup
  problem and SLURM held it. Check `scontrol show job <id>` for
  the reason field.

### `nvcc` not found inside the slurm job

The script does `module load cuda` which should fix this. If not,
verify on a GPU node:

```bash
srun -p gpu --gres=gpu:1 --pty bash
which nvcc
module avail cuda
```

If `module avail cuda` shows a different module name than `cuda`
(e.g., `cuda/12.4`), edit the slurm scripts to `module load cuda/12.4`.

### `import torch` says CUDA is not available on the GPU node

torch's CUDA build does not match the node's CUDA. Reinstall torch
matching the node's CUDA version:

```bash
srun -p gpu --gres=gpu:1 --pty bash
nvidia-smi | head -3   # note "CUDA Version: 12.X"
exit
source student_pkg/.venv/bin/activate
uv pip install "torch==2.4.0" --index-url https://download.pytorch.org/whl/cu121
# or cu118 for older CUDA
```

### Triton fails to import / "Kernel path: reference (CPU)"

Triton needs CUDA. On the GPU node:

```bash
python -c "import triton; print(triton.__version__)"
```

If this errors, install/upgrade Triton:

```bash
uv pip install --upgrade triton
```

If `triton` imports but `import triton.language as tl` fails inside
the kernel module, your torch and triton are mismatched. Pin both:

```bash
uv pip install "torch==2.4.0" "triton==3.0.0"
```

### OOM during training

L40S has 48GB. A 340M model in FP32 with ctx 2048 fits, but tight.

If you OOM:

1. Drop micro batch from 4 to 2 in `run_train.slurm` `MICRO_BATCH`.
2. Or switch to BF16 model parameters: in `train.py`, after
   `model = cfg["model_fn"](...)`, add `.to(torch.bfloat16)`.
   The SKA kernel still runs FP32 internally, so accuracy is fine.
3. Or drop ctx_len from 2048 to 1024 in the config.

### OOM during S-NIAH eval at 32K

Expected for FP32. Either:

1. Add `.bfloat16()` to the model load in `eval_sniah.py`.
2. Drop the 32K context length from `--lengths` and document
   the limitation.

### Job hangs after the GPU header line, no `step 1` output

Most likely the first SlimPajama stream pull. Wait 2-5 minutes; if
the first log line appears, you were just being impatient.

If 10 minutes pass with no progress, the issue is usually HF auth:

```bash
# On a login node:
source ~/git/CholeskykernelC-/student_pkg/.venv/bin/activate
python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"
```

If this errors with 401, your token didn't get picked up. Set
HF_TOKEN explicitly (section 4) and resubmit:

```bash
scancel <jobid>
echo 'export HF_TOKEN=hf_XXXX' >> ~/.bashrc
source ~/.bashrc
sbatch experiments/run_train_sanity.slurm
```

Also helpful for diagnosis while a job is running:

```bash
# Is the GPU actually doing anything?
ssh oat-01 "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 2"
# - util > 0 + memory > 0: training is initializing, just hasn't hit log_every yet.
# - util = 0 + memory > 0: stuck in Triton JIT compile or similar. Wait longer.
# - util = 0 + memory = 0: not started yet, probably HF download.

# Stderr often has more clues than stdout:
tail -n 50 logs/ska-sanity-<jobid>.err
```

### SlimPajama streaming hangs or times out

Pre-download a subset to a fixed dir on a login node:

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('cerebras/SlimPajama-627B', split='train', streaming=True).take(5_000_000)
out = '~/CholeskykernelC-/data/slimpajama_5M'
ds.save_to_disk(out)
print('saved to', out)
"
```

Then patch `train.py`'s `SlimPajamaStream.__init__` to load from disk
instead of streaming.

### Disk quota exceeded

Farmshare home is typically 100GB. Checkpoints add up:

```bash
du -sh ~/CholeskykernelC-/checkpoints
du -sh ~/.cache/huggingface
```

Move the HF cache to scratch:

```bash
export HF_HOME=/scratch/users/$USER/hf_cache
```

Add this to `~/.bashrc` and re-source. Re-running the slurm scripts
will pick it up.

Or thin out old checkpoints:

```bash
# Keep only the final and the most recent two:
ls -t checkpoints/prod_1gpu/ckpt_*.pt | tail -n +4 | xargs rm -f
```

### Loss is reasonable but S-NIAH accuracy is zero everywhere

Two common causes:

1. The model didn't train enough at the longer contexts. Submit
   one more training segment (`bash experiments/submit_chain.sh prod_1gpu 1`).
2. The tokenizer used at eval differs from training. Confirm
   both `train.py` and `eval_sniah.py` use the same `--tokenizer`.

### Slurm time cap is shorter than 24h

Some Farmshare partitions cap at 12h or even less. Check:

```bash
sinfo -p gpu -o "%P %l"
```

If the limit is 12h, edit the slurm scripts:

```
#SBATCH --time=12:00:00
```

and double the chain length:

```bash
bash experiments/submit_chain.sh prod_1gpu 6
```

---

## 13. Cheat sheet

```bash
# Setup (once):
ssh <yoursunet>@rice.stanford.edu
cd CholeskykernelC-
bash experiments/setup_farmshare.sh
huggingface-cli login

# Sanity:
sbatch experiments/run_microbench.slurm
sbatch experiments/run_train_sanity.slurm

# Main run (chained, ~8B tokens, 4-5 days, 4 L40S):
RUN_EVAL=1 bash experiments/submit_chain.sh prod 3
# Fallback if no 4-GPU slots available:
# RUN_EVAL=1 bash experiments/submit_chain.sh prod_1gpu 6

# Monitor:
squeue -u $USER
tail -f logs/ska-train-$(ls -t logs/ska-train-*.out | head -1 | grep -o '[0-9]\+').out

# Manual eval (if not auto-launched):
sbatch --export=ALL,SKA_CONFIG=prod_1gpu experiments/run_eval_sniah.slurm
sbatch --export=ALL,SKA_CONFIG=prod_1gpu experiments/run_eval_lm.slurm

# Plot:
source student_pkg/.venv/bin/activate
python experiments/plot_sniah.py \
    --ours results/sniah_ckpt_final_seed0.csv \
    --out figures/

# Pull figure to laptop:
scp <yoursunet>@rice.stanford.edu:~/CholeskykernelC-/figures/*.pdf ./
```

---

## 14. Estimated wall clock vs cumulative cost

| Phase | Wall | Cumulative cost |
|---|---|---|
| Sections 1-4 (setup) | ~15 min | $0 |
| Section 5 (interactive sanity) | ~10 min | $0 |
| Section 6 (microbench batch) | ~30 min | $0 |
| Section 7 (sanity training) | ~4 hours | $0 |
| Section 8 (3 chained 48h jobs, 4 L40S, 8B tokens) | ~4-5 days | $0 |
| Sections 9-10 (eval + figure) | ~6 hours | $0 |
| **Total** | **~5-6 days** | **$0** |

---

## 15. What "done" looks like

You have:

1. `figures/sniah_extrapolation.pdf` -- main paper figure.
2. `figures/microbench.pdf` -- supplementary throughput figure.
3. `results/sniah_*.csv` -- raw long-context numbers.
4. `results/lm_eval_*/results.json` -- commonsense + ICR table.
5. `checkpoints/prod_1gpu/ckpt_final.pt` -- trained 340M model.
6. `logs/*.out` -- full training logs for the appendix.

The figure should show your SKA line at or above PGDN/PKDA at 2K
context, and clearly above them at 8K and beyond. That is the
core empirical claim of the paper.

If the figure looks right, you are done. Move on to writing
Section X of the paper using the math derived in earlier turns.
