# SKA vs PDN: One-Week Experiment Plan (Stanford Farmshare)

## Goal

Replicate PDN's S-NIAH long-context evaluation on a 340M-parameter language
model trained on SlimPajama, with **SKA** (exact double-sided Cholesky
preconditioning) replacing PDN's diagonal preconditioner. Headline figure:
S-NIAH accuracy vs context length, our line flat where PGDN drops.

## Hardware

Stanford Farmshare cluster, GPU partition. Free.

- **Training**: 4x L40S 48GB on a single node, DDP via torchrun.
- **Eval**: 1x L40S 48GB.
- **Time cap**: 24h per SLURM job in the `gpu` partition. Use
  `submit_chain.sh` to fire dependent jobs that span longer runs.

## Compute envelope (4x L40S)

| Phase | Wall | Notes |
|---|---|---|
| Setup (login node) | 5 min | uv venv install, build C++ extension, run CPU sanity |
| Sanity training (165M, 100M tok) | ~1.5h | one L40S, validates loss decreases |
| Production training (340M, 5B tok) | ~35h | 4 L40S, two chained 24h jobs |
| S-NIAH eval (5 lengths x 3 variants) | ~3h | one L40S |
| LM eval (commonsense + ICR via lm-eval-harness) | ~2h | one L40S |
| Total wall clock | **~42h** | About two days of GPU time |

If you have only 2 L40S, double the training time to ~70h, fit in three
chained 24h jobs.

If you have only 1 L40S, drop to 1.5B tokens (fits in one 24h job at
~14h wall) and document this scaling choice in the paper.

## Day-by-day schedule

### Day 1: Local correctness (laptop, no SLURM)

- Run `python experiments/ska_recurrence.py` on any machine with PyTorch.
- Confirm gradcheck returns `True` and the BPTT rollout error is `< 1e-8`.
- This is the same code that will run on Farmshare; if it fails locally,
  fix it before consuming GPU time.

### Day 1 (afternoon): Farmshare setup

- SSH into a Farmshare login node.
- `cd <repo_root>; bash experiments/setup_farmshare.sh`
- Login to HuggingFace once for SlimPajama: `huggingface-cli login`
- **Exit criterion**: setup script finishes without error and prints
  the model parameter counts.

### Day 2: Sanity training

- `sbatch experiments/run_train_sanity.slurm`
- Should complete in ~1.5h.
- Inspect `logs/ska-sanity-<jobid>.out`. Loss should drop monotonically
  from ~10 to ~5 over 100M tokens.
- **Exit criterion**: final loss < 6.0; no NaNs in the log.

### Day 2-5: Main training run

- Submit chained training jobs:
  ```
  bash experiments/submit_chain.sh prod 2
  ```
  This fires two 24h jobs in sequence. The second auto-resumes from the
  checkpoint left by the first.
- Monitor: `squeue -u $USER` and tail the latest log.
- Save checkpoints land in `checkpoints/prod/ckpt_<tokens>.pt`.
- **Exit criterion**: final loss < 4.0 (proxy for ~LAMBADA-class quality).

### Day 6: Evaluation

- Once the chain finishes:
  ```
  sbatch experiments/run_eval_sniah.slurm
  sbatch experiments/run_eval_lm.slurm
  ```
- Or submit eval jobs as dependents of the training chain by setting
  `RUN_EVAL=1` when calling `submit_chain.sh`.
- Outputs: `results/sniah_*.csv`, `results/lm_eval_*/results.json`.
- **Exit criterion**: a CSV with non-zero accuracies at all three S-NIAH
  variants and all five context lengths.

### Day 7: Figure + writeup

- `python experiments/plot_sniah.py --ours results/sniah_*.csv --out figures/`
- Insert Section X (the theoretical material from the prior turn) plus the
  figure into the NeurIPS paper.
- Optional: write a small `microbench.py` for the local (CPU/single-GPU)
  per-token throughput plot reproducing PDN Figure 4.

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Job preempted before checkpoint write | Medium | We write every 1B tokens; lose at most ~5h |
| Throughput too low to finish 5B in two 24h chunks | Medium | Reduce to 3B tokens (still fits the figure narrative); or chain a third job |
| HuggingFace streaming stalls | Low | Pre-download a 5B-token subset to `$HF_HOME` once on a login node |
| Manual backward subtle bug at scale | Low | gradcheck passed at r=4; multi-step BPTT rollout passed at r=4 (extends by induction) |
| L40S OOM at 32K eval context | Medium | Use BF16 inference; or eval up to 16K only, mark 32K as "OOM, future work" |

## Outputs

By end of week:

1. `figures/sniah_extrapolation.pdf` — main paper figure
2. `results/sniah_*.csv` — raw long-context numbers
3. `results/lm_eval_*/results.json` — commonsense + ICR table
4. `checkpoints/prod/ckpt_final.pt` — trained model
5. `logs/ska-train-*.out`, `logs/ska-eval-*.out` — full experiment logs
6. `paper_section_X.tex` — new mathematical material
7. Reproduction commands documented in this PLAN.md

## Quick reference commands

```bash
# One-time setup (on a login node):
bash experiments/setup_farmshare.sh
huggingface-cli login   # for SlimPajama

# Sanity (1 GPU, ~1.5h):
sbatch experiments/run_train_sanity.slurm

# Main training (4 GPUs, ~35h, two chained jobs):
bash experiments/submit_chain.sh prod 2

# Or training + auto eval:
RUN_EVAL=1 bash experiments/submit_chain.sh prod 2

# Manual eval submission:
sbatch experiments/run_eval_sniah.slurm
sbatch experiments/run_eval_lm.slurm

# Figure:
python experiments/plot_sniah.py \
    --ours results/sniah_ckpt_final_seed0.csv \
    --out figures/

# Monitoring:
squeue -u $USER
tail -f logs/ska-train-<jobid>.out
```
