# SKA-vs-PDN experimental framework

This directory contains a self-contained framework for replicating PDN's
S-NIAH long-context evaluation with the SKA (Both-Sides-Now Cholesky)
preconditioner in place of PDN's diagonal approximation.

## Files

| File | Purpose |
|---|---|
| `PLAN.md` | Day-by-day schedule + Farmshare time-budget breakdown |
| `ska_recurrence.py` | `torch.autograd.Function` with manual closed-form backward |
| `ska_layer.py` | Sequence-mixing layer that replaces attention |
| `ska_model.py` | 165M sanity-check + 340M production transformer |
| `train.py` | Main training entry point (DDP via torchrun, supports `--resume`) |
| `eval_sniah.py` | S-NIAH evaluator at multiple context lengths |
| `eval_lm_harness.py` | Wrapper that exposes the SKA model to lm-evaluation-harness (commonsense + ICR) |
| `microbench.py` | Per-token Triton kernel throughput sweep; sanity-checks GPU path |
| `run_microbench.slurm` | 1x L40S microbench, 30 min |
| `plot_sniah.py` | Renders the headline figure (overlays SKA results on PDN's published numbers) |
| `setup_farmshare.sh` | One-shot login-node environment install (uv venv, build C++ extension) |
| `run_train.slurm` | 4x L40S DDP training, auto-resumes from latest checkpoint, 24h time cap |
| `run_train_sanity.slurm` | 1x L40S sanity training (165M, 100M tokens) |
| `run_eval_sniah.slurm` | 1x L40S S-NIAH evaluator |
| `run_eval_lm.slurm` | 1x L40S lm-evaluation-harness commonsense + ICR |
| `submit_chain.sh` | Submit chained training jobs (`afterok` dependency) to span the 24h cap |
| (legacy) `setup_runpod.sh`, `launch_training.sh`, `launch_eval.sh` | Earlier RunPod-flavored launchers; keep but not used on Farmshare |

## Quick reference (commands you would run on Farmshare)

```bash
# 1. One-time setup on a login node (~5 min, no GPU needed)
bash experiments/setup_farmshare.sh
huggingface-cli login                              # for SlimPajama gating

# 2. Sanity training: 165M, 100M tokens, 1 L40S, ~1.5h
sbatch experiments/run_train_sanity.slurm

# 3. Production training: 340M, 5B tokens, 4 L40S, two chained 24h jobs
bash experiments/submit_chain.sh prod 2

# 4. Evaluation (or chain it to step 3 with RUN_EVAL=1)
sbatch experiments/run_eval_sniah.slurm
sbatch experiments/run_eval_lm.slurm

# 5. Make the figure (local or login-node)
python experiments/plot_sniah.py \
    --ours results/sniah_ckpt_final_seed0.csv \
    --out figures/

# Monitoring:
squeue -u $USER
tail -f logs/ska-train-<jobid>.out
```

## Pre-launch verification

Run the SKA recurrence's self-tests on a CPU machine with PyTorch installed
(no GPU required) before spinning up the pod:

```bash
cd experiments
python ska_recurrence.py
```

The script runs four checks:

1. Triton-vs-reference forward agreement (skipped without CUDA).
2. Manual backward vs autograd-through-reference, single step (rel err `< 1e-10` in float64).
3. Formal `torch.autograd.gradcheck` on a scalar loss (returns `True`).
4. Multi-step BPTT rollout over 8 tokens (rel err `< 1e-8` in float64).

If any of these fail, do not launch training.

## Critical caveats

**Backward correctness and speed.** `_SKAUpdate.backward` is a hand-derived
closed-form gradient. Five upper-triangular solves, three lower-triangular
solves, and three `(r,r)` matmuls per token, all `O(r^2)`. No Cholesky in
the backward pass. Expect ~2-3x training slowdown vs PGDN (PGDN's diagonal
update is `O(r)` per token whereas SKA is `O(r^2)`); the asymptotic ratio
is exactly the architectural cost difference, not an implementation gap.

If throughput is still too low to fit the budget at 5B tokens, reduce
context to 1024 (training-time only; eval still runs at 2K-32K) or drop
to the 165M variant.

**Float precision.** All kernel and reference ops run in float32. Param
storage in `train.py` is float32 by default. Drift bound from your paper §3.2
is `O(N delta) = O(Nru)`. At `N = 2048, r = 128, u = 2^{-23}`, drift is
~3e-2, well below the LM loss scale, so no periodic refactorization is
needed for ctx 2048. If you ever extrapolate training to longer contexts,
bound `N` and add a refactorization every `N = 50` steps.

**Tokenizer.** Default is Llama-2's tokenizer. PDN uses GPT-2 BPE (50272 vocab)
in their flash-linear-attention setup. Either is fine for the long-context
recall figure; GPT-2 reduces vocab head FLOPs.

**SlimPajama streaming.** First epoch streams from HuggingFace, so the pod
needs working internet. If you anticipate dropouts, pre-download a 5B-token
subset to local disk first.

## Kernel utilization

The framework is wired to take full advantage of the Triton kernel:

- **Forward path uses `_fused_doublesided_kernel` directly**, capturing
  `v_w` from the kernel's mutated buffer instead of redoing a triangular
  solve outside (`ska_recurrence._kernel_call`).
- **Inner loop pre-permutes** `q, k, v` to `(T, BH, r)` layout once per
  layer call, so the per-step slice is a contiguous view (no memcpy in
  the hot loop).
- **Streaming inference uses state passing** in `eval_sniah.py` and
  `eval_lm_harness.py`, so generation runs in `O(N)` total kernel calls
  instead of `O(N^2)`. Includes short-conv history buffers for boundary
  correctness.
- **Single triangular solve outside the kernel** per token, only for
  `k_tilde = L_+^{-T} v_z` (the kernel does not compute this).
- **Reference forward** stays as a CPU-only fallback for correctness
  testing and `gradcheck`; never used on GPU.

To verify on Farmshare:

```bash
sbatch experiments/run_microbench.slurm
# Look for "Kernel path: triton" in the log; reference (CPU) means
# CUDA was not available and you should investigate before training.
```

## Math contract recap

The SKA recurrence (per token) maintains:

```
G_t = lambda I + sum_{s<=t} k_s k_s^T = L_t L_t^T
M_t = sum_{s<=t} k_s v_s^T
A_w_t = L_t^{-1} M_t L_t^{-T}
S_t = M_t^T G_t^{-1} = L_t A_w_t^T L_t^{-1}
```

The kernel returns `v_z = L_t^{-1} k_t`. From this:

```
k_tilde_t = L_t^{-T} v_z = G_t^{-1} k_t = P_t k_t      # exact preconditioned key
S_t = S_{t-1} + (v_t - S_{t-1} k_t) k_tilde_t^T        # ATK update (exact)
o_t = S_t q_t                                          # output
```

This is PDN's Eq. 9 with `B_t * k_t` replaced by `L_t^{-T} v_z`, which is
exactly `G_t^{-1} k_t` (no diagonal approximation, no squash, no `mu`, no `x`).
See Theorems 6-10 in the paper-section material for the formal containment.
