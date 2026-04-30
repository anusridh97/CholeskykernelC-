# SKA experiments

Replicates Preconditioned DeltaNet's S-NIAH long-context evaluation, with
the SKA (exact double-sided Cholesky) preconditioner in place of PDN's
diagonal approximation. End deliverable: a single figure showing accuracy
vs context length, SKA vs PDN baselines.

## Layout

```
experiments/
├── backend/        # cluster-agnostic core (model, kernel wrapper, training, evals)
├── farmshare/      # Stanford Farmshare slurm scripts + runbook
├── scg/            # Stanford SCG slurm scripts + runbook
└── runpod/         # legacy paid-cloud launchers
```

The `backend/` folder is the source of truth for the algorithm:

| File | Purpose |
|---|---|
| `ska_recurrence.py` | `torch.autograd.Function` wrapping the Triton kernel + manual closed-form backward |
| `ska_layer.py` | Sequence-mixing layer that replaces attention |
| `ska_model.py` | 165M sanity + 340M production transformer |
| `train.py` | Training entry point (single-GPU or `torchrun` DDP) |
| `eval_sniah.py` | S-NIAH long-context evaluator |
| `eval_lm_harness.py` | Adapter exposing the model to lm-evaluation-harness |
| `plot_sniah.py` | Renders the headline figure |
| `microbench.py` | Per-token kernel throughput sweep |

The `farmshare/`, `scg/`, and `runpod/` folders each contain a `setup.sh`,
a small set of slurm or bash launchers, a `submit_chain.sh` wrapper for
chained training, and a `README.md` with the cluster-specific runbook.
The slurm scripts in those folders never touch algorithmic logic; they
just activate the venv, configure threading, and invoke
`backend/{train,eval_*}.py`.

## Pick a cluster, then go to its README

| Cluster | Path | Cost | Speed | Doc |
|---|---|---|---|---|
| Stanford Farmshare | `farmshare/` | free | 4x L40S, ~3-5 days for 8B tokens | [farmshare/README.md](farmshare/README.md) |
| Stanford SCG | `scg/` | free (with PI account) | 8x H100/H200 if available, ~16-24h for 15B tokens | [scg/README.md](scg/README.md) |
| RunPod (legacy) | `runpod/` | ~$170 / run | 4x H100, ~14h for 5B tokens | this README |

Recommended workflow: pick the cheapest cluster you have access to, follow
its README, ship the figure.

## High-level pipeline (cluster-agnostic)

1. **Setup**: `bash <cluster>/setup.sh` on a login node. Installs the venv,
   builds the C++ extension, runs CPU-side gradcheck on the manual backward.
2. **Sanity**: `sbatch <cluster>/run_train_sanity.slurm` (Farmshare/SCG) or
   `bash runpod/launch_training.sh sanity`. 165M model on 100M tokens, single
   GPU, ~1-2 hours. Confirms the kernel fires, loss decreases.
3. **Production training**: chained jobs that auto-resume from the latest
   checkpoint. Wrapped by `<cluster>/submit_chain.sh`.
4. **Eval**: S-NIAH at 5 context lengths plus lm-eval-harness commonsense + ICR.
   Auto-fired after training when you set `RUN_EVAL=1`.
5. **Plot**: `python backend/plot_sniah.py --ours results/sniah_*.csv --out figures/`.

## What the figure should show

Accuracy on S-NIAH-{1,2,3} at context lengths $\{2K, 4K, 8K, 16K, 32K\}$,
with SKA at or above PGDN/PKDA at 2-4K and clearly above them at 8K and
beyond. PDN's published numbers from arXiv:2604.21100v1 Table 5 are
overlaid as baselines (replace placeholder values in
`backend/plot_sniah.py` `PDN_NUMBERS` with the real ones from the paper
before locking the figure).

## Running the local correctness checks

You don't need a GPU or cluster to verify the math:

```bash
cd experiments/backend
python ska_recurrence.py
```

Expected output: forward consistency check, manual-backward vs autograd
agreement (rel err < 1e-10 in float64), formal `gradcheck = True`,
multi-step BPTT (rel err < 1e-8). Run this on any machine with PyTorch
before consuming GPU time.

## Math contract (one-line summary)

```
G_t = lambda I + sum_{s<=t} k_s k_s^T = L_t L_t^T          # exact Cholesky
M_t = sum_{s<=t} k_s v_s^T
A_w_t = L_t^{-1} M_t L_t^{-T}                              # whitened cross-Gram
S_t = M_t^T G_t^{-1} = L_t A_w_t^T L_t^{-1}               # closed-form LS state
k_tilde_t = L_t^{-T} v_z = G_t^{-1} k_t                   # exact normalized write key
```

PDN's Eq. 9 is the diagonal restriction of this with the squash kicker;
see `paper_section_X.tex` (or the prior turn's theorem material) for the
formal containment statement.
