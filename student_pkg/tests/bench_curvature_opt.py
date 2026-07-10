"""Benchmark: incremental curvature optimizer vs AdamW and periodic refact.

Trains a small Transformer encoder on a synthetic sequence-classification task,
entirely GPU-resident, and compares three optimizers:

  * AdamW                       — the diagonal baseline people actually run.
  * curvature (incremental)     — one batched O(r^2) chart refresh per step.
  * curvature (periodic_refact) — the Shampoo/SOAP-style ablation: re-cholesky
                                  the same charts every K steps, O(r^3) amortized.

The net has dozens of weight blocks, so a single training step is dozens of
independent rank-1 whitened-curvature updates — the T-batch the kernel wants.
We report wall-clock-to-target-loss and the per-step optimizer cost, plus the
batched-vs-sequential refresh speedup that is the whole point of the kernel.

Run from student_pkg/:

    python -m tests.bench_curvature_opt

Runs on CPU (reference refresh) or GPU (batched Triton kernel) automatically;
the GPU numbers are the ones that matter. Output mirrors to tests/logs/.
"""

import os
import sys
import time
import datetime
import pathlib
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.curvature_optimizer import IncrementalCurvatureOptimizer

_log_file = None


def setup_logging():
    global _log_file
    log_dir = pathlib.Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", f"local_{os.getpid()}")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"bench_curvature_opt_{job_id}_{ts}.log"
    _log_file = open(path, "w")
    log(f"Log: {path}")
    return path


def log(msg=""):
    print(msg, flush=True)
    if _log_file is not None:
        print(msg, file=_log_file, flush=True)


# --------------------------------------------------------------------------- #
# Model + synthetic task
# --------------------------------------------------------------------------- #


class TinyTransformer(nn.Module):
    """Small Transformer encoder → dozens of independent weight blocks."""

    def __init__(self, vocab=32, d_model=64, nhead=4, layers=3, seqlen=16, nclass=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.randn(1, seqlen, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=2 * d_model, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, nclass)

    def forward(self, x):
        h = self.embed(x) + self.pos
        h = self.encoder(h)
        return self.head(h.mean(dim=1))


def make_task(batch, seqlen, vocab, nclass, device, seed=0):
    """Deterministic synthetic task: class = (sum of tokens) mod nclass."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randint(0, vocab, (batch, seqlen), generator=g)
    y = (x.sum(dim=1) % nclass)
    return x.to(device), y.to(device)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def train(make_opt, device, steps, target, batch=128, seqlen=16, vocab=32, nclass=4, seed=0):
    torch.manual_seed(seed)
    model = TinyTransformer(vocab=vocab, seqlen=seqlen, nclass=nclass).to(device)
    opt = make_opt(model)
    loss_fn = nn.CrossEntropyLoss()

    x, y = make_task(batch, seqlen, vocab, nclass, device, seed=seed)

    opt_times = []
    reached = None
    t0 = time.perf_counter()
    for step in range(steps):
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()
        ts = time.perf_counter()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        opt_times.append(time.perf_counter() - ts)

        if reached is None and loss.item() < target:
            reached = (step + 1, time.perf_counter() - t0)
    total = time.perf_counter() - t0

    opt_times.sort()
    med_opt_ms = opt_times[len(opt_times) // 2] * 1e3
    final_loss = loss.item()
    n_blocks = getattr(opt, "num_blocks", lambda: None)()
    return {
        "final_loss": final_loss,
        "reached": reached,
        "total_s": total,
        "med_opt_ms": med_opt_ms,
        "n_blocks": n_blocks,
    }


def fmt_reached(reached):
    if reached is None:
        return "  not reached"
    step, t = reached
    return f"step {step:>4} @ {t:6.2f}s"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device:       {device}")
    if device.type == "cuda":
        log(f"GPU:          {torch.cuda.get_device_name(0)}")
    log(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'n/a')}")
    log(f"Started:      {datetime.datetime.now().isoformat(timespec='seconds')}")
    # CPU reference refresh is slow; keep the CPU smoke run tiny.
    steps = 300 if device.type == "cuda" else 30
    target = 0.05
    rank = 32
    log(f"steps={steps}  target_loss={target}  rank={rank}")
    log("")

    configs = [
        ("AdamW", lambda m: torch.optim.AdamW(m.parameters(), lr=3e-3)),
        (
            "curvature (incremental)",
            lambda m: IncrementalCurvatureOptimizer(
                m.parameters(), lr=1e-2, rank=rank, refresh_mode="incremental"
            ),
        ),
        (
            "curvature (periodic_refact K=20)",
            lambda m: IncrementalCurvatureOptimizer(
                m.parameters(), lr=1e-2, rank=rank,
                refresh_mode="periodic_refact", refact_every=20,
            ),
        ),
    ]

    log(f"{'optimizer':<34} {'final loss':>10} {'to target':>18} {'opt ms/step':>12} {'blocks':>7}")
    log("-" * 86)
    for name, make_opt in configs:
        r = train(make_opt, device, steps, target)
        log(
            f"{name:<34} {r['final_loss']:>10.4f} {fmt_reached(r['reached']):>18} "
            f"{r['med_opt_ms']:>11.3f} {str(r['n_blocks']):>7}"
        )
    log("")
    log("Note: 'opt ms/step' is the optimizer.step() cost only (backward excluded).")
    log("On GPU the incremental refresh is one batched kernel launch across all")
    log("same-rank blocks; periodic_refact pays an O(r^3) re-cholesky every K steps.")


if __name__ == "__main__":
    log_path = setup_logging()
    try:
        run()
    finally:
        if _log_file is not None:
            _log_file.close()
        print(f"Log saved to {log_path}")
