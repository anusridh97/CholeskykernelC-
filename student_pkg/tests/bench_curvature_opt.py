"""Benchmark: incremental curvature optimizer vs AdamW and periodic refact.

Trains a small Transformer encoder on a synthetic sequence-classification task,
entirely GPU-resident, and compares three optimizers:

  * AdamW                       — the diagonal baseline people actually run.
  * curvature (incremental)     — one batched O(r^2) chart refresh per step.
  * curvature (periodic_refact) — the Shampoo/SOAP-style ablation: re-cholesky
                                  the same charts every K steps, O(r^3) amortized.

It also measures the batched-vs-sequential chart-refresh scaling that is the
whole point of the kernel: refreshing all of a net's same-rank blocks in one
launch versus one block at a time.

Returns a JSON-serializable results dict (also written to results/) that
``experiments.viz`` turns into figures. Run directly on any GPU:

    python -m tests.bench_curvature_opt

No scheduler required. Runs on CPU (reference refresh, tiny) or GPU (batched
Triton kernel) automatically; the GPU numbers are the ones that matter.
"""

import os
import sys
import json
import time
import datetime
import pathlib
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.curvature_optimizer import IncrementalCurvatureOptimizer

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"


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
    loss_curve = []
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
        loss_curve.append(loss.item())

        if reached is None and loss.item() < target:
            reached = {"step": step + 1, "seconds": time.perf_counter() - t0}
    total = time.perf_counter() - t0

    opt_times.sort()
    return {
        "loss_curve": loss_curve,
        "final_loss": loss_curve[-1],
        "reached": reached,
        "total_s": total,
        "med_opt_ms": opt_times[len(opt_times) // 2] * 1e3,
        "n_blocks": getattr(opt, "num_blocks", lambda: None)(),
    }


# --------------------------------------------------------------------------- #
# Batched-vs-sequential chart refresh scaling
# --------------------------------------------------------------------------- #


def bench_refresh_scaling(device, r=32, T_sizes=(1, 4, 16, 64, 128, 256), reps=20):
    """Refresh T equal-rank charts: one batched launch vs T sequential launches.

    This is the per-step cost the optimizer pays across a net's same-rank
    blocks. On CPU both paths use the reference loop, so skip it there.
    """
    if device.type != "cuda":
        return {"available": False}
    from doublesided_cholesky import (
        incremental_update_triton,
        incremental_update_triton_batched,
    )

    def make(T):
        torch.manual_seed(0)
        A = torch.randn(T, r, r, device=device)
        G = A @ A.mT + r * torch.eye(r, device=device).unsqueeze(0)
        L = torch.linalg.cholesky(G).contiguous()
        Aw = torch.randn(T, r, r, device=device)
        z = torch.randn(T, r, device=device) * 0.1
        w = torch.randn(T, r, device=device) * 0.1
        return L, Aw, z, w

    def timed(fn):
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        return ts[len(ts) // 2]

    rows = []
    for T in T_sizes:
        L, Aw, z, w = make(T)
        for _ in range(3):  # warmup
            incremental_update_triton_batched(L.clone(), Aw.clone(), z.clone(), w.clone())
        seq = timed(lambda: [
            incremental_update_triton(L[t].clone(), Aw[t].clone(), z[t].clone(), w[t].clone())
            for t in range(T)
        ])
        bat = timed(lambda: incremental_update_triton_batched(
            L.clone(), Aw.clone(), z.clone(), w.clone()))
        rows.append({"T": T, "seq_ms": seq, "bat_ms": bat, "speedup": seq / bat if bat else None})
    return {"available": True, "r": r, "rows": rows}


# --------------------------------------------------------------------------- #


def run(save=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = 300 if device.type == "cuda" else 30
    target = 0.05
    rank = 32

    configs = [
        ("AdamW", lambda m: torch.optim.AdamW(m.parameters(), lr=3e-3)),
        ("curvature (incremental)", lambda m: IncrementalCurvatureOptimizer(
            m.parameters(), lr=1e-2, rank=rank, refresh_mode="incremental")),
        ("curvature (periodic_refact K=20)", lambda m: IncrementalCurvatureOptimizer(
            m.parameters(), lr=1e-2, rank=rank,
            refresh_mode="periodic_refact", refact_every=20)),
    ]

    print(f"device: {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"steps={steps}  target_loss={target}  rank={rank}\n")
    print(f"{'optimizer':<34} {'final loss':>10} {'to target':>18} {'opt ms/step':>12} {'blocks':>7}")
    print("-" * 86)

    training = {}
    for name, make_opt in configs:
        r = train(make_opt, device, steps, target)
        training[name] = r
        reached = ("step %d @ %.2fs" % (r["reached"]["step"], r["reached"]["seconds"])
                   if r["reached"] else "not reached")
        print(f"{name:<34} {r['final_loss']:>10.4f} {reached:>18} "
              f"{r['med_opt_ms']:>11.3f} {str(r['n_blocks']):>7}")

    print("\nBatched vs sequential chart refresh (across a net's same-rank blocks):")
    refresh = bench_refresh_scaling(device)
    if refresh.get("available"):
        print(f"{'blocks T':>9} {'seq ms':>9} {'batched ms':>11} {'speedup':>9}")
        for row in refresh["rows"]:
            print(f"{row['T']:>9} {row['seq_ms']:>9.3f} {row['bat_ms']:>11.3f} "
                  f"{row['speedup']:>8.2f}x")
    else:
        print("  (GPU only — skipped on CPU)")

    results = {
        "kind": "curvature_optimizer",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "steps": steps, "target": target, "rank": rank,
        "training": training,
        "refresh_scaling": refresh,
    }
    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / "curvature_optimizer.json"
        path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {path}")
    return results


if __name__ == "__main__":
    run()
