"""
Per-token kernel throughput microbenchmark.

Two purposes:
  1. Sanity-check that the Triton kernel is actually being called on GPU
     (not silently falling back to the CPU reference path).
  2. Produce the supplementary "kernel runtime vs. sequence length" plot
     analogous to PDN's Figure 4.

Usage:
  python microbench.py                                  # default sweep
  python microbench.py --r 128 --bh 32 --T 2048         # one config
  python microbench.py --csv out.csv --plot fig.pdf

Reports per-token median wall time across (r, BH, T) configurations,
plus the implied tokens/second for that head/batch shape. If you see
the "kernel path: triton" line, you're using the GPU path; if you see
"reference (CPU)", figure out why CUDA is not available.
"""

import argparse
import csv
import statistics
import time
from pathlib import Path

import torch

from ska_recurrence import (
    _HAVE_TRITON,
    _forward_reference,
    _forward_triton,
    ska_step,
)


def run_step(L, Aw, k, v, use_triton):
    fn = _forward_triton if use_triton else _forward_reference
    return fn(L, Aw, k, v)


def make_batch(BH, r, dtype, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    A = torch.randn(BH, r, r, generator=g, dtype=dtype, device=device)
    G = A @ A.transpose(-1, -2) + r * torch.eye(r, dtype=dtype, device=device)
    L = torch.linalg.cholesky(G).contiguous()
    Aw = torch.zeros(BH, r, r, dtype=dtype, device=device)
    return L, Aw


def time_sequence(BH, r, T, dtype, device, use_triton, repeats=3, warmup=2):
    """Measure median wall time for a T-token rollout at (BH, r)."""
    times = []
    for trial in range(warmup + repeats):
        torch.manual_seed(trial)
        L, Aw = make_batch(BH, r, dtype, device, seed=trial)
        ks = torch.randn(T, BH, r, dtype=dtype, device=device) * 0.1
        vs = torch.randn(T, BH, r, dtype=dtype, device=device) * 0.1
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for t in range(T):
            L, Aw, _, _, _ = run_step(L, Aw, ks[t], vs[t], use_triton=use_triton)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if trial >= warmup:
            times.append(elapsed)
    return statistics.median(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--r", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--bh", type=int, nargs="+", default=[8, 32, 128])
    p.add_argument("--T", type=int, nargs="+", default=[512, 2048, 8192])
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--plot", type=str, default=None)
    p.add_argument("--force_reference", action="store_true",
                   help="Force the CPU reference path (for cross-check)")
    args = p.parse_args()

    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    use_triton = (cuda and _HAVE_TRITON and not args.force_reference)
    print(f"Device:    {device}")
    print(f"GPU:       {torch.cuda.get_device_name(0) if cuda else 'n/a'}")
    print(f"Triton:    {'available' if _HAVE_TRITON else 'NOT available'}")
    print(f"Kernel path: {'triton' if use_triton else 'reference (CPU)'}")
    print()

    rows = []
    print(f"{'r':>5} {'BH':>5} {'T':>6} {'tot ms':>10} {'us/tok':>10} {'tok/s':>12}")
    print("-" * 60)
    for r in args.r:
        for BH in args.bh:
            for T in args.T:
                t = time_sequence(BH, r, T, dtype, device, use_triton, repeats=args.repeats)
                tot_ms = t * 1000.0
                us_per_tok = tot_ms * 1000.0 / T
                tok_per_s = T * BH / t
                print(f"{r:>5} {BH:>5} {T:>6} {tot_ms:>10.2f} {us_per_tok:>10.2f} {tok_per_s:>12.0f}")
                rows.append({
                    "r": r, "bh": BH, "T": T,
                    "wall_s": f"{t:.6f}",
                    "us_per_token": f"{us_per_tok:.3f}",
                    "tokens_per_second": f"{tok_per_s:.0f}",
                    "kernel_path": "triton" if use_triton else "reference",
                })

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {out}")

    if args.plot:
        import matplotlib.pyplot as plt
        out = Path(args.plot)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        for BH in args.bh:
            for r in args.r:
                xs = [d["T"] for d in rows if d["r"] == r and d["bh"] == BH]
                ys = [float(d["us_per_token"]) for d in rows if d["r"] == r and d["bh"] == BH]
                if xs:
                    ax.plot(xs, ys, marker="o", label=f"r={r}, BH={BH}")
        ax.set_xlabel("Sequence length T")
        ax.set_ylabel("us / token")
        ax.set_xscale("log", base=2)
        ax.set_title("SKA kernel per-token wall time (Triton)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
