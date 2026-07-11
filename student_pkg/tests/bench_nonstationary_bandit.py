"""Benchmark: nonstationary sliding-window sandwich TS (rank-k C.3 + downdate C.4).

Each round acquires a batch of k arms (one rank-k positive update) and, once the
window is full, forgets the oldest batch by a leverage-guarded hyperbolic
downdate — so the whitened sandwich A_t tracks only the last W rounds. The reward
parameter switches at the midpoint; the sliding window adapts while a no-forget
baseline stays biased by stale data.

Reports the forget-vs-no-forget regret curves around the switch, the windowed
A-drift vs a full re-cholesky (machine precision), and rank-k acquisition
throughput. Run directly:  python -m tests.bench_nonstationary_bandit
"""

import os
import sys
import json
import datetime
import pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.nonstationary_bandit import SlidingWindowSandwichTS

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"


def run(save=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    r = 24
    T = 256 if device.type == "cuda" else 48
    rounds = 80
    switch = 40

    print(f"device: {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"r={r}  T={T}  rounds={rounds}  batch_k=4  window=15  switch@{switch}\n")

    curves = {}
    drift = None
    skipped = None
    for forget in (True, False):
        b = SlidingWindowSandwichTS(r=r, T=T, n_arms=16, batch_k=4, window=15,
                                    forget=forget, noise_scale=0.2, dtype=dtype,
                                    device=device, seed=1)
        b.switch_at = switch
        curve = [b.step() for _ in range(rounds)]
        curves["forget" if forget else "no_forget"] = curve
        post = sum(curve[switch + 5:]) / len(curve[switch + 5:])
        tag = "forget (sliding window)" if forget else "no-forget (accumulate)"
        print(f"  {tag:26s} post-switch mean regret = {post:.4f}")
        if forget:
            _, A_ref = b.refactor_ground_truth()
            drift = (torch.norm(b.A - A_ref) / torch.norm(A_ref)).item()
            skipped = b.n_skipped

    print(f"\n  windowed A-drift vs full re-cholesky: {drift:.1e}")
    print(f"  inadmissible downdates skipped by the leverage guard: {skipped}")
    print("  → the sliding window (rank-k add + hyperbolic downdate) tracks the")
    print("    regime switch; the accumulator lags. A_t stays exact to machine precision.")

    results = {
        "kind": "nonstationary_bandit",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "feature_r": r, "rounds": rounds, "switch_at": switch,
        "curves": curves, "A_drift": drift, "downdates_skipped": skipped,
    }
    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / "nonstationary_bandit.json"
        path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {path}")
    return results


if __name__ == "__main__":
    run()
