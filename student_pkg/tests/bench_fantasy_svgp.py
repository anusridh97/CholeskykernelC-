"""Benchmark: batched fantasy-conditioned SVGP-PDTS (systems / parallelism).

J fantasy branches per task, each carrying a whitened covariance A^{(j)}, all
conditioned on their own pending batch in one launch. Sweeps the number of
fantasy branches J and measures branch-update throughput and per-round latency,
comparing explicit double-sided covariance maintenance against a square-root /
refactor baseline. Reports A-drift vs a full re-cholesky (machine precision).

Run directly on any GPU:  python -m tests.bench_fantasy_svgp
"""

import os
import sys
import json
import time
import datetime
import pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.svgp_pdts import BatchedFantasySVGP

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_rounds(fp, rounds):
    _sync(fp.device)
    t0 = time.perf_counter()
    for _ in range(rounds):
        fp.round()
    _sync(fp.device)
    return (time.perf_counter() - t0) / rounds * 1e3


def time_refactor(fp, rounds):
    """Baseline: recompute L=chol(G) and A each round instead of transporting."""
    _sync(fp.device)
    t0 = time.perf_counter()
    for _ in range(rounds):
        z = 0.05 * torch.randn(fp.B, fp.r, generator=fp.gen).to(fp.device, fp.dtype)
        fp.G = fp.G + z.unsqueeze(2) * z.unsqueeze(1)
        fp.L = torch.linalg.cholesky(fp.G).contiguous()
        Y = torch.linalg.solve_triangular(fp.L, fp.S, upper=False)
        fp.A = torch.linalg.solve_triangular(fp.L, Y.mT, upper=False).mT
        fp.fantasy_condition(fp.random_pending())
    _sync(fp.device)
    return (time.perf_counter() - t0) / rounds * 1e3


def run(save=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    r = 128
    T = 8
    k = 4
    rounds = 20
    J_sizes = [16, 64, 256, 1024] if device.type == "cuda" else [8, 32]

    torch.manual_seed(0)
    X = torch.rand(r, 2)

    print(f"device: {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"inducing r={r}  tasks T={T}  pending k={k}  rounds={rounds}\n")
    print(f"{'J (branches)':>13} {'total':>8} {'incr ms/rd':>11} {'refact ms/rd':>13} "
          f"{'speedup':>9} {'branch upd/s':>13} {'A-drift':>9}")
    print("-" * 84)

    scaling = []
    for J in J_sizes:
        fi = BatchedFantasySVGP(X, T=T, J=J, k=k, dtype=dtype, device=device, seed=0)
        fi.round()  # warmup
        incr_ms = time_rounds(fi, rounds)
        _, A_ref = fi.refactor_ground_truth()
        drift = (torch.norm(fi.A - A_ref) / torch.norm(A_ref)).item()

        fr = BatchedFantasySVGP(X, T=T, J=J, k=k, dtype=dtype, device=device, seed=0)
        fr.round()
        refact_ms = time_refactor(fr, rounds)

        speedup = refact_ms / incr_ms if incr_ms else None
        upd_per_s = (T * J) / (incr_ms / 1e3)
        scaling.append({
            "J": J, "total_branches": T * J, "incr_ms": incr_ms, "refact_ms": refact_ms,
            "speedup": speedup, "branch_updates_per_sec": upd_per_s, "A_drift": drift,
        })
        print(f"{J:>13} {T*J:>8} {incr_ms:>11.3f} {refact_ms:>13.3f} "
              f"{(('%.2fx'%speedup) if speedup else 'n/a'):>9} {upd_per_s:>13.0f} "
              f"{drift:>9.1e}")

    print("\nEach branch is one sequential Givens sweep; all T·J branches update in one")
    print("launch. The covariance-form fantasy conditioning consumes the full A^(j).")

    results = {
        "kind": "fantasy_svgp",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "inducing_r": r, "tasks": T, "pending_k": k, "rounds": rounds,
        "scaling": scaling,
    }
    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / "fantasy_svgp.json"
        path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {path}")
    return results


if __name__ == "__main__":
    run()
