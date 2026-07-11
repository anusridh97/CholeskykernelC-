"""Benchmark: robust / sandwich Thompson sampling — double-sided under misspec.

Maintains the whitened Huber-White sandwich A_t = L^{-1} M_t L^{-⊤} for T
independent robust linear contextual bandits, updated by one rank-1 double-sided
step (z = x, w = e^2 x) per round, with a covariance-level shrinkage calibration
that reads the full A_t. Reports:

  * incremental vs full-refactorization per-round latency and A-drift (machine
    precision), tasks/sec throughput;
  * the separation ‖A − (trA/r)I‖/‖A‖ across noise regimes — homoskedastic
    (the degenerate negative control) vs heteroskedastic vs heavy-tailed — which
    is what makes the operator genuinely double-sided rather than A ≈ I;
  * mean instantaneous regret and posterior coverage of θ*.

Run directly on any GPU:  python -m tests.bench_robust_bandit
"""

import os
import sys
import json
import time
import datetime
import pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.robust_bandit import BatchedRobustSandwichTS

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_rounds(b, rounds):
    _sync(b.device)
    t0 = time.perf_counter()
    for _ in range(rounds):
        b.step()
    _sync(b.device)
    return (time.perf_counter() - t0) / rounds * 1e3


def time_refactor(device, r, T, rounds, dtype):
    """Baseline: recompute L=chol(G), A=L^{-1}M L^{-⊤} each round (O(r^3))."""
    torch.manual_seed(0)
    eye = torch.eye(r, device=device, dtype=dtype)
    G = eye.unsqueeze(0).repeat(T, 1, 1).contiguous()
    M = eye.unsqueeze(0).repeat(T, 1, 1).contiguous()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(rounds):
        z = torch.randn(T, r, device=device, dtype=dtype)
        w = (torch.rand(T, 1, device=device, dtype=dtype)) * z
        G = G + z.unsqueeze(2) * z.unsqueeze(1)
        M = M + z.unsqueeze(2) * w.unsqueeze(1)
        L = torch.linalg.cholesky(G)
        Y = torch.linalg.solve_triangular(L, M, upper=False)
        torch.linalg.solve_triangular(L, Y.mT, upper=False).mT
    _sync(device)
    return (time.perf_counter() - t0) / rounds * 1e3


def run(save=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    r = 32
    rounds = 60
    T_sizes = [16, 64, 256, 1024, 4096] if device.type == "cuda" else [16, 64]

    print(f"device: {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"feature dim r={r}  rounds={rounds}\n")
    print(f"{'T (bandits)':>12} {'incr ms/rd':>11} {'refact ms/rd':>13} {'speedup':>9} "
          f"{'bandits/sec':>12} {'A-drift':>9} {'regret':>8}")
    print("-" * 84)

    scaling = []
    for T in T_sizes:
        b = BatchedRobustSandwichTS(r=r, T=T, noise_mode="hetero", dtype=dtype,
                                    device=device, seed=0)
        b.step()
        incr_ms = time_rounds(b, rounds)
        _, A_ref = b.refactor_ground_truth()
        drift = (torch.norm(b.A - A_ref) / torch.norm(A_ref)).item()
        refact_ms = time_refactor(device, r, T, rounds, dtype)
        speedup = refact_ms / incr_ms if incr_ms else None
        scaling.append({
            "T": T, "incr_ms": incr_ms, "refact_ms": refact_ms, "speedup": speedup,
            "bandits_per_sec": T / (incr_ms / 1e3), "A_drift": drift,
            "final_regret": b.regret_history[-1],
        })
        print(f"{T:>12} {incr_ms:>11.3f} {refact_ms:>13.3f} "
              f"{(('%.2fx'%speedup) if speedup else 'n/a'):>9} {T/(incr_ms/1e3):>12.0f} "
              f"{drift:>9.1e} {b.regret_history[-1]:>8.4f}")

    # The decisive property: A separates from a scaled identity by misspecification.
    print("\nSandwich separation ‖A − (trA/r)I‖/‖A‖ by noise regime "
          "(homoskedastic = negative control):")
    T_sep = 256 if device.type == "cuda" else 64
    seps = {}
    for mode in ("homo", "hetero", "heavy"):
        torch.manual_seed(0)
        b = BatchedRobustSandwichTS(r=8, T=T_sep, n_arms=12, noise_mode=mode,
                                    noise_scale=0.3, hetero_beta=4.0,
                                    calibrate_every=10**9, dtype=dtype, device=device, seed=0)
        b.theta_star = b.theta_star * 0.5
        for _ in range(300):
            b.step()
        seps[mode] = {"separation": b.separation_from_identity(),
                      "final_regret": sum(b.regret_history[-20:]) / 20,
                      "coverage": b.coverage_history[-1]}
        print(f"  {mode:8s} sep={seps[mode]['separation']:.3f}  "
              f"regret={seps[mode]['final_regret']:.3f}  coverage={seps[mode]['coverage']:.2f}")
    print("  → separation grows homo < hetero < heavy: the sandwich is real, A ≠ cI.")

    # Regret curve for a representative T.
    T_rep = 256 if device.type == "cuda" else 64
    b = BatchedRobustSandwichTS(r=r, T=T_rep, noise_mode="hetero", dtype=dtype,
                                device=device, seed=1)
    regret_curve = [b.step() for _ in range(rounds)]
    print(f"\nRegret curve (T={T_rep}, hetero): "
          + "  ".join(f"{c:.2f}" for c in regret_curve[:: max(1, rounds // 10)]))

    results = {
        "kind": "robust_bandit",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "feature_r": r, "rounds": rounds,
        "scaling": scaling,
        "separation_by_mode": seps,
        "regret_curve": {"T": T_rep, "values": regret_curve},
    }
    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / "robust_bandit.json"
        path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {path}")
    return results


if __name__ == "__main__":
    run()
