"""Benchmark: non-symmetric double-sided dynamics TS — the flagship experiment.

Maintains a NON-SYMMETRIC whitened dynamics operator A_t = L^{-1} M L^{-⊤} for
T independent online Bayesian linear systems, updated by one rank-1 double-sided
step (z ≠ w) per round, and runs model-based Thompson sampling that reads A_t.

Because A_t ≠ A_t^⊤, no covariance root R with A_t = R R^⊤ exists — the
"root-only" baseline that makes ordinary parallel Thompson sampling one-sided is
mathematically inapplicable here. We report:

  * incremental vs full-refactorization per-round latency (the O(r^2) vs O(r^3)
    core cost) and A-drift (matches to machine precision);
  * batched-vs-sequential throughput across T (the massive-parallelism story,
    now on a genuinely double-sided primitive);
  * the operator symmetry gap ‖A−A^⊤‖/‖A‖ (≈ O(1), i.e. no root exists);
  * mean simple regret vs round (the loop actually controls).

Run directly on any GPU:  python -m tests.bench_dynamics_thompson
"""

import os
import sys
import json
import time
import datetime
import pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.dynamics_thompson import BatchedDynamicsThompson

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_round(gp, rounds):
    _sync(gp.device)
    t0 = time.perf_counter()
    for _ in range(rounds):
        gp.step()
    _sync(gp.device)
    return (time.perf_counter() - t0) / rounds * 1e3


def time_refactor_core(device, r, T, rounds, dtype):
    """Per-round cost when A is recomputed from scratch (chol + two solves)."""
    torch.manual_seed(0)
    G = (torch.eye(r, device=device, dtype=dtype) * 1.0).unsqueeze(0).repeat(T, 1, 1).contiguous()
    M = torch.zeros(T, r, r, device=device, dtype=dtype)
    z = torch.randn(T, r, device=device, dtype=dtype)
    w = torch.randn(T, r, device=device, dtype=dtype)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(rounds):
        G = G + z.unsqueeze(2) * z.unsqueeze(1)
        M = M + z.unsqueeze(2) * w.unsqueeze(1)
        L = torch.linalg.cholesky(G)
        Y = torch.linalg.solve_triangular(L, M, upper=False)
        torch.linalg.solve_triangular(L, Y.mT, upper=False).mT      # A = L^{-1} M L^{-⊤}
    _sync(device)
    return (time.perf_counter() - t0) / rounds * 1e3


def run(save=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    r = 128
    rounds = 30
    T_sizes = [16, 64, 256, 1024, 4096] if device.type == "cuda" else [8, 32]

    print(f"device: {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"feature dim r={r}  rounds={rounds}\n")
    print(f"{'T (systems)':>12} {'incr ms/rd':>11} {'refact ms/rd':>13} "
          f"{'speedup':>9} {'sys/sec':>10} {'sym gap':>8} {'A-drift':>9} {'regret':>8}")
    print("-" * 92)

    scaling = []
    regret_curve = None
    for T in T_sizes:
        gp = BatchedDynamicsThompson(r=r, T=T, dtype=dtype, device=device, seed=0)
        gp.step()  # warmup
        incr_ms = time_round(gp, rounds)
        refact_ms = time_refactor_core(device, r, T, rounds, dtype)
        _, A_ref = gp.refactor_ground_truth()
        drift = (torch.norm(gp.A - A_ref) / torch.norm(A_ref)).item()
        sys_per_sec = T / (incr_ms / 1e3)
        speedup = refact_ms / incr_ms if incr_ms else None
        final_regret = gp.regret_history[-1]
        scaling.append({
            "T": T, "incr_ms": incr_ms, "refact_ms": refact_ms,
            "speedup": speedup, "sys_per_sec": sys_per_sec,
            "symmetry_gap": gp.symmetry_gap(), "A_drift": drift,
            "final_regret": final_regret,
        })
        print(f"{T:>12} {incr_ms:>11.3f} {refact_ms:>13.3f} "
              f"{(('%.2fx'%speedup) if speedup else 'n/a'):>9} {sys_per_sec:>10.0f} "
              f"{gp.symmetry_gap():>8.3f} {drift:>9.1e} {final_regret:>8.4f}")

    # Regret curve for a representative T.
    T_rep = 256 if device.type == "cuda" else 32
    gp = BatchedDynamicsThompson(r=r, T=T_rep, dtype=dtype, device=device, seed=1)
    regret_curve = [gp.step() for _ in range(rounds)]
    print(f"\nRegret curve (T={T_rep}): "
          + "  ".join(f"{c:.3f}" for c in regret_curve[:: max(1, rounds // 10)]))
    print("\nNote: A is non-symmetric (sym gap ≈ O(1)), so no root A=RR^⊤ exists —")
    print("the one-sided root-transport baseline is mathematically inapplicable here.")

    results = {
        "kind": "dynamics_thompson",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "feature_r": r, "rounds": rounds,
        "scaling": scaling,
        "regret_curve": {"T": T_rep, "values": regret_curve},
    }
    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / "dynamics_thompson.json"
        path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {path}")
    return results


if __name__ == "__main__":
    run()
