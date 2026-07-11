"""Benchmark: adaptive-inducing full-covariance SVGP-PDTS (double-sided flagship).

Maintains the whitened variational covariance A_t = L^{-1} S_t L^{-⊤} for T
independent SVGP tasks and, each round, (1) transports A_t under an inducing-
basis change, (2) does a covariance-form Kalman conditioning, and (3) a
trust-region damping — all reading/modifying the full A_t. Compares:

  * incremental — double-sided transport at O(r^2) (our primitive);
  * refactor    — recompute L=chol(G), A=L^{-1}S L^{-⊤} each round at O(r^3).

Reports per-round latency, tasks/sec throughput, A-drift vs a full re-cholesky
(machine precision), and mean simple regret. Run directly on any GPU:

    python -m tests.bench_svgp_pdts
"""

import os
import sys
import json
import time
import datetime
import pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.svgp_pdts import BatchedAdaptiveSVGP

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"


def objective(X):
    return torch.sin(6 * X[:, 0]) * torch.cos(6 * X[:, 1]) + 0.3 * X[:, 0]


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_rounds(gp, rounds):
    _sync(gp.device)
    t0 = time.perf_counter()
    for _ in range(rounds):
        gp.step()
    _sync(gp.device)
    return (time.perf_counter() - t0) / rounds * 1e3


def run(save=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    r = 128
    rounds = 30
    T_sizes = [16, 64, 256, 1024] if device.type == "cuda" else [8, 32]

    torch.manual_seed(0)
    X = torch.rand(r, 2)

    print(f"device: {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"inducing r={r}  rounds={rounds}\n")
    print(f"{'T (tasks)':>10} {'incr ms/rd':>11} {'refact ms/rd':>13} {'speedup':>9} "
          f"{'tasks/sec':>10} {'A-drift':>9} {'regret':>8}")
    print("-" * 80)

    scaling = []
    for T in T_sizes:
        gi = BatchedAdaptiveSVGP(X, objective, T=T, transport_mode="incremental",
                                 dtype=dtype, device=device, seed=0)
        gi.step()  # warmup
        incr_ms = time_rounds(gi, rounds)
        _, A_ref = gi.refactor_ground_truth()
        drift = (torch.norm(gi.A - A_ref) / torch.norm(A_ref)).item()
        final_regret = gi.regret_history[-1]

        gr = BatchedAdaptiveSVGP(X, objective, T=T, transport_mode="refactor",
                                 dtype=dtype, device=device, seed=0)
        gr.step()
        refact_ms = time_rounds(gr, rounds)

        speedup = refact_ms / incr_ms if incr_ms else None
        tasks_per_sec = T / (incr_ms / 1e3)
        scaling.append({
            "T": T, "incr_ms": incr_ms, "refact_ms": refact_ms,
            "speedup": speedup, "tasks_per_sec": tasks_per_sec,
            "A_drift": drift, "final_regret": final_regret,
        })
        print(f"{T:>10} {incr_ms:>11.3f} {refact_ms:>13.3f} "
              f"{(('%.2fx'%speedup) if speedup else 'n/a'):>9} {tasks_per_sec:>10.0f} "
              f"{drift:>9.1e} {final_regret:>8.4f}")

    T_rep = 256 if device.type == "cuda" else 32
    gp = BatchedAdaptiveSVGP(X, objective, T=T_rep, dtype=dtype, device=device, seed=1)
    regret_curve = [gp.step() for _ in range(rounds)]
    print(f"\nRegret curve (T={T_rep}): "
          + "  ".join(f"{c:.3f}" for c in regret_curve[:: max(1, rounds // 10)]))
    print("\nA_t is the authoritative variational state: conditioning and trust-region")
    print("damping read the full covariance, so a sampling root R (A=RRᵀ) is not a")
    print("sufficient representation — the damping step would force an O(r^3) rebuild.")

    results = {
        "kind": "svgp_pdts",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "inducing_r": r, "rounds": rounds,
        "scaling": scaling,
        "regret_curve": {"T": T_rep, "values": regret_curve},
    }
    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / "svgp_pdts.json"
        path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {path}")
    return results


if __name__ == "__main__":
    run()
