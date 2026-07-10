"""Benchmark: massively parallel GP Thompson sampling vs BoTorch-style baselines.

Runs many independent GP arms over a shared candidate set and, each round,
conditions every arm on one new observation. Compares per-round conditioning:

  * batched        — one fused kernel launch for all T arms (our path), scaled
                     to thousands of arms.
  * sequential     — T single-arm rank-1 Cholesky updates in a loop (our kernel,
                     one arm at a time).
  * botorch-style  — one exact-GP model per arm, re-conditioned from its own
                     observations each round at O(n^3), looped over arms. This is
                     what the idiomatic BoTorch/GPyTorch path does and it is
                     exactly what has a hard time here: T independent cubic
                     conditionings with no shared launch. Uses real BoTorch if
                     installed, otherwise an equivalent exact-GP implementation.

Reports ms/round, batched-vs-baseline speedup, total-arm throughput (arms/sec),
and the mean simple-regret curve. Returns a JSON-serializable results dict
(written to results/) that ``experiments.viz`` turns into figures.

Run directly on any GPU — no scheduler required:

    python -m tests.bench_gp_thompson
"""

import os
import sys
import json
import time
import datetime
import pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.parallel_gp_thompson import BatchedGPThompson, rbf_kernel

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"


def make_candidates(r, device, dtype, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(r, 2, generator=g).to(device, dtype)


def objective(X):
    return torch.sin(6 * X[:, 0]) * torch.cos(6 * X[:, 1]) + 0.3 * X[:, 0]


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_batched(gp, rounds):
    _sync(gp.device)
    t0 = time.perf_counter()
    for _ in range(rounds):
        gp.step()
    _sync(gp.device)
    return (time.perf_counter() - t0) / rounds * 1e3


def time_sequential(X, T, rounds, device, dtype):
    """Per-arm sequential conditioning: T single-arm rank-1 updates per round."""
    from doublesided_cholesky import incremental_update_reference

    gp = BatchedGPThompson(X, objective, T=T, force_reference=True)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(rounds):
        f = gp.thompson_sample()
        idx = f.argmax(dim=1)
        z = torch.zeros(T, gp.r, device=device, dtype=dtype)
        z.scatter_(1, idx.unsqueeze(1), 1.0 / gp.noise)
        w = torch.zeros_like(z)
        for t in range(T):
            incremental_update_reference(gp.L[t], gp.Aw[t], z[t].clone(), w[t].clone())
    _sync(device)
    return (time.perf_counter() - t0) / rounds * 1e3


# --------------------------------------------------------------------------- #
# BoTorch-style baseline: one exact GP model per arm, re-conditioned per round.
# --------------------------------------------------------------------------- #


def _has_botorch():
    try:
        import botorch  # noqa: F401
        import gpytorch  # noqa: F401
        return True
    except Exception:
        return False


def time_botorch_style(X, T, rounds, device, dtype, lengthscale=0.1, noise=0.1):
    """Time the idiomatic per-model path: T independent exact GPs, cubic conditioning.

    Uses real BoTorch SingleTaskGP posteriors when available; otherwise an
    equivalent fixed-hyperparameter exact-GP conditioning (same O(n^3) work,
    same per-arm sequential structure). Returns (ms_per_round, backend_name).
    """
    r = X.shape[0]
    f_true = objective(X)
    gen = torch.Generator(device="cpu").manual_seed(0)

    use_bo = _has_botorch()
    if use_bo:
        from botorch.models import SingleTaskGP
        from gpytorch.mlls import ExactMarginalLogLikelihood  # noqa: F401

    # Per-arm observation stores.
    obs_idx = [[] for _ in range(T)]
    obs_y = [[] for _ in range(T)]
    K = rbf_kernel(X, lengthscale)

    def exact_gp_posterior_sample(idxs, ys):
        """Fixed-hyperparameter exact GP posterior draw over all candidates."""
        if not idxs:
            L = torch.linalg.cholesky(K)
            xi = torch.randn(r, generator=gen).to(device, dtype)
            return (L @ xi)
        n = len(idxs)
        ii = torch.tensor(idxs, device=device)
        Knn = K[ii][:, ii] + noise ** 2 * torch.eye(n, device=device, dtype=dtype)
        Ksn = K[:, ii]  # (r, n)
        yv = torch.tensor(ys, device=device, dtype=dtype)
        Ln = torch.linalg.cholesky(Knn)
        alpha = torch.cholesky_solve(yv.unsqueeze(1), Ln)  # (n,1)
        mu = (Ksn @ alpha).squeeze(1)
        v = torch.linalg.solve_triangular(Ln, Ksn.mT, upper=False)  # (n, r)
        cov = K - v.mT @ v
        cov = cov + 1e-4 * torch.eye(r, device=device, dtype=dtype)
        Ls = torch.linalg.cholesky(cov)
        xi = torch.randn(r, generator=gen).to(device, dtype)
        return mu + Ls @ xi

    _sync(device)
    t0 = time.perf_counter()
    for _ in range(rounds):
        for t in range(T):
            if use_bo and obs_idx[t]:
                trX = X[torch.tensor(obs_idx[t], device=device)]
                trY = torch.tensor(obs_y[t], device=device, dtype=dtype).unsqueeze(1)
                model = SingleTaskGP(trX, trY)
                post = model.posterior(X)
                sample = post.rsample(torch.Size([1])).reshape(-1)
            else:
                sample = exact_gp_posterior_sample(obs_idx[t], obs_y[t])
            i = int(sample.argmax())
            y = float(f_true[i] + noise * torch.randn(1, generator=gen).item())
            obs_idx[t].append(i)
            obs_y[t].append(y)
    _sync(device)
    backend = "botorch" if use_bo else "exact-gp"
    return (time.perf_counter() - t0) / rounds * 1e3, backend


# --------------------------------------------------------------------------- #


def run(save=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    r = 64
    rounds = 30
    # Massively parallel: scale to thousands of arms on GPU.
    T_sizes = [16, 64, 256, 1024, 4096, 16384] if device.type == "cuda" else [4, 16, 64]
    seq_max = 256      # sequential (python-loop) comparison only up to here
    botorch_max = 64   # per-model baseline is slow; cap it

    print(f"device: {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"candidates r={r}  rounds={rounds}  botorch_backend={'botorch' if _has_botorch() else 'exact-gp'}\n")
    print(f"{'T (arms)':>9} {'batched ms':>11} {'seq ms':>9} {'botorch ms':>11} "
          f"{'speedup':>9} {'arms/sec':>12} {'regret':>8}")
    print("-" * 82)

    X = make_candidates(r, device, dtype)
    scaling = []
    for T in T_sizes:
        gp = BatchedGPThompson(X, objective, T=T)
        gp.step()  # warmup
        bat_ms = time_batched(gp, rounds)
        final_regret = gp.regret_history[-1]
        arms_per_sec = T / (bat_ms / 1e3)

        seq_ms = time_sequential(X, T, rounds, device, dtype) if T <= seq_max else None
        bo_ms, backend = (time_botorch_style(X, T, rounds, device, dtype)
                          if T <= botorch_max else (None, None))

        base = bo_ms if bo_ms is not None else seq_ms
        speedup = base / bat_ms if base else None
        scaling.append({
            "T": T, "batched_ms": bat_ms, "seq_ms": seq_ms, "botorch_ms": bo_ms,
            "botorch_backend": backend, "speedup_vs_baseline": speedup,
            "arms_per_sec": arms_per_sec, "final_regret": final_regret,
        })
        print(f"{T:>9} {bat_ms:>11.3f} "
              f"{('%.3f' % seq_ms) if seq_ms else 'n/a':>9} "
              f"{('%.3f' % bo_ms) if bo_ms else 'n/a':>11} "
              f"{('%.1fx' % speedup) if speedup else 'n/a':>9} "
              f"{arms_per_sec:>12.0f} {final_regret:>8.4f}")

    # Regret curve for a representative arm count.
    T_rep = 256 if device.type == "cuda" else 16
    gp = BatchedGPThompson(X, objective, T=T_rep)
    regret_curve = [gp.step() for _ in range(rounds)]
    print(f"\nRegret curve (T={T_rep}): "
          + "  ".join(f"{c:.3f}" for c in regret_curve[:: max(1, rounds // 10)]))

    results = {
        "kind": "gp_thompson",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "candidates_r": r, "rounds": rounds,
        "botorch_backend": "botorch" if _has_botorch() else "exact-gp",
        "scaling": scaling,
        "regret_curve": {"T": T_rep, "values": regret_curve},
    }
    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / "gp_thompson.json"
        path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {path}")
    return results


if __name__ == "__main__":
    run()
