"""Benchmark: batched vs sequential GP Thompson-sampling conditioning.

Runs T independent GP arms over a shared candidate set and, each round,
conditions every arm on one new observation. Compares three ways to do the
per-round rank-1 conditioning across arms:

  * batched      — one fused kernel launch for all T arms (our path).
  * sequential   — T single-arm rank-1 Cholesky updates in a loop (the
                   BoTorch/GPyTorch-style per-model conditioning).
  * refactor     — full re-cholesky of every arm's precision each round
                   (O(r^3) per arm), the naive "recompute the model" baseline.

Sweeps over the number of arms T to show the batched kernel scaling far past
the MLX section's T = 16. Also prints the mean simple-regret curve to confirm
the arms are actually optimizing.

Run from student_pkg/:

    python -m tests.bench_gp_thompson

Runs on CPU (reference refresh, small T) or GPU (batched Triton kernel).
"""

import os
import sys
import time
import datetime
import pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.parallel_gp_thompson import BatchedGPThompson, rbf_kernel  # noqa: F401

_log_file = None


def setup_logging():
    global _log_file
    log_dir = pathlib.Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", f"local_{os.getpid()}")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"bench_gp_thompson_{job_id}_{ts}.log"
    _log_file = open(path, "w")
    log(f"Log: {path}")
    return path


def log(msg=""):
    print(msg, flush=True)
    if _log_file is not None:
        print(msg, file=_log_file, flush=True)


def make_candidates(r, device, dtype, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    X = torch.rand(r, 2, generator=g).to(device, dtype)
    return X


def objective(X):
    # Simple 2-D multi-modal objective on the candidate grid.
    return torch.sin(6 * X[:, 0]) * torch.cos(6 * X[:, 1]) + 0.3 * X[:, 0]


def time_round(gp, rounds):
    if gp.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rounds):
        gp.step()
    if gp.device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rounds * 1e3  # ms/round


def time_sequential_reference(X, T, rounds, device, dtype):
    """Per-arm sequential conditioning: T single-arm rank-1 Cholesky updates."""
    from doublesided_cholesky import incremental_update_reference

    gp = BatchedGPThompson(X, objective, T=T, force_reference=True)
    # Reuse gp's state but drive conditioning one arm at a time.
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rounds):
        f = gp.thompson_sample()
        idx = f.argmax(dim=1)
        z = torch.zeros(T, gp.r, device=device, dtype=dtype)
        z.scatter_(1, idx.unsqueeze(1), 1.0 / gp.noise)
        w = torch.zeros_like(z)
        for t in range(T):
            incremental_update_reference(gp.L[t], gp.Aw[t], z[t].clone(), w[t].clone())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rounds * 1e3


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    log(f"device:       {device}")
    if device.type == "cuda":
        log(f"GPU:          {torch.cuda.get_device_name(0)}")
    log(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'n/a')}")
    log(f"Started:      {datetime.datetime.now().isoformat(timespec='seconds')}")

    r = 64  # candidate-set size per arm
    rounds = 30
    T_sizes = [16, 64, 256, 1024, 4096] if device.type == "cuda" else [4, 16]
    log(f"candidates r={r}  rounds={rounds}")
    log("")

    X = make_candidates(r, device, dtype)

    log(f"{'T (arms)':>10} {'batched ms/rd':>14} {'seq ms/rd':>12} {'speedup':>9} {'final regret':>13}")
    log("-" * 64)
    for T in T_sizes:
        gp = BatchedGPThompson(X, objective, T=T)
        # warmup
        gp.step()
        bat_ms = time_round(gp, rounds)
        final_regret = gp.regret_history[-1]

        # Sequential reference only for modest T (it is O(T) python loop).
        if T <= 256:
            seq_ms = time_sequential_reference(X, T, rounds, device, dtype)
            speedup = f"{seq_ms / bat_ms:7.2f}x" if bat_ms > 0 else "   inf"
            seq_str = f"{seq_ms:12.3f}"
        else:
            seq_str = "         n/a"
            speedup = "      n/a"

        log(f"{T:>10} {bat_ms:>14.3f} {seq_str} {speedup:>9} {final_regret:>13.4f}")

    log("")
    log("Regret curve (T=256, mean simple regret vs round):")
    gp = BatchedGPThompson(X, objective, T=min(256, T_sizes[-1]))
    curve = [gp.step() for _ in range(rounds)]
    log("  " + "  ".join(f"{c:.3f}" for c in curve[:: max(1, rounds // 10)]))


if __name__ == "__main__":
    log_path = setup_logging()
    try:
        run()
    finally:
        if _log_file is not None:
            _log_file.close()
        print(f"Log saved to {log_path}")
