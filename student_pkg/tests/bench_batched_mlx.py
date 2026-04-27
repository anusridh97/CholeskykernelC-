"""
MLX/Metal benchmark: T sequential single calls vs one batched call.

Mirror of tests/bench_batched.py for the MLX path. Sweeps over batch size
T and matrix size r to show where the batched kernel wins.

Run from student_pkg:

    python -m tests.bench_batched_mlx

Requires Apple Silicon + macOS + mlx. Each cell is the median of REPS timed
runs after WARMUP runs. Times are in milliseconds; speedup is sequential /
batched.

Output is mirrored to tests/logs/bench_batched_mlx_<timestamp>.log.
"""

import datetime
import os
import pathlib
import sys
import time

import numpy as np

try:
    import mlx.core as mx
except ImportError:
    print("MLX not available (this script needs Apple Silicon + macOS).")
    sys.exit(0)

from doublesided_cholesky_mlx import (
    incremental_update_mlx,
    incremental_update_mlx_batched,
    _HAS_FAST_CPP,
)

# Optional CPU baseline (fast_update.cpp built as fast_cholesky). When
# available we time the C++ batched path alongside the MLX paths.
if _HAS_FAST_CPP:
    import torch
    import fast_cholesky

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log_file = None


def setup_logging():
    global _log_file
    log_dir = pathlib.Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"bench_batched_mlx_{ts}.log"
    _log_file = open(path, "w")
    log(f"Log: {path}")
    return path


def log(msg=""):
    print(msg, flush=True)
    if _log_file is not None:
        print(msg, file=_log_file, flush=True)


# ---------------------------------------------------------------------------

REPS    = 20
WARMUP  = 3
T_SIZES = [1, 4, 16, 64, 128, 256, 512]
R_SIZES = [16, 64, 128, 256]


def make_batch(T, r, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((T, r, r)).astype(np.float32)
    # Per-batch SPD matrix:  A_t A_t^T + r I
    G = np.einsum("tij,tkj->tik", A, A) + r * np.eye(r, dtype=np.float32)[None]
    L = np.linalg.cholesky(G).astype(np.float32)
    Aw = rng.standard_normal((T, r, r)).astype(np.float32)
    z  = (rng.standard_normal((T, r)) * 0.1).astype(np.float32)
    w  = (rng.standard_normal((T, r)) * 0.1).astype(np.float32)
    return mx.array(L), mx.array(Aw), mx.array(z), mx.array(w)


def time_call(fn, reps, warmup=WARMUP):
    """Median ms over `reps` runs after `warmup` warmups. Forces mx.eval so
    we don't measure graph construction time."""
    for _ in range(warmup):
        out = fn()
        # out is a tuple of mx.arrays; eval each.
        mx.eval(*out)

    times = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        out = fn()
        mx.eval(*out)
        times.append((time.perf_counter_ns() - t0) / 1e6)
    times.sort()
    return times[len(times) // 2]


def time_sequential(L, Aw, z, w, reps):
    T = L.shape[0]

    def call():
        outs_L, outs_Aw, outs_vz = [], [], []
        for t in range(T):
            L_new, Aw_new, vz = incremental_update_mlx(L[t], Aw[t], z[t], w[t])
            outs_L.append(L_new)
            outs_Aw.append(Aw_new)
            outs_vz.append(vz)
        # Stack so we have a single tuple to mx.eval.
        return (mx.stack(outs_L), mx.stack(outs_Aw), mx.stack(outs_vz))

    return time_call(call, reps)


def time_batched(L, Aw, z, w, reps):
    def call():
        return incremental_update_mlx_batched(L, Aw, z, w)

    return time_call(call, reps)


def time_cpp_par(L, Aw, z, w, reps, warmup=WARMUP):
    """Time fast_cholesky.fast_incremental_update_batched (C++ at::parallel_for).
    NOTE: benchmarks consistently show this is SLOWER than time_cpp_loop on
    this hardware — the thread-pool overhead exceeds the gains at our T."""
    L_np  = np.asarray(L,  dtype=np.float32)
    Aw_np = np.asarray(Aw, dtype=np.float32)
    z_np  = np.asarray(z,  dtype=np.float32)
    w_np  = np.asarray(w,  dtype=np.float32)
    Aw_t  = torch.from_numpy(Aw_np)
    w_t   = torch.from_numpy(w_np)

    def call():
        L_t = torch.from_numpy(L_np.copy())
        z_t = torch.from_numpy(z_np.copy())
        return fast_cholesky.fast_incremental_update_batched(L_t, Aw_t, z_t, w_t)

    for _ in range(warmup):
        call()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        call()
        times.append((time.perf_counter_ns() - t0) / 1e6)
    times.sort()
    return times[len(times) // 2]


def time_cpp_loop(L, Aw, z, w, reps, warmup=WARMUP):
    """Time T sequential calls to fast_cholesky.fast_incremental_update.
    Faster than time_cpp_par on benchmarks (no at::parallel_for overhead)
    and matches the path used by incremental_update_reference_batched."""
    L_np  = np.asarray(L,  dtype=np.float32)
    Aw_np = np.asarray(Aw, dtype=np.float32)
    z_np  = np.asarray(z,  dtype=np.float32)
    w_np  = np.asarray(w,  dtype=np.float32)
    T = L_np.shape[0]

    def call():
        L_buf = L_np.copy()
        z_buf = z_np.copy()
        for t in range(T):
            fast_cholesky.fast_incremental_update(
                torch.from_numpy(L_buf[t]),
                torch.from_numpy(Aw_np[t]),
                torch.from_numpy(z_buf[t]),
                torch.from_numpy(w_np[t]),
            )

    for _ in range(warmup):
        call()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        call()
        times.append((time.perf_counter_ns() - t0) / 1e6)
    times.sort()
    return times[len(times) // 2]


def run_sweep():
    log(f"MLX:           {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")
    log(f"fast_cholesky: {'available' if _HAS_FAST_CPP else 'not built'}")
    log(f"Started:       {datetime.datetime.now().isoformat(timespec='seconds')}")
    log(f"REPS={REPS}  WARMUP={WARMUP}")
    log()
    if _HAS_FAST_CPP:
        log(f"{'r':>6} {'T':>6}  {'mlx_seq':>9}  {'mlx_bat':>9}  "
            f"{'cpp_loop':>9}  {'cpp_par':>9}  {'mlx vs cpp_loop':>16}")
        log("-" * 78)
    else:
        log(f"{'r':>6} {'T':>6}  {'mlx_seq (ms)':>13}  {'mlx_bat (ms)':>13}  {'bat/seq':>8}")
        log("-" * 56)

    for r in R_SIZES:
        for T in T_SIZES:
            L, Aw, z, w = make_batch(T, r)
            seq_ms = time_sequential(L, Aw, z, w, REPS)
            bat_ms = time_batched(L, Aw, z, w, REPS)
            if _HAS_FAST_CPP:
                cpp_loop_ms = time_cpp_loop(L, Aw, z, w, REPS)
                cpp_par_ms  = time_cpp_par(L, Aw, z, w, REPS)
                mlx_vs_loop = cpp_loop_ms / bat_ms if bat_ms > 0 else float("inf")
                log(f"{r:>6} {T:>6}  {seq_ms:>9.3f}  {bat_ms:>9.3f}  "
                    f"{cpp_loop_ms:>9.3f}  {cpp_par_ms:>9.3f}  "
                    f"{mlx_vs_loop:>15.2f}x")
            else:
                seq_speedup = seq_ms / bat_ms if bat_ms > 0 else float("inf")
                log(f"{r:>6} {T:>6}  {seq_ms:>13.3f}  {bat_ms:>13.3f}  "
                    f"{seq_speedup:>7.2f}x")
        log()


if __name__ == "__main__":
    log_path = setup_logging()
    try:
        run_sweep()
    finally:
        if _log_file is not None:
            _log_file.close()
        print(f"Log saved to {log_path}")
