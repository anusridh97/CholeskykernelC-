"""
MLX/Metal microbenchmark: single-update kernel vs full re-factorization.

Mirror of tests/bench_timing.py for the MLX path. Times a single
incremental_update_mlx call against `mx.linalg.cholesky + mx.linalg.inv`
(the equivalent of the PyTorch refact baseline) at a sweep of r.

Run from the student_pkg directory:

    python -m tests.bench_timing_mlx

MLX is lazy. Each timed call ends with mx.eval(...) so the wall time
actually includes kernel execution.
"""

import sys
import time

import numpy as np

try:
    import mlx.core as mx
except ImportError:
    print("MLX not available (this script needs Apple Silicon + macOS).")
    sys.exit(0)

from doublesided_cholesky_mlx import incremental_update_mlx, _HAS_FAST_CPP

# Optional CPU baseline (fast_update.cpp built as fast_cholesky). When
# available we time it alongside the MLX kernel to show how the Metal
# port compares to the AVX-512 CPU path.
if _HAS_FAST_CPP:
    import torch
    import fast_cholesky


def make_problem(r, dtype=mx.float32, seed=42):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((r, r)).astype(np.float32)
    G = (A @ A.T + r * np.eye(r, dtype=np.float32)).astype(np.float32)
    L = np.linalg.cholesky(G).astype(np.float32)
    M = rng.standard_normal((r, r)).astype(np.float32)
    Linv = np.linalg.inv(L).astype(np.float32)
    Aw = (Linv @ M @ Linv.T).astype(np.float32)
    z = (rng.standard_normal(r) * 0.1).astype(np.float32)
    w = (rng.standard_normal(r) * 0.1).astype(np.float32)
    return (
        mx.array(G), mx.array(L), mx.array(M),
        mx.array(Aw), mx.array(z), mx.array(w),
    )


def time_call(fn, reps, warmup=5):
    for _ in range(warmup):
        out = fn()
        mx.eval(*(out if isinstance(out, tuple) else (out,)))

    times = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        out = fn()
        mx.eval(*(out if isinstance(out, tuple) else (out,)))
        times.append((time.perf_counter_ns() - t0) / 1e6)  # ms
    times.sort()
    return times[len(times) // 2]


def refactor_update(G, M, z, w):
    G_new = G + mx.outer(z, z)
    M_new = M + mx.outer(z, w)
    # mx.linalg.{cholesky,inv} are CPU-only in current MLX; matmul stays on GPU.
    L_new = mx.linalg.cholesky(G_new, stream=mx.cpu)
    Linv_new = mx.linalg.inv(L_new, stream=mx.cpu)
    return L_new, Linv_new @ M_new @ Linv_new.T


def time_cpp(L, Aw, z, w, reps, warmup=5):
    """Time fast_cholesky.fast_incremental_update on CPU. Uses time.perf_counter
    since fast_cholesky is synchronous (no lazy graph)."""
    L_np  = np.asarray(L,  dtype=np.float32)
    Aw_np = np.asarray(Aw, dtype=np.float32)
    z_np  = np.asarray(z,  dtype=np.float32)
    w_np  = np.asarray(w,  dtype=np.float32)
    Aw_t  = torch.from_numpy(Aw_np)
    w_t   = torch.from_numpy(w_np)

    def call():
        # Independent buffers each call: fast_incremental_update mutates L and z.
        L_t = torch.from_numpy(L_np.copy())
        z_t = torch.from_numpy(z_np.copy())
        return fast_cholesky.fast_incremental_update(L_t, Aw_t, z_t, w_t)

    for _ in range(warmup):
        call()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        call()
        times.append((time.perf_counter_ns() - t0) / 1e6)
    times.sort()
    return times[len(times) // 2]


def main():
    if _HAS_FAST_CPP:
        print(f"{'r':>6}  {'MLX (ms)':>10}  {'Refact (ms)':>12}  {'C++ (ms)':>10}  "
              f"{'MLX vs Refact':>14}  {'MLX vs C++':>11}")
        print("-" * 76)
    else:
        print(f"{'r':>6}  {'MLX (ms)':>10}  {'Refact (ms)':>12}  {'Speedup':>9}")
        print("-" * 46)
        print("(fast_cholesky not available — build with setup_ext.py for CPU baseline)")

    for r in [32, 64, 128, 256, 512, 1024]:
        G, L, M, Aw, z, w = make_problem(r)

        def ours():
            return incremental_update_mlx(L, Aw, z, w)

        def refact():
            return refactor_update(G, M, z, w)

        n = 100 if r <= 256 else 20
        mlx_ms = time_call(ours, n)
        refact_ms = time_call(refact, n)
        if _HAS_FAST_CPP:
            cpp_ms = time_cpp(L, Aw, z, w, n)
            print(f"{r:>6}  {mlx_ms:>10.3f}  {refact_ms:>12.3f}  {cpp_ms:>10.3f}  "
                  f"{refact_ms / mlx_ms:>13.2f}x  {cpp_ms / mlx_ms:>10.2f}x")
        else:
            speedup = refact_ms / mlx_ms if mlx_ms > 0 else float("inf")
            print(f"{r:>6}  {mlx_ms:>10.3f}  {refact_ms:>12.3f}  {speedup:>8.1f}x")


if __name__ == "__main__":
    main()
