"""
Microbenchmark harness.

Run with:
    python -m tests.bench_timing

Prints one row per r, showing median wall time for the fast path vs the
ground-truth re-factorization. Use this to measure whether a change you
made speeds things up or slows them down. Always keep single-threaded
BLAS while benchmarking so results are interpretable.

Running under 'taskset -c 0' and with OMP_NUM_THREADS=1, MKL_NUM_THREADS=1,
OPENBLAS_NUM_THREADS=1 in the environment is recommended.
"""
import time
import numpy as np
import torch


def bench_one(r, n_repeats, use_fast_cholesky):
    torch.manual_seed(42)
    A = torch.randn(r, r)
    G = A @ A.T + r * torch.eye(r)
    L = torch.linalg.cholesky(G)
    M = torch.randn(r, r)
    Linv = torch.linalg.inv(L)
    Aw = Linv @ M @ Linv.T
    z = torch.randn(r) * 0.1
    w = torch.randn(r) * 0.1

    if use_fast_cholesky:
        import fast_cholesky
        # Warm up.
        for _ in range(5):
            fast_cholesky.fast_incremental_update(L.clone(), Aw, z.clone(), w)

    times_fast = []
    times_refact = []
    for _ in range(n_repeats):
        if use_fast_cholesky:
            t0 = time.perf_counter()
            fast_cholesky.fast_incremental_update(L.clone(), Aw, z.clone(), w)
            times_fast.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        K = G + torch.outer(z, z)
        M2 = M + torch.outer(z, w)
        L2 = torch.linalg.cholesky(K)
        Li = torch.linalg.inv(L2)
        _ = Li @ M2 @ Li.T
        times_refact.append(time.perf_counter() - t0)

    med_refact = float(np.median(times_refact) * 1000)
    if use_fast_cholesky:
        med_fast = float(np.median(times_fast) * 1000)
        return med_fast, med_refact
    return None, med_refact


def main():
    torch.set_num_threads(1)
    try:
        import fast_cholesky  # noqa: F401
        have_fast = True
    except ImportError:
        have_fast = False
        print("NOTE: fast_cholesky not built, only showing refact timings")

    print(f"{'r':>6}  {'Ours (ms)':>12}  {'Refact (ms)':>12}  {'Speedup':>9}")
    print("-" * 46)

    for r in [32, 64, 128, 256, 512, 1024]:
        n = 100 if r <= 256 else 20
        med_fast, med_refact = bench_one(r, n, have_fast)
        if med_fast is not None:
            speedup = med_refact / med_fast
            print(f"{r:>6}  {med_fast:>12.3f}  {med_refact:>12.3f}  {speedup:>8.1f}x")
        else:
            print(f"{r:>6}  {'(n/a)':>12}  {med_refact:>12.3f}  {'--':>9}")


if __name__ == "__main__":
    main()
