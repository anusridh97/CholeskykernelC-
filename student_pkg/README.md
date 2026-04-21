# Double-Sided Cholesky Propagation

An O(r^2) incremental update for the whitened operator A_w = L^{-1} M L^{-T}
when L comes from a symmetric positive definite Gram matrix G and M is an
associated square matrix, both receiving a rank-1 update.

Given inputs L, A_w, z, w, the update produces L+ and A_w+ such that

    G+ = G + z z^T,   M+ = M + z w^T,   A_w+ = (L+)^{-1} M+ (L+)^{-T}

in O(r^2) time, versus O(r^3) for a full re-factorization.

## What is in this package

`doublesided_cholesky.py` has the pure-PyTorch reference implementation
(slow, used for correctness checking) and the fused Triton GPU kernel.

`fast_update.cpp` is the CPU extension used for benchmarking. Build it
with `setup_ext.py`.

`tests/test_correctness.py` checks that each implementation matches a
full re-factorization to floating-point precision.

`tests/bench_timing.py` measures update time against the refact baseline.

`TODO.md` lists the open optimization problems, ranked by difficulty.

## Setup

```
pip install -r requirements.txt
python setup_ext.py build_ext --inplace
```

The second command produces `fast_cholesky.*.so` in the current directory.
Keep it next to the `.py` files so `import fast_cholesky` works.

## Running tests

```
python -m pytest tests/test_correctness.py -v
```

The C++ tests skip automatically if the extension is not built, and the
Triton tests skip if CUDA is not available.

## Running the benchmark

Always pin single-threaded BLAS before benchmarking, otherwise the refact
baseline starts using multiple cores and the numbers are not comparable:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -m tests.bench_timing
```

Expected output on a reasonable CPU, float32, r = 128: the fast path is
around 100x faster than refact. The speedup grows with r.

## What to work on

See TODO.md. The two easiest wins are (1) removing the column-major
transpose copy in Phase 1 of `fast_update.cpp`, and (2) inlining the
triangular solve in Phase 4 instead of calling LAPACK. Both are small
edits with measurable impact. The Triton right-sweep rewrite is the
larger project and is described inside `doublesided_cholesky.py`.

## Conventions

Run everything under `torch.no_grad()`. The update modifies L in place.
Matrices are row-major contiguous. Float32 is the default; float64 is
supported and used for correctness tests.
