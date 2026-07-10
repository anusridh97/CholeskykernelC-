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

On FarmShare, once you have created a venv using uv, run `source .venv/bin/activate`.

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

Expected output depends heavily on the CPU, PyTorch/LAPACK build, and
whether the Cholesky factor is stored row-major as described below. On
FarmShare with the current PyTorch environment, the single-update CPU
extension is roughly 5x faster than refact at r = 128 and roughly 8x
faster at r = 1024. A 100x result at r = 128 is not expected with this
benchmark when the PyTorch refact baseline is already sub-millisecond.

## What to work on

See TODO.md. The two easiest wins are (1) removing the column-major
transpose copy in Phase 1 of `fast_update.cpp`, and (2) inlining the
triangular solve in Phase 4 instead of calling LAPACK. Both are small
edits with measurable impact. The Triton right-sweep rewrite is the
larger project and is described inside `doublesided_cholesky.py`.

## GPU application experiments

Two GPU experiments live in `experiments/`, each built on the batched Triton
kernel. Both lean into the kernel's real strength — device-residency plus
batching *many independent* whitened-operator updates into one launch — rather
than raw speed on a single latency-bound Givens sweep.

**1. Incremental curvature optimizer** (`experiments/curvature_optimizer.py`).
Turns the §3.5 natural-gradient chart into a working optimizer. A deep net gives
one low-rank whitened-Fisher chart *per parameter block*, so one training step is
dozens of independent rank-1 whitened-curvature updates — the T-batch the kernel
wants. Every step refreshes all charts in one batched launch (O(r^2) each),
versus the amortized O(r^3) root/eigendecomposition Shampoo/SOAP/K-FAC pay
periodically. The `periodic_refact` mode reproduces exactly that amortized
baseline (re-cholesky every K steps, stale factor between) for a head-to-head
comparison — the Table-4 drift/core-update story, on GPU. Benchmark against
AdamW and the ablation:

```
python -m tests.bench_curvature_opt        # or: sbatch run_curvature_opt_bench.slurm
```

Blocks are bucketed by rank so each launch stays dense; ragged per-block ranks
and the fact that a single chart's sweep is latency-bound are documented in the
module. This competes in the "explicit low-rank Fisher chart" family — it is not
a drop-in for Shampoo's Kronecker mechanics or Muon's Newton-Schulz step.

**2. Massively parallel GP Thompson sampling** (`experiments/parallel_gp_thompson.py`).
Hundreds to thousands of independent GP surrogates (bandit arms / trust regions /
BO seeds) on one GPU, each taking a rank-1 conditioning update per round, all in
one launch — the purest expression of "T is where we win", scaling past the MLX
section's T = 16. The idiomatic BoTorch/GPyTorch path conditions one model object
at a time at O(r^3); here every arm shares a launch at O(r^2). Sweeps T and prints
the mean simple-regret curve to confirm the arms actually optimize:

```
python -m tests.bench_gp_thompson          # or: sbatch run_gp_thompson_bench.slurm
```

Both modules fall back to the pure-PyTorch reference refresh when CUDA/Triton is
absent, so the loops and their correctness checks run on CPU too:

```
python -m pytest tests/test_experiments.py -v
```

## Conventions

Run everything under `torch.no_grad()`. The update modifies L in place.
Matrices are row-major contiguous. Float32 is the default; float64 is
supported and used for correctness tests.

**Contiguity requirement.** `torch.linalg.cholesky` returns a column-major
tensor (LAPACK convention, strides `(1, r)`). The C++ extension assumes
row-major input: if a non-contiguous tensor is passed, `L.contiguous()`
inside the wrapper silently allocates a temporary copy, modifies that copy,
and the caller's tensor is never updated. Always call `.contiguous()` once
on the initial Cholesky factor before storing it (e.g. as an optimizer
buffer). After that, `fast_incremental_update` keeps L row-major in-place
and no further copies are made.
