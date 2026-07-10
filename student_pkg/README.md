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

Two GPU experiments live in `experiments/`, both faithful to the paper's math
and built on the batched Triton kernel. The GPU angle they add on top of the
paper's CPU experiments is batching *many independent* whitened-operator updates
into one launch — device-residency, not raw speed on a single latency-bound
Givens sweep. Run everything directly (no scheduler) and auto-generate figures:

```
python run_experiments.py            # both experiments → results/ JSON → figures/ report.html
python run_experiments.py --only gp  # gp | curvature
python run_experiments.py --figures-only
```

**1. Transported-subspace natural gradient optimizer**
(`experiments/curvature_optimizer.py`). A faithful implementation of §3.5: per
parameter block a chart `F ≈ λI + U C Uᵀ` with an **adaptive orthonormal basis
U**, refreshed by an `(r+1)`-eigendecomposition when residual energy
`ρ²/‖g‖² > τ` (Fig. 2). In steady state the double-sided primitive maintains the
whitened core `C_w = I − λ L⁻¹L⁻ᵀ` (with `z = w = Uᵀg`), and preconditioning is
Woodbury `F⁻¹g = λ⁻¹(g − Ua) + U G⁻¹a` using the maintained factor (`G⁻¹a =
L⁻ᵀ·v_z`, and `v_z = L⁻¹a` comes free from the kernel). The GPU contribution:
one chart per block ⇒ a training step is dozens of independent rank-1 core
updates, refreshed in a **single batched launch** (bucketed by rank). The
`refresh_mode` switch gives the paper's ablation on identical charts —
`incremental` (O(r²) primitive) vs `refact` (full re-cholesky every step) vs
`periodic_refact` (every N steps, stale between). Baselines: **SOAP** (headline
second-order), SGD, AdamW.

```
python -m tests.bench_curvature_opt
```

**2. Massively parallel GP Thompson sampling**
(`experiments/parallel_gp_thompson.py`). Faithful whitened-SVGP streaming
conditioning (§4.4/§4.5) using the **same primitive** — `z = w = φ/σ` with
`φ = L_uu⁻¹ k_u(x)` the whitened cross-covariance, not an ad-hoc precision
filter. Thousands of independent GP arms are each conditioned by one rank-1
whitened update per round, all fused into one launch — the "T is where we win"
story, scaled past the MLX section's T = 16. Baseline is a **BoTorch** per-model
loop (real `SingleTaskGP` when installed, else an equivalent exact-GP), which
conditions one model at a time at O(r³); we share a launch at O(r²). Reports
per-round cost vs T, arms/sec throughput, and the simple-regret curve.

```
python -m tests.bench_gp_thompson
```

Both modules fall back to the pure-PyTorch reference refresh when CUDA/Triton is
absent, so the loops and their correctness checks (incremental core matches a
full re-cholesky to machine precision; Woodbury preconditioner is exact) run on
CPU too:

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
