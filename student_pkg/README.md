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

## Linear-attention experiment (SKA vs PDN)

The `experiments/` directory at the repo root contains a self-contained
framework for replicating the Preconditioned DeltaNet (Tumma et al.,
arXiv:2604.21100) S-NIAH long-context benchmark using this kernel as
the exact-preconditioning recurrence in place of PDN's diagonal
approximation. See `experiments/README.md` for the math contract,
`experiments/PLAN.md` for the one-week schedule and budget, and
`experiments/ska_recurrence.py` for the autograd.Function wrapper with
hand-derived closed-form backward through the double-sided Cholesky
update.

Quick start on Stanford Farmshare (run from the repo root):

```bash
python experiments/ska_recurrence.py        # local correctness + gradcheck
bash   experiments/setup_farmshare.sh       # one-shot login-node setup
sbatch experiments/run_train_sanity.slurm   # 165M, 100M tokens, 1 L40S
bash   experiments/submit_chain.sh prod 2   # 340M, 5B tokens, 4 L40S chained
sbatch experiments/run_eval_sniah.slurm     # S-NIAH at 5 ctx lengths
```

The legacy RunPod scripts (`setup_runpod.sh`, `launch_training.sh`,
`launch_eval.sh`) still work if you want to rent paid GPUs instead.

## Conventions

The bare kernel modifies L in place and is intended to be called under
`torch.no_grad()` for the GP / BO / NG applications in the paper. For
the linear-attention experiment in `experiments/`, the kernel is wrapped
in a `torch.autograd.Function` (see `experiments/ska_recurrence.py`)
with a hand-derived closed-form backward, so it is differentiable in
that context. Matrices are row-major contiguous. Float32 is the
default; float64 is supported and used for correctness tests.

**Contiguity requirement.** `torch.linalg.cholesky` returns a column-major
tensor (LAPACK convention, strides `(1, r)`). The C++ extension assumes
row-major input: if a non-contiguous tensor is passed, `L.contiguous()`
inside the wrapper silently allocates a temporary copy, modifies that copy,
and the caller's tensor is never updated. Always call `.contiguous()` once
on the initial Cholesky factor before storing it (e.g. as an optimizer
buffer). After that, `fast_incremental_update` keeps L row-major in-place
and no further copies are made.
