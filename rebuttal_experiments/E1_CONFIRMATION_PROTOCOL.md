# E1 paired confirmation protocol

## Purpose

This campaign replaces the old E1 wall-clock comparison with a paired,
source-locked confirmation experiment.  It tests whether the optimized exact
positive Cholesky update improves *application-level* natural-gradient
training time relative to fresh refactorization while preserving the intended
FP32 computation and held-out accuracy.

The isolated kernel comparison remains E6.  E1 includes forward/backward,
chart projection and refresh, preconditioning, momentum, parameter updates,
and Ours' scheduled numerical resets.  Its application timing must not be
reported as isolated kernel timing.

## Frozen design

- Data: `Cholesky-main-2/data/protein.mat`, fixed randomized 80/20 split with
  split seed 42, with normalization fitted only on the training partition.
- Model: `9-512-512-128-1`; only the first layer is preconditioned.
- Ranks: 256, 512, and 1024.
- Refresh threshold: `tau=0.1`.
- Training: ten epochs, batch size 256, learning rate 0.1, damping 1.0,
  momentum 0.9, and an exact Ours reset every 50 steps.
- Precision: CPU FP32 for both methods.
- Thread modes: equal 1-thread and equal 10-thread runs.  Each array task
  reserves ten physical CPU slots, launches one worker, pins that worker to
  exactly the requested number of distinct allocated cores on one socket,
  and checks the live ATen, OpenMP, and MKL thread counts.
- Pairing: one task runs both methods sequentially from byte-identical model,
  curvature, velocity, and minibatch-permutation state.  Method order is
  parity-counterbalanced.  Each rank/thread confirmation cell has five
  Ours-first and five Refact-first pairs.
- Pilots: seeds 1000--1002, 18 paired tasks.  Pilots are operational checks,
  never evaluate the test set, and cannot alter the already-frozen
  configuration.
- Confirmation: untouched seeds 2000--2009, 60 paired tasks.  Each method
  evaluates test MSE exactly once, after all training, with no test-based
  selection.

Ours calls `fast_incremental_update` with explicit `metric_sign=1` and
`operator_sign=1`.  In the locked corrected implementation this dispatches
to `positive_update_impl`, the row-major, virtual-padding, contiguous SIMD
path (AVX-512 when supported).  Refact performs `G += aa^T`, a fresh
Cholesky factorization, complete rewhitening, and two triangular solves at
each non-refresh step.

## Measured regions and retained evidence

Every training step stores raw samples for batch fetch, forward/backward,
projection and refresh decision, complete refresh work, non-refresh core
work, numerical reset, natural-gradient direction construction, parameter
update, total step time, and unaccounted Python/loop time.  Data loading,
paired initialization, warm-up, state cloning, and final train/test
evaluation are recorded separately from training.
The predeclared application time is the sum of complete per-step intervals;
the enclosing instrumented-loop wall time is also retained as a harness
diagnostic.

Each artifact also contains:

- source, binary, data, driver, and Slurm-script SHA-256 hashes;
- the exact git commit and complete source lock;
- requested and runtime-observed ATen/OpenMP/MKL thread counts;
- process CPU affinity and Slurm allocation metadata;
- initial-state and permutation hashes shared by the pair;
- every refresh decision, reset drift, final metric, and final-state hash;
- a same-rank FP32 positive-kernel check against fresh factorization.

Long nonlinear FP32 trajectories need not be bit-identical.  The campaign
therefore reports refresh-trace mismatches, final model/output relative
differences, and paired MSE differences without filtering seeds or rerunning
on the basis of confirmation outcomes.

## Immutable workflow

The harness and all locked sources must first be committed and clean.  The
binary is not required to be tracked, but its exact bytes are locked.

```bash
PYTHONPATH="$PWD/Cholesky-main-2/corrected_incremental_cholesky:$PWD:$PWD/rebuttal_deps" \
  /usr/bin/python3 rebuttal_experiments/e1_confirmation.py \
  --action self-test

PYTHONPATH="$PWD/Cholesky-main-2/corrected_incremental_cholesky:$PWD:$PWD/rebuttal_deps" \
  /usr/bin/python3 rebuttal_experiments/e1_confirmation.py \
  --action freeze-lock
```

Commit and push the newly written
`rebuttal_results/e1_confirmation/e1-confirm-fastpositive-v1/source_lock.json`
before any pilot.  Runtime verification requires that immutable lock to be
tracked and clean.  The lock commit may be an ancestor of the run commit, so
unrelated E3, E6, or black-hole work can continue in parallel; every locked
file must still match its exact locked bytes.  Each run records both commits.

After review, submit the pilot array:

```bash
sbatch rebuttal_experiments/slurm/e1_confirmation_pilot.sbatch
```

Only after all 18 pilots finish:

```bash
PYTHONPATH="$PWD/Cholesky-main-2/corrected_incremental_cholesky:$PWD:$PWD/rebuttal_deps" \
  /usr/bin/python3 rebuttal_experiments/e1_confirmation.py \
  --action summarize-pilot

PYTHONPATH="$PWD/Cholesky-main-2/corrected_incremental_cholesky:$PWD:$PWD/rebuttal_deps" \
  /usr/bin/python3 rebuttal_experiments/e1_confirmation.py \
  --action approve-pilot --reviewer "NAME" \
  --note "Reviewed all pilot diagnostics; proceed unchanged."
```

The confirmation script refuses to run without that write-once approval:

```bash
sbatch rebuttal_experiments/slurm/e1_confirmation.sbatch
```

After all 60 confirmation tasks finish:

```bash
PYTHONPATH="$PWD/Cholesky-main-2/corrected_incremental_cholesky:$PWD:$PWD/rebuttal_deps" \
  /usr/bin/python3 rebuttal_experiments/e1_confirmation.py \
  --action aggregate-confirmation
```

Every JSON publication is atomic and write-once.  A partial campaign may be
resumed by submitting only missing array indices; completed artifacts cannot
be replaced.
