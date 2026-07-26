# E1 refresh-threshold sensitivity confirmation

## Question and estimands

This source-locked campaign asks how the fixed refresh threshold changes the
optimized method's refresh frequency and final accuracy.  It is separate from
the paired E1 method comparison: E1 compares Ours with Refact at
`tau=0.1`, whereas this experiment changes only Ours' refresh policy.

The primary cost axis is the raw refresh fraction
`refresh_count / training_steps`.  It is independent of thread scheduling.
Application time is retained as a descriptive secondary outcome under one
equal, controlled ten-thread setting; it is not an isolated kernel
comparison.  E1 and E6 separately measure application and kernel performance.

## Frozen design

- Data and preprocessing: `Cholesky-main-2/data/protein.mat`, fixed 80/20
  split seed 42, with normalization fitted on training rows only.
- Model and optimizer: FP32 `9-512-512-128-1`; only the first layer is
  preconditioned; ten epochs, batch size 256, learning rate 0.1, damping 1.0,
  momentum 0.9, and an exact reset every 50 steps.
- Ranks: 256 and 1024.
- Thresholds: `0.01, 0.02, 0.05, 0.1, 0.2, 0.5`.
- Confirmation seeds: the previously unobserved seeds 3000--3009.
- Threading: ten threads for every run.  Each seed/rank block receives an
  exclusive one-node allocation, one task/process, ten physical cores on one
  socket, exact CPU pinning, and observed ATen/OpenMP/MKL thread validation.
- Pairing: one task runs all six thresholds from byte-identical initial
  model, basis, metric, factors, whitened operator, and velocity state, using
  byte-identical stored minibatch permutations.  The six forward cyclic
  orders followed by four reverse cyclic orders put every threshold in every
  execution position once or twice.
- Test policy: all six trainings and all pre-test paired invariants must pass
  before held-out tensors are materialized.  Each final model is then
  evaluated on test exactly once.  There is no epochwise test access, early
  stopping, checkpoint selection, threshold selection, omitted cell, or
  outcome-driven rerun.
- Exactly-once policy: each seed/rank task publishes a permanent attempt claim
  before training.  A claimed task is never retried; a failure retires the
  campaign tag.

Both one- and ten-thread sensitivity grids are unnecessary.  Refresh fraction
and fixed-run final accuracy are the scientific estimands, while E1 already
measures both thread modes at the fixed comparison threshold and E6 isolates
the kernel.  Repeating the full sensitivity grid at one thread would duplicate
the algorithmic condition and double compute without resolving a new claim.

## Retained measurements

Every threshold retains all step-level refresh decisions, unexplained energy,
batch loss, total step time, batch fetch, forward/backward, projection and
decision, refresh, non-refresh core, reset, direction, parameter update, and
unaccounted time.  Each block records refresh count/fraction, reset drift,
final train/test MSE, terminal state hashes, threshold order, shared
initial-state and permutation hashes, kernel equivalence, exact source,
binary, data, runtime and Slurm hashes, live thread counts, CPU affinity, and
the exactly-once claim.

The immutable aggregate reports every seed, mean and sample standard
deviation, plus paired threshold-minus-0.1 final-MSE and refresh-fraction
differences with two-sided 95% Student-t intervals.  All cells are reported;
the held-out results do not select a preferred threshold.

## Immutable workflow

Commit the driver, protocol, audit, and Slurm scripts first.  Then:

```bash
sbatch rebuttal_experiments/slurm/e1_tau_freeze.sbatch
```

Audit and commit the write-once `source_lock.json`.  Next:

```bash
sbatch rebuttal_experiments/slurm/e1_tau_preflight.sbatch
```

Audit and commit the write-once `preflight_receipt.json`.  Only after it
passes, submit:

```bash
sbatch rebuttal_experiments/slurm/e1_tau_confirmation.sbatch
```

After all 20 blocks finish exactly once:

```bash
PYTHONPATH="$PWD/Cholesky-main-2/corrected_incremental_cholesky:$PWD:$PWD/rebuttal_deps" \
  /usr/bin/python3 rebuttal_experiments/e1_tau_sensitivity.py \
  --action aggregate-confirmation
```
