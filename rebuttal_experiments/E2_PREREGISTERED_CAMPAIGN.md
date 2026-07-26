# Preregistered E2 positive-update confirmation

This campaign tests the optimized positive same-vector implementation without
reusing or tuning against any completed E2 evaluation outcome.  It is a new
campaign (`e2-positive-preregistered-v2`) with new data-free implementation
work, new Protein pilot seeds, untouched confirmation seeds, and a new
mechanically derived held-out split.

## Why there are two experiments

The two questions are different and are not combined into one budget-accuracy
table.

1. **Same-rank accuracy equivalence.** At rank 1024, Ours and Refact start
   from one shared model/chart initialization, consume identical minibatch
   permutations, and train for exactly ten epochs of full-size, drop-last
   minibatches.  The incomplete tail batch is predeclaredly omitted in both
   methods.  No checkpoint is selected.  Each final model is evaluated on the
   fixed test partition once.
   The primary paired contrast is test MSE(Ours) minus test MSE(Refact).
2. **Time-budget throughput.** A common application-training budget is frozen
   from excluded pilot timing only.  Each method times exactly one batch that
   crosses the budget, then reports only scalar counters snapshotted after the
   preceding under-budget batch.  The crossing batch is excluded, and no
   selected model state or accuracy is materialized.  The primary paired
   contrast is completed batches(Ours) minus completed batches(Refact).  This
   stage never constructs a test tensor or evaluates a test outcome.

Thus a wall-clock cutoff cannot change the state used for the accuracy
comparison, and a test metric cannot affect the throughput comparison.

## Mathematical and implementation identity

On a non-refresh training step, both methods form

\[
G^+ = G + aa^\top,\qquad M^+ = M + aa^\top.
\]

Refact recomputes the Cholesky factor and whitening from these matrices.
Ours applies the positive rank-one recurrence.  Because the metric and
operator updates use the same vector, \(w=z=a\), the two forward-solve
vectors satisfy

\[
v_w=(L^+)^{-1}w=(L^+)^{-1}z=v_z.
\]

The specialization therefore removes a redundant solve without changing the
exact-arithmetic algorithm.  It also uses caller-owned ping-pong storage for
the whitened operator, a caller-owned \(4r\) workspace, contiguous row-major
sweeps, and OpenMP/SIMD parallelism.  Chart refreshes use the same fresh
factorization path in both methods.  Ours is reset from its accumulated
\(G\) every 50 steps, and the maximum pre-reset whitening drift is recorded.

The methods are mathematically identical, not bitwise identical: FP32
reassociation can perturb gradients and can place the two trajectories on
opposite sides of the refresh threshold.  The campaign records refresh
counts and reset drift and does not guarantee that empirical equivalence will
be established.

## Frozen split, preprocessing, and seeds

The Protein `.mat` SHA-256 is frozen in the protocol and lock manifest.
Completed outcomes from the historical seed-42 split had already been viewed,
so this campaign does not reuse it.  The new split seed is mechanically
derived before execution as the first eight bytes of
`SHA256("NeurIPS-29421|e2-positive-preregistered-v2|heldout-test-split")`,
interpreted big-endian and reduced modulo \(2^{31}-1\), giving 381857132.
No v2 held-out outcome was viewed when selecting the split or protocol.  This
permutation gives an 80% development partition and a 20% held-out test
partition.  The five pilots split development 90/10 with seed
314159, fit normalization on inner training only, and may compute one final
inner-validation MSE.  They never index fixed test rows into feature/label
tensors.  Confirmation fits normalization on the full development partition.

| Role | Seeds | Test evaluations |
|---|---:|---:|
| Pilot | 51001--51005 | 0 |
| Fixed-epoch accuracy | 52001--52020 | 1 per method and seed |
| Time-budget throughput | 53001--53020 | 0 |

All three sets are mutually disjoint and disjoint from prior E2 seeds
0--2/1000--1002 and synthetic candidate seeds 6100--6102/7777.

Task ID \(i\) maps only to element \(i\) of the corresponding frozen seed
list.  Even task IDs run Ours then Refact; odd task IDs reverse that order.
The methods run sequentially in one process.

## Frozen time budget

For each pilot, Ours runs ten epochs on the inner-training partition.  The
calibrator computes its application seconds per completed batch.  The budget
is

\[
\operatorname{median}_{s\in\text{five pilots}}
  \frac{T_{\text{Ours},s}}{\text{batches}_{s}}
\times
\text{batches in ten full-development epochs}.
\]

Validation values are present only for implementation diagnosis and are not
read by the calibration rule.  Any code or protocol change after a pilot
artifact exists abandons this campaign tag and requires a new preregistration.

## Frozen analysis

Accuracy reports all 20 paired values, the mean paired difference, sample
standard deviation, and Student-\(t\) 95% confidence interval.  A formal
two-one-sided 5% equivalence decision uses the paired 90% interval and the
predeclared absolute normalized-MSE margin \(\pm0.01\), one percent of a
standardized target-variance unit.  This is not a superiority test.

Throughput reports all 20 paired completed-batch differences and ratios, a
paired 95% interval for the difference, and a 95% interval for the geometric
mean ratio using paired log ratios.

The strict aggregator refuses any missing, duplicate, extra, or
non-predeclared artifact.  It cannot produce the confirmatory summary from
fewer than all 20 pairs for each estimand.

## Runtime and provenance contract

Each pair runs under account `marlowe-m000151-pm06` on partition `batch`, on
one exclusive node in one process with exactly ten distinct
physical cores on one socket, ten OpenMP/BLAS threads, one interop thread, and
one hardware thread per core.  Both methods are warmed twice on disposable
training-only states, in opposite orders, before timing.  Runtime thread
counts and every worker affinity are verified and stored.

Python user-site loading and bytecode writes are disabled.  The Python
executable plus decisive Torch, NumPy, and SciPy files, versions, and explicit
site roots are verified against the tracked runtime contract.  The candidate
is compiled with explicit AVX-512F/DQ/BW/VL and 512-bit vector-width flags.
Before any Protein access on every actual node, the driver verifies those CPU
flags and proves that `zmm` operands occur inside the candidate core's own
symbol regions, not merely elsewhere in the extension.

The driver requires:

- a detached immutable 40-character Git commit;
- an externally supplied exact SHA-256 of the lock manifest;
- exact hashes for the driver, protocol, data, candidate source and binary,
  build description, Slurm script, and submission DAG; and
- atomic write-once output paths.

The pilot submission first runs a data-free algebra self-test on a small case
and 64 cumulative updates at the actual rank 1024, with the production reset
after update 50, under the same ten-thread runtime contract.  Every pilot pair
repeats that exact-node preflight before loading Protein.  Requeue is disabled,
and every compute stage/task atomically writes a permanent one-attempt claim
before preflight or data access.

The initial submission stops after all five pilot pairs.  It does not
calibrate, authorize, or submit confirmation.  After inspecting all pilot
artifacts, a human may explicitly run `record_e2_pilot_gate.sh`.  That command
recomputes the pilot-only budget and writes an immutable gate containing exact
pilot and budget hashes plus a fixed manual attestation.  Only
`submit_e2_confirmation.sh`, supplied with the exact gate SHA-256, can submit
the two 20-pair confirmation arrays and strict aggregate.

No job is launched merely by importing or testing these files.  After
committing the frozen files into a detached worktree, the operator may run:

```bash
E2_TORCH_SITE=/absolute/frozen/torch_site \
E2_NUMERIC_SITE=/absolute/frozen/numeric_site \
rebuttal_experiments/submit_e2_preregistered.sh
```

## Interpretation boundary

The exact-arithmetic identity motivates equivalence, but does not guarantee
that the finite-sample interval will fit inside the practical margin.  The
confirmation must report the complete paired result whether it establishes
equivalence, is inconclusive, or detects a systematic discrepancy.  There is
no optional stopping, outcome-dependent rerun, rank change, or held-out
hyperparameter tuning under this campaign tag.
