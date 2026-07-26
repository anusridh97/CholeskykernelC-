# E1 confirmation prior-outcome audit

This note records the outcome inventory and design provenance before the
`e1-confirm-samevector-out-v5` pilot launch.

## Inventory

- At the `origin/main` base commit `6e3e226`, the only file below
  `rebuttal_results/e1_confirmation/` is the v1 source lock.  There are no v1
  pilot artifacts, pilot summaries, approval gates, confirmation artifacts,
  or confirmation summaries.
- The v5 pilot seeds are 63001--63003 and the untouched confirmation seeds are
  62001--62010.  Repository-wide inspection found no experimental outcomes
  for those seeds before this protocol was frozen.
- Earlier reconstructed E1 results use seeds 0--2.  They are prior evidence,
  not pilot or confirmation observations for this campaign.

## Frozen-choice provenance

The v5 grid is unchanged from the source-locked v1/v4 protocol committed in
`00f2b85` and locked in `f8c7a9c`: Protein, fixed split seed 42, ranks
256/512/1024, `tau=0.1`, ten epochs, batch size 256, learning rate 0.1,
damping 1.0, momentum 0.9, and resets every 50 steps.  These are the existing
matched Ours/Refact E1 comparison settings; they were not selected after
seeing a v1, v2, v3, v4, or v5 pilot.  In particular, `tau=0.1` is not the best
old E1 cell by either mean MSE or wall time.  V2 changes execution control and
provenance, not an experimental hyperparameter.

The old seed-0--2 E1 table motivated the need for a controlled confirmation
because it mixed a serial C++ incremental call with shared, multithreaded
BLAS/LAPACK workers.  It was not used to add, remove, or tune v4 cells.  The
driver intentionally exposes no rank, threshold, optimizer, epoch, batch,
split, precision, or seed override.

The v2 exact-node job 394128 compiled the extension and created its immutable
source/runtime lock, then failed while trying to require that newly created
lock to already be Git-tracked.  It ended before kernel probes, pilot training,
or any test access.  V2 is retired and its lock/logs are preserved.  V3 changes
only the control flow by separating lock generation from tracked-lock
preflight validation; no experimental setting or outcome-dependent choice
changed.

The separate v3 preflight job 394185 verified the committed source/runtime
lock and the exclusive Slurm allocation, then failed at CPU-affinity
validation before loading Protein or running any kernel probe.  The job had
ten physical CPU slots, but the driver first queried affinity after importing
PyTorch/OpenMP; `OMP_PROC_BIND=close` had narrowed the calling thread to one
OpenMP place.  No preflight receipt, pilot artifact, confirmation claim, or
test evaluation was produced.  V4 changes only this launch/runtime-control
path: it captures Slurm's ten-core task mask before all numerical imports,
restores the selected mask to every extant runtime thread, audits the worker
affinity union after the workload, removes the distribution override, and
uses the same absolute single-process `srun` pattern that succeeded in E2 on
node n29.  Ranks, thresholds, seeds, precision, training, and method order are
unchanged, and the v3 lock, structured failure record, and logs remain
preserved.

V4 then completed all 60 claimed pairs once.  Its result is retained rather
than replaced.  The post-run reconciliation found that v4 used
`fast_update.cpp` SHA-256
`b1c92d7fcbca869eba625623b5d7ab440a38eceba79b715db2b8087dbf2d87b4`,
whose public API allocates and returns new outputs and executes the general
two-vector path.  This was inconsistent with the already-selected E1-specific
candidate.  That candidate was fixed independently before any v4 confirmation
outcome: source/protocol and synthetic pilots were frozen by 17:39 PDT on
2026-07-25, the source was committed in `112f8a3` at 20:01 PDT, and the
preregistered E2 confirmation completed by 21:53 PDT.  V4 confirmation began
only at 22:24 PDT.  V5 therefore changes only the implementation to the exact
pre-existing source SHA-256
`a0d7bba11f178b40fb5ab5c2674640ee8b2f7c5a63836eb931ac9749db0dc8b0`
and its caller-owned out-buffer contract.  It does not use v4 to select or
tune source code, cells, seeds, method order, hyperparameters, or analysis.

## Prospective boundary

Pilots may be used only to detect operational defects: launch failure, wrong
runtime, wrong affinity, missing data/binary, invariant failure, non-finite
values, or invalid instrumentation.  They may not change the grid, select
seeds, choose a favorable method order, or justify dropping a cell.  Any
methodological change retires this campaign tag and requires a new disclosed
protocol.

Confirmation remains manually gated.  No confirmation array may be submitted
until all 18 pilot artifacts are present, the immutable pilot summary passes
its audits, and a human creates the write-once approval.  Confirmation
outcomes may be analyzed only after all predeclared tasks finish; tasks are
never retried after their exactly-once claim.
