# E1 confirmation prior-outcome audit

This note records the outcome inventory and design provenance before the
`e1-confirm-fastpositive-v2` pilot launch.

## Inventory

- At the `origin/main` base commit `6e3e226`, the only file below
  `rebuttal_results/e1_confirmation/` is the v1 source lock.  There are no v1
  pilot artifacts, pilot summaries, approval gates, confirmation artifacts,
  or confirmation summaries.
- The v2 pilot seeds are 1000--1002 and the untouched confirmation seeds are
  2000--2009.  Repository-wide inspection found no E1 outcomes for those
  seeds.  Seeds 1000--1002 do occur in a separate E2 pilot campaign; those
  training-only E2 measurements are not read by this driver and cannot alter
  any E1 setting.
- Earlier reconstructed E1 results use seeds 0--2.  They are prior evidence,
  not pilot or confirmation observations for this campaign.

## Frozen-choice provenance

The v2 grid is unchanged from the source-locked v1 protocol committed in
`00f2b85` and locked in `f8c7a9c`: Protein, fixed split seed 42, ranks
256/512/1024, `tau=0.1`, ten epochs, batch size 256, learning rate 0.1,
damping 1.0, momentum 0.9, and resets every 50 steps.  These are the existing
matched Ours/Refact E1 comparison settings; they were not selected after
seeing a v1 or v2 pilot.  In particular, `tau=0.1` is not the best old E1 cell
by either mean MSE or wall time.  V2 changes execution control and provenance,
not an experimental hyperparameter.

The old seed-0--2 E1 table motivated the need for a controlled confirmation
because it mixed a serial C++ incremental call with shared, multithreaded
BLAS/LAPACK workers.  It was not used to add, remove, or tune v2 cells.  The
driver intentionally exposes no rank, threshold, optimizer, epoch, batch,
split, precision, or seed override.

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
