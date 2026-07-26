# E1 threshold-sensitivity prior-outcome audit

This audit was completed before launching `e1-tau-sensitivity-v1`.

## Old sweep

The old E1 threshold table used three seeds (0--2).  Every threshold was an
independent process rather than an explicitly paired block.  The CPU batch
job reserved 100 CPUs and `worker.py` launched ten concurrent worker
processes.  Each E1 process exported `OMP_NUM_THREADS=10`,
`MKL_NUM_THREADS=10`, and `OPENBLAS_NUM_THREADS=10`, but processes were not
individually pinned and the JSONs did not record observed runtime thread
counts, affinity, process concurrency, source/binary/data hashes, or the full
runtime.  The positive-update call was serial C++ while refresh work used
multithreaded BLAS/LAPACK, so the wall column mixed different threaded
components under co-scheduled workers.

The driver also materialized and evaluated test after every epoch.  The
threshold table reported final-epoch MSE rather than selecting its cell by
test, but this access pattern later enabled the invalid budget/checkpoint
analysis.  The worker allowed retries, and artifacts lacked the write-once
attempt policy required for a held-out confirmation.

Thus the raw old refresh fractions remain useful preliminary evidence, but
the wall seconds are not a controlled cost axis and the three-seed accuracy
uncertainty is weak.

## Prospective inventory and choices

Repository-wide inspection found no E1 outcomes for seeds 3000--3009.  The
six thresholds and two ranks exactly reproduce the old threshold grid; they
were not selected after observing these confirmation seeds.  The architecture,
data split, optimizer, reset interval, precision, and positive-update
implementation are the frozen E1 v4 settings.

The campaign runs ten threads only because refresh fraction and fixed-run
accuracy are the threshold-sensitivity estimands.  E1 v4 already measures
one- and ten-thread application performance at `tau=0.1`, and E6 provides
isolated kernel timing.  Timing here is descriptive and no recommendation is
based on seconds.

There is no adaptive pilot.  Freeze and preflight may only validate source,
runtime, data, kernel equivalence, CPU affinity, and launch integrity.  Any
scientific change requires a new disclosed campaign tag.  Confirmation
outcomes are analyzed only after every predeclared task publishes, and a
claimed task is never retried.

## Preserved pre-science freeze failure

Freeze job 394325 built the extension and then failed before producing a
source lock because the inherited Slurm validator invoked `scontrol` by its
bare name and the compute-step `PATH` did not contain Marlowe's Slurm binary
directory.  It did not load Protein, run a kernel probe, train a model,
materialize or evaluate test, create a confirmation claim, or publish an
outcome.  The structured failure record and hashes of both logs are preserved
under the campaign result directory.  The retry changes only launcher
plumbing by exporting Marlowe's canonical `SLURM_CONF` and Slurm binary
directory; the complete frozen scientific design is unchanged.
