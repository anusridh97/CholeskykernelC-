# Rebuttal result provenance

This directory contains the machine-readable outputs and paper-ready report
for the NeurIPS 2026 rebuttal experiments.

## Planned E1--E6 suite

The canonical raw artifact contains exactly 145 planned worker outputs:

- `ng/`: 48 reconstructed natural-gradient runs used for E1 and the
  original matched-rank equivalence check supporting E2;
- `representation/`: E3 representation-crossover microbenchmarks;
- `stability/`: E4 conditioning and downdate experiments;
- `cuda/`: 11 planned E5 CUDA runs; and
- `rank_k/`: E6 repeated-rank-one versus fresh rank-\(k\) comparisons.

The budget-truncated E2 table was withdrawn from the report because its
checkpoint boundary and test-set selection rule do not support a robust
budget--accuracy claim.  Its raw training histories remain under `ng/`.

## Corrected post-audit E2 campaign

The report's final E2 evidence comes from the separate recovered-data campaign
under `Cholesky-main-2/corrected_incremental_cholesky/`:

- campaign tag `e2proper-fastpositive-v1`, Slurm job `393642`, host `n02`;
- driver `e2_budget_protein.py` and batch script
  `slurm/e2_budget_protein.sbatch`;
- 17 artifacts under
  `results/e2_budget_protein/e2proper-fastpositive-v1/`;
- pilot seeds 1000--1002, excluded from evaluation and used only to freeze the
  18.5620677 s ceiling; and
- evaluation seeds 0--2, each tested exactly once after selecting the last
  completed batch at or below the frozen ceiling.

All six incremental pilot/evaluation JSONs record the optimized source hash
`b1c92d7fcbca869eba625623b5d7ab440a38eceba79b715db2b8087dbf2d87b4`.
The comparison is a time ceiling plus a ten-epoch cap, not strict
equal-compute training: the lower-rank refactorization runs finish the cap
with most of the ceiling unused.  The superseded
`e2proper-dataf073752` campaign is retained only for the report's explicitly
host-confounded implementation-timing diagnostic and is not pooled with the
final evaluation.  Because that diagnostic had already used seeds 0--2 and
the same test split, the corrected run eliminates within-run test checkpoint
selection but is not a globally untouched test-set confirmation.  The prior
campaign's raw JSONs survive under their campaign tag, but they record neither
the source nor binary hash of the overwritten historical extension; that
executable therefore cannot be reconstructed exactly.

## Authoritative post-audit E5 follow-ups

The report additionally uses a curated set of post-audit CUDA outputs from
the following Slurm jobs:

- `393048`: selected baseline-compatible parallel configurations;
- `393047`: phase profiles for those parallel configurations;
- `393321`: 50-update production-path stability;
- `393087` and `393161`: right-hand-side width and solve-orientation probes;
- `393425`: corrected H100 test suite (56 passed);
- `393427`: seed-0 cached-inverse configuration selection;
- `393470`: held-out cached-inverse timing, seeds 1--3;
- `393428`: 500-update well-conditioned stability;
- `393454`: condition-number stress test;
- `393472`: three-seed refresh-inclusive timing and pre-refresh error;
- `393466`: large-rank tile selection;
- `393491`: final recurrent inverse-versus-refactor ladder through
  rank 49,152; and
- `393509`: final rank-65,536 comparison.

Those jobs are retained as pilots and diagnostics.  All six final E5 tables
use the ten-seed campaign `e5n10-2ee46ba` from job `393591`: 380 tagged JSON
files, seeds 1--10, run serially in one 33:48 allocation on one H100
(`0.5633` allocated H100-hours).  Regenerate and strictly audit those tables
with:

```bash
python3 -m rebuttal_experiments.aggregate_e5n10
```

Configuration-selection jobs `393427` and `393466` are not mixed into the
held-out or final-ladder summary statistics.  Smoke jobs are likewise excluded.

The early inverse-factor jobs `393301` and `393306` used an invalid
Fortran-strided inverse-factor input and are superseded.  Job `393380` exposed
two test defects and is superseded by the passing test job `393425`.
Early refresh job `393437` predates complete pre-refresh diagnostics and is
superseded by `393472`.

## Locked The Well black-hole ladder

The final real-data E5 extension is campaign
`blackhole-s4-v1-locked-rerun2`, under
`blackhole/blackhole-s4-v1-locked-rerun2/`.  It uses the official
Polymathic AI `post_neutron_star_merger` scenario 4 trajectory, a frozen DCT
feature map, ten predeclared transitions, and ranks 2,048 through 65,536.
Slurm array `394033` ran the seven ranks on seven distinct H100 UUIDs.  The
directory retains all seven raw result JSONs, seven source/result
attestations, the feature manifest, the campaign lock, worker logs, and the
write-once strict aggregate.

All seven rows passed the frozen `1e-3` correctness gate.  The audited paired
speedup rises from `1.7119 ± 0.0085x` at rank 2,048 to
`71.7546 ± 0.2247x` at rank 65,536.  The latter has a 95% Student-t interval
of `[71.5939, 71.9153]` across the ten transition-level ratios.  Timing
repeats are retained but are not treated as independent observations.

Reconcile every displayed report cell and regenerate the figure with:

```bash
python3 rebuttal_experiments/audit_blackhole_report.py
python3 rebuttal_experiments/plot_blackhole_rank_scaling.py
```

The exact raw HDF5 file and the 47 MB feature trace are identified by hashes
in the lock and attestations but are not duplicated in the result directory.
The mixed-HEAD diagnostic campaign and the pre-execution failed first locked
attempt remain excluded from report claims.

The repository intentionally does not version local `summary.json` or
`SUMMARY.md` snapshots.  They predate the final CUDA audit and can be
regenerated from the canonical raw files with:

```bash
python3 -m rebuttal_experiments.aggregate
```

The aggregator enumerates the planned task matrix and uses an explicit
11-file planned-E5 manifest.  Smoke, tuning, diagnostic, and invalid JSON
files therefore cannot leak into the regenerated summary.

## Dataset and native extensions

E1 uses `data/uci_protein/CASP.csv`, the UCI Protein regression dataset.
Its source, CC BY 4.0 attribution, and SHA-256 checksum are recorded alongside
the CSV.  Corrected E2 uses the recovered Protein `.mat` file identified in
its run metadata by SHA-256 prefix `5ce538fa`; the dataset itself is not a
result artifact.  E3 requires the `one_sided_cholesky` extension, E1 requires
the tracked `student_pkg/fast_update.cpp` extension, and corrected E2 records
the source path/hash of its local positive-update extension.  Build
instructions for the planned suite are in `rebuttal_experiments/README.md`.

## Report

`rebuttal_experiment_report.tex` is the source of the paper-ready report and
`rebuttal_experiment_report.pdf` is its compiled form.  Recompile it from this
directory with:

```bash
tectonic rebuttal_experiment_report.tex
```

LaTeX auxiliary files, general Slurm logs, Python environments, compiled
extensions, and unrelated local datasets are not versioned.  The locked
black-hole campaign's seven worker logs and seven no-claim preflight logs are
the exception: they are retained beside its attestations as execution
provenance.
