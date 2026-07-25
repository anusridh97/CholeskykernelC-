# Rebuttal result provenance

This directory contains the machine-readable outputs and paper-ready report
for the NeurIPS 2026 rebuttal experiments.

## Planned E1--E6 suite

The canonical raw artifact contains exactly 145 planned worker outputs:

- `ng/`: 48 reconstructed natural-gradient runs used for E1 and the
  matched-rank E2 equivalence argument;
- `representation/`: E3 representation-crossover microbenchmarks;
- `stability/`: E4 conditioning and downdate experiments;
- `cuda/`: 11 planned E5 CUDA runs; and
- `rank_k/`: E6 repeated-rank-one versus fresh rank-\(k\) comparisons.

The budget-truncated E2 table was withdrawn from the report because its
checkpoint boundary and test-set selection rule do not support a robust
budget--accuracy claim.  Its raw training histories remain under `ng/`.

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

Configuration-selection jobs `393427` and `393466` are not mixed into the
held-out or final-ladder summary statistics.  Smoke jobs are likewise excluded.

The early inverse-factor jobs `393301` and `393306` used an invalid
Fortran-strided inverse-factor input and are superseded.  Job `393380` exposed
two test defects and is superseded by the passing test job `393425`.
Early refresh job `393437` predates complete pre-refresh diagnostics and is
superseded by `393472`.

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
the CSV.  E3 requires the `one_sided_cholesky` extension and E1 requires the
tracked `student_pkg/fast_update.cpp` extension; build instructions are in
`rebuttal_experiments/README.md`.

## Report

`rebuttal_experiment_report.tex` is the source of the paper-ready report and
`rebuttal_experiment_report.pdf` is its compiled form.  Recompile it from this
directory with:

```bash
tectonic rebuttal_experiment_report.tex
```

LaTeX auxiliary files, Slurm logs, Python environments, compiled extensions,
and unrelated local datasets are not versioned.
