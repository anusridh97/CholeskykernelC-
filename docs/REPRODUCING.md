# Reproduction commands

Apple Silicon users: [MLX validation and timing](MLX.md) covers the corrected
Metal kernel and all 18 rank/batch cells. The preserved [AMD report](../results/reproduced/amd-report-20260727/README.md)
is historical evidence; an AMD source backend has not been recovered.

Run from a checkout. Every experiment script accepts --help, and PYTHON selects the
interpreter (default: python). Experiment output directories must be new:
existing evidence is never overwritten. No script starts or rents a GPU.

## Regenerate the saved tables — no GPU or Torch needed

```bash
bash scripts/summarize_results.sh --output-dir results/local/tables
```

This reads results/reproduced/b200-20261010 and writes symmetric.csv,
asymmetric.csv, well-tiled.csv, well-inplace.csv and a source-hash manifest.
Raw timing samples are retained. Synthetic ratios use sample medians; The Well
uses the median of the ten per-transition speedup ratios. These are different
statistics.

To regenerate the earlier archived H100 and B200 tables as well:

```bash
python -m benchmarks.the_well.aggregate results/h100/the_well --output-dir results/local/archived-h100
python -m benchmarks.the_well.aggregate results/b200/the_well --output-dir results/local/archived-b200
```

This recalculates tables from saved measurements; fresh hardware timings require
running the experiments below. See the imported [evidence manifest](../results/reproduced/b200-20261010/manifest.json)
and [result notes](../results/reproduced/b200-20261010/README.md).

## CPU kernels

Use Python >=3.10, a C++17 compiler and a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
bash scripts/setup.sh cpu
bash scripts/reproduce_cpu.sh --output-dir results/local/cpu
```

The runner executes all selected CPU tests and the rank-32-to-1,024 CPU
microbenchmark with one BLAS/OpenMP thread. A missing native extension, failed
command or skipped selected test stops the run. Each step has a log; run.json
records the command, revision, status and wall time. The CPU fresh baseline
uses an explicit inverse and matrix multiplies, unlike the GPU two-solve baseline.

## BO and NG — CPU only

These scripts package the **recovered corrected application implementation**.
They do not assert reproduction of the original submitted BO/NG tables or the
later fixed-time NG budget study. The older drivers used different algorithms,
datasets and parameters. The port preserves the corrected numerical routines;
[provenance.json](../benchmarks/applications/provenance.json) records source and
ported hashes.

The applications require signed updates and cyclic factor transport. They build
a separate CPU extension, benchmarks.applications._corrected._C, without
changing the release kernel API. There is no CUDA dependency.

```bash
bash scripts/setup.sh applications
# Quick checks of the full runners using explicitly synthetic data:
bash scripts/reproduce_bo.sh --smoke --output-dir results/local/bo-smoke
bash scripts/reproduce_ng.sh --smoke --output-dir results/local/ng-smoke
```

Setup builds the native backend and runs its numerical tests, including signed
updates, cyclic transport, GP predictions versus refactorization and both NG
preconditioners versus direct-solve oracles. --backend torch is an explicit
reference implementation option; it must not be confused with native timings.
Synthetic output is labeled and never substituted for missing real data.

For the corrected GB1 BO experiment, download and encode the original measured
GB1 supplement once (about 6.6 MB; no GPU). The private handoff already includes
the verified arrays, so the same command verifies and reuses them:

```bash
bash scripts/prepare_gb1.sh
bash scripts/prepare_gb1.sh --verify
```

The recipe fixes source/output hashes, all 149,361 measured rows in worksheet
order, 80 one-hot features and raw float32 fitness. See
[the application audit](APPLICATION_AUDIT.md) for attribution, exact encoding and
the mapping to the old paper tables. Run the corrected BO driver with:

```bash
bash scripts/reproduce_bo.sh --dataset gb1 --ranks 64 --initial 96 \
  --rounds 30 --batch-size 96 --candidate-count 5000 --bo-seeds 5 \
  --output-dir results/local/bo-gb1
```

The five seeds are 0–4. Defaults include RBF lengthscale 5, noise standard
deviation 0.1, UCB mean + 2 standard deviations, and refactor interval 50.
Both methods use the same initial observations and candidate pools, exclude
already observed entries, and may choose different observations thereafter.
At least initial + rounds*batch-size examples are required.
For rank 128, explicitly choose an initial count >=128; no rank is clipped.
The historical arrays were absent; the new public-data recipe makes the corrected
experiment reproducible, without asserting that the lost encoding was identical.
The runner verifies and embeds gb1_manifest.json when present and always records
input hashes. Custom arrays without that manifest have unspecified encoding.

For the corrected Protein NG experiment, supply data/protein.mat with a matrix
named data, feature columns first and regression target last.
The recovered MAT used in local validation has SHA-256
5ce538fa05f6834de3acc727e220e777a02fb1ebc3a7bfebcb5aaad9769fed13. It is included in the private handoff ZIP, and remains
excluded from Git. In a plain checkout, supply it with --data-dir.



```bash
bash scripts/reproduce_ng.sh --dataset protein --ranks 64 \
  --epochs 5 --hidden-dims 128 --ng-seeds 42 \
  --chart-modes inverse sandwich --output-dir results/local/ng-protein
```

The inverse chart is the NG-style inverse preconditioner. The sandwich chart
maintains a separate transported operator. Both compare ours, fresh
refactorization and SGD with matched initialization and minibatch permutations.
Parameters are batch size 256, learning rate 0.01, ridge 1, refresh threshold
0.1, refactor interval 50, and sandwich beta 0.9. The split is 80/20 by seed;
feature ranges and target standardization are fit on training data only.
These parameters describe the preserved driver, not a claim of matched
wall-time budgets. --ng-seeds accepts an explicit list.

BO timing covers online rounds after initial conditioning. NG timing covers
training after model/chart construction, excluding final test evaluation.
The preserved driver runs ours before refact (and then SGD for NG), with no
separate timing warmup. These are recorded CPU experiments, not new speedup
claims or a counterbalanced timing study.

Both scripts accept --data-dir, --threads (default 1), --dtype,
--ranks and --dry-run. They save raw per-seed curves, summary.csv and run.json
with dataset/code hashes, actual backend, software, configuration and status.
Repeated timings depend on CPU and thread settings. A completed run records
execution success; use the numerical tests and application metrics to assess it.

## Full corrected BO/Protein NG sweep

```bash
bash scripts/reproduce_applications.sh --dry-run
bash scripts/reproduce_applications.sh --output-dir results/local/applications-1thread
```

This explicit sweep uses 15-round GB1 BO at ranks 64/128 with five seeds, then
Protein NG at all five ranks 64–1,024 for ten epochs with hidden widths
512,512,128. BO rank 128 explicitly uses 128 initial examples. It requires the
verified GB1 recipe and exact recovered Protein MAT. It runs on CPU, with one
thread by default; --threads 10 requests another CPU configuration. Large NG
ranks can be slow. --experiments bo runs only BO; --experiments ng --ng-ranks 64
runs one complete ten-epoch NG comparison. The
[application audit](APPLICATION_AUDIT.md) records differences from the submitted
paper; this command does not certify its old values or reproduce the separate
Elevators/EMA and time-ceiling studies.

## The Well features — CPU preparation, 14 GB download

Use a separate Python 3.12 environment to avoid changing the GPU environment:

```bash
bash scripts/setup.sh well
bash scripts/prepare_well.sh --workers 32 --output-dir results/local/well-preparation
```

The source revision and raw-data hash are pinned in
benchmarks/the_well/protocol.json. Required raw bytes: 14,109,638,656. Allow
additional disk space for caches, processing and the 47 MB feature trace.
The producer versions are NumPy 2.2.6, SciPy 1.15.3 and h5py 3.14.0.
The regenerated trace matched the archived digest under Python 3.12.3:

```
bdf1619a70af70585af317d25aa69452fcb8e94d8417e525de9d50f878e530b4
```

Default path: data/the_well/processed/post_neutron_star_merger_scenario_4_dct65536.npy;
shape (181,65536), float32, C order. An existing verified trace can be reused.
The repository includes its manifest, not the dataset or trace. Feature
preparation is not included in GPU timing.

## Recompute the B200 experiments — GPU required

Use a preconfigured Python 3.12 CUDA environment with PyTorch 2.8.0+cu128,
Triton 3.4.0, CUDA toolkit/nvcc 12.8.93, and a compatible C++ compiler.
The completed run used Python 3.12.3, GCC 13.3.0, NumPy 2.1.2 and driver 595.91.07.
The setup script preserves the installed Torch/Triton pair.

```bash
bash scripts/setup.sh b200
bash scripts/reproduce_b200.sh --dry-run
bash scripts/reproduce_b200.sh --output-dir results/local/b200
```

The full runner requires a B200 with at least 128 GiB free GPU memory, a prepared
trace and roughly 35 GB of host memory plus runtime overhead. It runs, in order:

1. Full tests including both opt-in rank-65,536 regressions, with no skips accepted.
2. Symmetric synthetic ranks 128–65,536: seed 0, two warmups, 7/5/3 repeats by rank.
3. Asymmetric synthetic ranks 8,192/32,768/65,536: seed 0, two warmups, five repeats,
   reporting both default and matched tuned in-place/tiled settings.
4. All seven The Well ranks 2,048–65,536 for tiled and in-place methods, each rank
   in a separate process, followed by aggregation.

Each rank finishes its correctness checks before its output is accepted.
The runner stops at the first failure and leaves logs and partial evidence.
There is no automatic resume or overwrite. Use the printed commands to run
only missing ranks into a separate directory if needed.

For a smaller integration check, --smoke selects reduced rank sets and excludes
large tests. It requires a compatible CUDA GPU (sm_80+) and 4 GiB free memory;
it is not a full timing reproduction. --trace PATH selects a verified trace
stored elsewhere. Full runs can take tens of minutes; the largest rank alone
includes 30 fresh factorizations per method ladder, plus setup and validation.

The Well timing includes G/M outer updates and the method core, excluding I/O,
state construction/restoration and correctness checks. Full dense validation
covers ranks <=32,768; larger ranks use the frozen probe checks. B2 synthetic
timing remains core-only. Do not compare these regions as if they were identical.

The new B200 run includes both methods on the same hardware. Recomputing the
historical H100 ladder requires H100 hardware separately; after a compatible
CUDA build, run:

```bash
for rank in 2048 4096 8192 16384 32768 49152 65536; do
  python -m benchmarks.the_well.rank_ladder --method inplace --ranks "$rank" \
    --output-dir results/local/h100-inplace --run-tag reproduction-h100 || break
done
python -m benchmarks.the_well.aggregate results/local/h100-inplace
```

Inspect all seven rank files and correctness flags before treating a partial
H100 aggregate as complete. H100 and B200 results are separate hardware claims.

## Validation scope

The original B200 commands completed before the pod was stopped: 73 tests
passed, all 140 The Well transition/method checks passed, 22 snapshot comparison
cases were bitwise equal, and paired small-rank timing checks were saved.
The imported synthetic artifacts retain their original status strings and
source commits; they are not relabeled or assigned to the current script revision.

The new orchestration and CPU application code can be validated locally.
The new combined B200 wrapper has not been executed end to end on GPU.
A fresh run records its own commit and measurements. Timings are not expected
to be bitwise reproducible.

Local validation of the new scripts used Python 3.14.2, Torch 2.11.0,
NumPy 2.5.3 and SciPy 1.18.1 on macOS arm64. CPU compilation, numerical
tests, the CPU runner, both native application smoke runs and the corrected
Protein NG experiment (rank 64, seed 42, five epochs, both charts) completed.
The two new Well CSVs also match the saved RunPod aggregates byte for byte.
The [application artifacts](../results/reproduced/cpu-applications-20261010)
are retained separately from the B200 results.
