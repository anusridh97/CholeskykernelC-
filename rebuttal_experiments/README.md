# Rebuttal experiment harness

This directory contains the self-contained runners added for the
reviewer-directed E1--E6 experiments.

## Artifact audit

The copied `Cholesky-main/` directory is not, by itself, a reproducible copy of
the submitted experiments:

- its `refresh.py` imports a missing `benchmarks` module;
- the UCI Protein Google Drive object is no longer publicly downloadable;
- the GB1 arrays used by `run_protein.py` are absent;
- the compiled `fast_cholesky` extension is absent until built; and
- its natural-gradient prototype initializes the curvature core incorrectly,
  uses an incorrect Woodbury scale, and uses stale coordinates after a chart
  refresh.

The harness fills the mechanical gaps as follows:

- `data/uci_protein/CASP.csv` is the official UCI CASP/Protein dataset;
- `data/elevators.mat` is the dataset referenced by the copied GPyTorch loader;
- `student_pkg/fast_cholesky*.so` is built from `student_pkg/fast_update.cpp`;
- `one_sided_cholesky*.so` is built from `one_sided_ext.cpp`; and
- `ng_sweep.py` ports the corrected chart initialization and preconditioner from
  git commit `eb99659`, whose commit message identifies it as the faithful
  Section 3.5 rewrite.

The reconstructed natural-gradient runner is suitable for the new threshold
sensitivity comparison, and incremental/refactor methods use identical chart
logic. It is not represented as an exact reproduction of the submitted Setup B:
the original Protein `.mat` artifact and exact training driver are not present,
and the submitted test MSE is not reproduced by the copied files.

## Environment and native extensions

The recorded runs used PyTorch 2.12.0 with CUDA 12.6; CUDA kernels additionally
require the Triton version bundled with that PyTorch environment.  Install the
auxiliary Python packages into the target used by the Slurm scripts and build
the two CPU extensions from the repository root:

```bash
python3 -m pip install --target rebuttal_deps -r requirements-rebuttal.txt
(cd student_pkg && python3 setup_ext.py build_ext --inplace)
(cd rebuttal_experiments && python3 setup_one_sided.py build_ext --inplace)
```

The compiled `.so` files and `rebuttal_deps/` are machine-specific build
products and are intentionally not versioned.

## Run

From the repository root:

```bash
export PYTHONPATH="$PWD/rebuttal_deps:$PWD"
python3 -m rebuttal_experiments.task_matrix --group ng --count
python3 -m rebuttal_experiments.task_matrix --group synthetic --count
python3 -m rebuttal_experiments.task_matrix --group gpu --count
sbatch rebuttal_experiments/slurm/cpu_worker.sbatch
sbatch rebuttal_experiments/slurm/gpu_worker.sbatch
```

Both workers skip existing result files, retry failures, and write one log per
task under `rebuttal_logs/workers/`. After completion:

```bash
python3 -m rebuttal_experiments.aggregate
```

This creates `rebuttal_results/summary.json` and
`rebuttal_results/SUMMARY.md`.
