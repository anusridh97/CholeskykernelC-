# Reproducing the release candidate

Run commands from this checkout after installing the package. CPU installation,
CPU tests, CPU benchmark and result aggregation were exercised locally. Commands
labelled **requires GPU**, **requires B200/H100** or **14 GB download** were not
executed here. CUDA compilation and GPU equivalence remain release gates.

## Build and CPU checks

Python >=3.10, a C++17 compiler, and PyTorch >=2.8 are required. CPU native builds
use -O3, -ffast-math and native architecture flags without forcing AVX-512.
The tested local environment is Python 3.14.2 / PyTorch 2.11.0 / macOS arm64.
Linux and clean-machine macOS CI remain to be run after upload.

```bash
python -m pip install 'torch>=2.8' 'setuptools>=69' wheel numpy pytest
python -m pip install --no-build-isolation -e .
python -m pytest -m 'not cuda and not large'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m benchmarks.cpu_synthetic
```

CPU tests take under a second after imports in the local environment; native
compilation takes longer and depends on the machine. The CPU benchmark was run
at ranks 32,64,128,256,512,1024; it completes in seconds locally. On Linux,
optionally prefix its command with `taskset -c 0`. It preserves the original
CPU baseline (explicit inverse and matrix multiplies), which differs from the
GPU two-triangular-solve baseline. No CPU benchmark numbers are shipped as claims.

## CUDA build and correctness (requires GPU)

Use a compatible CUDA-enabled Torch/Triton pair and toolkit. For the archived
B200 software, use Python 3.12.3, PyTorch 2.8.0+cu128, nvcc 12.8.93 and GCC
13.3.0. The original H100 run recorded Python 3.10.12, PyTorch 2.12.0+cu126 and
CUDA 12.6. Availability of those exact builds is environment-dependent.

```bash
python -m pip install --no-build-isolation -e .
python -m pytest -m 'not large'
DSC_RUN_LARGE=1 python -m pytest tests/test_large_rank.py -v
```

Correctness needs sm_80 or newer. The large symmetric test requires approximately
40 GB of GPU memory (it checks at least 36 GiB free); the asymmetric comparison
needs an 80 GB class GPU (at least 72 GiB free). Both explicitly skip unless
`DSC_RUN_LARGE=1`. An ordinary CPU run skips all CUDA tests. `needs_ext` skips
native tests if the corresponding extension is missing; inspect skip reasons,
and do not count skips as validation.

Release gates: one-step factor/operator errors <=5e-6; symmetric 100-step errors
<=1e-4; tiled/in-place comparisons <=1e-6; rank-65,536 symmetric probes <=1e-3
and the formerly overflowing last factor tile row <=1e-5. See the test files
for all parameter grids. GPU wall-clock duration is unmeasured; reserve time
for Triton compilation and repeated dense factorizations.

The local-only `.dev/compare_with_snapshot.py` builds both source versions in
the same environment. CPU bitwise equivalence passed; GPU comparisons at the
plan's ranks/configurations are pending. This script is excluded from Git and
must not be added to the public release. A golden mismatch outside documented
fixes requires investigation before publication.

## B1: symmetric synthetic (requires B200 for archived timings)

```bash
python -m benchmarks.synthetic_symmetric --seed 0 \
  --ranks 128 256 512 1024 2048 4096 8192 16384 32768 65536 \
  --output results/local/symmetric-b200
```

The default sweep uses 7 repeats through rank 8,192, 5 through 32,768, and 3
at 65,536, with 2 warmups. Persistent buffers at rank 65,536 occupy about
103.1 GB plus 0.69 GB workspace when both methods are resident. Leave additional
space for solver/runtime allocations. On an 80 GB GPU, run separate processes
with the same rank and seed:

```bash
python -m benchmarks.synthetic_symmetric --ranks 65536 --seed 0 \
  --method incremental --output results/local/symmetric-incremental
python -m benchmarks.synthetic_symmetric --ranks 65536 --seed 0 \
  --method fresh --output results/local/symmetric-fresh
```

Each output stem produces JSON (all samples, versions, source hashes, exact
persistent byte counts, measured peak memory) and CSV (medians). Take the ratio
of fresh to incremental medians only for matching rank/seed/configuration.
Status stays `unvalidated-measurement` until the correctness gates pass.

Expected archived medians: [full table](../results/README.md). Investigate
changes exceeding ±5% on the same GPU for unaffected ranks. Rank 65,536 must
receive a new measurement; **239.55× is not an acceptance target**. Preserve
its historical record and replace the published row only after correctness
and timing checks, setting status `release-rerun`. Largest-rank timed calls
alone take tens of seconds; total wall time is unmeasured and includes warmup,
restoration and compilation. Whole-sweep timing is still to be recorded.

## B2: asymmetric synthetic (requires B200 for archived timings)

```bash
python -m benchmarks.synthetic_asymmetric --ranks 8192 32768 65536 \
  --seed 0 --repeats 5 --warmups 2 --output results/local/asymmetric-b200
python -m benchmarks.synthetic_asymmetric --ranks 8192 --fresh \
  --output results/local/asymmetric-with-fresh
python -m benchmarks.profile_symmetric_phases --rank 65536 \
  --output results/local/symmetric-phases.json
```

At rank 65,536, B2 retains four dense matrices (~68.7 GB) plus tiled workspace
(~1.08 GB), so allow an 80 GB class GPU. Both default and tuned in-place baselines
are reported. `tiled_ms` retains the archived tile-8/4-warp inverse-factor settings;
`tiled_tuned_ms` uses `InplaceConfig.for_rank` for that factor pass (the descriptor
tile remains 64). Compare `inplace_tuned_ms` with `tiled_tuned_ms` to use the same
rank-ladder settings; JSON records all method configurations. The additional
measurement restores the same state and uses the same warmups/repeats.
The archived default comparison was 477.1596 / 109.7262 ms,
about 4.35×. Allow ±5% on B200 or explain changes. The optional fresh region
includes G/M outer updates; incremental B2 regions remain core-only to preserve
the archived protocol. Whole-run time is unmeasured, likely dominated by JIT
compilation and restoration. Profiling times are operator phases only.

## B3: The Well preparation (14 GB download)

The default is local regeneration; this repository does not host derived data.
Install extras and pin the numerical producer versions for the recorded trace:

```bash
python -m pip install --no-build-isolation -e '.[the-well]'
python -m pip install numpy==2.2.6 scipy==1.15.3 h5py==3.14.0
python -m benchmarks.the_well.download
python -m benchmarks.the_well.prepare_features --workers 32
```

Use Python 3.10.12 for exact archived producer replication. Raw bytes:
14,109,638,656; raw SHA-256:
`21e587c90c006c4b7d37e2b04899bd7e928967ae506e9ceb5afd05dfca6909fd`.
The pinned dataset revision is in `benchmarks/the_well/protocol.json`.
Allow disk space for the download, cache, temporary processing and 47 MB trace.
A rough planning estimate is one hour of DCT work with 32 workers; **unmeasured**,
and it must be replaced after a real run. Download duration depends on bandwidth.

Expected trace: `(181,65536)`, float32, C order, SHA-256
`bdf1619a70af70585af317d25aa69452fcb8e94d8417e525de9d50f878e530b4`.
A mismatched hash stops the pipeline. Retry the exact producer versions and
32 workers; if it still differs, stop and investigate. Do not change the expected
hash or silently accept a numerically similar trace.

## B3: rank ladder (requires B200/H100 and prepared trace)

R1 uses tiled on B200; R2 uses original in-place on H100. R3 is a recommended
new original-kernel run on B200, separating hardware and algorithm changes.
Run each rank in its own process to match the archived timing isolation:

```bash
for rank in 2048 4096 8192 16384 32768 49152 65536; do
  python -m benchmarks.the_well.rank_ladder --method tiled --ranks "$rank" \
    --output-dir results/local/b200-tiled --run-tag release-b200-tiled
done
```

For R2, use `--method inplace`, an H100, and a distinct output directory/tag.
For R3, use `--method inplace` on B200 and another distinct output directory/tag.
Existing output is refused unless `--overwrite` is supplied. Input trace SHA-256
and recomputed scales are checked before allocation. Default trace path is
`data/the_well/processed/post_neutron_star_merger_scenario_4_dct65536.npy`.

At rank 65,536 allow an 80 GB GPU and at least ~35 GB of host memory for the two
factor/operator backups, plus Python, feature data and runtime overhead. The
four GPU matrices use ~68.7 GB and C4 workspace ~1.08 GB. Full dense validation
is limited to ranks <=32,768; probe checks cover larger ranks. The pinned protocol
contains transitions, repeat counts, probes, rank scales and 1e-3 thresholds.
Warmup, alternating method order, state construction and post-timing validation
preserve the archived numerical sequence.

The largest B200 row has ten transitions with three repeats per method and one
warmup. Timed fresh calls alone take roughly five minutes at archived speeds;
restoration, initialization and validation make total runtime longer. Total
wall time remains unmeasured. Use ±5% timing tolerance on the same GPU model,
with every correctness flag passing. Exact tables are in `results/README.md`.

## Aggregation (CPU; exercised locally)

```bash
python -m benchmarks.the_well.aggregate results/b200/the_well \
  --output-dir results/local/aggregate-b200
python -m benchmarks.the_well.aggregate results/h100/the_well \
  --output-dir results/local/aggregate-h100
python -m pytest tests/test_aggregate.py
```

Aggregation takes less than a second locally. B200 CSV regeneration is
byte-identical including CRLF. H100 CSV regeneration also passes byte comparison.
Extra JSON statistics use ten per-transition ratios, sample SD, and t(9)=2.2621571627
for the 95% interval. H100's headline is mean 71.7546×; B200's is median 112.27×.

## Publication gates

- [x] Frozen source hashes and seven result files per GPU verified.
- [x] macOS CPU extension build, CPU correctness and bitwise golden checks.
- [x] Aggregates reproduced; result identifiers scrubbed; inputs left untouched.
- [ ] Clean CPU Linux/macOS CI on GitHub.
- [ ] CUDA compilation, full GPU tests and GPU golden comparison.
- [ ] The Well trace regenerated with the expected digest.
- [ ] Symmetric large-rank checks and complete B200 timing sweep; replace superseded row.
- [ ] B2 default/tuned comparisons and recommended B200 original-kernel R3 run.
- [ ] Formatting with Ruff (not available in the offline local environment).
- [ ] Owner supplies license, authorship and citation metadata.

The original snapshot-preservation branch is an owner action and has not been
created here. No GitHub publication or changes to either input tree were made.
