# Archived measurements

These files describe the archived implementations, not newly verified release
kernels. The rank-65,536 symmetric result is marked `superseded-F1` because of
int32 tile-offset overflow. All symmetric publication requires the one-step,
trajectory and rank-65,536 correctness gates plus B200 remeasurement. Smaller
ranks do not cross the overflow boundary, but have not been revalidated on GPU.

## B200 symmetric synthetic

The statistic is the ratio of the incremental and fresh **sample medians**.
The timing region includes packed G/M updates. Samples and workspace sizes are
in `b200/symmetric_synthetic_samples.jsonl`; the source benchmark did not retain
fresh sample arrays. Archived seeds were not recorded. Release runs record seeds.

| Rank | Incremental ms | Fresh ms | Speedup | Status |
|---:|---:|---:|---:|---|
| 128 | 0.466720 | 0.148512 | 0.32× | archived |
| 256 | 0.342080 | 0.258624 | 0.76× | archived |
| 512 | 0.246560 | 0.487904 | 1.98× | archived |
| 1,024 | 0.269632 | 0.965472 | 3.58× | archived |
| 2,048 | 0.447296 | 2.213088 | 4.95× | archived |
| 4,096 | 0.718656 | 6.943968 | 9.66× | archived |
| 8,192 | 1.661376 | 33.570881 | 20.21× | archived |
| 16,384 | 3.652896 | 200.366531 | 54.85× | archived |
| 32,768 | 12.068288 | 1399.636108 | 115.98× | archived |
| 65,536 | 43.597954 | 10444.065430 | 239.55× | superseded-F1 |

## B200 asymmetric synthetic

The original in-place baseline used library defaults (tile 8, sweep warps 4).
It was not the tuned rank-ladder configuration. These are core-only timings;
raw samples do not survive. The release benchmark reports both default and tuned
baselines. Its optional fresh column includes G/M outer updates and is explicitly
labelled with that different timing region.

| Rank | Original defaults ms | Tiled ms | Speedup |
|---:|---:|---:|---:|
| 8,192 | 13.8415 | 4.8910 | 2.83× |
| 32,768 | 123.0665 | 28.5312 | 4.31× |
| 65,536 | 477.1596 | 109.7262 | 4.35× |

## The Well — B200 tiled asymmetric

Per-transition times are CUDA-event medians; the CSV reports medians across ten
transitions and the median of their speedup ratios. Timed regions include G/M
outer updates. Aggregation forms paired ratios before computing statistics.

| Rank | Incremental ms | Fresh ms | Median speedup | Mean speedup |
|---:|---:|---:|---:|---:|
| 2,048 | 0.971496 | 2.164936 | 2.2315× | 2.2281× |
| 4,096 | 1.666864 | 6.909096 | 4.1463× | 4.1067× |
| 8,192 | 5.205744 | 32.406137 | 6.2239× | 6.2217× |
| 16,384 | 12.138048 | 198.344208 | 16.3391× | 16.3259× |
| 32,768 | 31.520161 | 1378.788025 | 43.7423× | 43.7389× |
| 49,152 | 58.001841 | 4439.277344 | 76.5337× | 76.5168× |
| 65,536 | 91.540623 | 10277.684570 | 112.2732× | 112.2732× |

## The Well — H100 original in-place

Per-transition times are CUDA-event medians; the CSV reports medians across ten
transitions and the median of their speedup ratios. Timed regions include G/M
outer updates. Aggregation forms paired ratios before computing statistics.

| Rank | Incremental ms | Fresh ms | Median speedup | Mean speedup |
|---:|---:|---:|---:|---:|
| 2,048 | 1.490920 | 2.557576 | 1.7153× | 1.7119× |
| 4,096 | 3.569872 | 7.651424 | 2.1433× | 2.1395× |
| 8,192 | 7.948760 | 39.803569 | 5.0078× | 5.0076× |
| 16,384 | 19.795824 | 247.011459 | 12.4787× | 12.4802× |
| 32,768 | 58.108208 | 1755.941589 | 30.2178× | 30.2255× |
| 49,152 | 113.979584 | 5691.624756 | 49.9550× | 49.9896× |
| 65,536 | 183.749901 | 13183.537598 | 71.7627× | 71.7546× |

All archived The Well correctness flags pass. Across B200 ranks, maximum
cross-method errors are 4.59e-5 (operator) and 2.63e-7 (factor). Maximum inverse
method validation errors are 1.12e-4 and 7.02e-6. The threshold is 1e-3.
The original H100 write-up used the **mean** speedup: 71.7546× with sample SD
0.2247× at rank 65,536. B200's 112.27× headline uses the **median**.
The JSON aggregator additionally reports 95% Student-t intervals (df=9).

Changing from H100/original to B200/tiled changes hardware and software together.
It does not measure a kernel-only improvement. Run the original kernel on B200
as an additional controlled comparison.

## Recorded environments

- B200: Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8, cuDNN 91002,
  glibc 2.39. Archived binaries identify nvcc 12.8.93, GCC 13.3.0 and sm_100.
  NumPy and Triton versions were not recorded; confirm them on rebuild. The
  matching Torch wheel normally supplied Triton 3.4.0, but this is unverified.
- H100: Python 3.10.12, PyTorch 2.12.0+cu126, CUDA 12.6, glibc 2.35.
  In-place tiles are 8/8/8/16/32/32/32, with one sweep warp.
- Feature producer: Python 3.10.12, NumPy 2.2.6, SciPy 1.15.3,
  h5py 3.14.0 and 32 DCT workers.

The original B200 feature-manifest file is unavailable. The supplied H100
reference manifest records the same trace hash and exactly matching per-rank
scales. Archived relative source-path metadata identifies historical sources;
it does not imply those research directories are shipped here. B200 ladder
JSONs and aggregate CSV remain byte-identical; H100 path/host/scheduler metadata
was scrubbed without changing numbers.

## Historical optimization evidence

These rank-65,536 observations predate F1 and are not release measurements.
Full update progressed 64.008 → 60.178 → 43.598 ms with symmetry reuse and shared
memory padding. The following operator-only phase times describe padding:

| Phase | Before, ms | After, ms |
|---|---:|---:|
| Descriptor | 9.768 | 4.056 |
| North scan | 0.488 | 0.488 |
| West scan | 4.164 | 4.164 |
| Apply | 18.142 | 7.309 |
| Total | 32.562 | 16.017 |

Rejected variants: explicit float4 full update 61.199 ms; four independent
operator chains 43.517 ms; cooperative factor update 24.563 ms versus about
14.06 ms for Triton; cooperative operator update 57.23 ms. These implementations
are not included. These historical values were transcribed from the frozen
optimization summary, whose digest is recorded in `docs/INPUT_PROVENANCE.json`.
