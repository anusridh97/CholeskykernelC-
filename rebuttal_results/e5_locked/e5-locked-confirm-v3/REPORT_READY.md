# Report-ready E5 locked confirmation

The strict audit passed all 380 JSON records for ten predeclared fresh seeds
(5001--5010) on ten distinct H100 GPUs. `aggregate_v2.md` contains every
displayed main-timing, cached-inverse, repeated-stability, condition-stress,
refresh, and large-rank row as mean plus/minus sample SD. `aggregate_v2.json`
is the machine-precision source, including all seed-level values and global
worst cases.

For timing comparisons below, speedup is formed within seed as
`fresh / incremental` and then summarized across the ten seeds. The 95%
interval is the two-sided Student-t interval for the mean paired speedup
(`t_9 = 2.262157`). These intervals describe between-seed/GPU variation;
they are not ten independent application datasets.

## Full-state large-rank ladder

Both methods include the rank-one updates to the recurrent `G` and `M`
matrices. The incremental path updates the inverse factor and whitened
operator; the fresh baseline updates the same `G,M`, performs Cholesky, and
performs both triangular solves. Times are medians of the frozen raw
CUDA-event repetitions within seed, followed by a mean and sample SD across
seeds.

| Rank | Incremental ms (mean +/- SD) | Fresh ms (mean +/- SD) | Paired speedup (mean +/- SD) | 95% paired-speedup CI |
|---:|---:|---:|---:|---:|
| 2,048 | 1.441755 +/- 0.023267 | 2.541294 +/- 0.006809 | 1.763003 +/- 0.025318 | [1.744891, 1.781115] |
| 4,096 | 3.536262 +/- 0.018457 | 7.686518 +/- 0.019115 | 2.173670 +/- 0.010238 | [2.166346, 2.180994] |
| 8,192 | 7.923710 +/- 0.030791 | 39.707469 +/- 0.112068 | 5.011286 +/- 0.023187 | [4.994699, 5.027873] |
| 16,384 | 19.755491 +/- 0.091079 | 246.706956 +/- 0.652537 | 12.488214 +/- 0.056352 | [12.447902, 12.528525] |
| 32,768 | 57.818052 +/- 0.150538 | 1,753.595374 +/- 6.040393 | 30.329825 +/- 0.152846 | [30.220486, 30.439164] |
| 49,152 | 112.712058 +/- 0.370440 | 5,706.866504 +/- 21.691639 | 50.632803 +/- 0.265299 | [50.443019, 50.822586] |
| 65,536 | 182.934569 +/- 0.621766 | 13,051.979102 +/- 47.913455 | 71.348402 +/- 0.324612 | [71.116188, 71.580616] |

This is the clean confirmation of the approximately 71x rank-65,536 result.
The monotone growth from 1.76x to 71.35x is the expected practical
manifestation of quadratic update work versus cubic refactorization work.

## Displayed kernel grids with paired intervals

| Implementation | Rank | Batch | Incremental us (mean +/- SD) | Fresh us (mean +/- SD) | Paired speedup (mean +/- SD) | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| fused | 64 | 1 | 109.930 +/- 4.364 | 132.336 +/- 6.193 | 1.2039 +/- 0.0339 | [1.1797, 1.2282] |
| fused | 64 | 4 | 107.344 +/- 3.612 | 227.418 +/- 7.089 | 2.1188 +/- 0.0195 | [2.1049, 2.1328] |
| fused | 64 | 16 | 108.341 +/- 4.127 | 202.501 +/- 8.532 | 1.8690 +/- 0.0243 | [1.8516, 1.8863] |
| fused | 64 | 64 | 107.621 +/- 4.042 | 205.211 +/- 9.118 | 1.9070 +/- 0.0540 | [1.8683, 1.9456] |
| fused | 128 | 1 | 190.446 +/- 3.680 | 172.094 +/- 5.155 | 0.9035 +/- 0.0156 | [0.8924, 0.9147] |
| fused | 128 | 4 | 191.992 +/- 4.372 | 376.592 +/- 10.215 | 1.9615 +/- 0.0291 | [1.9407, 1.9823] |
| fused | 128 | 16 | 191.667 +/- 4.279 | 340.091 +/- 8.192 | 1.7744 +/- 0.0129 | [1.7651, 1.7836] |
| fused | 128 | 64 | 193.416 +/- 4.873 | 376.299 +/- 6.390 | 1.9461 +/- 0.0321 | [1.9231, 1.9690] |
| parallel | 256 | 1 | 157.885 +/- 5.958 | 280.640 +/- 6.363 | 1.7785 +/- 0.0353 | [1.7533, 1.8038] |
| parallel | 512 | 1 | 270.158 +/- 5.010 | 506.493 +/- 4.722 | 1.8751 +/- 0.0221 | [1.8594, 1.8909] |
| parallel | 1,024 | 1 | 500.013 +/- 6.570 | 1,024.418 +/- 6.054 | 2.0490 +/- 0.0189 | [2.0355, 2.0625] |

The single exception is fused rank 128, batch 1: kernel-launch and fixed
overheads dominate this tiny problem, and fresh refactorization is
approximately 1.11x faster. It must not be described as a universal win.

The cached-state inverse-factor rows are:

| Rank | Incremental us (mean +/- SD) | Fresh us (mean +/- SD) | Paired speedup (mean +/- SD) | 95% CI |
|---:|---:|---:|---:|---:|
| 256 | 90.771 +/- 10.967 | 280.774 +/- 6.019 | 3.1258 +/- 0.3023 | [2.9095, 3.3421] |
| 512 | 91.899 +/- 11.219 | 505.243 +/- 4.940 | 5.5655 +/- 0.6178 | [5.1236, 6.0074] |
| 1,024 | 93.587 +/- 10.142 | 1,026.099 +/- 8.888 | 11.0737 +/- 1.1278 | [10.2669, 11.8805] |

These cached-state rows intentionally isolate update-kernel cost and are not
an end-to-end same-work comparison: the inverse state is already available.
Use the full-state ladder and refresh experiment for self-contained evidence.

## Stability and high-rank validation

Dense elementwise comparison against fresh factorization was performed
through rank 32,768. Across those ladder rows, the global worst dense relative
errors were `1.3012e-7` for the whitened operator and `5.0154e-8` for the
inverse factor; every element of every output was finite.

At ranks 49,152 and 65,536, a fifth dense FP32 rank-65,536 output cannot fit
beside the four required 16-GiB state matrices on an 80-GB H100. Therefore
each method performed an untimed, predeclared 16-column PCG64 Rademacher
sketch (`seed = 29421 + rank`) after timing. Across all ten seeds:

| Rank | Max cross-method inverse-factor action error | Max cross-method whitened-operator action error | Max defining-identity residual |
|---:|---:|---:|---:|
| 49,152 | 9.743e-8 | 7.806e-7 | 7.088e-6 |
| 65,536 | 7.568e-8 | 8.740e-7 | 8.019e-6 |

Every value is far below the locked `1e-3` gate, every full state was finite,
and the strict aggregate independently reopened all 40 NPZ files and
recomputed the cross-method comparisons. This replaces the old
``finite-only'' high-rank scope with a deterministic `O(rq)` validation, but
it should still be described as sketch validation rather than a dense
elementwise comparison.

Repeated-update stability remains good in the ordinary-condition regime:
after 500 steps, the largest observed whitened-operator relative error among
the four displayed paths was `1.4591e-5`. The condition stress test is the
important boundary: at target condition numbers `1e2`, `1e4`, and `1e6`, the
largest observed whitened-operator errors were `5.2934e-6`, `1.9701e-4`, and
`4.2630e-2`, respectively. The `1e6` row is not a success regime; it
motivates periodic refresh or a condition-aware fallback.

The full refresh table is in `aggregate_v2.md`. Its main report-ready result
at rank 1,024 is that no refresh gives `9.377 +/- 1.066x`, while refresh every
50 updates gives `8.258 +/- 0.687x` with a 95% paired-speedup interval
`[7.766, 8.749]` and a global maximum pre-refresh whitened-operator error of
`1.4162e-6`. Refresh every 10 updates reduces that maximum to `4.6816e-7`
but lowers speedup to `4.810 +/- 0.605x`. This is the honest speed/stability
tradeoff.
