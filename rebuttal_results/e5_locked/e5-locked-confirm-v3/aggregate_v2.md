# E5 ten-seed campaign: `e5-locked-confirm-v3`

Audit: **PASS** — 380 tagged JSONs; seeds 5001–5010; GPU `NVIDIA H100 80GB HBM3`. Columns marked mean ± SD are arithmetic means and sample standard deviations across seed-level measurements. Paired speedups are formed within seed before aggregation. Columns marked fixed are deterministic and identical across runs. Every stochastic correctness/drift cell shown also reports mean ± SD; for refresh rows, the seed-level statistic being aggregated is that seed's maximum pre-refresh error. The JSON additionally preserves global worst-case bounds over all ten seeds and all applicable checkpoints.

## Main timing (11 frozen rows)

| Impl. | r (fixed) | T (fixed) | Incremental µs (mean ± SD) | Fresh µs (mean ± SD) | Incremental k/s (mean ± SD) | Fresh k/s (mean ± SD) | Paired speedup (mean ± SD) | BW GB/s (mean ± SD) | Workspace KiB (fixed) | L err. (mean ± SD) | Aw err. (mean ± SD) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fused | 64 | 1 | 109.9 ± 4.4 | 132.3 ± 6.2 | 9.1 ± 0.4 | 7.6 ± 0.4 | 1.2 ± 0.0× | 0.6 ± 0.0 | 18.5 | 1.3e-07 ± 9.0e-09 | 2.5e-07 ± 1.4e-08 |
| fused | 64 | 4 | 107.3 ± 3.6 | 227.4 ± 7.1 | 37.3 ± 1.2 | 17.6 ± 0.5 | 2.1 ± 0.0× | 2.5 ± 0.1 | 69.0 | 9.3e-08 ± 4.2e-09 | 1.9e-07 ± 4.1e-09 |
| fused | 64 | 16 | 108.3 ± 4.1 | 202.5 ± 8.5 | 147.9 ± 5.5 | 79.1 ± 3.3 | 1.9 ± 0.0× | 10.0 ± 0.4 | 276.0 | 9.2e-08 ± 1.3e-09 | 2.2e-07 ± 1.7e-09 |
| fused | 64 | 64 | 107.6 ± 4.0 | 205.2 ± 9.1 | 595.4 ± 21.2 | 312.4 ± 13.5 | 1.9 ± 0.1× | 40.2 ± 1.4 | 1104.0 | 9.1e-08 ± 5.7e-10 | 2.2e-07 ± 1.2e-09 |
| fused | 128 | 1 | 190.4 ± 3.7 | 172.1 ± 5.2 | 5.3 ± 0.1 | 5.8 ± 0.2 | 0.9 ± 0.0× | 1.4 ± 0.0 | 66.5 | 1.3e-07 ± 7.6e-09 | 2.6e-07 ± 1.1e-08 |
| fused | 128 | 4 | 192.0 ± 4.4 | 376.6 ± 10.2 | 20.8 ± 0.5 | 10.6 ± 0.3 | 2.0 ± 0.0× | 5.5 ± 0.1 | 266.0 | 9.2e-08 ± 2.5e-09 | 2.0e-07 ± 3.8e-09 |
| fused | 128 | 16 | 191.7 ± 4.3 | 340.1 ± 8.2 | 83.5 ± 1.9 | 47.1 ± 1.1 | 1.8 ± 0.0× | 22.2 ± 0.5 | 1064.0 | 9.2e-08 ± 1.2e-09 | 2.3e-07 ± 2.0e-09 |
| fused | 128 | 64 | 193.4 ± 4.9 | 376.3 ± 6.4 | 331.1 ± 8.2 | 170.1 ± 2.9 | 1.9 ± 0.0× | 88.1 ± 2.2 | 4256.0 | 9.1e-08 ± 4.5e-10 | 2.3e-07 ± 7.5e-10 |
| parallel | 256 | 1 | 157.9 ± 6.0 | 280.6 ± 6.4 | 6.3 ± 0.2 | 3.6 ± 0.1 | 1.8 ± 0.0× | 6.7 ± 0.3 | 521.0 | 1.4e-07 ± 3.5e-09 | 2.7e-07 ± 3.7e-09 |
| parallel | 512 | 1 | 270.2 ± 5.0 | 506.5 ± 4.7 | 3.7 ± 0.1 | 2.0 ± 0.0 | 1.9 ± 0.0× | 15.6 ± 0.3 | 2066.0 | 1.3e-07 ± 2.4e-09 | 2.6e-07 ± 3.6e-09 |
| parallel | 1024 | 1 | 500.0 ± 6.6 | 1024.4 ± 6.1 | 2.0 ± 0.0 | 1.0 ± 0.0 | 2.0 ± 0.0× | 33.6 ± 0.4 | 8228.0 | 1.4e-07 ± 2.0e-09 | 3.0e-07 ± 2.6e-09 |

## Cached inverse-factor timing (3 frozen rows)

| r (fixed) | Inverse µs (mean ± SD) | Fresh µs (mean ± SD) | Paired speedup (mean ± SD) | Linv err. (mean ± SD) | Aw err. (mean ± SD) |
|---:|---:|---:|---:|---:|---:|
| 256 | 90.8 ± 11.0 | 280.8 ± 6.0 | 3.1 ± 0.3× | 1.9e-07 ± 2.9e-09 | 2.7e-07 ± 3.7e-09 |
| 512 | 91.9 ± 11.2 | 505.2 ± 4.9 | 5.6 ± 0.6× | 1.8e-07 ± 3.4e-09 | 2.6e-07 ± 3.6e-09 |
| 1024 | 93.6 ± 10.1 | 1026.1 ± 8.9 | 11.1 ± 1.1× | 2.1e-07 ± 1.9e-09 | 3.0e-07 ± 2.6e-09 |

## Repeated-update stability (4 frozen rows)

| Path | r (fixed) | Initial κ(G) (mean ± SD) | Final Linv err. (mean ± SD) | Final Aw err. (mean ± SD) | Final inverse residual (mean ± SD) |
|---|---:|---:|---:|---:|---:|
| single_block | 256 | 5.0 ± 0.1 | 6.8e-06 ± 6.3e-08 | 1.2e-05 ± 6.4e-08 | 6.9e-06 ± 6.4e-08 |
| single_block | 512 | 5.0 ± 0.1 | 1.0e-05 ± 1.1e-07 | 1.2e-05 ± 7.5e-08 | 1.1e-05 ± 1.3e-07 |
| single_block | 1024 | 5.0 ± 0.0 | 7.6e-06 ± 3.9e-08 | 1.3e-05 ± 7.8e-08 | 7.7e-06 ± 3.9e-08 |
| inplace | 1024 | 5.0 ± 0.0 | 7.6e-06 ± 3.9e-08 | 1.4e-05 ± 7.6e-08 | 7.7e-06 ± 3.9e-08 |

## Condition stress (3 frozen rows)

| Target κ(G) (fixed) | Measured κ(G) (mean ± SD) | Final Linv err. (mean ± SD) | Final Aw err. (mean ± SD) | Final inverse residual (mean ± SD) |
|---:|---:|---:|---:|---:|
| 1.0e+02 | 1.0e+02 ± 1.3e-03 | 3.0e-06 ± 6.3e-08 | 5.1e-06 ± 1.4e-07 | 3.1e-06 ± 5.0e-08 |
| 1.0e+04 | 1.0e+04 ± 9.2e-01 | 1.3e-04 ± 5.3e-06 | 1.9e-04 ± 8.9e-06 | 6.9e-05 ± 2.1e-06 |
| 1.0e+06 | 1.0e+06 ± 3.9e+03 | 2.3e-02 ± 6.4e-04 | 4.0e-02 ± 1.6e-03 | 8.9e-03 ± 2.0e-04 |

## Refresh cost and drift (15 frozen rows)

| r (fixed) | q (fixed) | Refreshes (fixed) | Event µs/update (mean ± SD) | Fresh µs/update (mean ± SD) | Paired speedup (mean ± SD) | Per-seed max pre-refresh Linv err. (mean ± SD) | Per-seed max pre-refresh Aw err. (mean ± SD) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | none | 0 | 109.8 ± 11.8 | 251.2 ± 6.4 | 2.3 ± 0.2× | 2.8e-06 ± 4.6e-08 | 4.9e-06 ± 5.8e-08 |
| 256 | 10 | 20 | 129.5 ± 13.3 | 251.2 ± 6.4 | 2.0 ± 0.2× | 2.8e-07 ± 2.7e-09 | 4.4e-07 ± 6.1e-09 |
| 256 | 50 | 4 | 111.0 ± 13.2 | 251.2 ± 6.4 | 2.3 ± 0.2× | 7.8e-07 ± 9.7e-09 | 1.4e-06 ± 1.8e-08 |
| 256 | 100 | 2 | 107.9 ± 12.2 | 251.2 ± 6.4 | 2.3 ± 0.2× | 1.4e-06 ± 2.0e-08 | 2.5e-06 ± 3.6e-08 |
| 256 | 200 | 1 | 106.5 ± 12.2 | 251.2 ± 6.4 | 2.4 ± 0.2× | 2.8e-06 ± 4.6e-08 | 4.9e-06 ± 5.8e-08 |
| 512 | none | 0 | 106.7 ± 11.5 | 480.3 ± 4.4 | 4.5 ± 0.4× | 4.1e-06 ± 6.4e-08 | 4.9e-06 ± 5.2e-08 |
| 512 | 10 | 20 | 142.2 ± 13.9 | 480.3 ± 4.4 | 3.4 ± 0.3× | 3.2e-07 ± 4.3e-09 | 4.4e-07 ± 6.5e-09 |
| 512 | 50 | 4 | 114.4 ± 11.9 | 480.3 ± 4.4 | 4.2 ± 0.4× | 1.1e-06 ± 1.2e-08 | 1.4e-06 ± 1.3e-08 |
| 512 | 100 | 2 | 109.0 ± 11.5 | 480.3 ± 4.4 | 4.4 ± 0.4× | 2.1e-06 ± 2.9e-08 | 2.6e-06 ± 3.0e-08 |
| 512 | 200 | 1 | 108.6 ± 12.7 | 480.3 ± 4.4 | 4.5 ± 0.5× | 4.1e-06 ± 6.4e-08 | 4.9e-06 ± 5.2e-08 |
| 1024 | none | 0 | 106.4 ± 14.0 | 984.4 ± 7.8 | 9.4 ± 1.1× | 3.1e-06 ± 1.6e-08 | 5.1e-06 ± 2.7e-08 |
| 1024 | 10 | 20 | 209.0 ± 38.1 | 984.4 ± 7.8 | 4.8 ± 0.6× | 3.0e-07 ± 2.0e-09 | 4.7e-07 ± 1.8e-09 |
| 1024 | 50 | 4 | 120.1 ± 11.6 | 984.4 ± 7.8 | 8.3 ± 0.7× | 8.5e-07 ± 9.6e-09 | 1.4e-06 ± 9.4e-09 |
| 1024 | 100 | 2 | 113.7 ± 10.4 | 984.4 ± 7.8 | 8.7 ± 0.7× | 1.6e-06 ± 1.6e-08 | 2.6e-06 ± 2.6e-08 |
| 1024 | 200 | 1 | 110.7 ± 10.6 | 984.4 ± 7.8 | 9.0 ± 0.8× | 3.1e-06 ± 1.6e-08 | 5.1e-06 ± 2.7e-08 |

## Self-contained large-rank pairs (7 frozen rows)

| r (fixed) | Inverse ms (mean ± SD) | Fresh ms (mean ± SD) | Paired speedup (mean ± SD) | State MiB (fixed) | Linv err. (mean ± SD) | Aw err. (mean ± SD) |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 1.4 ± 0.0 | 2.5 ± 0.0 | 1.8 ± 0.0× | 64 | 4.0e-08 ± 6.1e-10 | 9.6e-08 ± 3.0e-09 |
| 4096 | 3.5 ± 0.0 | 7.7 ± 0.0 | 2.2 ± 0.0× | 256 | 4.9e-08 ± 6.9e-10 | 1.2e-07 ± 2.8e-09 |
| 8192 | 7.9 ± 0.0 | 39.7 ± 0.1 | 5.0 ± 0.0× | 1,024 | 3.8e-08 ± 4.5e-10 | 1.0e-07 ± 1.6e-09 |
| 16384 | 19.8 ± 0.1 | 246.7 ± 0.7 | 12.5 ± 0.1× | 4,096 | 4.6e-08 ± 2.3e-10 | 1.2e-07 ± 4.3e-09 |
| 32768 | 57.8 ± 0.2 | 1753.6 ± 6.0 | 30.3 ± 0.2× | 16,384 | 3.4e-08 ± 2.8e-10 | 1.0e-07 ± 1.5e-09 |
| 49152 | 112.7 ± 0.4 | 5706.9 ± 21.7 | 50.6 ± 0.3× | 36,864 | n/a | n/a |
| 65536 | 182.9 ± 0.6 | 13052.0 ± 47.9 | 71.3 ± 0.3× | 65,536 | n/a | n/a |
