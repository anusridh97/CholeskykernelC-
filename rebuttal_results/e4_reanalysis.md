# E4 fresh-versus-incremental reanalysis

This is a deterministic, read-only reanalysis of the canonical E4 JSONs. No experiment or timing kernel was rerun. Fresh diagnostics were reconstructed from the saved seeds and configuration.

For non-reset checkpoints, `raw − fresh` is a signed difference between two scalar relative errors. It is not a direct norm between the incremental and fresh tensors because E4 did not save those tensors.

## Audit

- Canonical artifacts: 72 controlled, 4 kernel, 6 downdate.
- Maximum absolute reconstruction mismatch at protocol-fresh controlled checkpoints: 0.000e+00.
- Maximum absolute reconstruction mismatch at protocol-fresh kernel checkpoints: 0.000e+00.
- Excluded 3 noncanonical smoke/development artifacts: rebuttal_results/stability/downdate_r32_seed0.json, rebuttal_results/stability/kernel_r32_float32_cumulative.json, rebuttal_results/stability/stability_r32_cond10000_float32_reset10_seed0.json.
- Existing summary counts are {'positive_updates': 73, 'near_duplicate_kernels': 5, 'downdates': 7}; canonical counts are {'positive_updates': 72, 'near_duplicate_kernels': 4, 'downdates': 6}.
- Existing-summary canonical final-cell mismatches: 0.

## Controlled-spectrum FP32 checkpoint means

All values are means over three seeds. `reset50` and `reset200` show the raw trajectories; before their first reset they are identical to the no-reset trajectory by construction.

| target κ | step | fresh | no reset | no reset − fresh | reset50 | reset200 |
|---:|---:|---:|---:|---:|---:|---:|
| 1e2 | 0 | 1.115e-06 | 1.115e-06 | 0.000e+00 | 1.115e-06 | 1.115e-06 |
| 1e2 | 1 | 1.162e-06 | 1.120e-06 | -4.138e-08 | 1.120e-06 | 1.120e-06 |
| 1e2 | 10 | 1.119e-06 | 1.322e-06 | 2.029e-07 | 1.322e-06 | 1.322e-06 |
| 1e2 | 50 | 1.142e-06 | 3.298e-06 | 2.156e-06 | 1.142e-06 | 3.298e-06 |
| 1e2 | 100 | 1.140e-06 | 5.890e-06 | 4.749e-06 | 1.140e-06 | 5.890e-06 |
| 1e2 | 200 | 1.111e-06 | 1.070e-05 | 9.585e-06 | 1.111e-06 | 1.111e-06 |
| 1e4 | 0 | 4.208e-05 | 4.208e-05 | 0.000e+00 | 4.208e-05 | 4.208e-05 |
| 1e4 | 1 | 4.136e-05 | 4.206e-05 | 6.927e-07 | 4.206e-05 | 4.206e-05 |
| 1e4 | 10 | 4.176e-05 | 4.190e-05 | 1.460e-07 | 4.190e-05 | 4.190e-05 |
| 1e4 | 50 | 4.142e-05 | 4.126e-05 | -1.562e-07 | 4.142e-05 | 4.126e-05 |
| 1e4 | 100 | 4.130e-05 | 4.053e-05 | -7.615e-07 | 4.130e-05 | 4.053e-05 |
| 1e4 | 200 | 3.800e-05 | 3.991e-05 | 1.910e-06 | 3.800e-05 | 3.800e-05 |
| 1e6 | 0 | 2.480e-03 | 2.480e-03 | 0.000e+00 | 2.480e-03 | 2.480e-03 |
| 1e6 | 1 | 2.398e-03 | 2.400e-03 | 2.396e-06 | 2.400e-03 | 2.400e-03 |
| 1e6 | 10 | 1.989e-03 | 2.092e-03 | 1.028e-04 | 2.092e-03 | 2.092e-03 |
| 1e6 | 50 | 1.224e-03 | 1.254e-03 | 2.941e-05 | 1.224e-03 | 1.254e-03 |
| 1e6 | 100 | 8.178e-04 | 8.486e-04 | 3.086e-05 | 8.178e-04 | 8.486e-04 |
| 1e6 | 200 | 5.387e-04 | 5.444e-04 | 5.655e-06 | 5.387e-04 | 5.387e-04 |
| 1e8 | 0 | 1.558e-01 | 1.558e-01 | 0.000e+00 | 1.558e-01 | 1.558e-01 |
| 1e8 | 1 | 1.425e-01 | 1.432e-01 | 6.857e-04 | 1.432e-01 | 1.432e-01 |
| 1e8 | 10 | 7.680e-02 | 7.382e-02 | -2.981e-03 | 7.382e-02 | 7.382e-02 |
| 1e8 | 50 | 8.494e-03 | 8.748e-03 | 2.538e-04 | 8.494e-03 | 8.748e-03 |
| 1e8 | 100 | 3.067e-03 | 3.148e-03 | 8.174e-05 | 3.067e-03 | 3.148e-03 |
| 1e8 | 200 | 1.401e-03 | 1.401e-03 | 1.620e-07 | 1.401e-03 | 1.401e-03 |

## Controlled-spectrum FP64 no-reset summary

| target κ | largest mean | step | final mean | fresh mean |
|---:|---:|---:|---:|---:|
| 1e2 | 4.929e-15 | 200 | 4.929e-15 | 0.000e+00 |
| 1e4 | 1.435e-13 | 200 | 1.435e-13 | 0.000e+00 |
| 1e6 | 6.619e-12 | 1 | 1.594e-12 | 0.000e+00 |
| 1e8 | 4.170e-10 | 1 | 4.254e-12 | 0.000e+00 |

## Near-duplicate kernel summary

The largest signed gap is selected over the two non-reset checkpoints (steps 1 and 10). Step 100 follows a fresh reset.

| precision | operator | δ | initial fresh | largest raw − fresh | step | final fresh/reset |
|---|---|---:|---:|---:|---:|---:|
| float32 | cumulative | 1e-01 | 9.113e-02 | 7.334e-03 | 10 | 8.484e-02 |
| float32 | cumulative | 1e-02 | 1.058e-01 | 3.838e-03 | 1 | 1.028e-01 |
| float32 | cumulative | 1e-03 | 1.104e-01 | 3.829e-03 | 10 | 1.021e-01 |
| float32 | cumulative | 1e-04 | 1.013e-01 | 3.704e-03 | 1 | 1.018e-01 |
| float32 | cumulative | 1e-05 | 9.564e-02 | 5.970e-03 | 1 | 9.460e-02 |
| float32 | cumulative | 1e-06 | 8.801e-02 | 4.885e-03 | 10 | 8.164e-02 |
| float32 | random | 1e-01 | 8.007e-02 | 8.268e-03 | 1 | 6.579e-02 |
| float32 | random | 1e-02 | 1.133e-01 | 2.091e-02 | 1 | 9.808e-02 |
| float32 | random | 1e-03 | 1.224e-01 | 1.582e-02 | 1 | 1.098e-01 |
| float32 | random | 1e-04 | 1.264e-01 | 5.225e-03 | 10 | 1.248e-01 |
| float32 | random | 1e-05 | 1.329e-01 | 2.584e-02 | 10 | 1.187e-01 |
| float32 | random | 1e-06 | 1.267e-01 | 1.897e-02 | 10 | 1.188e-01 |
| float64 | cumulative | 1e-01 | 0.000e+00 | 2.586e-10 | 10 | 0.000e+00 |
| float64 | cumulative | 1e-02 | 0.000e+00 | 3.437e-10 | 10 | 0.000e+00 |
| float64 | cumulative | 1e-03 | 0.000e+00 | 3.441e-10 | 10 | 0.000e+00 |
| float64 | cumulative | 1e-04 | 0.000e+00 | 3.498e-10 | 10 | 0.000e+00 |
| float64 | cumulative | 1e-05 | 0.000e+00 | 3.538e-10 | 10 | 0.000e+00 |
| float64 | cumulative | 1e-06 | 0.000e+00 | 3.368e-10 | 10 | 0.000e+00 |
| float64 | random | 1e-01 | 0.000e+00 | 2.678e-10 | 10 | 0.000e+00 |
| float64 | random | 1e-02 | 0.000e+00 | 3.488e-10 | 10 | 0.000e+00 |
| float64 | random | 1e-03 | 0.000e+00 | 4.207e-10 | 10 | 0.000e+00 |
| float64 | random | 1e-04 | 0.000e+00 | 4.414e-10 | 10 | 0.000e+00 |
| float64 | random | 1e-05 | 0.000e+00 | 4.358e-10 | 10 | 0.000e+00 |
| float64 | random | 1e-06 | 0.000e+00 | 4.123e-10 | 10 | 0.000e+00 |

## Supplemental signed single-update comparison

This artifact is supplemental, FP64-only, and not part of the canonical E4 task matrix. Its stored errors are direct comparisons to fresh factorization, averaged over five seeds; operator signs are mixed and cannot be disaggregated.

| r | metric sign | L mean | L max | Aw mean | Aw max |
|---:|---:|---:|---:|---:|---:|
| 16 | -1 | 1.382e-16 | 1.477e-16 | 3.874e-16 | 4.606e-16 |
| 16 | +1 | 1.381e-16 | 1.678e-16 | 3.853e-16 | 4.390e-16 |
| 64 | -1 | 2.392e-16 | 2.526e-16 | 4.907e-16 | 5.081e-16 |
| 64 | +1 | 2.277e-16 | 2.501e-16 | 4.799e-16 | 5.062e-16 |
| 128 | -1 | 3.259e-16 | 3.713e-16 | 6.519e-16 | 7.111e-16 |
| 128 | +1 | 3.137e-16 | 3.197e-16 | 6.349e-16 | 6.446e-16 |
| 256 | -1 | 2.043e-16 | 2.117e-16 | 5.162e-16 | 5.246e-16 |
| 256 | +1 | 1.954e-16 | 2.002e-16 | 5.144e-16 | 5.190e-16 |

## Comparisons the raw campaign cannot support

- direct incremental-state versus fresh-state tensor norm at non-reset checkpoints: raw E4 JSONs contain diagnostics but not L/Aw tensors.
- signed downdate trajectory versus fresh: canonical E4 downdate files contain amplification analysis, not executions of the signed update implementation.
- near-duplicate kernel no-reset versus reset: only reset_interval=50 was run for the kernel family.
