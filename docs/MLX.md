# Apple Silicon MLX update

The optional `doublesided_cholesky.mlx` module exposes the corrected functional
Metal implementation. It preserves its inputs and returns `(L_new, Aw_new, vz)`.
The CPU API updates `L` in place; these return conventions are intentionally
different. The MLX path is FP32 and supports ranks up to 1,024.

```python
from doublesided_cholesky import mlx

L_new, Aw_new, vz = mlx.incremental_update_mlx(L, Aw, z, w)
# Batch-leading arrays (T, r, r) and (T, r):
L_new, Aw_new, vz = mlx.incremental_update_mlx_batched(L, Aw, z, w)
```

Install the package with its optional dependency on Apple Silicon:

```bash
python -m pip install 'torch>=2.8' 'setuptools>=69' wheel numpy mlx
python -m pip install --no-build-isolation -e '.[mlx]'
bash scripts/reproduce_mlx.sh --output-dir results/local/mlx-check
# Collect new timings only after matched-output correctness passes:
bash scripts/reproduce_mlx.sh --output-dir results/local/mlx-timing --time --warmups 5 --repeats 100
```

`PYTHON` selects the interpreter. Each output directory must be new. No command
starts cloud resources. GPU execution requires a working local Metal device.

## Fix and validation

The recovered kernel failed rank/batch cells `(512,16)`, `(768,16)` and
`(1024,16)` on the tested Apple M4. During each Cholesky iteration, lane `k`
could overwrite `L[k,k]` and `z_work[k]` before other SIMD groups read them.
A threadgroup/device-memory barrier now precedes that write. The corrected
kernel is byte-identical to the validated snapshot, SHA-256
`68a4455163f7874c5cc7e03b9827b88d419b28212506bab93772b155da998389`.

The new comparison passed all 18 cells: ranks 64/128/256/512/768/1024, batches
1/4/16. It constructs consistent `L`, `M`, and `Aw`; checks updated `L` and `Aw`
against independent FP64 Cholesky/two-solve results; and checks defining
residuals, finiteness, factor structure, auxiliary `vz`, and cross-method
agreement for every batch item. The tolerance is fixed at `5e-5`.
All checks also pass after timing. Sixty additional calls on the three
previously failing inputs passed; these are repeated calls, not independent
seeds or a sequential trajectory.

Maximum full-grid MLX errors: factor `5.714e-8`, operator `1.001e-7`, factor
identity `9.973e-8`, operator identity `1.472e-7`, auxiliary `vz` `5.276e-6`.

## Hardware, timing, and scope

The recorded device is **Apple M4, 24 GiB**, Mac16,1, macOS 15.6. The saved
environment used Python 3.9.6, MLX 0.29.3, NumPy 2.0.2, and Torch 2.8.0.
The installable package requires Python >=3.10; the historical validation
driver and snapshot ran directly in the recorded Python environment.
CPU thread limits were requested as one; Torch reports one thread. Accelerate
worker counts were not independently observed.

Timing uses five warmups and 100 synchronized host wall-clock samples per
method/cell. Figures report the ratio of arithmetic mean latencies. Raw
samples, sample SD and medians are retained. CPU baselines use Cholesky,
explicit inverse and two matmuls, including G/M outer products. This does not
measure an optimized triangular-solve baseline or a GPU refactor baseline.
At rank 1024, batch 16: MLX 17.7693 ms, NumPy CPU 541.6197 ms, PyTorch CPU
232.1099 ms. Small cells include slowdowns.

These are new Apple M4 measurements, not retroactive certification of the
historical M4 Pro timing-only heatmaps. The original and corrected run records,
source snapshots, input hashes, logs and figures are in
[the saved evidence](../results/reproduced/mlx-m4-20261014/README.md).

The package exposes positive rank-one CPU/CUDA/MLX updates. Signed updates
used by BO/NG remain in the separate corrected CPU application backend.
There is no shared update/downdate/automatic-guard API across all devices.
