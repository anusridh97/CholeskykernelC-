# Matched-output Apple M4 evidence

Imported unchanged from the local 2026-10-14 evidence recovery. This is a new
Apple M4 (24 GiB) run; it does not certify old M4 Pro measurements.

- `mlx-matched/`: recovered original kernel; three batch-16 cells failed.
- `mlx-matched-fixed/`: corrected kernel; all 18 cells passed before/after timing.
- `mlx-fixed-repeat-checks.json`: 60 additional identical-input calls passed.
- `mlx-fixed-source/`: executed corrected driver/kernel and gate tests.
- `mlx_synchronization_fix.patch`: one additional barrier in the Cholesky loop.
- `mlx_summary.json`, `mlx_timing_table.txt`: full-grid values and maximum errors.
- `mlx_speedups.png`, `mlx_speedups.svg`: replacement heatmaps labeled Apple M4.

Each run retains environment/source manifests, exact executed source copies,
input hashes and per-item correctness. Corrected cells contain 100 timing
samples per method. Recorded absolute paths identify the original environment;
they are not required to use the sources in this repository.

To regenerate the saved figures (requires matplotlib):

```bash
python results/reproduced/mlx-m4-20261014/summarize_mlx.py
```

The summary script verifies raw-result hashes, all 18 passing cells and
matching input hashes before generating figures. Full methodology and current
reproduction commands: [MLX documentation](../../../docs/MLX.md).
