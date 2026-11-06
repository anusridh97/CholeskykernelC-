# Corrected real-data CPU application runs — 2026-10-10

All runs used the native signed CPU extension, float32, one CPU thread,
Python 3.14.2 and Torch 2.11.0 on macOS arm64. The stopped B200 was not used.
These are corrected reconstructions, not replications of the historical tables.

- `bo-and-ng-r64`: GB1 BO ranks 64/128, five seeds (0–4), 15 rounds, batch 96,
  5,000 candidates; initial observations 96/128 respectively. Also Protein NG
  rank 64, seed 42, ten epochs, inverse chart, hidden widths 512,512,128.
- `ng-r128-to-1024`: remaining Protein NG ranks 128/256/512/1024 with the same
  settings. All three methods completed 1,420 training steps at every rank.
- `bo-all-seeds.csv` and `ng-all-ranks.csv` concatenate the child summary.csv
  files in the two completed campaigns; these do not replace the raw results.
- `validation/cpu-tests.xml`: 78 selected CPU tests passed, 43 CUDA/large tests
  deselected; no selected tests skipped. The targeted application suite had
  also passed all 48 data/model/orchestration tests.
- `validation/gb1-shared-stream.json`: shared 1,440-update stream, seed 2,
  5,000 prediction probes every 96 updates, ranks 64/128. Maximum sampled mean
  and variance relative differences are below 1e-4. This is a numerical
  diagnostic, not proof of identical acquisition decisions.
- `validation/sweep-plan.json`: all seven full sweep commands passed dry-run.

GB1 comes from the checksum-pinned Wu et al. original measured supplement:
149,361 rows, 80 one-hot features, raw float32 fitness. The downloader was
executed successfully, and its arrays were byte-identical to an independent
conversion of the downloaded XLSX. Preparation metadata is embedded in each
BO run.json. Protein uses the exact recovered MAT, SHA-256 5ce538fa...fed13.

Both BO methods reported zero fallback refactorizations. Rank-64 closed-loop
fitness outcomes differ for seeds 2 and 3; rank-128 final best fitness agrees
on all five seeds. These differences are retained, not rewritten as equality.
NG test MSE is approximately 0.7457 for the two chart methods at all ranks;
these values do not reproduce the old paper's rank-dependent MSE trend.

The measured local CPU times do not establish the historical paper speedups.
Method order is fixed, with no separate timing warmup; elapsed time excludes
initial model/chart construction. See [the application audit](../../../docs/APPLICATION_AUDIT.md) for exact scope.

Raw run.json files preserve their original source hashes. Later additions of
orchestration/documentation do not relabel those hashes. The campaign was
executed in two parts to validate the small run before starting large ranks.
The full one-command plan was checked, while its jobs were executed across
those two recorded invocations. No original raw result is overwritten.
