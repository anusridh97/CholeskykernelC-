# BO and NG reproducibility audit

The corrected CPU applications now have a complete real-GB1 input recipe and a
fixed sweep command. The historical submitted tables remain **unreproduced**.
This distinction comes from inspecting the paper and recovered sources, not
from a requirement for B200 hardware. These applications execute on CPU.

## Run the corrected real-data experiments

The private handoff includes `data/protein.mat` and the prepared GB1 arrays.
In a checkout without GB1 data, prepare it once:

```bash
bash scripts/setup.sh applications
bash scripts/prepare_gb1.sh
bash scripts/prepare_gb1.sh --verify
bash scripts/reproduce_applications.sh --dry-run
bash scripts/reproduce_applications.sh --output-dir results/local/applications-1thread
```

The sweep runs GB1 BO at ranks 64 and 128 (five seeds each), then Protein NG at
ranks 64, 128, 256, 512, 1024. It stops on failure, preserves logs and raw curves,
refuses existing output directories, and records each command. Large NG ranks
can take substantial CPU time. `--experiments bo` runs only BO;
`--experiments ng --ng-ranks 64` runs one complete ten-epoch NG comparison.
`--threads 10` in a separate output directory requests a ten-thread CPU run.
That setting applies to all Torch operations; the native update is sequential.
No command allocates cloud resources or uses CUDA.

The original smaller corrected runs are still available through
`reproduce_bo.sh` (30 rounds by default) and `reproduce_ng.sh` (rank 64,
128 hidden units, five epochs, both chart modes). These differ from the sweep.

## GB1 data, pinned and reproducible

Source: Wu et al., *Adaptation in protein fitness landscapes is facilitated by
indirect paths*, eLife (2016), [Supplementary file 1](https://doi.org/10.7554/eLife.16965.024).
Use the original measured-data XLSX, not the separate imputed-data supplement.
The data is attributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- Original XLSX: 6,633,543 bytes; SHA-256
  `ac9a16295cd24110ad3ef4bcc57741f023ff9f3f5c3d9f5fdc0a94b996dca9b7`.
- Keep all 149,361 measured variants, including zero-fitness variants, in the
  original worksheet order. No sort, imputation, shuffle, or normalization.
- Sites 39, 40, 41, 54, in that order; alphabet `ACDEFGHIKLMNPQRSTVWY`.
  Site-major one-hot encoding gives 80 float32 features.
- Targets are the published raw fitness values cast to float32. Wild type VDGV
  has fitness 1. The variant identifiers are preserved in a third NPY file.
- `prepare_gb1.py` fixes the source and output checksums; `gb1_manifest.json`
  records the full recipe. An altered source, partial dataset, or altered output
  fails verification. Existing verified data can be reused offline.
- To regenerate without network, pass `--source /path/to/elife-16965-supp1-v4.xlsx`.

The old `gb1_X.npy`/`gb1_y.npy` and their preprocessing recipe were absent from
the recovered snapshots and local Git history. The new encoding is explicit,
but equality with those lost arrays cannot be established. The BO runner checks
the new manifest when present and embeds it in the result metadata. Custom
arrays without that manifest retain file hashes but have unspecified encoding;
the fixed sweep refuses them.

## Mapping to the submitted paper

Audited local source: `Neurips2026__Incremental_Cholesky-5.pdf`, SHA-256
`5108870d38a592a8ae8b887955225d16f970e43c87bc09c7d6a0c00cd0e969d6`.
Relevant sections: 3.5.2–3.5.3, 4.4–4.5; Tables 4–6 on PDF pages 9–11.

| Result | Paper settings | Recovered evidence and current treatment |
| --- | --- | --- |
| Table 4, GB1 BO | 96 initial observations; ranks 64/128; 15 rounds; batch 96; UCB coefficient 2; five seeds; 1/10-thread comparisons | The older `run_full.py` uses `min(r,96)`, so its row labeled 128 actually uses rank 96. Its acquisition mean pairs inducing points with unrelated leading observation labels, and can reacquire observed entries. The corrected sweep uses the maintained posterior and excludes observations. Rank 64 starts with 96 examples; rank 128 explicitly starts with 128. This changed initialization and the newly frozen encoding prevent an exact Table 4 claim. |
| Table 5, NG Setup A | Elevators; one 128-unit hidden layer; rank 32; 20 epochs; Refact, EMA and SGD | The existing generic driver covers corrected inverse/sandwich charts and SGD on Protein/synthetic data. It does not reproduce the Elevators/EMA table. This table is outside the new Protein sweep; it is not silently replaced by Protein results. |
| Table 6, NG Setup B | Protein; five ranks 64–1024; 10 epochs; batch 256; threshold 0.1; reset every 50 | The paper prints `128 → 512 → 512 → 128 → 1`, but states first-layer parameter count `9×512+512=5120`. The sweep explicitly chooses `9 → 512 → 512 → 128 → 1`, consistent with that parameter count, and fixes seed 42. This is a documented reconstruction, not proof of the missing original driver. |
| Later E2 time-ceiling table | Recovered Protein MAT; separate excluded pilot seeds and frozen time ceiling | A different driver, native implementation, model protocol and checkpoint-selection experiment. The generic ten-epoch sweep does not reproduce it. Original repository audit also records prior exposure of its evaluation split. |

The recovered Protein MAT is exact (SHA-256
`5ce538fa05f6834de3acc727e220e777a02fb1ebc3a7bfebcb5aaad9769fed13`),
and its attribution is retained in the handoff. Its historical conversion from
the public CASP CSV was not preserved. The fixed sweep requires this exact MAT
rather than silently switching datasets or row order.

The corrected algorithms fix chart initialization, Woodbury scaling and stale
coordinates. Those fixes are scientifically material; using their outputs to
certify the old table values would be misleading. The real-data runs did not give identical rank-64 BO outcomes on every seed.
A separate shared-stream diagnostic replays 1,440 fixed observations and checks
5,000 probes after each batch. Regenerate it with:

```bash
python -m benchmarks.applications.verify_gb1 --output results/local/gb1-shared-stream.json
```

Small posterior differences or acquisition ties can affect closed-loop choices;
the diagnostic does not establish the cause of every differing selection. End-to-end timing includes acquisition/training and depends on CPU,
BLAS, thread count and method order; a B200 is not needed for these CPU runs.

To close *historical numerical matching*, the authors would need to provide the
original BO arrays/preprocessor, the exact training drivers and random seeds,
and resolve the rank/architecture discrepancies. Until then, the scripts make
the corrected experiment repeatable and preserve the historical claim boundary.
