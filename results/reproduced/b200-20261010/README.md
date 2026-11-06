# Completed B200 run, 2026-10-10 UTC

The JSON artifacts are unmodified copies from the verified final RunPod archive.
manifest.json records original member paths and SHA-256 digests. No dataset,
derived trace, native binary or private comparison source is included.

Regenerate tables from the repository root:

```bash
bash scripts/summarize_results.sh --output-dir results/local/tables
```

| Rank 65,536 workload | Incremental (ms) | Comparator (ms) | Ratio |
|---|---:|---:|---:|
| Symmetric synthetic | 43.5593 | fresh 10439.0615 | 239.6519 |
| The Well, tiled | 90.5970 | fresh 10273.6465 | 113.3989 |
| The Well, in-place | 128.3520 | fresh 10273.9429 | 80.0435 |

The Well ratios are medians of ten transition ratios. The synthetic ratio is
the ratio of timing medians. Matched asymmetric synthetic at rank 65,536 measured
105.2329 ms tuned in-place versus 67.5499 ms tuned tiled (1.5579x).

Source revisions differ intentionally: symmetric b07b0e2, asymmetric b2d8b84,
and The Well/validation c24a9d2. Raw JSON preserves full hashes and configurations.
The synthetic rows' original unvalidated-measurement strings are retained;
later test evidence is separate, not silently substituted into those records.

validation/tests.json summarizes the final XML: 73 passed, zero failures/errors/skips.
All seven ranks for both Well methods contain ten cases and passing method and
cross-method checks. Above rank 32,768 those checks use probes, not full dense
comparisons. validation/golden.json records 22 passing bitwise comparisons on
the tested CPU/GPU paths and ranks; it does not cover every possible input.
The comparison harness itself belongs to the private snapshot audit.

validation/paired-small-ranks.json contains the separate same-GPU snapshot versus
release check. It did not establish a release-code speedup at small symmetric
ranks: the release was slower at the smallest ranks and similar at larger ranks.
Do not attribute improvements over old archived timings to a code optimization
based on this evidence.
