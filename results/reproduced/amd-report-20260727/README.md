# AMD historical report transcription

These files preserve two AMD experiment reports supplied from the historical
conversation on 2026-10-14. They are **not recovered original run artifacts**.
The original AMD source, driver, hashes, JSON/CSV archive and compiler outputs
remain unavailable. No AMD experiment was rerun during recovery.

`amd_report_excerpts.txt` contains the two report bodies. The structured
`amd_transcript_evidence.json` labels every setting and correctness result as
reported, and recomputes medians from the pasted timing samples. Verify with:

```bash
python results/reproduced/amd-report-20260727/recover_amd_transcript.py
```

At rank 28,672, the samples give 624.674108 ms optimized versus 6862.878285 ms
for the earlier Triton implementation, a ratio of 10.986334 (10.99x). The timer
was synchronized host `perf_counter`; GPU-event timing reportedly returned
zero. The region covers Linv/P and Aw updates, excluding G/M outer products.
The comparator uses the original transport with pure-Triton setup replacing
unsupported ATen operations. This is not a fresh-refactor speedup.

Reported device: Radeon Pro V520 / Navi12, gfx1011:xnack-, 8,422,162,432 bytes
(7.84375 GiB) VRAM. Reported software: Python 3.12.13, Torch 2.6.0+rocm6.2.4,
Triton 3.2.0; full reported versions are retained. Reported one-step tests at
ranks 129/257/1025 and a 100-step rank-129 trajectory passed. Those checks do
not certify every large timing rank. Available wheels and rocBLAS lacked
gfx1011 support, so no valid GPU fresh-refactor result was produced.

The package does not claim an AMD backend based on this transcription. The
exact remote paths and unresolved protocol details are retained in the JSON.
