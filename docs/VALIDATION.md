# Validation record

CPU golden comparison passed bitwise for ranks 33, 128 and 1024, FP32 and FP64,
against the frozen original CPU source, using matching native compiler flags.
See `cpu_golden.json`. The local test environment is Python 3.14.2, PyTorch 2.11.0,
macOS arm64. Dependencies were reused from local installations because network
package installation was unavailable.

The final B200 record at c24a9d2 contains 73 passing tests with zero skips,
22 bitwise snapshot comparison cases, a matching regenerated Well feature digest
and both complete seven-rank Well ladders. The verified raw records are now
imported under results/reproduced/b200-20261010. Earlier sections below record
intermediate stages; their pending statements are superseded by the final
record at the end of this document. Skipped GPU tests are not validation of
the CUDA implementations. The development-only golden comparison
tool was excluded from Git and distributions and must be restored separately.

The supplied `flop_analysis.md` was absent; roofline discussion uses the formulas
in the release plan instead. No numerical material was inferred from that file.

The batched original CPU entry point cloned L without writing it back. The release
updates each caller-owned L slice, in addition to F2's dtype/device/contiguity checks
and scratch copy of z. Its arithmetic loop remains unchanged.

Initial local verification (before B200 and application work):

- 28 CPU tests passed, including float64 descriptor emulation, CPU mutation/shape
  checks, bounded-memory layout actions, exact large-rank memory formulas, frozen
  protocol checks and both aggregate CSV regenerations.
- 43 CUDA tests are collected and skipped on this machine, including the opt-in
  large-rank gates. They are not treated as passes.
- 35 selected Python kernel/numerical sections match the frozen inputs after
  normalizing names, docstrings and allowed int64 casts; all CUDA kernel bodies
  match after whitespace and `tile_id` renaming. This static check supplements,
  and does not replace, GPU golden equivalence.
- Editable installation, native CPU compilation and a macOS arm64 wheel build
  succeeded. A separate isolated wheel installation executed a native CPU update
  against the reference and verified the CUDA source/header package data. CPU
  benchmark and every benchmark CLI help entry point executed.
- Identifying-path scan of published results and documentation is empty. No
  runtime module modifies `sys.path` or depends on scheduler/campaign tooling.

Python sources were expanded with AST-preserving formatting and C++/CUDA sources
formatted with clang-format. Ruff is unavailable locally; the requested Ruff
formatting pass remains pending. The GPU tests and numerical sections use the
same floating-point expression order as the archived code.

## B200 hardware check: packed bookkeeping reference (2026-10-10 UTC)

Evidence: owner-supplied build, pytest and diagnostic console output from commit
`92aee085d49175c03e296fb4c614e536fd525963`. Python 3.12.3, PyTorch 2.8.0+cu128,
Triton 3.4.0, NumPy 2.1.2, nvcc 12.8.93, NVIDIA driver 595.91.07; target sm_100.
Both native extensions compiled and the editable installation succeeded.

The first standard test run stopped with 35 passes and one failure; two large
cases were deselected. Symmetric one-step checks through rank 4097, split-K
checks and the 100-step trajectory passed. This was not a complete GPU-suite pass.

`test_packed_outer[1]` required bitwise equality to `torch.addr`. A diagnostic on
ranks 1, 65 and 300 established exact packing round trips and identical G/M
updates. At all three ranks the kernel output exactly matched the FP64 update
rounded once to FP32. At rank 1, `torch.addr` matched a separately rounded
FP32 multiply/add and differed from the kernel by one FP32 ULP.

| Rank | Max absolute difference from torch.addr | Max absolute error from FP64 |
|---:|---:|---:|
| 1 | 5.960464477539063e-08 | 1.5978912415448576e-08 |
| 65 | 2.384185791015625e-07 | 2.2691432377541787e-07 |
| 300 | 9.5367431640625e-07 | 4.602427452482516e-07 |

The test now uses the once-rounded FP64 reference and retains `torch.equal`,
including exact agreement between the G and M outputs. This corrects the
T-SYM-5 reference assumption in the release plan; no kernel, compilation flag,
or numerical tolerance changed. The corrected test and remainder of the suite
still need to run on the B200. Large-rank correctness, golden equivalence and
performance claims remain unvalidated.

## B200 suite completion and initial timings (2026-10-10 UTC)

Owner-provided output at `b07b0e2217b7f90ed032455581760bef6f1890cf` records
69 standard passes (16.50 seconds) and both large-rank passes (3.01 seconds),
including the rank-65536 overflow regression and tuned asymmetric comparison.
No selected tests were skipped. GPU snapshot equivalence remains outstanding.

The first symmetric sweep reports 43.559265 ms incremental and 10439.061523 ms
fresh at rank 65536: 239.651920x, with 120993612800 peak allocated bytes. The
full raw JSON/sample artifact has not yet been imported. Ranks 8192 through
32768 reproduce archived medians within 5%; smaller incremental ranks differ
by 7-73% and require investigation. These observations do not replace the
archived results table or finalize the release claim.

The asymmetric default comparison reproduces all three archived ranks within
4%. At rank 65536, default in-place is 469.628845 ms, tuned in-place is
105.233505 ms, and default tiled is 107.807266 ms. Thus the default-baseline
ratio is 4.356189x, while the tuned in-place path is slightly faster than tiled
at its defaults in this run. No statistical significance is inferred.
The benchmark now additionally reports `tiled_tuned_ms`, using the existing
rank-ladder inverse-factor settings. All historical columns, timed operations,
state restoration and production kernels are unchanged; the new measurement
and raw-artifact import are pending.

## The Well reporting import fix

A pre-run inspection found that the rank-ladder driver used `statistics.median`
without importing `statistics`. The import is restored. Two CPU regression
checks pass: known probe errors and their median, and conversion/aggregation of
unsorted timing samples using mocked CUDA events. Python 3.10 syntax and a
global-name check also pass. Numerical kernels and the timing protocol are
unchanged; these reporting checks do not constitute a GPU ladder run.

## Completed hardware evidence and reproduction entry points

The final RunPod archive SHA-256 is
b50fe8391ec65060218bf06e746f66ef3c7988de90fef9cbf0bde5780c1eaa9d.
[Imported artifacts](../results/reproduced/b200-20261010/manifest.json) retain
their original hashes and source revisions. The symmetric and asymmetric
samples precede c24a9d2; their original commit identifiers remain intact.

At rank 65,536 the symmetric rerun reports 239.651920x, the matched tuned
asymmetric ratio is 1.557856x, and The Well transition-median speedups are
113.398879x tiled and 80.043484x in-place on the same B200. All 140 Well cases
pass; ranks above 32,768 use probes. Twenty-two snapshot cases passed exact
torch.equal comparisons on the tested configurations.

Paired same-GPU small-rank measurements do not support a release-code speedup:
the release has roughly 20 microseconds additional overhead at the smallest
symmetric ranks and is similar at large ranks. The cause of the difference
from older archived absolute timings has not been established.

The new scripts preserve the benchmark timing regions and run each Well rank
in its own process. They retain logs, refuse overwrites and reject missing or
failed correctness evidence. Offline tables regenerate from raw samples.
The combined GPU wrapper has not been run after the pod was stopped.

CPU BO/NG routines were recovered from a separate corrected snapshot, with
numerical functions and C++ source preserved and hashes recorded. Both Python
and native backends pass signed-update, cyclic-transport, GP/refactor and
NG/direct-solve tests. Native smoke runs and a real Protein run also complete;
see [CPU application evidence](../results/reproduced/cpu-applications-20261010).
The first handoff had no real GB1 data. A subsequent public-data reconstruction
now pins the original measured supplement and its 80-feature encoding. Native
GB1 BO completed at ranks 64/128, 15 rounds, batch 96, seeds 0–4. Corrected
Protein NG completed all five ranks 64–1,024 with hidden widths 512,512,128,
ten epochs and seed 42, using one CPU thread. These results are preserved in
[real-data CPU evidence](../results/reproduced/cpu-real-applications-20261010).
The data/model/orchestration tests passed (48 cases). The downloader was tested
both online and from the pinned local XLSX. Prepared arrays matched byte for byte.
The [application audit](APPLICATION_AUDIT.md) details the verified recipe and
historical table discrepancies; exact original-paper reproduction remains unproven.
Full commands are in [REPRODUCING.md](REPRODUCING.md).

Final CPU regression run: **78 passed, 43 deselected, zero selected skips**.
All seven commands in the fixed application sweep passed dry-run.

## Main integration, 2026-11-06

Both CPU extensions built from this combined checkout. The selected CPU suite
passed **83 tests**, with 43 CUDA/large tests deselected and zero selected skips.
The integrated MLX validation command passed all 18 rank/batch cells on the
local Apple M4. The packaged MLX kernel is byte-identical to the corrected
October snapshot; the driver now locates it under `src/doublesided_cholesky`.
This integration check collects correctness only and does not replace the
October timing measurements. CUDA kernels were not rerun on new hardware.

The application sweep dry-run passed. All 117 pending release files were
carried into the combined checkout, and the later MLX/AMD evidence was added.
Raw integration checks and their manifest are saved under
[main integration evidence](../results/reproduced/main-integration-20261106).
The historical AMD transcription does not supply a missing AMD backend.
