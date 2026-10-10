# Validation record

CPU golden comparison passed bitwise for ranks 33, 128 and 1024, FP32 and FP64,
against the frozen original CPU source, using matching native compiler flags.
See `cpu_golden.json`. The local test environment is Python 3.14.2, PyTorch 2.11.0,
macOS arm64. Dependencies were reused from local installations because network
package installation was unavailable.

The B200 CPU/CUDA build and all 71 existing tests have now passed; see below.
GPU golden comparisons, full The Well feature regeneration, investigation of
small-rank timing changes and final result import remain pending. Skipped GPU tests are not
validation of the CUDA implementations. The development-only golden comparison
tool was excluded from Git and distributions and must be restored separately.

The supplied `flop_analysis.md` was absent; roofline discussion uses the formulas
in the release plan instead. No numerical material was inferred from that file.

The batched original CPU entry point cloned L without writing it back. The release
updates each caller-owned L slice, in addition to F2's dtype/device/contiguity checks
and scratch copy of z. Its arithmetic loop remains unchanged.

Final local verification:

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
