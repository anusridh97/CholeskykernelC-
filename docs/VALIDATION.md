# Validation record

CPU golden comparison passed bitwise for ranks 33, 128 and 1024, FP32 and FP64,
against the frozen original CPU source, using matching native compiler flags.
See `cpu_golden.json`. The local test environment is Python 3.14.2, PyTorch 2.11.0,
macOS arm64. Dependencies were reused from local installations because network
package installation was unavailable.

GPU builds, GPU correctness tests, GPU golden comparisons, full The Well feature
regeneration, and B200 remeasurement remain pending. Skipped GPU tests are not
validation of the CUDA implementations. The local-only golden comparison tool is
in `.dev/compare_with_snapshot.py`, ignored by Git and excluded from distributions.

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
