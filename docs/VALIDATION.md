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
