# Locked E5 ten-seed confirmation protocol

This campaign repeats the exact displayed E5 grid with ten fresh, predeclared
seeds `5001..5010`. It does not replace or alter the original campaign. No
configuration may be tuned, no seed may be replaced, and no run may be
stopped or repeated based on an observed numerical outcome. All ten seeds are
aggregated.

The timed regions, warmups, repetitions, ranks, batches, policies, launch
parameters, and alternating large-rank method order are unchanged from the
original ten-seed E5 scripts. Each timing JSON retains every raw CUDA-event
sample. The job uses one H100, four CPUs, and 64 GiB per array task on
`marlowe-m000151-pm06`, `batch`, `medium`.

Large-rank correctness remains a dense direct comparison through rank 32,768.
At ranks 49,152 and 65,536, each method additionally performs an untimed,
deterministic 16-column Rademacher sketch after timing. It checks the defining
metric and whitening identities and saves the actions of the inverse factor
and whitened operator. The attestation compares those actions across the two
independent method processes at relative tolerance `1e-3`. This costs
`O(rq)` storage with `q=16`.

A full extra dense FP32 rank-65,536 output cannot coexist with the four
required recurrent state matrices on an 80-GB H100: each matrix is exactly
17,179,869,184 bytes (16 GiB); four consume 68,719,476,736 bytes (64 GiB),
and a fifth raises the exact tensor total to 85,899,345,920 bytes, already
853,868,544 bytes above the GPU's 85,045,477,376 bytes before allocator and
library workspace. The sketch therefore strengthens the old finite-only check
without changing the benchmark timing or requiring another dense output.

The campaign lock binds all numerical and audit sources, Git commit, runtime,
GPU model/memory/driver, scheduler request, and complete grid. A data-free GPU
preflight must pass before the numerical array is submitted. Each array task
is write-once and produces a final attestation containing the exact source and
lock hashes, GPU UUID, runtime, CPU affinity, Slurm record, every raw artifact
hash, and the probe checks.
