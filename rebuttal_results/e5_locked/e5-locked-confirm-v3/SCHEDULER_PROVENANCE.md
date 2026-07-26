# E5 locked confirmation: scheduler and recovery provenance

The scientific campaign is `e5-locked-confirm-v3`, with ten predeclared seeds
5001--5010 and Slurm array indices 0--9. The immutable numerical
implementation/wrapper commit is
`b3d5b066cd9b7c480de88bea7f8d3fcbb786cfb7`; the checkout containing the
pre-run campaign lock is
`799caf46780bdd853403bf263b83d2204c451030`; and the campaign-lock SHA-256 is
`e9cbc1ec5ddc9f45efc64f6d6b8db9a8b3ae450e7b88fd010f6fa5bd01a85838`.
The exact Git bundle SHA-256 is
`bab78970a4ebbc56bf9c1aabbe4aa642d2af9735fb71cd760211e99d016313ba`.

All submitted jobs used account `marlowe-m000151-pm06`, partition `batch`,
QoS `medium`, one H100 per task, and the frozen CUDA-12.6 runtime. Numerical
tasks requested four CPUs, 64 GiB, and one hour. The data-free preflight
receipt records the exact GPU UUID/model/memory, driver 565.57.01, runtime,
CPU affinity, Slurm job record, source hashes, and lock hash.

## Preserved attempt ledger

1. Preflight job 394534 (`v1`) failed after 4 seconds, before importing any
   numerical driver, because the lock recorded `/usr/bin/python3` but the
   verifier compared its canonical path `/usr/bin/python3.10`. Binary hash,
   Python version, Torch, and CUDA otherwise matched. No receipt, claim, or
   numerical result was produced.
2. Preflight job 394540 (`v2`) completed successfully in 5 seconds. The
   subsequent `sbatch` command was rejected synchronously with
   `Invalid job array specification`: this cluster does not permit array
   indices 5001--5010. Slurm issued no v2 numerical job ID, no task began, and
   no v2 claim or result exists.
3. `v3` froze the scheduler-only mapping `index i -> seed 5001+i`; no
   scientific grid, seed, timed region, or algorithm changed. Preflight job
   394554 completed successfully in 5 seconds.
4. Numerical array job 394555 produced all 380 expected JSON records and all
   40 expected probe NPZ artifacts. Task 8 was first allocated on node n01
   from 22:53:49 to 22:55:23, then was requeued/held for
   `user_env_retrieval_failed`. Before release it had no stdout/stderr, claim,
   JSON, NPZ, or numerical outcome. It was released in place, preserving
   `Restarts=1`, and ran on n07 from 22:57:30 to 23:00:51.
5. Every numerical task completed its full 38-JSON plus 4-NPZ protocol and
   then exited 1 in the final fail-closed attestation wrapper. The wrapper had
   incorrectly assumed `slurm_array_task_id` was top-level in every JSON;
   `cuda_scaling` and `cuda_inverse_refresh` correctly store it inside their
   `environment` object. No numerical row failed and no numerical task was
   rerun.
6. Audit-only commit
   `4de5cac013a5455efb547257c095bda2d2468316`, locked by SHA-256
   `ea327ea3bdea77e1f844f123fd11827147544b2c18baafc28815ac03318e6561`,
   generated the ten write-once attestations from the already complete
   outputs. It revalidated the original in-task source/runtime/GPU record,
   every raw hash, and all probe checks; it never calls a numerical driver.
   Aggregate-renderer commit
   `2cb6b3f45540c6fb0c8166b421f1137998ab6f38`, locked by SHA-256
   `dd4d6644bbe15c849525b253f44e40647278ce3a0832e9ba583d939acb4ae75d`,
   corrected only the rendered seed label from the old hard-coded `1--10` to
   `5001--5010`.

Slurm records all ten final array elements as `FAILED (1:0)` because the
fail-closed attestation was the last command in each script. Their numerical
completeness is established independently by the exact artifact counts,
per-file hashes, ten posthoc attestations, and the strict aggregate's PASS.

## Compute and storage

The ten successful numerical allocations sum to 2,301 H100-seconds
(0.6392 H100-hours). Task 8's pre-script environment failure consumed another
94 allocated H100-seconds, and the three data-free preflights consumed 14
H100-seconds, for 2,409 accounted H100-seconds (0.6692 H100-hours) overall.
The main nine tasks took 3:32--4:13 each; the recovered task took 3:21.

The 380 raw JSONs total 1,871,860 bytes and are committed. The 40 untimed
probe arrays total 440,434,640 bytes and remain at
`/scratch/m000151-pm06/e5-locked-confirm-v3/repo/rebuttal_results/cuda_tuning/`;
their exact path, byte count, and SHA-256 are committed in each seed's
attestation. They are intentionally not added as 440 MB of ordinary Git
objects.
