# Locked black-hole secondary stability trace

## Purpose

`blackhole-s4-v1-stability-locked-rerun2` is an immutable provenance rerun of
the secondary 100-transition stability trace predeclared in
`BLACKHOLE_PROTOCOL.md`.  It executes the unmodified numerical driver at Git
commit `96c8018cdd0b56ffe1fd93bddf48fa4affb34697`.

Two earlier `blackhole-s4-v1` stability JSONs are preserved.  Their numerical
driver, feature producer, and kernel hashes match the frozen files, but their
live `git_commit` metadata was taken after unrelated repository commits.  No
earlier numerical outcome was used to change this rerun.

Locked rerun 1 stopped during its data-free remote preflight: its dependency
lock named the invoked `/usr/bin/python3` symlink while the verifier compared
the normalized `/usr/bin/python3.10` target.  It created no Slurm task claim,
submitted no job, and ran no numerical code.  Its prepared directory and lock
remain preserved.  Rerun 2 changes only that dependency-path spelling.

## Frozen scientific design

- Dataset: official Polymathic AI `post_neutron_star_merger`, revision
  `1fd582f3e29749ac4d0c14bc816bddb2394258cc`, scenario 4.
- Feature manifest SHA-256:
  `3b8c6e32da3842b8a9f21ee722e8b9fb1093899744d283d4e939427dd116a5c2`.
- Feature trace SHA-256:
  `bdf1619a70af70585af317d25aa69452fcb8e94d8417e525de9d50f878e530b4`.
- Ranks: 8,192 and 32,768, mapped to array tasks 0 and 1.
- Initial state: transitions 0--63; warm-up transition 64.
- Measured sequential trace: transitions 64--163, exactly 100 updates.
- Fixed method order: fresh refactorization, inverse-factor update without
  refresh, inverse-factor update with refresh every 50 steps.
- Fixed checkpoints: steps 1, 10, 25, 50, 75, and 100.
- Refresh work is included in the corresponding timed q=50 samples.
- Fixed FP32, ridge 1, TF32 disabled, 16 Rademacher probes with seed 29,421,
  and correctness threshold `1e-3`.
- Fixed Triton tiles: 8 at rank 8,192 and 32 at rank 32,768; setup warps 4,
  sweep warps 1.

The inner driver refuses to overwrite an artifact.  The outer launcher first
creates a persistent write-once claim; a failed claim is retained and is not
deleted merely to seek another timing outcome.  A new protocol identity would
be required for an authorized infrastructure rerun.

## Provenance and execution lock

The committed `campaign_lock.json` binds:

- every numerical source blob at the locked Git commit;
- every outer harness blob at its pre-lock Git commit;
- exact input byte counts and hashes;
- all scientific choices above;
- Python, Torch, CUDA, cuDNN, NumPy, and Triton versions;
- hashes of package `METADATA`, package `RECORD`, and critical Python/CUDA
  implementation files;
- account `marlowe-m000151-pm06`, partition `batch`, QOS `medium`, array
  `0-1%2`, four CPUs, one H100, 64 GiB host memory, and four-hour limit.

The original driver predates the authorized allocation and requires
`SLURM_JOB_PARTITION=preempt` as metadata.  To keep the scientific source
byte-for-byte unchanged, the launcher sets that single environment variable
only for the inner Python process.  The numerical implementation never reads
partition metadata.  Before and after numerical execution, the outer harness
independently requires and records the actual `batch` account, partition,
array mapping, CPU count, node, one-H100 identity, and software fingerprint.
The raw result explicitly records the legacy label; the attestation explicitly
records both labels.

## Preparation

After the harness is committed and pushed, create and commit its lock:

```bash
/usr/bin/python3 rebuttal_experiments/blackhole_stability_locked.py make-lock \
  --source-repo /path/to/repository-containing-the-locked-commit \
  --manifest /path/to/feature_manifest.json \
  --trace /path/to/post_neutron_star_merger_scenario_4_dct65536.npy
```

On the authorized account, prepare the detached worktree and campaign:

```bash
/usr/bin/python3 rebuttal_experiments/blackhole_stability_locked.py prepare \
  --control-repo "$control_repo" \
  --source-repo "$source_repo" \
  --locked-worktree "$locked_worktree" \
  --campaign-dir "$campaign_dir" \
  --manifest "$manifest" \
  --trace "$trace"
```

Preparation validates hashes and creates directories; it does not run any
experiment.

## Submission

```bash
BLACKHOLE_STABILITY_CONTROL_REPO="$control_repo" \
BLACKHOLE_STABILITY_LOCKED_WORKTREE="$locked_worktree" \
BLACKHOLE_STABILITY_CAMPAIGN_DIR="$campaign_dir" \
BLACKHOLE_STABILITY_TORCH_SITE="$torch_site" \
BLACKHOLE_STABILITY_DEP_ROOT="$dep_root" \
sbatch \
  --account=marlowe-m000151-pm06 \
  --partition=batch \
  --qos=medium \
  --array=0-1%2 \
  --cpus-per-task=4 \
  --mem=64G \
  --output="$campaign_dir/logs/stability-%A_%a.out" \
  --error="$campaign_dir/logs/stability-%A_%a.err" \
  "$control_repo/rebuttal_experiments/slurm/cuda_blackhole_stability_locked.sbatch"
```

The batch script and verifier reject changed account, partition, array shape,
task mapping, CPU count, source, data, dependency, or campaign identity before
numerical work.

## Audit

After both tasks complete:

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH="$torch_site:$dep_root:$locked_worktree" \
/usr/bin/python3 rebuttal_experiments/blackhole_stability_locked.py aggregate \
  --control-repo "$control_repo" \
  --source-repo "$locked_worktree" \
  --campaign-dir "$campaign_dir"
```

The aggregate is write-once.  It imports the original strict auditor from the
locked worktree, requires exactly both raw results, claims, and attestations,
and emits paired speedups only for methods passing the frozen correctness
gate.  The 100 transitions are a dependent time series; their descriptive
sample dispersion is not presented as 100 independent experimental seeds.
