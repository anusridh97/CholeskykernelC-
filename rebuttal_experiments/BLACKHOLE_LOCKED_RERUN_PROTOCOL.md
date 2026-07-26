# Locked black-hole ladder rerun 2

## Purpose

`blackhole-s4-v1-locked-rerun2` is a provenance-only rerun of the seven-rank
ladder from `blackhole-s4-v1`.  It exists because the original serial Slurm
job read `git rev-parse HEAD` separately at each rank while unrelated commits
were landing on `main`.  The driver, kernel, feature producer, feature trace,
manifest, transitions, ranks, repeats, tiles, precision, and correctness
checks did not change, but the recorded live commit did.  The original raw
files are preserved in `rebuttal_results/blackhole/blackhole-s4-v1/`.

This rerun executes the **unmodified v1 driver** from a detached worktree at
commit `96c8018cdd0b56ffe1fd93bddf48fa4affb34697`.  Consequently every inner
result correctly retains `campaign_tag: blackhole-s4-v1` and the v1 schema.
The outer directory, lock, attestations, and aggregate use the distinct rerun
tag.  The v1 feature manifest is copied byte-for-byte; its embedded campaign
tag is intentionally not edited.

The first locked attempt, `blackhole-s4-v1-locked-rerun1` (Slurm array
`393982`), failed closed before any numerical code ran.  Marlowe's `batch`
partition allocated 12 CPU cores to each one-H100 task even though four were
requested: its 17,000-MiB maximum memory per CPU rounded the requested 192 GiB
to 12 allocated CPUs.  The four-core lock therefore rejected every task after
it created its persistent claim.  All seven failed claims and logs are
retained and no result JSON exists for that attempt.  Rerun 2 records the
scheduler's actual 12-core allocation; this is an execution-metadata
correction only and does not alter the unmodified numerical driver or any
frozen experimental choice.
The outer verifier now rejects any account other than
`marlowe-m000151-pm06`, any partition other than `batch`, and any allocated
CPU count other than 12 before invoking the numerical driver.  Preparation,
task verification, attestation, and aggregation also require the campaign
directory basename to be exactly `blackhole-s4-v1-locked-rerun2`, preventing
failed rerun-1 claims from being mistaken for rerun-2 state.

## Frozen experiment

- Official source: Polymathic AI `post_neutron_star_merger`, revision
  `1fd582f3e29749ac4d0c14bc816bddb2394258cc`, scenario 4.
- Feature trace SHA-256:
  `bdf1619a70af70585af317d25aa69452fcb8e94d8417e525de9d50f878e530b4`.
- Feature manifest SHA-256:
  `3b8c6e32da3842b8a9f21ee722e8b9fb1093899744d283d4e939427dd116a5c2`.
- Ranks, in array-task order:
  `2048, 4096, 8192, 16384, 32768, 49152, 65536`.
- All numerical and timing settings are those enforced by the unchanged v1
  driver.  No result from job 393886 was used to change the protocol.
- Array task `i` may run only rank `ranks[i]`.  Seven tasks may occupy seven
  H100s concurrently, but concurrency is a scheduling property rather than a
  statistical condition.

Each task atomically creates a persistent claim directory before invoking the
driver or hashing the 47 MB trace.  Existing claims, results, or attestations
cause failure.  Failed claims are retained for investigation; they must never
be removed merely to seek a more favorable timing.  The driver already
refuses to overwrite a result.  A successful task adds a write-once
attestation that hashes the raw result and records its Slurm account,
partition, array identity, node, and GPU identity.  The aggregate is emitted
only after all seven raw results and all seven attestations pass the original
v1 audit plus the rerun audit.

## Preparation

The control repository may be Ravid's checkout while the source repository,
detached worktree, output directory, and submission belong to an explicitly
authorized Alex account.  Do not share credentials or submit as another user
without authorization.

Set paths appropriate for the submitting account:

```bash
control_repo=/users/ravid/git/CholeskykernelC-
source_repo=/path/to/submitters/CholeskykernelC-
locked_worktree=/path/to/submitters/CholeskykernelC--blackhole-locked
campaign_dir=/path/to/submitters/results/blackhole-s4-v1-locked-rerun2

/usr/bin/python3 \
  "$control_repo/rebuttal_experiments/blackhole_locked_rerun.py" prepare \
  --control-repo "$control_repo" \
  --source-repo "$source_repo" \
  --locked-worktree "$locked_worktree" \
  --campaign-dir "$campaign_dir"
```

`prepare` verifies every hash in the committed campaign lock, creates (or
verifies) the detached worktree, links the already-downloaded trace into that
worktree, and copies the lock and v1 manifest byte-for-byte into the new
campaign directory.  It does not execute an experiment.

The manifest and trace need not live in the submitter's clone.  They are
readable at the frozen paths under Ravid's checkout; pass `--manifest` and
`--trace` to `prepare` when using copied locations.  For an isolated
dependency tree owned by the submitting account:

```bash
mkdir -p /path/to/submitters/rebuttal_deps
rsync -a /users/ravid/git/CholeskykernelC-/rebuttal_deps/ \
  /path/to/submitters/rebuttal_deps/
```

Point `BLACKHOLE_DEP_ROOT` at that copy.  Dependencies are not treated as
source code, but each result records Python, Torch, CUDA, and cuDNN versions;
the audit requires an identical environment across all seven tasks.

## Submission

Rerun 2 is assigned to Alex's authorized `marlowe-m000151-pm06` allocation,
the `batch` partition, and the `medium` QOS.  The partition allocates 12 CPU
cores with one H100, and both the batch script and outer audit require the
resulting `SLURM_CPUS_PER_TASK=12`.  The output and error paths must be
writable by the submitting account.

```bash
mkdir -p "$campaign_dir/logs"

BLACKHOLE_CONTROL_REPO="$control_repo" \
BLACKHOLE_LOCKED_WORKTREE="$locked_worktree" \
BLACKHOLE_CAMPAIGN_DIR="$campaign_dir" \
BLACKHOLE_DEP_ROOT="$control_repo/rebuttal_deps" \
sbatch \
  --account="marlowe-m000151-pm06" \
  --partition="batch" \
  --qos="medium" \
  --cpus-per-task=12 \
  --mem=192G \
  --array=0-6%7 \
  --output="$campaign_dir/logs/ladder-%A_%a.out" \
  --error="$campaign_dir/logs/ladder-%A_%a.err" \
  "$control_repo/rebuttal_experiments/slurm/cuda_blackhole_rank_scaling_locked.sbatch"
```

Do not submit this target-specific rerun script under a different account or
partition, and do not submit a second copy of an index whose claim exists.
The strict audit requires seven distinct H100 UUIDs and seven distinct Slurm
element job IDs.  Submit only when the allocation can place all seven array
elements without recycling one physical GPU; a staggered array that reuses a
GPU is deliberately rejected rather than silently changing the hardware
design.

## Audit

After all seven tasks succeed, supply the account and partition selected
before submission:

```bash
PYTHONPATH="$dep_root:$locked_worktree" /usr/bin/python3 \
  "$control_repo/rebuttal_experiments/blackhole_locked_rerun.py" aggregate \
  --control-repo "$control_repo" \
  --source-repo "$locked_worktree" \
  --locked-worktree "$locked_worktree" \
  --campaign-dir "$campaign_dir" \
  --expected-account "marlowe-m000151-pm06" \
  --expected-partition "batch"
```

The audit accepts multiple hosts because the ladder is intentionally
parallel.  It requires one Slurm array, exact task-to-rank mapping, identical
H100/Torch/CUDA/cuDNN/Python/platform contexts, the locked commit and source
hashes, exact data hashes, complete raw samples, and all original correctness
checks.  It never imports computational code from mutable `main`.
