# Locked black-hole ladder rerun

## Purpose

`blackhole-s4-v1-locked-rerun1` is a provenance-only rerun of the seven-rank
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
campaign_dir=/path/to/submitters/results/blackhole-s4-v1-locked-rerun1

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

Choose the account and H100 partition before submission.  The examples below
show placeholders because allocation names are account-specific.  The output
and error paths must be writable by the submitting account.

```bash
mkdir -p "$campaign_dir/logs"

BLACKHOLE_CONTROL_REPO="$control_repo" \
BLACKHOLE_LOCKED_WORKTREE="$locked_worktree" \
BLACKHOLE_CAMPAIGN_DIR="$campaign_dir" \
BLACKHOLE_DEP_ROOT="$control_repo/rebuttal_deps" \
sbatch \
  --account="<authorized-account>" \
  --partition="<authorized-h100-partition>" \
  --array=0-6%7 \
  --output="$campaign_dir/logs/ladder-%A_%a.out" \
  --error="$campaign_dir/logs/ladder-%A_%a.err" \
  "$control_repo/rebuttal_experiments/slurm/cuda_blackhole_rank_scaling_locked.sbatch"
```

The same script can be used on the current account with its authorized H100
partition.  Do not submit a second copy of an index whose claim exists.

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
  --expected-account "<authorized-account>" \
  --expected-partition "<authorized-h100-partition>"
```

The audit accepts multiple hosts because the ladder is intentionally
parallel.  It requires one Slurm array, exact task-to-rank mapping, identical
H100/Torch/CUDA/cuDNN/Python/platform contexts, the locked commit and source
hashes, exact data hashes, complete raw samples, and all original correctness
checks.  It never imports computational code from mutable `main`.
