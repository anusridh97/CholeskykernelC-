# Scheduler provenance: locked secondary stability trace

This is a post-run transcription of `sacct` plus the two write-once
attestations.  The strict machine-readable audit is `aggregate.json`.

## Final array

Slurm array `394122` ran the two preregistered ranks concurrently under
account `marlowe-m000151-pm06`, partition `batch`, QOS `medium`.  Each task
received four CPUs, 64 GiB host memory, and one H100; both stderr files are
empty.

| task | rank | element job | state | elapsed | start (PDT) | end (PDT) | node | physical GPU |
|---:|---:|---:|:---|---:|:---|:---|:---|:---|
| 0 | 8,192 | 394123 | COMPLETED 0:0 | 00:00:29 | 2026-07-25 20:01:10 | 2026-07-25 20:01:39 | n05 | `GPU-8e6fd757-27ce-6a0a-08d5-2ad20c2188a2` |
| 1 | 32,768 | 394122 | COMPLETED 0:0 | 00:04:07 | 2026-07-25 20:01:10 | 2026-07-25 20:05:17 | n05 | `GPU-68e8f641-9860-500b-4e86-84bc3398b52d` |

The final audit passed, all method/rank correctness gates passed, and no
headline was withheld.

## Immutable identities

- Locked numerical commit:
  `96c8018cdd0b56ffe1fd93bddf48fa4affb34697`.
- Campaign-lock Git commit:
  `dc7989b57ff81d77adf2e512b985f2d391cb08f3`.
- Campaign-lock SHA-256:
  `f34cbf424cc7cf6022c8825d166f17ae3b6f8b2d701454e0a96eb95cc81d4c69`.
- Feature-manifest SHA-256:
  `3b8c6e32da3842b8a9f21ee722e8b9fb1093899744d283d4e939427dd116a5c2`.
- Feature-trace SHA-256:
  `bdf1619a70af70585af317d25aa69452fcb8e94d8417e525de9d50f878e530b4`.
- Audited aggregate SHA-256:
  `06aac4121afe196184dcd1b587f493488d4bacd79bf58581dc1a7013fea7dfab`.

## Preserved fail-closed attempts

- Rerun 1 stopped in a data-free remote preflight because its dependency
  lock named the invoked `/usr/bin/python3` symlink while verification used
  its normalized `/usr/bin/python3.10` target.  It submitted no job, created
  no task claim, and ran no numerical code.
- Rerun 2 array `394115` exposed an array-ID ambiguity in the outer scheduler
  query.  Task 0 (rank 8,192) completed and was attested; task 1 stopped in
  the outer verifier before numerical entry because its bare job ID also
  selected the array master.  Both claims, logs, and the completed artifact
  remain preserved in Alex's scratch campaign.  No numerical outcome from
  that incomplete attempt changed rerun 3.

Rerun 3 changed only the scheduler query to the unambiguous
`ArrayJobID_TaskID` selector and reran both fixed ranks in one array context.
