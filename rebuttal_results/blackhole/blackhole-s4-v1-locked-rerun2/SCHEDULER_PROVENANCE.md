# Slurm provenance for locked rerun 2

Captured from Marlowe `sacct` at `2026-07-25T19:15:46-07:00`, after every
array element had reached a terminal state. The result and attestation JSONs
remain the authoritative scientific artifacts.

| task | rank | element job | state | elapsed | start (PDT) | end (PDT) | node | requested/allocated CPUs | restarts | attested H100 UUID |
|---:|---:|---:|---|---:|---|---|---|---:|---:|---|
| 0 | 2,048 | 394034 | COMPLETED 0:0 | 00:00:16 | 18:44:23 | 18:44:39 | n18 | 12/12 | 0 | `GPU-14ceacf9-e3ae-c2f4-b342-57140e72e13c` |
| 1 | 4,096 | 394035 | COMPLETED 0:0 | 00:00:18 | 18:44:24 | 18:44:42 | n20 | 12/12 | 0 | `GPU-5a602863-fe24-7536-01b4-b1a2a50431f4` |
| 2 | 8,192 | 394036 | COMPLETED 0:0 | 00:00:46 | 18:44:24 | 18:45:10 | n01 | 12/12 | 0 | `GPU-b3b93b47-6d20-f656-2dbe-81a59966815f` |
| 3 | 16,384 | 394037 | COMPLETED 0:0 | 00:00:58 | 18:47:57 | 18:48:55 | n30 | 12/12 | 1 | `GPU-8f520a8e-8086-dccf-1a67-296b05677f3b` |
| 4 | 32,768 | 394038 | COMPLETED 0:0 | 00:04:30 | 18:44:24 | 18:48:54 | n02 | 12/12 | 0 | `GPU-b5dee3be-8167-2a39-48b0-f5c7e5b330d3` |
| 5 | 49,152 | 394039 | COMPLETED 0:0 | 00:06:28 | 18:44:24 | 18:50:52 | n02 | 12/12 | 0 | `GPU-4bd52887-92ae-9aa7-409d-6bd874718796` |
| 6 | 65,536 | 394033 | COMPLETED 0:0 | 00:13:55 | 18:44:24 | 18:58:19 | n02 | 12/12 | 0 | `GPU-2856ad6f-b9db-8bf9-9682-7ec6b27a8aa6` |

## Task 3 infrastructure recovery

The first allocation of task 3 was held by Slurm with
`user_env_retrieval_failed_requeued_held` before the batch script started.
At that point there was no worker output, persistent claim, result, or
attestation for rank 16,384. The same array element and element job ID were
released with nodes `n01,n18,n20` excluded; it then executed the scientific
script once on `n30`.

The completed accounting record therefore has `Restarts=1`, while the
scientific campaign has exactly one claim, one result, one attestation, and
one four-line worker log for task 3. This was an infrastructure recovery, not
an outcome-triggered rerun. The strict aggregate requires one array, exact
task-to-rank mapping, seven unique element job IDs, and seven distinct H100
UUIDs; all of those checks passed.
