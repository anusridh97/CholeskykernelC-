"""
Benchmark: T sequential single Triton calls vs one batched Triton call.

Sweeps over batch size T and matrix size r to show where the batched kernel
wins. Run from the student_pkg directory:

    python -m tests.bench_batched

Requires CUDA. Each cell is the median of REPS timed runs after a warmup pass.
Times are in milliseconds; speedup is sequential / batched.

Output is mirrored to tests/logs/bench_batched_<job_id>_<timestamp>.log.
"""

import os
import sys
import datetime
import pathlib
import torch

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

_log_file = None


def setup_logging():
    global _log_file
    log_dir = pathlib.Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", f"local_{os.getpid()}")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"bench_batched_{job_id}_{ts}.log"
    _log_file = open(path, "w")
    log(f"Log: {path}")
    return path


def log(msg=""):
    print(msg, flush=True)
    if _log_file is not None:
        print(msg, file=_log_file, flush=True)


# ---------------------------------------------------------------------------

if not torch.cuda.is_available():
    log("CUDA not available — skipping GPU benchmark.")
    sys.exit(0)

from doublesided_cholesky import incremental_update_triton, incremental_update_triton_batched

REPS    = 20
WARMUP  = 3
T_SIZES = [1, 4, 16, 64, 128]
R_SIZES = [16, 64, 128, 256]


def make_batch(T, r, device="cuda"):
    torch.manual_seed(0)
    A = torch.randn(T, r, r, device=device)
    G = A @ A.mT + r * torch.eye(r, device=device).unsqueeze(0)
    L = torch.linalg.cholesky(G).contiguous()
    Aw = torch.randn(T, r, r, device=device)
    z = torch.randn(T, r, device=device) * 0.1
    w = torch.randn(T, r, device=device) * 0.1
    return L, Aw, z, w


def time_sequential(L, Aw, z, w, reps):
    T = L.shape[0]
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for t in range(T):
            incremental_update_triton(
                L[t].clone(), Aw[t].clone(), z[t].clone(), w[t].clone()
            )
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def time_batched(L, Aw, z, w, reps):
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        incremental_update_triton_batched(
            L.clone(), Aw.clone(), z.clone(), w.clone()
        )
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def run_sweep():
    log(f"GPU:           {torch.cuda.get_device_name(0)}")
    log(f"SLURM_JOB_ID:  {os.environ.get('SLURM_JOB_ID', 'n/a')}")
    log(f"Started:       {datetime.datetime.now().isoformat(timespec='seconds')}")
    log(f"REPS={REPS}  WARMUP={WARMUP}")
    log()
    log(f"{'r':>6} {'T':>6}  {'seq (ms)':>10}  {'bat (ms)':>10}  {'speedup':>8}")
    log("-" * 50)

    for r in R_SIZES:
        for T in T_SIZES:
            if r > 256 and T > 64:
                continue
            L, Aw, z, w = make_batch(T, r)
            for _ in range(WARMUP):
                for t in range(T):
                    incremental_update_triton(
                        L[t].clone(), Aw[t].clone(), z[t].clone(), w[t].clone()
                    )
                incremental_update_triton_batched(
                    L.clone(), Aw.clone(), z.clone(), w.clone()
                )
            seq_ms = time_sequential(L, Aw, z, w, REPS)
            bat_ms = time_batched(L, Aw, z, w, REPS)
            speedup = seq_ms / bat_ms if bat_ms > 0 else float("inf")
            log(f"{r:>6} {T:>6}  {seq_ms:>10.3f}  {bat_ms:>10.3f}  {speedup:>7.2f}x")
        log()


if __name__ == "__main__":
    log_path = setup_logging()
    try:
        run_sweep()
    finally:
        if _log_file is not None:
            _log_file.close()
        print(f"Log saved to {log_path}")
