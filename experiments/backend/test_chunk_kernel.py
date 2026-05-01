"""GPU correctness + speed test for the fused chunk kernel.

The kernel in student_pkg/doublesided_cholesky.py:_fused_chunk_kernel
processes C sequential per-token rank-1 SKA updates inside ONE GPU program
per (b,h), instead of C separate Python-launched kernel calls. This script:

  1. Runs the OLD per-token Triton path (calling _forward_triton C times
     plus solve_triangular for T per token), accumulating the same
     (L_seq, Aw_seq, vz_seq, vw_seq, kt_seq, T_seq) tensors that the new
     chunk kernel returns.
  2. Runs the NEW chunk kernel (incremental_chunk_update_triton) on the
     same inputs.
  3. Asserts every output tensor matches to fp32 tolerance.
  4. Benchmarks both paths and reports speedup.

CUDA-only: skips with a message if CUDA isn't available.

Run:
    python test_chunk_kernel.py
    python test_chunk_kernel.py --bh 12 --c 64 --r 128
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# Make the kernel package importable from this directory.
_HERE = Path(__file__).resolve().parent
for _p in [_HERE, *_HERE.parents]:
    if (_p / "student_pkg").is_dir():
        sys.path.insert(0, str(_p / "student_pkg"))
        break

# Importing ska_recurrence pulls in the chunk kernel and the per-token path.
from ska_recurrence import _forward_triton, _HAVE_CHUNK_KERNEL  # noqa: E402

if _HAVE_CHUNK_KERNEL:
    from doublesided_cholesky import incremental_chunk_update_triton  # noqa: E402


def per_token_reference(L, Aw, K, V):
    """Reproduce the OLD ska_recurrence._chunk_forward Triton-CUDA path.

    Calls _forward_triton C times sequentially, plus the per-token
    solve_triangular for T_t. Returns the same six stacked tensors that
    incremental_chunk_update_triton returns.
    """
    BH, C, r = K.shape
    device, dtype = L.device, L.dtype

    L_seq  = torch.empty(BH, C + 1, r, r, device=device, dtype=dtype)
    Aw_seq = torch.empty(BH, C + 1, r, r, device=device, dtype=dtype)
    vz_seq = torch.empty(BH, C, r, device=device, dtype=dtype)
    vw_seq = torch.empty(BH, C, r, device=device, dtype=dtype)
    kt_seq = torch.empty(BH, C, r, device=device, dtype=dtype)
    T_seq  = torch.empty(BH, C, r, r, device=device, dtype=dtype)

    L_seq[:, 0]  = L
    Aw_seq[:, 0] = Aw

    L_curr = L
    Aw_curr = Aw
    for c in range(C):
        z_c = K[:, c]
        w_c = V[:, c]
        L_curr, Aw_curr, vz_c, vw_c, kt_c = _forward_triton(L_curr, Aw_curr, z_c, w_c)
        T_c = torch.linalg.solve_triangular(L_curr, L_seq[:, c], upper=False)
        L_seq[:, c + 1]  = L_curr
        Aw_seq[:, c + 1] = Aw_curr
        vz_seq[:, c]     = vz_c
        vw_seq[:, c]     = vw_c
        kt_seq[:, c]     = kt_c
        T_seq[:, c]      = T_c

    return L_seq, Aw_seq, vz_seq, vw_seq, kt_seq, T_seq


def make_inputs(BH, C, r, device, dtype, seed=0):
    """Bounded-magnitude inputs that keep the Cholesky update well-conditioned.

    G = LL^T must stay PD across the chunk. Using small random keys
    (norm ~ 0.1) plus a strong ridge keeps the spectrum bounded.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    A = torch.randn(BH, r, r, device=device, dtype=dtype, generator=g)
    G = A @ A.transpose(-1, -2) + (2.0 * r) * torch.eye(r, device=device, dtype=dtype)
    L = torch.linalg.cholesky(G).contiguous()
    Aw = torch.randn(BH, r, r, device=device, dtype=dtype, generator=g) * 0.1
    K = torch.randn(BH, C, r, device=device, dtype=dtype, generator=g) * 0.1
    V = torch.randn(BH, C, r, device=device, dtype=dtype, generator=g) * 0.1
    return L, Aw, K, V


def cmp_tensor(name, a, b, rtol=1e-4, atol=1e-5):
    """Scale-relative comparison; fails on NaN."""
    if torch.isnan(a).any() or torch.isnan(b).any():
        print(f"  FAIL  {name:14s}  NaN detected")
        return False
    diff  = (a - b).abs().max().item()
    scale = b.abs().max().item()
    rel = diff / max(1e-12, scale)
    ok = rel <= rtol or diff <= atol
    print(f"  {'OK  ' if ok else 'FAIL':4s}  {name:14s}  rel={rel:.3e}  (max_abs={diff:.3e}, scale={scale:.3e})")
    return ok


def correctness(BH, C, r, device):
    print(f"\n=== Correctness: BH={BH}, C={C}, r={r}, device={device} ===")
    L, Aw, K, V = make_inputs(BH, C, r, device, torch.float32, seed=42)

    with torch.no_grad():
        L_p, Aw_p, vz_p, vw_p, kt_p, T_p = per_token_reference(
            L.clone(), Aw.clone(), K.clone(), V.clone()
        )
        L_c, Aw_c, vz_c, vw_c, kt_c, T_c = incremental_chunk_update_triton(
            L.clone(), Aw.clone(), K.clone(), V.clone()
        )

    rtol = 1e-4
    ok = True
    ok &= cmp_tensor("L_seq",  L_c,  L_p,  rtol=rtol)
    ok &= cmp_tensor("Aw_seq", Aw_c, Aw_p, rtol=rtol)
    ok &= cmp_tensor("vz_seq", vz_c, vz_p, rtol=rtol)
    ok &= cmp_tensor("vw_seq", vw_c, vw_p, rtol=rtol)
    ok &= cmp_tensor("kt_seq", kt_c, kt_p, rtol=rtol)
    ok &= cmp_tensor("T_seq",  T_c,  T_p,  rtol=rtol)
    return ok


def speed(BH, C, r, device, n_warmup=5, n_runs=30):
    if device.type != "cuda":
        return
    print(f"\n=== Speed: BH={BH}, C={C}, r={r}, device={device} ===")
    L, Aw, K, V = make_inputs(BH, C, r, device, torch.float32, seed=7)

    def bench(name, fn):
        for _ in range(n_warmup):
            fn(L.clone(), Aw.clone(), K.clone(), V.clone())
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            fn(L.clone(), Aw.clone(), K.clone(), V.clone())
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000 / n_runs
        print(f"  {name:14s}  {ms:7.3f} ms / chunk_forward  ({n_runs} runs after {n_warmup} warmup)")
        return ms

    with torch.no_grad():
        ms_p = bench("per-token",  per_token_reference)
        ms_c = bench("chunk fused", incremental_chunk_update_triton)

    speedup = ms_p / max(1e-6, ms_c)
    print(f"  -> chunk fused is {speedup:.2f}x faster")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bh", type=int, default=12)
    p.add_argument("--c",  type=int, default=64)
    p.add_argument("--r",  type=int, default=128)
    p.add_argument("--matmul_precision", type=str, default="high",
                   choices=["highest", "high", "medium"])
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; skipping (this test requires the Triton kernel).")
        raise SystemExit(0)

    if not _HAVE_CHUNK_KERNEL:
        print("Chunk kernel import failed; cannot test.")
        raise SystemExit(1)

    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda")
    print(f"Device:           {device}")
    print(f"Torch:            {torch.__version__}")
    print(f"Matmul precision: {args.matmul_precision}")

    ok = True
    # Small case: catches obvious bugs fast.
    ok &= correctness(BH=4,  C=8,  r=16, device=device)
    # Production-shape: BH=12 (B=2*H=6), C=64, r=128.
    ok &= correctness(BH=args.bh, C=args.c, r=args.r, device=device)

    speed(BH=args.bh, C=args.c, r=args.r, device=device)

    print()
    if ok:
        print("PASS: chunk-fused kernel matches per-token Triton path.")
    else:
        print("FAIL: at least one mismatch above. The chunk kernel needs review.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
