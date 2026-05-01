"""Correctness + speed test for the compiled ATK within-chunk update.

The compiled path in ska_layer.py replaces the per-token Python einsum
loop with a torch.compile'd version that fuses the per-iteration ops
into a single device-side launch (CUDA-graph-friendly). This script:

  1. Runs the eager reference (`_atk_chunk_update_eager`) and the
     compiled wrapper (`atk_chunk_update`) on identical random inputs.
  2. Asserts forward outputs (S_final, out) agree to fp32 tolerance.
  3. Asserts gradients on all five inputs (S_init, K, V, Q, K_tilde)
     agree to fp32 tolerance.
  4. Reports a wall-clock comparison for a representative chunk size
     (BH=12, C=64, r=128, matching the 152M sanity model).

Run on CPU or any CUDA device:
    python test_atk_compile.py
    python test_atk_compile.py --cuda

Exits 0 on success, 1 on mismatch. Prints a one-line PASS/FAIL summary.
"""

import argparse
import math
import time

import torch

# Local import so this file is runnable from experiments/backend/.
from ska_layer import _atk_chunk_update_eager, atk_chunk_update, _get_atk_chunk_fn


def _make_inputs(BH: int, C: int, r: int, device, dtype, seed: int = 0):
    g = torch.Generator(device=device).manual_seed(seed)
    S = torch.randn(BH, r, r, device=device, dtype=dtype, generator=g)
    K = torch.randn(BH, C, r, device=device, dtype=dtype, generator=g)
    V = torch.randn(BH, C, r, device=device, dtype=dtype, generator=g)
    Q = torch.randn(BH, C, r, device=device, dtype=dtype, generator=g)
    KT = torch.randn(BH, C, r, device=device, dtype=dtype, generator=g)
    return S, K, V, Q, KT


def _to_grad(tensors):
    return [t.clone().detach().requires_grad_(True) for t in tensors]


def correctness_check(BH: int, C: int, r: int, device, fwd_rtol: float, bwd_rtol: float):
    """Run both paths and compare forward + backward.

    Uses **relative error vs the max magnitude of the reference tensor**
    rather than torch.allclose's mixed atol/rtol: in a long recurrent
    chain, op reordering by Inductor produces small per-element errors
    that look large relative to small elements but tiny relative to the
    tensor's overall scale. We care about scale-relative error.

    Forward and backward use separate thresholds because a 64-step
    sequential chain amplifies tiny forward errors in the gradient
    calculation; that's a property of the recurrence, not a bug.
    """
    print(f"\n=== Correctness: BH={BH}, C={C}, r={r}, device={device} ===")

    raw = _make_inputs(BH, C, r, device, torch.float32, seed=42)
    eager_inputs = _to_grad(raw)
    comp_inputs = _to_grad(raw)

    # Eager forward + backward.
    S_e, K_e, V_e, Q_e, KT_e = eager_inputs
    Sf_e, out_e = _atk_chunk_update_eager(S_e, K_e, V_e, Q_e, KT_e)
    # Mix output and final-state in the loss so all gradients are exercised.
    loss_e = out_e.sin().sum() + Sf_e.cos().sum() * 0.1
    loss_e.backward()

    # Compiled forward + backward.
    S_c, K_c, V_c, Q_c, KT_c = comp_inputs
    Sf_c, out_c = atk_chunk_update(S_c, K_c, V_c, Q_c, KT_c)
    loss_c = out_c.sin().sum() + Sf_c.cos().sum() * 0.1
    loss_c.backward()

    failures = []

    def cmp(name, a, b, rtol):
        diff = (a - b).abs().max().item()
        scale = b.abs().max().item()
        rel = diff / max(1e-12, scale)
        # Detect NaN explicitly so a silent NaN in either path fails fast.
        any_nan = bool(torch.isnan(a).any().item() or torch.isnan(b).any().item())
        ok = (not any_nan) and (rel <= rtol)
        marker = "OK  " if ok else "FAIL"
        nan_tag = "  NaN!" if any_nan else ""
        print(f"  {marker}  {name:20s}  rel={rel:.3e}  (max_abs={diff:.3e}, scale={scale:.3e}){nan_tag}")
        if not ok:
            failures.append(name)

    # Forward should be tight (single sweep, fp32 noise only).
    cmp("forward out",     out_c, out_e, fwd_rtol)
    cmp("forward S_final", Sf_c,  Sf_e,  fwd_rtol)
    # Backward through 64 sequential ops amplifies error; allow more.
    cmp("grad S_init",  S_c.grad,  S_e.grad,  bwd_rtol)
    cmp("grad K",       K_c.grad,  K_e.grad,  bwd_rtol)
    cmp("grad V",       V_c.grad,  V_e.grad,  bwd_rtol)
    cmp("grad Q",       Q_c.grad,  Q_e.grad,  bwd_rtol)
    cmp("grad K_tilde", KT_c.grad, KT_e.grad, bwd_rtol)

    if failures:
        print(f"  -> {len(failures)} mismatch(es): {', '.join(failures)}")
        return False
    print("  All matches within tolerance.")
    return True


def speed_check(BH: int, C: int, r: int, device, n_warmup: int = 5, n_runs: int = 30):
    """Wall-clock comparison of eager vs compiled forward+backward."""
    if device.type != "cuda":
        print("\n[speed] skipped on non-CUDA device")
        return

    print(f"\n=== Speed: BH={BH}, C={C}, r={r}, device={device} ===")

    raw = _make_inputs(BH, C, r, device, torch.float32, seed=7)

    def bench(fn_name: str, fn):
        inputs = _to_grad(raw)
        # Warmup (also pays compile cost the first time).
        for _ in range(n_warmup):
            for t in inputs:
                if t.grad is not None:
                    t.grad = None
            Sf, out = fn(*inputs)
            (out.sum() + Sf.sum() * 0.1).backward()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            for t in inputs:
                if t.grad is not None:
                    t.grad = None
            Sf, out = fn(*inputs)
            (out.sum() + Sf.sum() * 0.1).backward()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000 / n_runs
        print(f"  {fn_name:10s}  {ms:6.2f} ms / fwd+bwd  ({n_runs} runs after {n_warmup} warmup)")
        return ms

    ms_e = bench("eager", _atk_chunk_update_eager)
    ms_c = bench("compiled", atk_chunk_update)
    speedup = ms_e / max(1e-6, ms_c)
    print(f"  -> compiled is {speedup:.2f}x faster")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cuda", action="store_true", help="run on CUDA if available")
    p.add_argument("--fwd_rtol", type=float, default=1e-5,
                   help="relative-error tolerance for forward outputs")
    p.add_argument("--bwd_rtol", type=float, default=2e-2,
                   help="relative-error tolerance for gradients (looser due "
                        "to error amplification through the 64-step recurrence)")
    p.add_argument("--matmul_precision", type=str, default="highest",
                   choices=["highest", "high", "medium"],
                   help="torch.set_float32_matmul_precision. 'high' enables "
                        "TF32 on Ampere+ for ~2-4x speedup at slightly lower "
                        "precision; safe to use during training.")
    args = p.parse_args()

    # Match training-time precision so the test mirrors actual training.
    torch.set_float32_matmul_precision(args.matmul_precision)

    device = torch.device(
        "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    )
    print(f"Device:           {device}")
    print(f"Torch:            {torch.__version__}")
    print(f"Matmul precision: {args.matmul_precision}")
    print(f"Forward rtol:     {args.fwd_rtol}")
    print(f"Backward rtol:    {args.bwd_rtol}")

    # Force the compile to happen up front (rather than inside the timing).
    _ = _get_atk_chunk_fn()

    ok = True
    # Small test: cheap, broad gradient coverage.
    ok &= correctness_check(
        BH=4, C=8, r=16, device=device,
        fwd_rtol=args.fwd_rtol, bwd_rtol=args.bwd_rtol,
    )
    # Production-shape test: matches what 152M sanity uses (B=2,H=6 -> BH=12, r=128).
    ok &= correctness_check(
        BH=12, C=64, r=128, device=device,
        fwd_rtol=args.fwd_rtol, bwd_rtol=args.bwd_rtol,
    )

    if device.type == "cuda":
        speed_check(BH=12, C=64, r=128, device=device)

    print()
    if ok:
        print("PASS: compiled ATK matches eager (forward + 5 gradients).")
    else:
        print("FAIL: at least one mismatch above. Review and revert if needed.")

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
