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


def correctness_check(BH: int, C: int, r: int, device, atol: float, rtol: float):
    """Run both paths and compare forward + backward."""
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

    def cmp(name, a, b):
        diff = (a - b).abs().max().item()
        rel = diff / max(1e-12, b.abs().max().item())
        ok = torch.allclose(a, b, atol=atol, rtol=rtol)
        marker = "OK  " if ok else "FAIL"
        print(f"  {marker}  {name:20s}  max_abs={diff:.3e}  max_rel={rel:.3e}")
        if not ok:
            failures.append(name)

    cmp("forward out",  out_c, out_e)
    cmp("forward S_final", Sf_c, Sf_e)
    cmp("grad S_init",  S_c.grad,  S_e.grad)
    cmp("grad K",       K_c.grad,  K_e.grad)
    cmp("grad V",       V_c.grad,  V_e.grad)
    cmp("grad Q",       Q_c.grad,  Q_e.grad)
    cmp("grad K_tilde", KT_c.grad, KT_e.grad)

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
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--rtol", type=float, default=1e-4)
    args = p.parse_args()

    device = torch.device(
        "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    )
    print(f"Device: {device}")
    print(f"Torch:  {torch.__version__}")

    # Force the compile to happen up front (rather than inside the timing).
    _ = _get_atk_chunk_fn()

    ok = True
    # Small test: cheap, broad gradient coverage.
    ok &= correctness_check(BH=4, C=8, r=16, device=device, atol=args.atol, rtol=args.rtol)
    # Production-shape test: matches what 152M sanity uses (B=2,H=6 -> BH=12, r=128).
    ok &= correctness_check(BH=12, C=64, r=128, device=device, atol=args.atol, rtol=args.rtol)

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
