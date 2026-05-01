"""Memory leak diagnostic for the SKA training loop.

Bypasses HuggingFace dataset loading and tokenization. Builds the same
SKAModel that sanity uses, generates synthetic input ids, and runs 5
forward+backward+optim iterations on identical inputs. Prints CUDA
memory snapshots at every key boundary (after forward, after backward,
after optim step, after gc+empty_cache).

Patterns we distinguish:
  - LEAK: memory grows monotonically across iterations even after
          gc.collect() + torch.cuda.empty_cache(). Specific component
          (forward, backward) holding references that won't free.
  - SIZING: memory peaks at the same value every iteration. Workload is
            just too large for the GPU.
  - FRAGMENTATION: allocated stays bounded but reserved grows. Allocator
                   chunking. Mitigated by PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.

Usage:
    python debug_memory.py                      # ctx 1024, no checkpoint
    python debug_memory.py --grad_checkpoint    # turn checkpoint on
    python debug_memory.py --ctx_len 512        # try smaller context
    python debug_memory.py --bf16               # try bf16

The slurm wrapper runs three configurations sequentially so we can see
how memory scales with ctx_len.
"""

import argparse
import contextlib
import gc
import os
import sys
import time
from pathlib import Path

import torch

# Ensure local + student_pkg imports work.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
for _p in [_HERE, *_HERE.parents]:
    if (_p / "student_pkg").is_dir():
        sys.path.insert(0, str(_p / "student_pkg"))
        break


def _gb(b):
    return b / 1e9


def mem_snapshot(label: str) -> str:
    """One-line CUDA memory readout. Pad label for column alignment."""
    alloc = _gb(torch.cuda.memory_allocated())
    reserved = _gb(torch.cuda.memory_reserved())
    peak = _gb(torch.cuda.max_memory_allocated())
    return (
        f"{label:<42s} alloc={alloc:6.2f} GB  "
        f"reserved={reserved:6.2f} GB  peak={peak:6.2f} GB"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", type=int, default=5,
                        help="number of forward+backward+optim iterations")
    parser.add_argument("--ctx_len", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1,
                        help="micro-batch size (sanity uses 1)")
    parser.add_argument("--grad_checkpoint", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no_optim", action="store_true",
                        help="skip optim.step() to isolate forward+backward")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; this test requires a GPU.")
        sys.exit(1)

    # Apply the same env setup the real training does.
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.grad_checkpoint:
        # Disable ATK compile (matches train.py auto-set behavior).
        os.environ["SKA_NO_COMPILE_ATK"] = "1"

    # Now import ska_model AFTER setting the env var.
    from ska_model import make_165m_model
    from ska_recurrence import _SKAChunkUpdate  # for ctx-counting probe

    # Optional: count live _SKAChunkUpdate instances via Python gc.
    # Custom autograd.Functions are Python-managed; gc.get_objects sees them.
    def count_chunk_ctxes() -> int:
        n = 0
        for obj in gc.get_objects():
            try:
                if type(obj).__name__ == "_SKAChunkUpdateBackward":
                    n += 1
            except ReferenceError:
                pass
        return n

    device = torch.device("cuda")
    print(f"==== debug_memory.py ====")
    print(f"Device:           {device}  ({torch.cuda.get_device_name(0)})")
    print(f"Total VRAM:       {_gb(torch.cuda.get_device_properties(0).total_memory):.1f} GB")
    print(f"Torch:            {torch.__version__}")
    print(f"Args:             ctx_len={args.ctx_len}, batch={args.batch}, "
          f"grad_ckpt={args.grad_checkpoint}, bf16={args.bf16}, "
          f"n_iter={args.n_iter}, no_optim={args.no_optim}")
    print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}")
    print()

    # Build model.
    print(mem_snapshot("BASELINE (no model)"))
    model = make_165m_model().to(device)
    if args.grad_checkpoint:
        model.grad_checkpoint = True
    model.train()
    print(mem_snapshot("after model build"))

    if not args.no_optim:
        optim = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
        # Force optimizer state allocation by stepping once on dummy grads.
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        optim.step()
        optim.zero_grad(set_to_none=True)
        print(mem_snapshot("after optimizer init"))
    else:
        optim = None

    # Synthetic input. Same shape every iteration so we test true growth,
    # not shape variation.
    vocab_size = 50272
    input_ids = torch.randint(0, vocab_size, (args.batch, args.ctx_len), device=device)
    labels = torch.randint(0, vocab_size, (args.batch, args.ctx_len), device=device)
    print(mem_snapshot("after input tensors"))
    print()

    # The actual loop.
    for it in range(args.n_iter):
        torch.cuda.reset_peak_memory_stats()
        t_start = time.perf_counter()

        # FORWARD
        if args.bf16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids, labels=labels)
        else:
            out = model(input_ids, labels=labels)
        torch.cuda.synchronize()
        t_fwd = time.perf_counter() - t_start
        print(mem_snapshot(f"[iter {it}] after forward (t={t_fwd:5.1f}s)"))

        # BACKWARD
        loss = out["loss"]
        t_b = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        t_bwd = time.perf_counter() - t_b
        print(mem_snapshot(f"[iter {it}] after backward (t={t_bwd:5.1f}s)"))

        # OPTIM
        if optim is not None:
            optim.step()
            optim.zero_grad(set_to_none=True)
            print(mem_snapshot(f"[iter {it}] after optim.step"))

        # Drop references to the autograd graph and force GC.
        del out, loss
        gc.collect()
        torch.cuda.empty_cache()
        print(mem_snapshot(f"[iter {it}] after gc+empty_cache"))

        n_ctx = count_chunk_ctxes()
        print(f"           live _SKAChunkUpdate ctx objects: {n_ctx}")
        print()

    print("=" * 70)
    print(mem_snapshot("FINAL"))
    print()
    print("Detailed allocator summary:")
    print(torch.cuda.memory_summary(abbreviated=True))


if __name__ == "__main__":
    main()
