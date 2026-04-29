"""
SKA training entry point.

Usage:
  python train.py --config sanity   # 165M, 100M tokens (1h on 1xH100)
  python train.py --config prod     # 340M, 5B tokens (~14h on 4xH100)

Hyperparameters match PDN Appendix E.1:
  - AdamW, betas=(0.9, 0.95), weight decay 0.01
  - peak LR 4e-4, cosine to 4e-5
  - warmup over 0.5B tokens (scaled in sanity mode)
  - batch size 0.5M tokens
  - context length 2048
  - gradient clipping 1.0

Streams SlimPajama from HuggingFace; no on-disk dataset required.
Distributed via torchrun + DDP.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset, DataLoader

from ska_model import make_340m_model, make_165m_model

# ----------------------------- Configurations ---------------------------- #

CONFIGS = {
    "sanity": {
        "model_fn": make_165m_model,
        "total_tokens": int(100e6),         # 100M
        "warmup_tokens": int(5e6),
        "batch_tokens": int(0.25e6),         # 250K (smaller for tiny model)
        "ctx_len": 2048,
        "peak_lr": 4e-4,
        "min_lr": 4e-5,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "save_every_tokens": int(50e6),
        "log_every_steps": 10,
    },
    "prod": {                                # 4-GPU Farmshare default
        "model_fn": make_340m_model,
        "total_tokens": int(8e9),           # 8B; ~half PDN's 15B, fits 4-5 days on 4 L40S
        "warmup_tokens": int(0.4e9),         # 400M (5% of total; PDN uses 500M)
        "batch_tokens": int(0.5e6),         # PDN-matched global batch
        "ctx_len": 2048,
        "peak_lr": 3e-4,                    # slightly below PDN's 4e-4 for safety
        "min_lr": 3e-5,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "save_every_tokens": int(1e9),       # checkpoint every 1B tokens
        "log_every_steps": 50,
    },
    "prod_1gpu": {                          # single-L40S Farmshare default
        "model_fn": make_340m_model,
        "total_tokens": int(3e9),           # 3B; fits 5-6 chained 24h jobs at FP32
        "warmup_tokens": int(0.15e9),        # 150M (5% of total)
        "batch_tokens": int(0.25e6),         # smaller global batch on 1 GPU
        "ctx_len": 2048,
        "peak_lr": 2.5e-4,
        "min_lr": 2.5e-5,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "save_every_tokens": int(0.5e9),     # checkpoint every 500M tokens
        "log_every_steps": 50,
    },
}

# ------------------------------ Data pipeline --------------------------- #

class SlimPajamaStream(IterableDataset):
    """Streams SlimPajama via HuggingFace datasets, tokenizes on the fly,
    and yields (ctx_len + 1)-token windows packed across documents.
    """

    def __init__(self, tokenizer, ctx_len: int, seed: int = 0):
        from datasets import load_dataset
        self.ctx_len = ctx_len
        self.tokenizer = tokenizer
        self.ds = load_dataset(
            "cerebras/SlimPajama-627B",
            split="train",
            streaming=True,
        ).shuffle(seed=seed, buffer_size=10_000)

    def __iter__(self):
        buffer = []
        for ex in self.ds:
            ids = self.tokenizer.encode(ex["text"], add_special_tokens=False)
            ids.append(self.tokenizer.eos_token_id)
            buffer.extend(ids)
            while len(buffer) >= self.ctx_len + 1:
                window = buffer[: self.ctx_len + 1]
                buffer = buffer[self.ctx_len :]  # one-token overlap
                input_ids = torch.tensor(window[:-1], dtype=torch.long)
                labels = torch.tensor(window[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


def make_dataloader(tokenizer, ctx_len, micro_batch_size, num_workers=2, seed=0):
    ds = SlimPajamaStream(tokenizer, ctx_len, seed=seed)
    return DataLoader(
        ds,
        batch_size=micro_batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )


# ----------------------------- LR schedule ------------------------------ #

def cosine_lr(step_tokens: int, warmup_tokens: int, total_tokens: int, peak_lr: float, min_lr: float) -> float:
    if step_tokens < warmup_tokens:
        return peak_lr * step_tokens / warmup_tokens
    progress = (step_tokens - warmup_tokens) / max(1, total_tokens - warmup_tokens)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ------------------------------ Main loop ------------------------------- #

def setup_distributed():
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_world_size(), dist.get_rank()
    return 0, 1, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS), required=True)
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--micro_batch", type=int, default=4,
                        help="per-GPU micro batch size (sequences)")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    local_rank, world_size, global_rank = setup_distributed()
    is_main = global_rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed + global_rank)

    # -------------------- Tokenizer + model -------------------- #
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = cfg["model_fn"](vocab_size=tok.vocab_size).to(device)
    if is_main:
        print(f"Model: {model.num_parameters() / 1e6:.1f}M params")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["peak_lr"],
        betas=(0.9, 0.95),
        weight_decay=cfg["weight_decay"],
    )

    # -------------------- Data -------------------- #
    sequences_per_batch_global = cfg["batch_tokens"] // cfg["ctx_len"]
    grad_accum = max(1, sequences_per_batch_global // (args.micro_batch * world_size))
    if is_main:
        print(f"Global batch: {sequences_per_batch_global} seqs ({cfg['batch_tokens']} tokens)")
        print(f"Per-GPU micro batch: {args.micro_batch}, grad accum: {grad_accum}")

    loader = make_dataloader(tok, cfg["ctx_len"], args.micro_batch, num_workers=2, seed=args.seed + global_rank)

    # -------------------- Resume -------------------- #
    start_tokens = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        (model.module if world_size > 1 else model).load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        start_tokens = ckpt["tokens"]
        if is_main:
            print(f"Resumed from {args.resume} at {start_tokens} tokens")

    # -------------------- Training -------------------- #
    model.train()
    tokens_seen = start_tokens
    last_save = start_tokens
    step = 0
    grad_step = 0
    optim.zero_grad(set_to_none=True)
    t_start = time.time()

    loader_iter = iter(loader)

    while tokens_seen < cfg["total_tokens"]:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        # Forward + backward, scaled for grad accumulation.
        out = (model.module if world_size > 1 else model)(input_ids, labels=labels) \
              if False else model(input_ids, labels=labels)
        loss = out["loss"] / grad_accum
        loss.backward()
        step += 1

        # Step optimizer when we've accumulated a full global batch.
        if step % grad_accum == 0:
            grad_step += 1
            tokens_seen += cfg["batch_tokens"]
            lr = cosine_lr(
                tokens_seen,
                cfg["warmup_tokens"],
                cfg["total_tokens"],
                cfg["peak_lr"],
                cfg["min_lr"],
            )
            for pg in optim.param_groups:
                pg["lr"] = lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optim.step()
            optim.zero_grad(set_to_none=True)

            if is_main and grad_step % cfg["log_every_steps"] == 0:
                elapsed = time.time() - t_start
                tps = (tokens_seen - start_tokens) / max(1, elapsed)
                eta = (cfg["total_tokens"] - tokens_seen) / max(1, tps) / 3600
                print(
                    f"step {grad_step:6d} | tokens {tokens_seen/1e6:7.1f}M | "
                    f"loss {loss.item() * grad_accum:6.4f} | lr {lr:.2e} | "
                    f"tps {tps/1e3:5.1f}K | eta {eta:5.2f}h",
                    flush=True,
                )

            # Periodic checkpoint.
            if is_main and tokens_seen - last_save >= cfg["save_every_tokens"]:
                ckpt_path = out_dir / f"ckpt_{tokens_seen}.pt"
                torch.save(
                    {
                        "model": (model.module if world_size > 1 else model).state_dict(),
                        "optim": optim.state_dict(),
                        "tokens": tokens_seen,
                        "config": args.config,
                    },
                    ckpt_path,
                )
                last_save = tokens_seen
                print(f"Saved {ckpt_path}", flush=True)

    # Final checkpoint.
    if is_main:
        ckpt_path = out_dir / "ckpt_final.pt"
        torch.save(
            {
                "model": (model.module if world_size > 1 else model).state_dict(),
                "optim": optim.state_dict(),
                "tokens": tokens_seen,
                "config": args.config,
            },
            ckpt_path,
        )
        print(f"Done. Final checkpoint: {ckpt_path}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
