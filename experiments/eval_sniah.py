"""
S-NIAH evaluation: Single Needle-In-A-Haystack at multiple context lengths.

Three variants from RULER (Hsieh et al., 2024):
  S-NIAH-1: numeric needle (single number to recall)
  S-NIAH-2: word needle (single word in a long passage)
  S-NIAH-3: UUID needle (high-entropy string)

Output: CSV with columns (variant, context_length, accuracy, n_examples).

Usage:
  python eval_sniah.py --ckpt checkpoints/ckpt_final.pt \
      --tokenizer meta-llama/Llama-2-7b-hf \
      --out results/sniah.csv \
      --lengths 2048 4096 8192 16384 32768 \
      --variants 1 2 3 \
      --n_examples 100
"""

import argparse
import csv
import random
import string
import uuid
from pathlib import Path

import torch

from ska_model import make_340m_model, make_165m_model

PADDING_TEMPLATE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again. "
)


def build_prompt(variant: int, ctx_tokens: int, tokenizer, seed: int):
    """Construct an S-NIAH prompt of the requested token length, with a needle.

    Returns (prompt_str, needle_str, ground_truth_str).
    """
    rng = random.Random(seed)
    if variant == 1:
        needle_value = str(rng.randint(10_000_000, 99_999_999))
        needle_sentence = (
            f"The special magic number is {needle_value}. "
            "Remember this number."
        )
        question = "What is the special magic number? The special magic number is"
        ground_truth = needle_value
    elif variant == 2:
        word = "".join(rng.choices(string.ascii_lowercase, k=8))
        needle_sentence = (
            f"The secret keyword is {word}. "
            "Please remember this keyword."
        )
        question = "What is the secret keyword? The secret keyword is"
        ground_truth = word
    elif variant == 3:
        u = str(uuid.UUID(int=rng.getrandbits(128)))
        needle_sentence = f"The hidden uuid is {u}. Remember this uuid."
        question = "What is the hidden uuid? The hidden uuid is"
        ground_truth = u
    else:
        raise ValueError(f"unknown variant {variant}")

    # Pad to roughly ctx_tokens.
    pad_unit_len = len(tokenizer.encode(PADDING_TEMPLATE, add_special_tokens=False))
    needle_len = len(tokenizer.encode(needle_sentence, add_special_tokens=False))
    question_len = len(tokenizer.encode(question, add_special_tokens=False))

    target_pad = ctx_tokens - needle_len - question_len - 16  # safety
    n_units = max(1, target_pad // pad_unit_len)

    pad_blocks = [PADDING_TEMPLATE] * n_units
    needle_position = rng.randint(n_units // 4, 3 * n_units // 4)
    pad_blocks.insert(needle_position, needle_sentence)
    prompt = "".join(pad_blocks) + " " + question
    return prompt, needle_sentence, ground_truth


@torch.no_grad()
def greedy_generate(model, tokenizer, prompt: str, max_new_tokens: int = 32, device="cuda"):
    """Greedy decoding with state passing (O(N) instead of O(N^2)).

    Step 1: process the full prompt in one forward pass and capture the
    SKA state (L, Aw, S) plus short-conv buffers, per layer.
    Step 2: generate one token at a time, feeding both the new token and
    the state back to the model. Each step costs one Triton kernel
    launch per layer instead of len(prompt)+t launches.

    For 32K-context S-NIAH this is roughly a 50x speedup vs no-cache.
    """
    model.eval()
    prompt_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)],
        device=device,
        dtype=torch.long,
    )

    # Step 1: prompt processing.
    out = model(prompt_ids)
    states = out["states"]
    next_id = out["logits"][:, -1:].argmax(dim=-1)        # (1, 1)

    pieces = [tokenizer.decode(next_id[0].tolist())]
    eos_id = tokenizer.eos_token_id

    # Step 2: incremental generation, one token per kernel call.
    for _ in range(max_new_tokens - 1):
        if next_id.item() == eos_id:
            break
        out = model(next_id, states=states)
        states = out["states"]
        next_id = out["logits"][:, -1:].argmax(dim=-1)
        tok_str = tokenizer.decode(next_id[0].tolist())
        pieces.append(tok_str)
        if "\n" in tok_str:
            break

    return "".join(pieces).strip()


def matches_ground_truth(generation: str, gt: str) -> bool:
    return gt.lower() in generation.lower()


def evaluate_variant_at_length(
    model,
    tokenizer,
    variant: int,
    ctx_tokens: int,
    n_examples: int,
    device: str,
    seed: int = 0,
):
    correct = 0
    for i in range(n_examples):
        prompt, _needle, gt = build_prompt(
            variant=variant,
            ctx_tokens=ctx_tokens,
            tokenizer=tokenizer,
            seed=seed + i,
        )
        gen = greedy_generate(model, tokenizer, prompt, max_new_tokens=48, device=device)
        if matches_ground_truth(gen, gt):
            correct += 1
        if (i + 1) % 10 == 0:
            print(f"  variant {variant} L={ctx_tokens}  {i+1}/{n_examples}  acc={correct/(i+1):.3f}")
    return correct / n_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--config", choices=["sanity", "prod"], default="prod")
    parser.add_argument("--lengths", type=int, nargs="+",
                        default=[2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--variants", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--n_examples", type=int, default=100)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model_fn = make_340m_model if args.config == "prod" else make_165m_model
    model = model_fn(vocab_size=tok.vocab_size).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "context_length", "accuracy", "n_examples"])
        for variant in args.variants:
            for L in args.lengths:
                print(f"=== S-NIAH-{variant}, ctx={L} ===")
                acc = evaluate_variant_at_length(
                    model, tok, variant, L, args.n_examples, device, seed=args.seed
                )
                writer.writerow([variant, L, f"{acc:.4f}", args.n_examples])
                f.flush()
                print(f"  -> acc {acc:.3f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
