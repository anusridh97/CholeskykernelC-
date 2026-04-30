"""Synthetic retrieval data for long-context training.

Generates fact-distractor-query examples shaped like the S-NIAH eval used
by `eval_sniah.py`. Each example places a random key-value fact inside a
long distractor span, followed by a question that asks for the value
verbatim. Loss flows over the entire sequence so distractors still teach
ordinary LM behavior, while the (key, value) edge teaches long-range
recall through the SKA operator.

Layout per example:
    [pre_pad] [fact] [post_pad] [question + answer]
where (pre_pad + post_pad) sum to (ctx_len + 1 - len(fact) - len(qa))
and the fact is positioned uniformly in [25%, 75%] of the distractor
span, matching `eval_sniah.build_prompt`.

The padding template is identical to `eval_sniah.PADDING_TEMPLATE` so the
train and eval distractor distributions are aligned.

Usage:
    ds = RetrievalStream(tokenizer, ctx_len=2048, seed=0)
    for batch in DataLoader(ds, batch_size=4): ...

Mix with FineWeb at a configurable ratio via MixedStream:
    mixed = MixedStream(text_ds, retrieval_ds, secondary_ratio=0.15, seed=0)

Default in train.py is `--retrieval_ratio 0.0` (disabled), so existing
runs are unaffected.
"""

import random
import string
import uuid
from typing import Iterator, List, Optional

import torch
from torch.utils.data import IterableDataset


# Must match eval_sniah.PADDING_TEMPLATE exactly so train/eval distractor
# distributions are identical.
PADDING_TEMPLATE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again. "
)

DEFAULT_VARIANTS = ("numeric", "word", "uuid")


def _gen_value(rng: random.Random, variant: str) -> str:
    """Generate a single needle value of the requested S-NIAH variant."""
    if variant == "numeric":
        return str(rng.randint(10_000_000, 99_999_999))
    if variant == "word":
        return "".join(rng.choices(string.ascii_lowercase, k=8))
    if variant == "uuid":
        return str(uuid.UUID(int=rng.getrandbits(128)))
    raise ValueError(f"unknown variant {variant!r}")


def _build_strings(rng: random.Random, variant: str):
    """Return (fact_str, question_str, value_str) for one example.

    Templates mirror `eval_sniah.build_prompt` so the training distribution
    overlaps the eval distribution as much as possible.
    """
    value = _gen_value(rng, variant)
    if variant == "numeric":
        fact = (
            f"The special magic number is {value}. "
            "Remember this number."
        )
        question = "What is the special magic number? The special magic number is"
    elif variant == "word":
        fact = (
            f"The secret keyword is {value}. "
            "Please remember this keyword."
        )
        question = "What is the secret keyword? The secret keyword is"
    else:  # uuid
        fact = f"The hidden uuid is {value}. Remember this uuid."
        question = "What is the hidden uuid? The hidden uuid is"
    return fact, question, value


def _tile_to_length(unit_ids: List[int], target_len: int) -> List[int]:
    """Tile `unit_ids` and truncate to exactly `target_len` tokens."""
    if target_len <= 0:
        return []
    n_repeats = (target_len + len(unit_ids) - 1) // len(unit_ids)
    return (unit_ids * n_repeats)[:target_len]


class RetrievalStream(IterableDataset):
    """Infinite iterator of synthetic retrieval examples.

    Yields dicts with keys (input_ids, labels), each a 1-D long tensor of
    length ctx_len. Loss flows over every token.

    Args:
        tokenizer: a HuggingFace tokenizer.
        ctx_len: target sequence length (yields (ctx_len + 1)-token windows
            internally, then splits into input/labels of length ctx_len).
        seed: RNG seed.
        variants: subset of {"numeric", "word", "uuid"}; defaults to all.
    """

    def __init__(
        self,
        tokenizer,
        ctx_len: int,
        seed: int = 0,
        variants: Optional[List[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.ctx_len = int(ctx_len)
        self.seed = int(seed)
        self.variants = list(variants) if variants else list(DEFAULT_VARIANTS)
        # Cache the tokenized padding unit so we don't re-tokenize per example.
        self._pad_ids = tokenizer.encode(PADDING_TEMPLATE, add_special_tokens=False)
        if not self._pad_ids:
            raise ValueError("PADDING_TEMPLATE tokenizes to empty sequence")

    def __iter__(self) -> Iterator[dict]:
        # Worker-aware seeding so multi-worker DataLoader doesn't duplicate.
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = (
            self.seed
            if worker_info is None
            else self.seed + worker_info.id * 7919
        )
        rng = random.Random(worker_seed)

        target_len = self.ctx_len + 1

        while True:
            variant = rng.choice(self.variants)
            fact, question, value = _build_strings(rng, variant)

            fact_ids = self.tokenizer.encode(fact, add_special_tokens=False)
            question_ids = self.tokenizer.encode(
                question, add_special_tokens=False
            )
            # Leading space helps BPE tokenizers keep the value as one piece.
            value_ids = self.tokenizer.encode(
                " " + value, add_special_tokens=False
            )
            tail_ids = question_ids + value_ids

            budget = target_len - len(tail_ids) - len(fact_ids)
            if budget <= len(self._pad_ids) * 2:
                # ctx_len too small for this variant; resample.
                continue

            # Fact position: uniform in [25%, 75%] of distractor budget.
            pos_frac = 0.25 + 0.5 * rng.random()
            pre_len = int(budget * pos_frac)
            post_len = budget - pre_len

            ids = (
                _tile_to_length(self._pad_ids, pre_len)
                + fact_ids
                + _tile_to_length(self._pad_ids, post_len)
                + tail_ids
            )
            # Tile arithmetic should always land exactly on target_len.
            assert len(ids) == target_len, (
                len(ids),
                target_len,
                pre_len,
                post_len,
                len(fact_ids),
                len(tail_ids),
            )

            input_ids = torch.tensor(ids[:-1], dtype=torch.long)
            labels = torch.tensor(ids[1:], dtype=torch.long)
            yield {"input_ids": input_ids, "labels": labels}


class MixedStream(IterableDataset):
    """Round-robin two streams with a probabilistic mix ratio.

    On each iteration, yields from `secondary` with probability
    `secondary_ratio` and from `primary` otherwise. Both streams are
    restarted on StopIteration.

    Args:
        primary: the dominant stream (e.g. FineWeb text).
        secondary: the augmentation stream (e.g. RetrievalStream).
        secondary_ratio: probability of drawing from `secondary` per sample.
        seed: RNG seed for the mixer (independent of stream seeds).
    """

    def __init__(
        self,
        primary: IterableDataset,
        secondary: IterableDataset,
        secondary_ratio: float,
        seed: int = 0,
    ):
        if not (0.0 <= secondary_ratio <= 1.0):
            raise ValueError("secondary_ratio must be in [0, 1]")
        self.primary = primary
        self.secondary = secondary
        self.ratio = float(secondary_ratio)
        self.seed = int(seed)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = (
            self.seed
            if worker_info is None
            else self.seed + worker_info.id * 6151
        )
        rng = random.Random(worker_seed)

        p_iter = iter(self.primary)
        s_iter = iter(self.secondary)
        while True:
            if rng.random() < self.ratio:
                try:
                    yield next(s_iter)
                except StopIteration:
                    s_iter = iter(self.secondary)
                    yield next(s_iter)
            else:
                try:
                    yield next(p_iter)
                except StopIteration:
                    p_iter = iter(self.primary)
                    yield next(p_iter)


# ----------------------------- Self-test ------------------------------ #

if __name__ == "__main__":
    """Quick sanity check: tokenize a few examples and print shapes + a snippet."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.model_max_length = int(1e9)

    ctx_len = 1024
    ds = RetrievalStream(tok, ctx_len=ctx_len, seed=0)
    it = iter(ds)
    for i in range(3):
        ex = next(it)
        ids = ex["input_ids"]
        labels = ex["labels"]
        assert ids.shape == (ctx_len,)
        assert labels.shape == (ctx_len,)
        decoded_tail = tok.decode(ids[-32:])
        decoded_head = tok.decode(ids[:32])
        print(f"Example {i}: ctx={ctx_len}, head={decoded_head!r}")
        print(f"           tail={decoded_tail!r}")
        print()

    # Mix test.
    class _DummyText(IterableDataset):
        def __iter__(self):
            ids = torch.zeros(ctx_len, dtype=torch.long)
            while True:
                yield {"input_ids": ids, "labels": ids}

    mixed = MixedStream(_DummyText(), ds, secondary_ratio=0.5, seed=42)
    n_ret = 0
    for j, ex in enumerate(mixed):
        # Retrieval examples have nonzero ids; dummy text is all zeros.
        if (ex["input_ids"] != 0).any().item():
            n_ret += 1
        if j == 99:
            break
    print(f"MixedStream sanity: {n_ret}/100 from secondary (expected ~50)")
