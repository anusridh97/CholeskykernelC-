"""
Wrapper that exposes the SKA model to lm-evaluation-harness.

lm-eval-harness expects a `LM` subclass providing `loglikelihood`,
`loglikelihood_rolling`, and `generate_until`. We implement a minimal
HuggingFace-style adapter so the harness can use the SKA model just like
any GPT-2/Llama checkpoint.

Usage:
  python eval_lm_harness.py \
      --ckpt path/to/ckpt.pt \
      --tokenizer meta-llama/Llama-2-7b-hf \
      --tasks arc_easy,piqa,hellaswag \
      --batch_size 8 \
      --out_dir results/lm_eval/
"""

import argparse
import json
from pathlib import Path

import torch

from ska_model import make_165m_model, make_340m_model


class SKAEvalLM:
    """Minimal adapter for lm-eval-harness. Implements the bits the harness needs.

    Built against lm-eval-harness's `models.huggingface.HFLM` interface where
    possible, so most tasks "just work."
    """

    def __init__(self, model, tokenizer, device, max_length=2048, batch_size=8):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self._max_length = max_length
        self.batch_size = batch_size
        self.eot_token_id = tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 256

    def tok_encode(self, s):
        return self.tokenizer.encode(s, add_special_tokens=False)

    def tok_decode(self, ids):
        return self.tokenizer.decode(ids)

    def _model_forward(self, ids):
        with torch.no_grad():
            return self.model(ids)["logits"]

    def loglikelihood(self, requests):
        """Compute log P(continuation | context) per request."""
        results = []
        for ctx, cont in [(r.args[0], r.args[1]) for r in requests]:
            ctx_ids = self.tok_encode(ctx)
            cont_ids = self.tok_encode(cont)
            full = ctx_ids + cont_ids
            full = full[-self.max_length:]
            inp = torch.tensor([full], dtype=torch.long, device=self.device)
            logits = self._model_forward(inp)
            # Score the continuation tokens.
            cont_len = len(cont_ids)
            log_probs = torch.log_softmax(logits[0, -cont_len-1:-1], dim=-1)
            target = torch.tensor(cont_ids, device=self.device)
            ll = log_probs.gather(-1, target.unsqueeze(-1)).sum().item()
            # is-greedy check
            greedy = (log_probs.argmax(-1) == target).all().item()
            results.append((ll, greedy))
        return results

    def loglikelihood_rolling(self, requests):
        """For perplexity-style evals (e.g., wikitext)."""
        results = []
        for s in [r.args[0] for r in requests]:
            ids = self.tok_encode(s)
            ll_total = 0.0
            for start in range(0, len(ids), self.max_length - 1):
                window = ids[start:start + self.max_length]
                if len(window) < 2:
                    break
                inp = torch.tensor([window], dtype=torch.long, device=self.device)
                logits = self._model_forward(inp)
                target = torch.tensor(window[1:], device=self.device)
                log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
                ll_total += log_probs.gather(-1, target.unsqueeze(-1)).sum().item()
            results.append(ll_total)
        return results

    def generate_until(self, requests):
        """Greedy decoding with state passing (O(N), not O(N^2))."""
        results = []
        for ctx, gen_kwargs in [(r.args[0], r.args[1]) for r in requests]:
            until = gen_kwargs.get("until", [self.tokenizer.eos_token])
            if isinstance(until, str):
                until = [until]
            max_new = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            ids = self.tok_encode(ctx)
            ids = ids[-self.max_length + max_new:]
            prompt_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)

            # Step 1: process prompt, capture state.
            with torch.no_grad():
                out = self.model(prompt_tensor)
            states = out["states"]
            next_id = out["logits"][:, -1:].argmax(dim=-1)
            generated = [self.tokenizer.decode(next_id[0].tolist())]

            # Step 2: incremental decoding.
            for _ in range(max_new - 1):
                with torch.no_grad():
                    out = self.model(next_id, states=states)
                states = out["states"]
                next_id = out["logits"][:, -1:].argmax(dim=-1)
                tok = self.tokenizer.decode(next_id[0].tolist())
                generated.append(tok)
                joined = "".join(generated)
                if any(s in joined for s in until):
                    break
                if next_id.item() == self.eot_token_id:
                    break
            results.append("".join(generated))
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--config", choices=["sanity", "prod"], default="prod")
    parser.add_argument("--tasks", type=str, required=True,
                        help="Comma-separated lm-eval task names")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.model_max_length = int(1e9)

    model_fn = make_340m_model if args.config == "prod" else make_165m_model
    model = model_fn(vocab_size=tok.vocab_size).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])

    lm = SKAEvalLM(model, tok, device,
                   max_length=args.max_length,
                   batch_size=args.batch_size)

    # Wire into lm-eval-harness.
    import lm_eval
    from lm_eval import evaluator

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task_list,
        batch_size=args.batch_size,
        device=device,
    )

    out_file = out_dir / "results.json"
    with out_file.open("w") as f:
        json.dump(results.get("results", results), f, indent=2, default=str)

    print(f"Wrote {out_file}")
    print(json.dumps(results.get("results", {}), indent=2, default=str))


if __name__ == "__main__":
    main()
