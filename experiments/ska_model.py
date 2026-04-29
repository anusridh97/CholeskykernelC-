"""
340M Llama-style transformer with SKA replacing attention.

Architecture matches PDN's Appendix E.1 spec for the 340M setting:
  - d_model   = 1024
  - n_heads   = 8
  - head_dim  = 128
  - n_layers  = 24
  - SwiGLU MLP with intermediate dim 2560 (sized to match PDN's 340M)
  - RMSNorm pre-norm
  - tied input/output embeddings
  - LLaMA tokenizer (50272 vocab)

Parameter accounting:
  embeddings (tied):           50272 * 1024              = 51.5M
  per-layer SKA (q,k,v,o):     4 * 1024^2                =  4.2M
  per-layer SwiGLU (w1,w2,w3): 3 * 1024 * 2560           =  7.9M
  per-layer total                                         = ~12.1M
  24 layers                                               = 290M
  total                                                   = ~341M
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ska_layer import SKALayer


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # down
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # up

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SKABlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        d_ff: int,
        lambda_ridge: float,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.mixer = SKALayer(
            d_model=d_model,
            n_heads=n_heads,
            head_dim=head_dim,
            lambda_ridge=lambda_ridge,
        )
        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)

    def forward(self, x, state=None):
        y, state = self.mixer(self.norm1(x), state=state)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x, state


class SKAModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50272,
        d_model: int = 1024,
        n_heads: int = 8,
        head_dim: int = 128,
        n_layers: int = 22,
        d_ff: int | None = None,
        lambda_ridge: float = 1.0,
        tie_weights: bool = True,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                SKABlock(d_model, n_heads, head_dim, d_ff, lambda_ridge)
                for _ in range(n_layers)
            ]
        )
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.embed.weight

        # Init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, states=None):
        """
        Args:
          input_ids: (B, T) int64 token ids
          labels: (B, T) int64, -100 to ignore (optional, for loss)
          states: list of layer states for stateful continuation (optional)
        Returns:
          dict with 'logits' (B, T, V) and optionally 'loss'
        """
        x = self.embed(input_ids)
        new_states = [] if states is None else None
        for i, layer in enumerate(self.layers):
            s = None if states is None else states[i]
            x, s_out = layer(x, state=s)
            if states is None:
                new_states.append(s_out)
            else:
                states[i] = s_out
        x = self.norm_f(x)
        logits = self.lm_head(x)

        out = {"logits": logits, "states": new_states if states is None else states}
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            out["loss"] = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def make_340m_model(vocab_size: int = 50272) -> SKAModel:
    """340M-parameter SKA model matching PDN Appendix E.1."""
    return SKAModel(
        vocab_size=vocab_size,
        d_model=1024,
        n_heads=8,
        head_dim=128,
        n_layers=24,
        d_ff=2560,
        lambda_ridge=1.0,
    )


def make_165m_model(vocab_size: int = 50272) -> SKAModel:
    """165M sanity-check model. Same head_dim, smaller depth/width."""
    return SKAModel(
        vocab_size=vocab_size,
        d_model=768,
        n_heads=6,
        head_dim=128,
        n_layers=12,
        d_ff=3072,
        lambda_ridge=1.0,
    )


if __name__ == "__main__":
    m340 = make_340m_model()
    m165 = make_165m_model()
    print(f"340M model: {m340.num_parameters() / 1e6:.1f}M params")
    print(f"165M model: {m165.num_parameters() / 1e6:.1f}M params")
