"""
SKA layer: a sequence-mixing layer that replaces standard attention.

Architecture follows PDN's setup (Appendix E.1):
  - q, k, v projections from input
  - short conv on q, k, v (kernel size 4) for token-shift information
  - SiLU activation
  - L2 normalize q, k
  - SKA recurrent core (per-token kernel call, batched over B*H)
  - headwise RMSNorm
  - output projection

Per-head state at the end of the layer's forward:
  L_h:    (B, H, r, r) lower triangular Cholesky of G_t
  Aw_h:   (B, H, r, r) whitened cross-Gram, A_w = L^{-1} M L^{-T}
  S_h:    (B, H, r, r) regression state, S = L A_w^T L^{-1}
  conv_q: (B, K-1, D)  short-conv history for q-projection
  conv_k: (B, K-1, D)  short-conv history for k-projection
  conv_v: (B, K-1, D)  short-conv history for v-projection

State plumbing supports streaming inference: pass `state` in, get the
updated state out, feed it to the next call. The conv buffers ensure the
short conv has correct history at the prompt/answer boundary.

Per-token output (ATK realization, decay-free):
  k_tilde_t = L_+^{-T} v_z (from kernel)         # exact preconditioned key
  S_t = S_{t-1} + (v_t - S_{t-1} k_t) k_tilde_t^T
  o_t = S_t q_t

Inner loop is a Python `for t in range(T)` calling the batched Triton
kernel once per token, parallelizing over B*H. Q, K, V are pre-permuted
to (T, BH, r) layout before the loop so the per-step slice is a
contiguous view (no per-step memcpy).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ska_recurrence import ska_step


_CONV_KERNEL = 4  # K in the short conv; matches PDN's setup


def _l2_normalize(x, dim=-1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def _short_conv_with_state(x, weight, bias, prev):
    """Causal short conv with optional history buffer.

    Args:
      x: (B, T, D) current input.
      weight: (D, 1, K) depthwise conv weight.
      bias: optional (D,)
      prev: optional (B, K-1, D) previous tokens to use as causal padding.
            If None, we pad with zeros.

    Returns:
      y: (B, T, D) conv output.
      new_prev: (B, K-1, D) tail of `cat([prev, x])` for the next call.
    """
    B, T, D = x.shape
    K = weight.shape[-1]
    if prev is None:
        x_pad = F.pad(x.transpose(1, 2), (K - 1, 0))                # (B, D, K-1+T)
    else:
        x_full = torch.cat([prev, x], dim=1)                        # (B, K-1+T, D)
        x_pad = x_full.transpose(1, 2)
    y = F.conv1d(x_pad, weight, bias=bias, groups=D).transpose(1, 2)
    # Update the buffer to the last K-1 tokens of (prev || x).
    if prev is None:
        new_prev = x[:, max(0, T - (K - 1)):]
    else:
        joint = torch.cat([prev, x], dim=1)
        new_prev = joint[:, -(K - 1):]
    if new_prev.shape[1] < (K - 1):
        # Pad to K-1 if we have very few tokens (e.g., first call with T=1).
        pad_amt = (K - 1) - new_prev.shape[1]
        new_prev = F.pad(new_prev, (0, 0, pad_amt, 0))
    return y, new_prev


class SKALayer(nn.Module):
    """SKA sequence-mixing layer with full kernel utilization.

    Args:
      d_model: int, model dimension D.
      n_heads: int, number of heads H.
      head_dim: int, per-head dimension r (= d_k = d_v).
      lambda_ridge: float, ridge term added to G = sum k k^T + lambda I.
      conv_kernel: int, short-conv kernel size for q, k, v (default 4).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        lambda_ridge: float = 1.0,
        conv_kernel: int = _CONV_KERNEL,
    ):
        super().__init__()
        assert head_dim * n_heads == d_model, (
            f"head_dim {head_dim} * n_heads {n_heads} must equal d_model {d_model}"
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.lambda_ridge = lambda_ridge
        self.r = head_dim
        self.K = conv_kernel

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.q_conv_w = nn.Parameter(torch.empty(d_model, 1, conv_kernel))
        self.k_conv_w = nn.Parameter(torch.empty(d_model, 1, conv_kernel))
        self.v_conv_w = nn.Parameter(torch.empty(d_model, 1, conv_kernel))
        for w in [self.q_conv_w, self.k_conv_w, self.v_conv_w]:
            nn.init.normal_(w, std=0.02)

        self.head_norm = nn.RMSNorm(head_dim)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def _init_state(self, B, device, dtype):
        """Initialize all six state tensors for a fresh sequence."""
        r, H, D, K = self.r, self.n_heads, self.d_model, self.K
        L0 = math.sqrt(self.lambda_ridge) * torch.eye(
            r, device=device, dtype=dtype
        ).expand(B, H, r, r).contiguous()
        Aw0 = torch.zeros(B, H, r, r, device=device, dtype=dtype)
        S0 = torch.zeros(B, H, r, r, device=device, dtype=dtype)
        cq = torch.zeros(B, K - 1, D, device=device, dtype=dtype)
        ck = torch.zeros(B, K - 1, D, device=device, dtype=dtype)
        cv = torch.zeros(B, K - 1, D, device=device, dtype=dtype)
        return L0, Aw0, S0, cq, ck, cv

    def forward(self, x, state=None):
        """
        Args:
          x: (B, T, D)
          state: optional 6-tuple (L, Aw, S, conv_q, conv_k, conv_v).
        Returns:
          y: (B, T, D)
          state_out: 6-tuple, ready to feed into the next call.
        """
        B, T, D = x.shape
        H, r = self.n_heads, self.r

        if state is None:
            L, Aw, S, conv_q, conv_k, conv_v = self._init_state(B, x.device, x.dtype)
        else:
            L, Aw, S, conv_q, conv_k, conv_v = state

        # Projections + short conv (with history) + activation + L2 norm.
        q_proj = self.q_proj(x)
        k_proj = self.k_proj(x)
        v_proj = self.v_proj(x)

        q_post, conv_q_new = _short_conv_with_state(q_proj, self.q_conv_w, None, conv_q)
        k_post, conv_k_new = _short_conv_with_state(k_proj, self.k_conv_w, None, conv_k)
        v_post, conv_v_new = _short_conv_with_state(v_proj, self.v_conv_w, None, conv_v)

        q = _l2_normalize(F.silu(q_post))
        k = _l2_normalize(F.silu(k_post))
        v = F.silu(v_post)

        # Reshape into per-head views: (B, T, H, r).
        q = q.view(B, T, H, r)
        k = k.view(B, T, H, r)
        v = v.view(B, T, H, r)

        # KEY KERNEL UTILIZATION FIX:
        # Pre-permute to (T, BH, r) and make contiguous ONCE outside the loop,
        # so each per-step slice is a contiguous (BH, r) view with no memcpy.
        BH = B * H
        q_seq = q.permute(1, 0, 2, 3).reshape(T, BH, r).contiguous()
        k_seq = k.permute(1, 0, 2, 3).reshape(T, BH, r).contiguous()
        v_seq = v.permute(1, 0, 2, 3).reshape(T, BH, r).contiguous()

        L_flat = L.reshape(BH, r, r).contiguous()
        Aw_flat = Aw.reshape(BH, r, r).contiguous()
        S_flat = S.reshape(BH, r, r).contiguous()

        # Pre-allocate output in (T, BH, r) layout; fill in the inner loop.
        out_seq = torch.empty(T, BH, r, device=x.device, dtype=x.dtype)

        for t in range(T):
            k_t = k_seq[t]                                          # (BH, r) contiguous view
            v_t = v_seq[t]
            q_t = q_seq[t]

            # Triton kernel call (batched over BH).
            L_flat, Aw_flat, k_tilde = ska_step(L_flat, Aw_flat, k_t, v_t)

            # ATK state update: S = S + (v - S k) k_tilde^T
            Sk = torch.einsum("brc,bc->br", S_flat, k_t)
            err = v_t - Sk
            S_flat = S_flat + torch.einsum("br,bc->brc", err, k_tilde)

            # Output: o_t = S q_t
            out_seq[t] = torch.einsum("brc,bc->br", S_flat, q_t)

        # (T, BH, r) -> (B, T, H, r) -> (B, T, D)
        y = out_seq.view(T, B, H, r).permute(1, 0, 2, 3).contiguous()
        y = self.head_norm(y).view(B, T, D)
        y = self.o_proj(y)

        state_out = (
            L_flat.view(B, H, r, r),
            Aw_flat.view(B, H, r, r),
            S_flat.view(B, H, r, r),
            conv_q_new,
            conv_k_new,
            conv_v_new,
        )
        return y, state_out
