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

from ska_recurrence import ska_step, ska_chunk


_CONV_KERNEL = 4  # K in the short conv; matches PDN's setup


def _l2_normalize(x, dim=-1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


# ---------------------- Fused within-chunk ATK update ---------------------- #
#
# The ATK regression-state recurrence
#     S_t = S_{t-1} + (v_t - S_{t-1} k_t) k_tilde_t^T
#     o_t = S_t q_t
# is sequential in t, so it cannot be parallelized across the chunk.
# But the per-iteration body is just three small einsums plus a subtract,
# and a naive Python loop incurs ~4 CUDA kernel launches per token.
# At ctx_len=1024, n_layers=12, that is ~50,000 launches per forward pass,
# which dominates wall time on H200 where each math op is microseconds.
#
# The helper below is the same loop in a form torch.compile can ingest.
# With mode="reduce-overhead", Inductor unrolls C iterations and fuses the
# einsum chain into a small number of kernels (often captured in a CUDA
# graph), eliminating most of the launch overhead.
#
# Eager and compiled paths must produce numerically identical outputs and
# gradients; verified by experiments/backend/test_atk_compile.py.

def _atk_chunk_update_eager(S_init, K_chunk, V_chunk, Q_chunk, K_tilde):
    """Reference ATK regression-state update for one chunk.

    Args:
        S_init: (BH, r, r) state at the chunk boundary.
        K_chunk: (BH, C, r) keys.
        V_chunk: (BH, C, r) values.
        Q_chunk: (BH, C, r) queries.
        K_tilde: (BH, C, r) preconditioned keys from ska_chunk.
    Returns:
        S_final: (BH, r, r) state after C iterations.
        out: (BH, C, r) outputs o_t = S_t q_t.
    """
    BH, C, r = K_chunk.shape
    out_list = []
    S_curr = S_init
    for t in range(C):
        k_t = K_chunk[:, t]
        v_t = V_chunk[:, t]
        q_t = Q_chunk[:, t]
        kt_t = K_tilde[:, t]
        Sk = torch.einsum("brc,bc->br", S_curr, k_t)
        err = v_t - Sk
        S_curr = S_curr + torch.einsum("br,bc->brc", err, kt_t)
        out_list.append(torch.einsum("brc,bc->br", S_curr, q_t))
    out = torch.stack(out_list, dim=1)
    return S_curr, out


# Compile lazily so we don't pay the cost in import/inference where it's
# not needed. Falls back to eager if torch.compile is unavailable or
# fails (e.g. backend errors on a particular GPU).
_atk_chunk_compiled_fn = None
_atk_chunk_compile_attempted = False


def _get_atk_chunk_fn():
    """Return the (lazily-compiled) ATK chunk update fn."""
    global _atk_chunk_compiled_fn, _atk_chunk_compile_attempted
    if _atk_chunk_compile_attempted:
        return _atk_chunk_compiled_fn
    _atk_chunk_compile_attempted = True
    try:
        # mode="default": Inductor op fusion without CUDA graphs.
        # We deliberately avoid mode="reduce-overhead" because its CUDA-
        # graph buffer reuse produces NaNs on this 64-step recurrent loop
        # (verified empirically). "default" still fuses the per-iteration
        # einsum chain into a few kernels, giving a ~5x speedup at fp32
        # tolerance for forward + backward.
        _atk_chunk_compiled_fn = torch.compile(
            _atk_chunk_update_eager,
            mode="default",
            dynamic=False,
            fullgraph=False,
        )
    except Exception as e:                                  # pragma: no cover
        print(
            f"[ska_layer] torch.compile of ATK chunk failed ({type(e).__name__}: {e}); "
            f"falling back to eager Python loop."
        )
        _atk_chunk_compiled_fn = _atk_chunk_update_eager
    return _atk_chunk_compiled_fn


def atk_chunk_update(S_init, K_chunk, V_chunk, Q_chunk, K_tilde):
    """Public entry point: compiled ATK chunk update with eager fallback."""
    fn = _get_atk_chunk_fn()
    return fn(S_init, K_chunk, V_chunk, Q_chunk, K_tilde)


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
        chunk_size: int = 64,
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
        self.chunk_size = chunk_size

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

        # CHUNKWISE PROCESSING.
        # Lay out as (BH, T, r) so each chunk slice (BH, C, r) is contiguous.
        # Each chunk makes ONE ska_chunk autograd.Function call (instead of
        # C ska_step calls), reducing the autograd graph by C times.
        BH = B * H
        q_chunked = q.permute(0, 2, 1, 3).reshape(BH, T, r).contiguous()
        k_chunked = k.permute(0, 2, 1, 3).reshape(BH, T, r).contiguous()
        v_chunked = v.permute(0, 2, 1, 3).reshape(BH, T, r).contiguous()

        L_flat = L.reshape(BH, r, r).contiguous()
        Aw_flat = Aw.reshape(BH, r, r).contiguous()
        S_flat = S.reshape(BH, r, r).contiguous()

        out_chunked = torch.empty(BH, T, r, device=x.device, dtype=x.dtype)

        C = min(self.chunk_size, T)
        N_chunks_full = T // C
        T_full = N_chunks_full * C

        for ci in range(N_chunks_full):
            ts = slice(ci * C, (ci + 1) * C)
            K_chunk = k_chunked[:, ts]      # (BH, C, r)
            V_chunk = v_chunked[:, ts]
            Q_chunk = q_chunked[:, ts]

            # ONE autograd.Function call for the whole chunk.
            L_flat, Aw_flat, K_tilde = ska_chunk(L_flat, Aw_flat, K_chunk, V_chunk)

            # Within-chunk ATK state update. Compiled by torch.compile
            # (with eager fallback) to fuse the per-token einsum chain
            # into a single device-side launch. See atk_chunk_update
            # docstring above for the math; numerical equivalence to the
            # eager loop is verified by test_atk_compile.py.
            S_flat, out_slice = atk_chunk_update(
                S_flat, K_chunk, V_chunk, Q_chunk, K_tilde
            )
            out_chunked[:, ci * C : (ci + 1) * C] = out_slice

        # Handle any tail tokens (T % C != 0). Falls back to per-token ska_step.
        for t in range(T_full, T):
            k_t = k_chunked[:, t]
            v_t = v_chunked[:, t]
            q_t = q_chunked[:, t]

            L_flat, Aw_flat, k_tilde = ska_step(L_flat, Aw_flat, k_t, v_t)

            Sk = torch.einsum("brc,bc->br", S_flat, k_t)
            err = v_t - Sk
            S_flat = S_flat + torch.einsum("br,bc->brc", err, k_tilde)
            out_chunked[:, t] = torch.einsum("brc,bc->br", S_flat, q_t)

        # (BH, T, r) -> (B, H, T, r) -> (B, T, H, r)
        y = out_chunked.view(B, H, T, r).permute(0, 2, 1, 3).contiguous()
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
