"""
SKA recurrence with a manual backward through the double-sided Cholesky kernel.

Forward graph (per-token, batched over T = B*H):

    G_+ = L L^T + z z^T = L_+ L_+^T              # rank-1 Cholesky update
    T_  = L_+^{-1} L                              # transport matrix
    v_z = L_+^{-1} z,  v_w = L_+^{-1} w           # one-sided whitenings
    A_w_+ = T_ A_w T_^T + v_z v_w^T               # double-sided propagation
    k_tilde = L_+^{-T} v_z                        # exact preconditioned key

Inputs:  L (T,r,r), A_w (T,r,r), z (T,r), w (T,r).
Outputs: L_+, A_w_+, k_tilde (returned), plus v_z, v_w, T_ (saved for bwd).

Backward derivation (see paper-section material): five upper-triangular solves
plus three lower-triangular solves plus three r-by-r matmuls per token. No
Cholesky in backward.

The CPU path (no CUDA available) routes the forward through a high-level
matrix-op reference inside `torch.no_grad()`, so that gradcheck still tests
the manual backward against finite differences of the same forward. The CUDA
path uses the Triton kernel.
"""

import sys
from pathlib import Path

import torch

# Make the kernel importable. Walk upward from this file looking for the
# student_pkg directory so this module works whether located at experiments/,
# experiments/backend/, or anywhere else under the repo root.
_HERE = Path(__file__).resolve().parent
_STUDENT = None
for _p in [_HERE, *_HERE.parents]:
    if (_p / "student_pkg").is_dir():
        _STUDENT = _p / "student_pkg"
        break
if _STUDENT is None:
    raise RuntimeError(
        f"Cannot locate student_pkg/ above {_HERE}; expected the kernel package "
        f"to live somewhere on the path from this file to the repo root."
    )
if str(_STUDENT) not in sys.path:
    sys.path.insert(0, str(_STUDENT))

try:
    import triton
    from doublesided_cholesky import (
        incremental_update_triton_batched,
        incremental_update_triton_tiled,
        _fused_doublesided_kernel,
    )
    _HAVE_TRITON = True
except Exception:
    _HAVE_TRITON = False


# ---------------------------------------------------------------------------
# Forward: high-level reference (autograd-friendly; used as CPU fallback)
# ---------------------------------------------------------------------------

def _forward_reference(L, Aw, z, w):
    """Reference forward using only standard PyTorch ops.

    Returns (L_new, Aw_new, vz, vw, k_tilde).
    """
    LT = L.transpose(-1, -2)

    G = L @ LT
    G_new = G + z.unsqueeze(-1) * z.unsqueeze(-2)
    L_new = torch.linalg.cholesky(G_new)
    L_newT = L_new.transpose(-1, -2)

    M = L @ Aw @ LT
    M_new = M + z.unsqueeze(-1) * w.unsqueeze(-2)

    # Aw_new = L_new^{-1} M_new L_new^{-T}
    # Step 1: tmp = L_new^{-1} M_new                (left solve, lower)
    # Step 2: Aw_new = tmp @ L_new^{-T}             (right solve by L_new^{-T})
    #   Use the identity A @ L^{-T} = (L^{-1} A^T)^T:
    #     solve_triangular(L_new, tmp.T, upper=False).T
    tmp = torch.linalg.solve_triangular(L_new, M_new, upper=False)
    Aw_new = torch.linalg.solve_triangular(
        L_new, tmp.transpose(-1, -2), upper=False
    ).transpose(-1, -2)

    vz = torch.linalg.solve_triangular(
        L_new, z.unsqueeze(-1), upper=False
    ).squeeze(-1)
    vw = torch.linalg.solve_triangular(
        L_new, w.unsqueeze(-1), upper=False
    ).squeeze(-1)
    k_tilde = torch.linalg.solve_triangular(
        L_newT, vz.unsqueeze(-1), upper=True
    ).squeeze(-1)

    return L_new, Aw_new, vz, vw, k_tilde


# ---------------------------------------------------------------------------
# Forward: Triton kernel (in-place on a clone of L)
# ---------------------------------------------------------------------------

def _kernel_call(L_work, Aw, z_clone, w_clone):
    """Direct call into _fused_doublesided_kernel that captures v_w.

    The wrapper `incremental_update_triton_batched` returns only (Aw_out, vz)
    and discards the mutated `w_work` buffer that holds v_w after Phase 4.
    Calling the underlying kernel directly here lets us recover v_w without
    redoing the triangular solve outside.

    Args:
      L_work: (T, r, r) lower-triangular, MODIFIED IN PLACE to L_+.
      Aw:     (T, r, r) input whitened operator (read-only).
      z_clone: (T, r) RANK-1 update vector. Mutated by Phase 1 (do not reuse).
      w_clone: (T, r) value vector. Mutated by Phase 4 to hold v_w.

    Returns: (Aw_out, vz, vw)  where vw = L_+^{-1} w lives in w_clone.
    """
    T, r = L_work.shape[0], L_work.shape[1]
    BLOCK = triton.next_power_of_2(r)
    assert BLOCK <= 1024, f"r={r} too large for single-block kernel"

    vz = torch.empty(T, r, dtype=L_work.dtype, device=L_work.device)
    Aw_out = torch.empty(T, r, r, dtype=L_work.dtype, device=L_work.device)
    cs = torch.empty(T, r, dtype=L_work.dtype, device=L_work.device)
    ss = torch.empty(T, r, dtype=L_work.dtype, device=L_work.device)

    _fused_doublesided_kernel[(T,)](
        L_work, Aw, z_clone, w_clone, vz, Aw_out, cs, ss,
        r=r, BLOCK=BLOCK,
    )
    # w_clone now holds v_w (Phase 4 wrote into it).
    return Aw_out, vz, w_clone


def _forward_triton(L, Aw, z, w):
    """Triton-kernel forward. Returns (L_new, Aw_new, vz, vw, k_tilde).

    Single triangular solve outside the kernel: only k_tilde = L_+^{-T} v_z.
    v_w is taken directly from the kernel's mutated w buffer.
    """
    # Clone L (kernel mutates it). Clone z and w for kernel internal mutation.
    L_work = L.detach().contiguous().clone()
    Aw_in = Aw.detach().contiguous()
    z_clone = z.detach().contiguous().clone()
    w_clone = w.detach().contiguous().clone()

    Aw_new, vz, vw = _kernel_call(L_work, Aw_in, z_clone, w_clone)
    L_new = L_work
    L_newT = L_new.transpose(-1, -2)

    # Only one extra solve outside the kernel: k_tilde = L_+^{-T} v_z.
    k_tilde = torch.linalg.solve_triangular(
        L_newT, vz.unsqueeze(-1), upper=True
    ).squeeze(-1)
    return L_new, Aw_new, vz, vw, k_tilde


# ---------------------------------------------------------------------------
# autograd.Function: manual backward
# ---------------------------------------------------------------------------

def _per_token_backward_step(
    L, Aw, z, w,
    L_new, vz, vw, k_tilde, T_mat,
    grad_L_new, grad_Aw_new, grad_k_tilde,
):
    """Apply the seven-step manual backward for one SKA token.

    Inputs are the same tensors `_SKAUpdate.forward` would have saved
    (L, Aw, z, w are the inputs; L_new, vz, vw, k_tilde, T_mat are the
    forward outputs / intermediates) plus the upstream output gradients.

    Returns (grad_L, grad_Aw, grad_z, grad_w). The chunkwise backward
    calls this function C times in reverse token order.
    """
    if grad_L_new is None:
        grad_L_new = torch.zeros_like(L_new)
    if grad_Aw_new is None:
        grad_Aw_new = torch.zeros_like(L_new)
    if grad_k_tilde is None:
        grad_k_tilde = torch.zeros_like(vz)

    L_newT = L_new.transpose(-1, -2)
    T_T = T_mat.transpose(-1, -2)
    Aw_T = Aw.transpose(-1, -2)

    # --- Step 1: k_tilde = L_new^{-T} v_z ---
    grad_vz = torch.linalg.solve_triangular(
        L_new, grad_k_tilde.unsqueeze(-1), upper=False
    ).squeeze(-1)
    outer1 = k_tilde.unsqueeze(-1) * grad_k_tilde.unsqueeze(-2)
    update1 = torch.linalg.solve_triangular(
        L_new, outer1.transpose(-1, -2), upper=False
    ).transpose(-1, -2)
    grad_L_new = grad_L_new - update1

    # --- Step 2: A_w_new = T A_w T^T + v_z v_w^T ---
    grad_T = grad_Aw_new @ T_mat @ Aw_T + grad_Aw_new.transpose(-1, -2) @ T_mat @ Aw
    grad_Aw = T_T @ grad_Aw_new @ T_mat
    grad_vz = grad_vz + (grad_Aw_new @ vw.unsqueeze(-1)).squeeze(-1)
    grad_vw = (grad_Aw_new.transpose(-1, -2) @ vz.unsqueeze(-1)).squeeze(-1)

    # --- Step 3: T = L_new^{-1} L ---
    Linv_T_grad_T = torch.linalg.solve_triangular(L_newT, grad_T, upper=True)
    grad_L_new = grad_L_new - Linv_T_grad_T @ T_T
    grad_L_from_T = Linv_T_grad_T

    # --- Step 4: v_z = L_new^{-1} z ---
    Linv_T_grad_vz = torch.linalg.solve_triangular(
        L_newT, grad_vz.unsqueeze(-1), upper=True
    ).squeeze(-1)
    grad_L_new = grad_L_new - Linv_T_grad_vz.unsqueeze(-1) * vz.unsqueeze(-2)
    grad_z_direct = Linv_T_grad_vz

    # --- Step 5: v_w = L_new^{-1} w ---
    Linv_T_grad_vw = torch.linalg.solve_triangular(
        L_newT, grad_vw.unsqueeze(-1), upper=True
    ).squeeze(-1)
    grad_L_new = grad_L_new - Linv_T_grad_vw.unsqueeze(-1) * vw.unsqueeze(-2)
    grad_w = Linv_T_grad_vw

    # --- Step 6: G_new = L_new L_new^T (Cholesky adjoint) ---
    M_aux = L_newT @ grad_L_new
    P = torch.tril(M_aux, diagonal=-1) + 0.5 * torch.diag_embed(
        torch.diagonal(M_aux, dim1=-2, dim2=-1)
    )
    Y = torch.linalg.solve_triangular(L_newT, P, upper=True)
    grad_G_new = torch.linalg.solve_triangular(
        L_newT, Y.transpose(-1, -2), upper=True
    ).transpose(-1, -2)

    # --- Step 7: G_new = L L^T + z z^T ---
    H = grad_G_new + grad_G_new.transpose(-1, -2)
    grad_L = grad_L_from_T + H @ L
    grad_z = grad_z_direct + (H @ z.unsqueeze(-1)).squeeze(-1)

    return grad_L, grad_Aw, grad_z, grad_w


class _SKAUpdate(torch.autograd.Function):
    """Per-token SKA update with manual backward.

    Forward returns (L_new, Aw_new, k_tilde).
    Backward calls _per_token_backward_step.
    """

    @staticmethod
    def forward(ctx, L, Aw, z, w):
        use_triton = _HAVE_TRITON and L.is_cuda and Aw.is_cuda and z.is_cuda and w.is_cuda
        if use_triton:
            L_new, Aw_new, vz, vw, k_tilde = _forward_triton(L, Aw, z, w)
        else:
            with torch.no_grad():
                L_new, Aw_new, vz, vw, k_tilde = _forward_reference(L, Aw, z, w)

        T_mat = torch.linalg.solve_triangular(L_new, L, upper=False)
        ctx.save_for_backward(L, Aw, z, w, L_new, vz, vw, k_tilde, T_mat)
        return L_new.contiguous(), Aw_new.contiguous(), k_tilde.contiguous()

    @staticmethod
    def backward(ctx, grad_L_new, grad_Aw_new, grad_k_tilde):
        L, Aw, z, w, L_new, vz, vw, k_tilde, T_mat = ctx.saved_tensors
        return _per_token_backward_step(
            L, Aw, z, w, L_new, vz, vw, k_tilde, T_mat,
            grad_L_new, grad_Aw_new, grad_k_tilde,
        )


def ska_step(L, Aw, k, v):
    """One SKA recurrence step with manual backward.

    Args:
      L:  (BH, r, r) lower triangular Cholesky factor of G_t.
      Aw: (BH, r, r) whitened cross-Gram, A_w = L^{-1} M L^{-T}.
      k:  (BH, r) per-token key (z in the paper).
      v:  (BH, r) per-token value (w in the paper).

    Returns:
      L_new:   (BH, r, r)  Cholesky of G_t + k k^T.
      Aw_new:  (BH, r, r)  whitened cross-Gram after the update.
      k_tilde: (BH, r)     exact normalized write key, k_tilde = G_new^{-1} k.
    """
    return _SKAUpdate.apply(L, Aw, k, v)


# ---------------------------------------------------------------------------
# Chunkwise SKA update
#
# The per-token loop in ska_layer.py used to call ska_step (and thus _SKAUpdate
# .apply) once per token. With C tokens per chunk and 24 layers and ctx 2048,
# that means ~50K autograd.Function invocations per micro-step, which dominates
# wall time. The chunkwise version below lets the layer call ONE
# autograd.Function per chunk; internally it still calls the per-token Triton
# kernel C times, but the autograd graph collapses to chunk granularity.
#
# Numerically identical to running ska_step C times back-to-back.
# Math is unchanged; this is purely a graph-size optimization.
# ---------------------------------------------------------------------------


def _chunk_forward(L, Aw, K, V):
    """Run the per-token forward kernel C times, collecting all intermediates.

    Args:
      L:  (BH, r, r) initial Cholesky factor at the chunk start.
      Aw: (BH, r, r) initial whitened operator at the chunk start.
      K:  (BH, C, r) per-token keys for the chunk.
      V:  (BH, C, r) per-token values for the chunk.

    Returns:
      L_new:   (BH, r, r) at chunk end
      Aw_new:  (BH, r, r) at chunk end
      K_tilde: (BH, C, r) preconditioned write keys, one per token
      saved:   tuple of per-token intermediates needed for the backward.
               Each element is a list of length C of shape (BH, r, r) or (BH, r).
    """
    BH, C, r = K.shape
    L_curr = L
    Aw_curr = Aw

    L_seq = [L]              # L_0, L_1, ..., L_C  (length C+1)
    Aw_seq = [Aw]            # Aw_0, Aw_1, ..., Aw_C
    vz_seq = []              # length C
    vw_seq = []
    k_tilde_seq = []
    T_mat_seq = []

    use_triton = _HAVE_TRITON and L.is_cuda

    for c in range(C):
        z_c = K[:, c]
        w_c = V[:, c]
        if use_triton:
            L_curr, Aw_curr, vz_c, vw_c, k_tilde_c = _forward_triton(
                L_curr, Aw_curr, z_c, w_c
            )
        else:
            with torch.no_grad():
                L_curr, Aw_curr, vz_c, vw_c, k_tilde_c = _forward_reference(
                    L_curr, Aw_curr, z_c, w_c
                )
        T_c = torch.linalg.solve_triangular(L_curr, L_seq[-1], upper=False)
        L_seq.append(L_curr.contiguous())
        Aw_seq.append(Aw_curr.contiguous())
        vz_seq.append(vz_c)
        vw_seq.append(vw_c)
        k_tilde_seq.append(k_tilde_c)
        T_mat_seq.append(T_c)

    K_tilde = torch.stack(k_tilde_seq, dim=1)  # (BH, C, r)
    return L_curr, Aw_curr, K_tilde, (L_seq, Aw_seq, vz_seq, vw_seq, k_tilde_seq, T_mat_seq)


class _SKAChunkUpdate(torch.autograd.Function):
    """C-token SKA update; one autograd.Function per chunk (vs per token).

    Forward signature: (L, Aw, K, V) -> (L_new, Aw_new, K_tilde)
      L:  (BH, r, r)
      Aw: (BH, r, r)
      K:  (BH, C, r)
      V:  (BH, C, r)

    Backward applies _per_token_backward_step C times in reverse, recovering
    the per-token intermediates from the saved sequence built in forward.

    Memory cost: stores L_c, Aw_c, etc. for c = 0..C in a Python list.
    For C=64, BH=128, r=128, that's about 4 GB per chunk, which is fine.
    If memory becomes a bottleneck, we can switch to recompute-from-final
    via the inverse Givens rotations (orthogonal so cheap to invert).
    """

    @staticmethod
    def forward(ctx, L, Aw, K, V):
        L_new, Aw_new, K_tilde, saved = _chunk_forward(L, Aw, K, V)
        ctx.K = K
        ctx.V = V
        ctx.saved_seq = saved
        return L_new.contiguous(), Aw_new.contiguous(), K_tilde.contiguous()

    @staticmethod
    def backward(ctx, grad_L_new, grad_Aw_new, grad_K_tilde):
        K = ctx.K
        V = ctx.V
        L_seq, Aw_seq, vz_seq, vw_seq, k_tilde_seq, T_mat_seq = ctx.saved_seq

        BH, C, r = K.shape
        if grad_L_new is None:
            grad_L_new = torch.zeros(BH, r, r, dtype=K.dtype, device=K.device)
        if grad_Aw_new is None:
            grad_Aw_new = torch.zeros(BH, r, r, dtype=K.dtype, device=K.device)
        if grad_K_tilde is None:
            grad_K_tilde = torch.zeros(BH, C, r, dtype=K.dtype, device=K.device)

        grad_K = torch.zeros_like(K)
        grad_V = torch.zeros_like(V)

        # Walk backward through the C tokens.
        grad_L_curr = grad_L_new
        grad_Aw_curr = grad_Aw_new
        for c in reversed(range(C)):
            grad_L_c, grad_Aw_c, grad_z_c, grad_w_c = _per_token_backward_step(
                L=L_seq[c],
                Aw=Aw_seq[c],
                z=K[:, c],
                w=V[:, c],
                L_new=L_seq[c + 1],
                vz=vz_seq[c],
                vw=vw_seq[c],
                k_tilde=k_tilde_seq[c],
                T_mat=T_mat_seq[c],
                grad_L_new=grad_L_curr,
                grad_Aw_new=grad_Aw_curr,
                grad_k_tilde=grad_K_tilde[:, c],
            )
            grad_K[:, c] = grad_z_c
            grad_V[:, c] = grad_w_c
            grad_L_curr = grad_L_c
            grad_Aw_curr = grad_Aw_c

        return grad_L_curr, grad_Aw_curr, grad_K, grad_V


def ska_chunk(L, Aw, K, V):
    """C-token SKA update with one autograd.Function per chunk.

    Numerically equivalent to calling ska_step C times in a row, but
    creates only ONE autograd graph node per chunk instead of C.

    Args:
      L:  (BH, r, r) lower triangular Cholesky factor at chunk start.
      Aw: (BH, r, r) whitened cross-Gram at chunk start.
      K:  (BH, C, r) per-token keys for the chunk.
      V:  (BH, C, r) per-token values for the chunk.

    Returns:
      L_new:   (BH, r, r)
      Aw_new:  (BH, r, r)
      K_tilde: (BH, C, r) exact preconditioned write keys, one per token.
    """
    return _SKAChunkUpdate.apply(L, Aw, K, V)


# ---------------------------------------------------------------------------
# Sanity tests
# ---------------------------------------------------------------------------

def _make_problem(T=2, r=4, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    A = torch.randn(T, r, r, dtype=dtype)
    G = A @ A.transpose(-1, -2) + 2 * r * torch.eye(r, dtype=dtype)
    L = torch.linalg.cholesky(G).contiguous()
    Aw = torch.randn(T, r, r, dtype=dtype) * 0.1
    z = torch.randn(T, r, dtype=dtype) * 0.1
    w = torch.randn(T, r, dtype=dtype) * 0.1
    return L, Aw, z, w


def _check_forward_consistency(T=4, r=8):
    """Triton vs reference forward agreement (only if CUDA + Triton)."""
    L, Aw, z, w = _make_problem(T=T, r=r, dtype=torch.float32)
    L_ref, Aw_ref, vz_ref, vw_ref, kt_ref = _forward_reference(L.clone(), Aw, z, w)
    if torch.cuda.is_available() and _HAVE_TRITON:
        L_g, Aw_g, z_g, w_g = (x.cuda() for x in (L, Aw, z, w))
        L_t, Aw_t, vz_t, vw_t, kt_t = _forward_triton(L_g, Aw_g, z_g, w_g)
        for name, ref, kt in [
            ("L", L_ref, L_t.cpu()),
            ("Aw", Aw_ref, Aw_t.cpu()),
            ("vz", vz_ref, vz_t.cpu()),
            ("vw", vw_ref, vw_t.cpu()),
            ("k_tilde", kt_ref, kt_t.cpu()),
        ]:
            err = (ref - kt).norm() / (ref.norm() + 1e-30)
            print(f"  {name:<8}: rel err {err.item():.2e}")
    else:
        print("  (CUDA + Triton unavailable; skipping)")


def _check_manual_vs_autograd(T=2, r=4, dtype=torch.float64):
    """Manual backward should agree with autograd through the reference."""
    L, Aw, z, w = _make_problem(T=T, r=r, dtype=dtype)

    # Path A: autograd through the reference.
    L_a = L.clone().requires_grad_(True)
    Aw_a = Aw.clone().requires_grad_(True)
    z_a = z.clone().requires_grad_(True)
    w_a = w.clone().requires_grad_(True)
    L_n, Aw_n, vz_n, vw_n, kt_n = _forward_reference(L_a, Aw_a, z_a, w_a)
    loss_a = 1.3 * L_n.sum() + 0.7 * Aw_n.sum() + 2.1 * kt_n.sum()
    loss_a.backward()

    # Path B: ska_step (manual backward).
    L_m = L.clone().requires_grad_(True)
    Aw_m = Aw.clone().requires_grad_(True)
    z_m = z.clone().requires_grad_(True)
    w_m = w.clone().requires_grad_(True)
    L_n2, Aw_n2, kt_n2 = ska_step(L_m, Aw_m, z_m, w_m)
    loss_b = 1.3 * L_n2.sum() + 0.7 * Aw_n2.sum() + 2.1 * kt_n2.sum()
    loss_b.backward()

    rows = [("L", L_a.grad, L_m.grad),
            ("Aw", Aw_a.grad, Aw_m.grad),
            ("z", z_a.grad, z_m.grad),
            ("w", w_a.grad, w_m.grad)]
    max_err = 0.0
    for name, ga, gm in rows:
        err = (ga - gm).norm() / (ga.norm() + 1e-30)
        max_err = max(max_err, err.item())
        print(f"  grad_{name:<3}: rel err {err.item():.2e}")
    return max_err


def _gradcheck(T=1, r=4):
    L, Aw, z, w = _make_problem(T=T, r=r, dtype=torch.float64)
    L = L.clone().requires_grad_(True)
    Aw = Aw.clone().requires_grad_(True)
    z = z.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)

    def fn(L_, Aw_, z_, w_):
        L_n, Aw_n, kt = ska_step(L_, Aw_, z_, w_)
        return 1.3 * L_n.sum() + 0.7 * Aw_n.sum() + 2.1 * kt.sum()

    ok = torch.autograd.gradcheck(fn, (L, Aw, z, w), atol=1e-4, eps=1e-6)
    print(f"  gradcheck: {ok}")
    return ok


def _multi_step_check(T_seq=8, r=4, dtype=torch.float64):
    """Verify a multi-step rollout (BPTT) matches autograd through reference."""
    torch.manual_seed(0)
    B = 2
    L0 = torch.eye(r, dtype=dtype).expand(B, r, r).contiguous()
    Aw0 = torch.zeros(B, r, r, dtype=dtype)
    Ks = (torch.randn(T_seq, B, r, dtype=dtype) * 0.1)
    Vs = (torch.randn(T_seq, B, r, dtype=dtype) * 0.1)

    # Path A: reference rollout
    L_a, Aw_a = L0.clone(), Aw0.clone()
    Ks_a = Ks.clone().requires_grad_(True)
    Vs_a = Vs.clone().requires_grad_(True)
    loss_a = 0.0
    for t in range(T_seq):
        L_a, Aw_a, _, _, kt = _forward_reference(L_a, Aw_a, Ks_a[t], Vs_a[t])
        loss_a = loss_a + (kt.sum() + L_a.sum() + Aw_a.sum())
    loss_a.backward()

    # Path B: ska_step rollout
    L_b, Aw_b = L0.clone(), Aw0.clone()
    Ks_b = Ks.clone().requires_grad_(True)
    Vs_b = Vs.clone().requires_grad_(True)
    loss_b = 0.0
    for t in range(T_seq):
        L_b, Aw_b, kt = ska_step(L_b, Aw_b, Ks_b[t], Vs_b[t])
        loss_b = loss_b + (kt.sum() + L_b.sum() + Aw_b.sum())
    loss_b.backward()

    err_k = (Ks_a.grad - Ks_b.grad).norm() / (Ks_a.grad.norm() + 1e-30)
    err_v = (Vs_a.grad - Vs_b.grad).norm() / (Vs_a.grad.norm() + 1e-30)
    print(f"  multi-step grad_k rel err: {err_k.item():.2e}")
    print(f"  multi-step grad_v rel err: {err_v.item():.2e}")
    return max(err_k.item(), err_v.item())


def _check_chunk_matches_per_token(C=4, BH=2, r=4, dtype=torch.float64):
    """Chunkwise forward + backward must equal C per-token calls in a row."""
    torch.manual_seed(0)
    A = torch.randn(BH, r, r, dtype=dtype)
    G = A @ A.transpose(-1, -2) + 2 * r * torch.eye(r, dtype=dtype)
    L0 = torch.linalg.cholesky(G).contiguous()
    Aw0 = torch.randn(BH, r, r, dtype=dtype) * 0.1
    K = torch.randn(BH, C, r, dtype=dtype) * 0.1
    V = torch.randn(BH, C, r, dtype=dtype) * 0.1

    # Path A: per-token rollout via ska_step.
    L_a = L0.clone().requires_grad_(True)
    Aw_a = Aw0.clone().requires_grad_(True)
    K_a = K.clone().requires_grad_(True)
    V_a = V.clone().requires_grad_(True)
    L_curr = L_a
    Aw_curr = Aw_a
    K_tilde_a_list = []
    for c in range(C):
        L_curr, Aw_curr, kt = ska_step(L_curr, Aw_curr, K_a[:, c], V_a[:, c])
        K_tilde_a_list.append(kt)
    K_tilde_a = torch.stack(K_tilde_a_list, dim=1)
    loss_a = 1.3 * L_curr.sum() + 0.7 * Aw_curr.sum() + 2.1 * K_tilde_a.sum()
    loss_a.backward()

    # Path B: ska_chunk in one shot.
    L_b = L0.clone().requires_grad_(True)
    Aw_b = Aw0.clone().requires_grad_(True)
    K_b = K.clone().requires_grad_(True)
    V_b = V.clone().requires_grad_(True)
    L_n, Aw_n, K_tilde_b = ska_chunk(L_b, Aw_b, K_b, V_b)
    loss_b = 1.3 * L_n.sum() + 0.7 * Aw_n.sum() + 2.1 * K_tilde_b.sum()
    loss_b.backward()

    err_K_tilde = (K_tilde_a - K_tilde_b).norm() / (K_tilde_a.norm() + 1e-30)
    err_L = (L_curr - L_n).norm() / (L_curr.norm() + 1e-30)
    err_Aw = (Aw_curr - Aw_n).norm() / (Aw_curr.norm() + 1e-30)
    print(f"  forward K_tilde rel err: {err_K_tilde.item():.2e}")
    print(f"  forward L_new   rel err: {err_L.item():.2e}")
    print(f"  forward Aw_new  rel err: {err_Aw.item():.2e}")

    rows = [("L", L_a.grad, L_b.grad),
            ("Aw", Aw_a.grad, Aw_b.grad),
            ("K", K_a.grad, K_b.grad),
            ("V", V_a.grad, V_b.grad)]
    max_err = 0.0
    for name, ga, gb in rows:
        err = (ga - gb).norm() / (ga.norm() + 1e-30)
        max_err = max(max_err, err.item())
        print(f"  grad_{name:<2}: rel err {err.item():.2e}")
    return max_err


def _gradcheck_chunk(C=2, BH=1, r=4):
    """gradcheck on ska_chunk."""
    torch.manual_seed(0)
    dtype = torch.float64
    A = torch.randn(BH, r, r, dtype=dtype)
    G = A @ A.transpose(-1, -2) + 2 * r * torch.eye(r, dtype=dtype)
    L = torch.linalg.cholesky(G).contiguous().requires_grad_(True)
    Aw = (torch.randn(BH, r, r, dtype=dtype) * 0.1).requires_grad_(True)
    K = (torch.randn(BH, C, r, dtype=dtype) * 0.1).requires_grad_(True)
    V = (torch.randn(BH, C, r, dtype=dtype) * 0.1).requires_grad_(True)

    def fn(L_, Aw_, K_, V_):
        L_n, Aw_n, K_tilde = ska_chunk(L_, Aw_, K_, V_)
        return 1.3 * L_n.sum() + 0.7 * Aw_n.sum() + 2.1 * K_tilde.sum()

    ok = torch.autograd.gradcheck(fn, (L, Aw, K, V), atol=1e-4, eps=1e-6)
    print(f"  gradcheck (chunk): {ok}")
    return ok


if __name__ == "__main__":
    print("== Forward consistency (Triton vs reference) ==")
    _check_forward_consistency(T=4, r=8)
    print()
    print("== Manual backward vs autograd-through-reference ==")
    _check_manual_vs_autograd(T=2, r=4)
    print()
    print("== gradcheck (per-token) ==")
    _gradcheck(T=1, r=4)
    print()
    print("== Multi-step BPTT (per-token) ==")
    _multi_step_check(T_seq=8, r=4)
    print()
    print("== Chunkwise == per-token ==")
    _check_chunk_matches_per_token(C=4, BH=2, r=4)
    print()
    print("== gradcheck (chunkwise) ==")
    _gradcheck_chunk(C=2, BH=1, r=4)
    print()
    print("OK")
