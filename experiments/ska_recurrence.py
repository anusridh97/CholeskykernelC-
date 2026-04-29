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

# Make the kernel importable.
_HERE = Path(__file__).resolve().parent
_STUDENT = _HERE.parent / "student_pkg"
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

class _SKAUpdate(torch.autograd.Function):
    """Per-step SKA update with manual backward.

    Forward returns (L_new, Aw_new, k_tilde).
    Backward implements the seven-step adjoint computation derived in the
    docstring of `ska_recurrence.py` and the corresponding paper section.
    """

    @staticmethod
    def forward(ctx, L, Aw, z, w):
        use_triton = _HAVE_TRITON and L.is_cuda and Aw.is_cuda and z.is_cuda and w.is_cuda
        if use_triton:
            L_new, Aw_new, vz, vw, k_tilde = _forward_triton(L, Aw, z, w)
        else:
            with torch.no_grad():
                L_new, Aw_new, vz, vw, k_tilde = _forward_reference(L, Aw, z, w)

        # Transport matrix T = L_new^{-1} L (one lower-triangular solve).
        T_mat = torch.linalg.solve_triangular(L_new, L, upper=False)

        ctx.save_for_backward(L, Aw, z, w, L_new, vz, vw, k_tilde, T_mat)
        return L_new.contiguous(), Aw_new.contiguous(), k_tilde.contiguous()

    @staticmethod
    def backward(ctx, grad_L_new, grad_Aw_new, grad_k_tilde):
        L, Aw, z, w, L_new, vz, vw, k_tilde, T_mat = ctx.saved_tensors

        # Replace any None grad with zero (autograd convention).
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
        # grad_vz_init = L_new^{-1} grad_k_tilde
        grad_vz = torch.linalg.solve_triangular(
            L_new, grad_k_tilde.unsqueeze(-1), upper=False
        ).squeeze(-1)
        # grad_L_new -= k_tilde grad_k_tilde^T L_new^{-T}
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
        grad_L_from_T = Linv_T_grad_T  # combined later in Step 7

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
        # Standard formula: grad_G_new = L_new^{-T} P L_new^{-1}
        # Two solves:
        #   Y = L_new^{-T} P                        (left solve by L_new^{-T})
        #   grad_G_new = Y @ L_new^{-1}             (right solve by L_new^{-1})
        # For the right solve we use A @ L^{-1} = (L^{-T} A^T)^T:
        #   solve_triangular(L_new.T, Y.T, upper=True).T
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


def ska_step(L, Aw, k, v):
    """One SKA recurrence step with manual backward.

    Args:
      L:  (T, r, r) lower triangular Cholesky factor of G_t.
      Aw: (T, r, r) whitened cross-Gram, A_w = L^{-1} M L^{-T}.
      k:  (T, r) per-token key (z in the paper).
      v:  (T, r) per-token value (w in the paper).

    Returns:
      L_new:   (T, r, r)  Cholesky of G_t + k k^T.
      Aw_new:  (T, r, r)  whitened cross-Gram after the update.
      k_tilde: (T, r)     exact normalized write key, k_tilde = G_new^{-1} k.
    """
    return _SKAUpdate.apply(L, Aw, k, v)


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


if __name__ == "__main__":
    print("== Forward consistency (Triton vs reference) ==")
    _check_forward_consistency(T=4, r=8)
    print()
    print("== Manual backward vs autograd-through-reference ==")
    _check_manual_vs_autograd(T=2, r=4)
    print()
    print("== gradcheck ==")
    _gradcheck(T=1, r=4)
    print()
    print("== Multi-step BPTT ==")
    _multi_step_check(T_seq=8, r=4)
    print()
    print("OK")
