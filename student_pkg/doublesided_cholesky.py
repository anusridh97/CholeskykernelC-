"""
Double-Sided Cholesky Propagation: O(r^2) Whitened Operator Updates

Given:  L, A_w = L^{-1} M L^{-T}, z, w
Update: G+ = G + z z^T,  M+ = M + z w^T
Compute: L+, A_w+ = (L+)^{-1} M+ (L+)^{-T}

This file provides two things:

  1. A pure-PyTorch reference implementation (incremental_update_reference)
     used for correctness checking. It is slow and should not be benchmarked.

  2. A fused Triton kernel (incremental_update_triton) that runs as a single
     GPU block. Gated at r <= 1024 so that r fits in a single power-of-two
     BLOCK up to 1024.

For the fast CPU path, see fast_update.cpp / the fast_cholesky extension.
Intended to be called inside torch.no_grad(); no autograd graph is built.
"""

import torch
import triton
import triton.language as tl


def givens_params(a: float, b: float):
    """Givens (c, s) such that c*a + s*b = rho, -s*a + c*b = 0."""
    import math
    rho = math.sqrt(a * a + b * b)
    if rho == 0:
        return 1.0, 0.0
    return a / rho, b / rho


def cholesky_rank1_update(L, z):
    """Rank-1 Cholesky update via augmented Givens rotations.

    Operates on [L | z] by right-multiplying Givens rotations that zero
    each entry of the last column. Modifies L in place to L+. Returns
    (cs, ss) rotation parameters, which Phase 2 re-uses.
    """
    r = L.shape[0]
    cs = torch.zeros(r, dtype=L.dtype, device=L.device)
    ss = torch.zeros(r, dtype=L.dtype, device=L.device)

    for k in range(r):
        c, s = givens_params(L[k, k].item(), z[k].item())
        cs[k], ss[k] = c, s

        col_k = L[:, k].clone()
        col_z = z.clone()
        L[:, k] = c * col_k + s * col_z
        z[:] = -s * col_k + c * col_z

    return cs, ss


def doublesided_propagate(Aw, cs, ss):
    """Apply Q^T [Aw 0; 0 0] Q, return the top-left r x r block."""
    r = Aw.shape[0]
    Ah = torch.zeros(r + 1, r + 1, dtype=Aw.dtype, device=Aw.device)
    Ah[:r, :r] = Aw

    for k in range(r):
        c, s = cs[k].item(), ss[k].item()
        rk, rr = Ah[k].clone(), Ah[r].clone()
        Ah[k] = c * rk + s * rr
        Ah[r] = -s * rk + c * rr

    for k in range(r):
        c, s = cs[k].item(), ss[k].item()
        ck, cr = Ah[:, k].clone(), Ah[:, r].clone()
        Ah[:, k] = c * ck + s * cr
        Ah[:, r] = -s * ck + c * cr

    return Ah[:r, :r].clone()


def recover_vz(cs, ss, r):
    """Top r entries of Q^T e_{r+1} = (L+)^{-1} z."""
    vz = torch.zeros(r, dtype=cs.dtype, device=cs.device)
    e_last = 1.0
    for k in range(r):
        c, s = cs[k].item(), ss[k].item()
        vz[k] = s * e_last
        e_last = c * e_last
    return vz


def incremental_update_reference(L, Aw, z, w):
    """Full O(r^2) update. Modifies L in place to L+. Slow, for correctness."""
    r = L.shape[0]
    z_work = z.clone()
    cs, ss = cholesky_rank1_update(L, z_work)
    Aw_core = doublesided_propagate(Aw, cs, ss)
    vz = recover_vz(cs, ss, r)
    vw = torch.linalg.solve_triangular(L, w.unsqueeze(1), upper=False).squeeze(1)
    return Aw_core + torch.outer(vz, vw)


# Fused Triton Kernel
#
# Phase 2's right sweep updates column vectors directly. That avoids the
# old row-patching O(r^2 * BLOCK) memory traffic without paying for full
# global-memory transposes through a second scratch matrix.

@triton.jit
def _givens_params_tri(a, b):
    rho = tl.sqrt(a * a + b * b)
    zero = rho == 0.0
    c = tl.where(zero, 1.0, a / rho)
    s = tl.where(zero, 0.0, b / rho)
    return c, s


@triton.jit
def _fused_doublesided_kernel(
    L_ptr, Aw_ptr, z_ptr, w_ptr,
    vz_ptr, Aw_out_ptr, cs_ptr, ss_ptr,
    r: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Single-update fused kernel. BLOCK must be a power of two >= r.

    Phase 2 keeps the padded row/column virtual and only writes the returned
    r x r top-left block.
    """

    # Phase 1: Rank-1 Cholesky update.
    idx = tl.arange(0, BLOCK)

    for k in range(r):
        mask_r = idx < r
        L_col = tl.load(L_ptr + idx * r + k, mask=mask_r, other=0.0)
        z_vec = tl.load(z_ptr + idx, mask=mask_r, other=0.0)

        Lkk = tl.sum(tl.where(idx == k, L_col, 0.0))
        zk = tl.sum(tl.where(idx == k, z_vec, 0.0))
        c, s = _givens_params_tri(Lkk, zk)

        tl.store(cs_ptr + k, c)
        tl.store(ss_ptr + k, s)

        active = mask_r & (idx >= k)
        new_L = c * L_col + s * z_vec
        new_z = -s * L_col + c * z_vec
        tl.store(L_ptr + idx * r + k, new_L, mask=active)
        tl.store(z_ptr + idx, tl.where(active, new_z, z_vec), mask=mask_r)

    # Phase 2: Double-sided propagation. The padded row and column start as
    # zeros; only the top-left r x r block is materialized in Aw_out_ptr.
    cols_p = tl.arange(0, BLOCK)
    mask_r = cols_p < r
    pad_row = tl.zeros((BLOCK,), dtype=tl.float32)

    # Left sweep: Q_k^T mixes row k with the virtual padded row.
    for k in range(r):
        c = tl.load(cs_ptr + k)
        s = tl.load(ss_ptr + k)

        rk = tl.load(Aw_ptr + k * r + cols_p, mask=mask_r, other=0.0)
        rr = pad_row

        tl.store(Aw_out_ptr + k * r + cols_p, c * rk + s * rr, mask=mask_r)
        pad_row = -s * rk + c * rr

    # Right sweep: Q_k mixes columns k and r. Vectorize over rows so each
    # column is loaded and stored once per rotation. The padded column stays
    # in registers and is discarded because only the top-left block is needed.
    rows = tl.arange(0, BLOCK)
    row_mask = rows < r
    pad_col = tl.zeros((BLOCK,), dtype=tl.float32)

    for k in range(r):
        c = tl.load(cs_ptr + k)
        s = tl.load(ss_ptr + k)

        ck = tl.load(Aw_out_ptr + rows * r + k, mask=row_mask, other=0.0)
        cr = pad_col

        tl.store(Aw_out_ptr + rows * r + k, c * ck + s * cr, mask=row_mask)
        pad_col = -s * ck + c * cr

    # Phase 3: recover v_z.
    e_last = 1.0
    for k in range(r):
        c = tl.load(cs_ptr + k)
        s = tl.load(ss_ptr + k)
        tl.store(vz_ptr + k, s * e_last)
        e_last = c * e_last

    # Phase 4: forward substitution with f32 accumulation. Overwrites w_ptr
    # with v_w.
    for k in range(r):
        wk = tl.load(w_ptr + k).to(tl.float32)
        L_row = tl.load(L_ptr + k * r + idx, mask=idx < r, other=0.0)
        vw_prev = tl.load(w_ptr + idx, mask=idx < r, other=0.0)

        contrib = tl.where(idx < k, L_row.to(tl.float32) * vw_prev.to(tl.float32), 0.0)
        acc = tl.sum(contrib, axis=0)

        Lkk = tl.sum(tl.where(idx == k, L_row, 0.0))
        vw_k = ((wk - acc) / Lkk.to(tl.float32)).to(L_row.dtype)
        tl.store(w_ptr + k, vw_k)

    # Rank-1 correction: Aw_out += v_z v_w^T.
    for i in range(r):
        vzi = tl.load(vz_ptr + i)
        vw_vec = tl.load(w_ptr + cols_p, mask=cols_p < r, other=0.0)
        cur = tl.load(Aw_out_ptr + i * r + cols_p, mask=cols_p < r, other=0.0)
        tl.store(Aw_out_ptr + i * r + cols_p, cur + vzi * vw_vec, mask=cols_p < r)


def incremental_update_triton(L, Aw, z, w, num_warps=None):
    """Fused Triton update. Modifies L in place to L+.

    Requires r <= 1024 so r fits in a single BLOCK of at most 1024.
    For larger r, use the C++ extension or tile the kernel.

    Returns:
        Aw_new: (r, r) updated whitened operator.
        vz: (r,) the vector (L+)^{-1} z.
    """
    r = L.shape[0]

    BLOCK = triton.next_power_of_2(r)
    assert BLOCK <= 1024, f"r={r} too large for single-block kernel"

    vz = z.clone()
    Aw_out = torch.empty_like(Aw)
    cs = torch.empty(r, dtype=L.dtype, device=L.device)
    ss = torch.empty(r, dtype=L.dtype, device=L.device)

    w_work = w.clone()

    launch_kwargs = {"r": r, "BLOCK": BLOCK}
    if num_warps is not None:
        launch_kwargs["num_warps"] = num_warps

    _fused_doublesided_kernel[(1,)](
        L, Aw, vz, w_work, vz, Aw_out, cs, ss,
        **launch_kwargs,
    )

    return Aw_out, vz
