"""
Double-Sided Cholesky Propagation: O(r^2) Whitened Operator Updates

Given:  L, A_w = L^{-1} M L^{-T}, z, w
Update: G+ = G + z z^T,  M+ = M + z w^T
Compute: L+, A_w+ = (L+)^{-1} M+ (L+)^{-T}

This file provides two things:

  1. A pure-PyTorch reference implementation (incremental_update_reference)
     used for correctness checking. It is slow and should not be benchmarked.

  2. A fused Triton kernel (incremental_update_triton) that runs as a single
     GPU block with SRAM-resident working sets. Gated at r <= 1023 so that
     r+1 fits in a single power-of-two BLOCK up to 1024.

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
# KNOWN ISSUE (see TODO.md item 5): Phase 2's right sweep is currently
# implemented as a nested loop with a full row load/store per (k, i) pair.
# This is functionally correct but is O(r^2 * BLOCK) memory traffic for
# an O(r^2) algorithm. The intended implementation transposes Ah into
# scratch, sweeps rows (which are the original columns), then transposes
# back; the comment inside the kernel describes this but the code falls
# back to the slow row-by-row version. Fixing this is the main GPU-side
# project for students.

@triton.jit
def _givens_params_tri(a, b):
    abs_a = tl.abs(a)
    abs_b = tl.abs(b)
    zero = abs_b == 0.0
    b_big = abs_b > abs_a

    t1 = -a / (b + tl.where(zero, 1.0, 0.0))
    s1 = 1.0 / tl.sqrt(1.0 + t1 * t1)
    c1 = s1 * t1

    t2 = -b / (a + tl.where(abs_a == 0.0, 1.0, 0.0))
    c2 = 1.0 / tl.sqrt(1.0 + t2 * t2)
    s2 = c2 * t2

    c = tl.where(zero, 1.0, tl.where(b_big, c1, c2))
    s = tl.where(zero, 0.0, tl.where(b_big, s1, s2))
    return c, s


@triton.jit
def _fused_doublesided_kernel(
    L_ptr, Aw_ptr, z_ptr, w_ptr,
    vz_ptr, Ah_ptr, cs_ptr, ss_ptr,
    r: tl.constexpr,
    RP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """SRAM-resident fused kernel. BLOCK must be a power of two >= r+1.

    All matrix data should ideally be loaded into registers/SRAM once per
    phase and written back once at the end. See KNOWN ISSUE above for the
    right-sweep memory traffic problem.
    """

    # Phase 1: Rank-1 Cholesky update.
    cols = tl.arange(0, BLOCK)

    for k in range(r):
        mask_r = cols < r
        L_row = tl.load(L_ptr + k * r + cols, mask=mask_r, other=0.0)
        z_vec = tl.load(z_ptr + cols, mask=mask_r, other=0.0)

        Lkk = tl.sum(tl.where(cols == k, L_row, 0.0))
        zk = tl.sum(tl.where(cols == k, z_vec, 0.0))
        c, s = _givens_params_tri(Lkk, zk)

        tl.store(cs_ptr + k, c)
        tl.store(ss_ptr + k, s)

        active = mask_r & (cols >= k)
        new_L = c * L_row + s * z_vec
        new_z = -s * L_row + c * z_vec
        tl.store(L_ptr + k * r + cols, tl.where(active, new_L, L_row), mask=mask_r)
        tl.store(z_ptr + cols, tl.where(active, new_z, z_vec), mask=mask_r)

    # Phase 2: Double-sided propagation.
    cols_p = tl.arange(0, BLOCK)
    mask_rp = cols_p < RP

    for i in range(r):
        mask_aw = cols_p < r
        aw_row = tl.load(Aw_ptr + i * r + cols_p, mask=mask_aw, other=0.0)
        padded = tl.where(mask_aw, aw_row, 0.0)
        tl.store(Ah_ptr + i * RP + cols_p, padded, mask=mask_rp)

    tl.store(Ah_ptr + r * RP + cols_p, 0.0, mask=mask_rp)

    # Left sweep: Q_k^T mixes rows k and r.
    for k in range(r):
        c = tl.load(cs_ptr + k)
        s = tl.load(ss_ptr + k)

        rk = tl.load(Ah_ptr + k * RP + cols_p, mask=mask_rp, other=0.0)
        rr = tl.load(Ah_ptr + r * RP + cols_p, mask=mask_rp, other=0.0)

        tl.store(Ah_ptr + k * RP + cols_p, c * rk + s * rr, mask=mask_rp)
        tl.store(Ah_ptr + r * RP + cols_p, -s * rk + c * rr, mask=mask_rp)

    # Right sweep: SLOW PATH, see KNOWN ISSUE above. The intended plan is
    # to transpose Ah into a (RP x RP) scratch buffer, run the sweep over
    # rows of the transpose (which are columns of the original), and
    # transpose back. That requires a second SRAM-sized scratch region.
    # The version below is the fallback that works but wastes memory traffic.
    for k in range(r):
        c = tl.load(cs_ptr + k)
        s = tl.load(ss_ptr + k)

        for i in range(RP):
            row = tl.load(Ah_ptr + i * RP + cols_p, mask=mask_rp, other=0.0)

            val_k = tl.sum(tl.where(cols_p == k, row, 0.0))
            val_r = tl.sum(tl.where(cols_p == r, row, 0.0))

            new_k = c * val_k + s * val_r
            new_r = -s * val_k + c * val_r

            row = tl.where(cols_p == k, new_k, row)
            row = tl.where(cols_p == r, new_r, row)
            tl.store(Ah_ptr + i * RP + cols_p, row, mask=mask_rp)

    # Phase 3: recover v_z.
    e_last = 1.0
    for k in range(r):
        c = tl.load(cs_ptr + k)
        s = tl.load(ss_ptr + k)
        tl.store(vz_ptr + k, s * e_last)
        e_last = c * e_last

    # Phase 4: forward substitution with f64 accumulation. Overwrites w_ptr
    # with v_w.
    for k in range(r):
        wk = tl.load(w_ptr + k).to(tl.float64)
        L_row = tl.load(L_ptr + k * r + cols, mask=cols < r, other=0.0)
        vw_prev = tl.load(w_ptr + cols, mask=cols < r, other=0.0)

        contrib = tl.where(cols < k, L_row.to(tl.float64) * vw_prev.to(tl.float64), 0.0)
        acc = tl.sum(contrib, axis=0)

        Lkk = tl.sum(tl.where(cols == k, L_row, 0.0))
        vw_k = ((wk - acc) / Lkk.to(tl.float64)).to(L_row.dtype)
        tl.store(w_ptr + k, vw_k)

    # Rank-1 correction: Ah[:r, :r] += v_z v_w^T.
    for i in range(r):
        vzi = tl.load(vz_ptr + i)
        vw_vec = tl.load(w_ptr + cols_p, mask=cols_p < r, other=0.0)
        cur = tl.load(Ah_ptr + i * RP + cols_p, mask=cols_p < r, other=0.0)
        tl.store(Ah_ptr + i * RP + cols_p, cur + vzi * vw_vec, mask=cols_p < r)


def incremental_update_triton(L, Aw, z, w):
    """Fused Triton update. Modifies L in place to L+.

    Requires r <= 1023 so r+1 fits in a single BLOCK of at most 1024.
    For larger r, use the C++ extension or tile the kernel.

    Returns:
        Aw_new: (r, r) updated whitened operator.
        vz: (r,) the vector (L+)^{-1} z.
    """
    r = L.shape[0]
    rp = r + 1

    BLOCK = triton.next_power_of_2(rp)
    assert BLOCK <= 1024, f"r={r} too large for single-block kernel"

    vz = torch.empty(r, dtype=L.dtype, device=L.device)
    Ah = torch.empty(rp, rp, dtype=L.dtype, device=L.device)
    cs = torch.empty(r, dtype=L.dtype, device=L.device)
    ss = torch.empty(r, dtype=L.dtype, device=L.device)

    z_work = z.clone()
    w_work = w.clone()

    _fused_doublesided_kernel[(1,)](
        L, Aw, z_work, w_work, vz, Ah, cs, ss,
        r=r, RP=rp, BLOCK=BLOCK,
    )

    return Ah[:r, :r].contiguous(), vz
