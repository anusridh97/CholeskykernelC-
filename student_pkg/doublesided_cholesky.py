"""
Double-Sided Cholesky Propagation: O(r^2) Whitened Operator Updates

Given:  L, A_w = L^{-1} M L^{-T}, z, w
Update: G+ = G + z z^T,  M+ = M + z w^T
Compute: L+, A_w+ = (L+)^{-1} M+ (L+)^{-T}

This file provides three Triton entry points sharing two underlying kernels:

  1. A pure-PyTorch reference implementation (incremental_update_reference)
     used for correctness checking. It is slow and should not be benchmarked.

  2. A fused Triton kernel runnable in single-update (incremental_update_triton)
     and batched (incremental_update_triton_batched) modes — same SRAM-resident
     kernel, dispatched with grid=(1,) or grid=(T,). Gated at r <= 1023 so that
     r+1 fits in a single power-of-two BLOCK up to 1024.

  3. An experimental tiled Triton path (incremental_update_triton_tiled) that
     keeps dependency-limited setup work single-block, but parallelizes the
     double-sided propagation and rank-1 correction across row/column tiles.

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
# Phase 2's right sweep is implemented as a nested loop with a full row
# load/store per (k, i) pair. This is functionally correct but is
# O(r^2 * BLOCK) memory traffic for an O(r^2) algorithm. The intended
# implementation transposes Ah into scratch, sweeps rows (which are the
# original columns), then transposes back. Fixing this is the main
# GPU-side project for students.

@triton.jit
def _givens_params_tri_numerically_stable(a, b):
    """Two-branch Givens via cotangent/tangent — avoids sqrt(a^2+b^2) overflow.
    NOTE: signs are wrong as-written; kept for reference only. Do not call."""
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
def _givens_params_tri(a, b):
    """Givens (c, s) such that c*a + s*b = rho, -s*a + c*b = 0."""
    rho = tl.sqrt(a * a + b * b)
    zero = rho == 0.0
    safe_rho = rho + tl.where(zero, 1.0, 0.0)
    c = tl.where(zero, 1.0, a / safe_rho)
    s = tl.where(zero, 0.0, b / safe_rho)
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

    Supports batched execution: launch with grid=(T,) and pass (T,...) tensors.
    Each program instance handles one independent update via tl.program_id(0).
    grid=(1,) gives the original single-update behaviour (pid=0, all offsets 0).

    All matrix data should ideally be loaded into registers/SRAM once per
    phase and written back once at the end. See KNOWN ISSUE above for the
    right-sweep memory traffic problem.
    """
    pid     = tl.program_id(0)
    off_mat = pid * r * r      # stride for (r, r) tensors: L, Aw
    off_vec = pid * r          # stride for (r,) tensors: z, w, vz, cs, ss
    off_ah  = pid * RP * RP    # stride for (RP, RP) scratch: Ah

    # Phase 1: Rank-1 Cholesky update.
    #
    # 'rows' and 'cols' are the same tl.arange(0, BLOCK) vector — they are
    # aliases, not different data.  The two names exist to make the access
    # pattern self-documenting:
    #
    #   Phase 1  uses 'rows': the index sweeps row positions to load an
    #            entire column at once (L_ptr + rows*r + k  →  column k).
    #
    #   Phase 2 left sweep / Phase 4  use 'cols': the index sweeps column
    #            positions to load an entire row at once (L_ptr + k*r + cols
    #            →  row k).
    #
    # Both are just [0, 1, ..., BLOCK-1]; the name signals which matrix
    # dimension is being vectorised over.
    rows = tl.arange(0, BLOCK)
    cols = rows  # same vector; renamed below when sweeping columns not rows

    for k in range(r):
        mask_r = rows < r
        # Load column k of L (stride-r gather) and the full z vector.
        L_col = tl.load(L_ptr + off_mat + rows * r + k, mask=mask_r, other=0.0)
        z_vec = tl.load(z_ptr + off_vec + rows, mask=mask_r, other=0.0)

        Lkk = tl.sum(tl.where(rows == k, L_col, 0.0))
        zk  = tl.sum(tl.where(rows == k, z_vec, 0.0))
        c, s = _givens_params_tri(Lkk, zk)

        tl.store(cs_ptr + off_vec + k, c)
        tl.store(ss_ptr + off_vec + k, s)

        # Only update rows >= k (lower-triangular region of column k).
        active  = mask_r & (rows >= k)
        new_col = c * L_col + s * z_vec
        new_z   = -s * L_col + c * z_vec
        tl.store(L_ptr + off_mat + rows * r + k, tl.where(active, new_col, L_col), mask=mask_r)
        tl.store(z_ptr + off_vec + rows, tl.where(active, new_z, z_vec), mask=mask_r)

    # Phase 2: Double-sided propagation.
    cols_p = tl.arange(0, BLOCK)
    mask_rp = cols_p < RP

    for i in range(r):
        mask_aw = cols_p < r
        aw_row = tl.load(Aw_ptr + off_mat + i * r + cols_p, mask=mask_aw, other=0.0)
        padded = tl.where(mask_aw, aw_row, 0.0)
        tl.store(Ah_ptr + off_ah + i * RP + cols_p, padded, mask=mask_rp)

    tl.store(Ah_ptr + off_ah + r * RP + cols_p, 0.0, mask=mask_rp)

    # Left sweep: Q_k^T mixes rows k and r.
    for k in range(r):
        c = tl.load(cs_ptr + off_vec + k)
        s = tl.load(ss_ptr + off_vec + k)

        rk = tl.load(Ah_ptr + off_ah + k * RP + cols_p, mask=mask_rp, other=0.0)
        rr = tl.load(Ah_ptr + off_ah + r * RP + cols_p, mask=mask_rp, other=0.0)

        tl.store(Ah_ptr + off_ah + k * RP + cols_p, c * rk + s * rr, mask=mask_rp)
        tl.store(Ah_ptr + off_ah + r * RP + cols_p, -s * rk + c * rr, mask=mask_rp)

    # Right sweep: SLOW PATH, see KNOWN ISSUE above. The intended plan is
    # to transpose Ah into a (RP x RP) scratch buffer, run the sweep over
    # rows of the transpose (which are columns of the original), and
    # transpose back. That requires a second SRAM-sized scratch region.
    # The version below is the fallback that works but wastes memory traffic.
    for k in range(r):
        c = tl.load(cs_ptr + off_vec + k)
        s = tl.load(ss_ptr + off_vec + k)

        for i in range(RP):
            row = tl.load(Ah_ptr + off_ah + i * RP + cols_p, mask=mask_rp, other=0.0)

            val_k = tl.sum(tl.where(cols_p == k, row, 0.0))
            val_r = tl.sum(tl.where(cols_p == r, row, 0.0))

            new_k = c * val_k + s * val_r
            new_r = -s * val_k + c * val_r

            row = tl.where(cols_p == k, new_k, row)
            row = tl.where(cols_p == r, new_r, row)
            tl.store(Ah_ptr + off_ah + i * RP + cols_p, row, mask=mask_rp)

    # Phase 3: recover v_z.
    e_last = 1.0
    for k in range(r):
        c = tl.load(cs_ptr + off_vec + k)
        s = tl.load(ss_ptr + off_vec + k)
        tl.store(vz_ptr + off_vec + k, s * e_last)
        e_last = c * e_last

    # Phase 4: forward substitution with f64 accumulation. Overwrites w_ptr
    # with v_w.
    for k in range(r):
        wk = tl.load(w_ptr + off_vec + k).to(tl.float64)
        L_row = tl.load(L_ptr + off_mat + k * r + cols, mask=cols < r, other=0.0)
        vw_prev = tl.load(w_ptr + off_vec + cols, mask=cols < r, other=0.0)

        contrib = tl.where(cols < k, L_row.to(tl.float64) * vw_prev.to(tl.float64), 0.0)
        acc = tl.sum(contrib, axis=0)

        Lkk = tl.sum(tl.where(cols == k, L_row, 0.0))
        vw_k = ((wk - acc) / Lkk.to(tl.float64)).to(L_row.dtype)
        tl.store(w_ptr + off_vec + k, vw_k)

    # Rank-1 correction: Ah[:r, :r] += v_z v_w^T.
    for i in range(r):
        vzi = tl.load(vz_ptr + off_vec + i)
        vw_vec = tl.load(w_ptr + off_vec + cols_p, mask=cols_p < r, other=0.0)
        cur = tl.load(Ah_ptr + off_ah + i * RP + cols_p, mask=cols_p < r, other=0.0)
        tl.store(Ah_ptr + off_ah + i * RP + cols_p, cur + vzi * vw_vec, mask=cols_p < r)


@triton.jit
def _cholesky_update_kernel(
    L_ptr, z_ptr, vz_ptr, cs_ptr, ss_ptr,
    r: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Rank-1 Cholesky update plus v_z recovery.

    This phase remains single-block because each rotation depends on the
    updated padded column produced by all previous rotations.
    """
    idx = tl.arange(0, BLOCK)
    mask_r = idx < r

    for k in range(r):
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

    e_last = 1.0
    for k in range(r):
        c = tl.load(cs_ptr + k)
        s = tl.load(ss_ptr + k)
        tl.store(vz_ptr + k, s * e_last)
        e_last = c * e_last


@triton.jit
def _left_sweep_cols_kernel(
    Aw_ptr, Aw_mid_ptr, cs_ptr, ss_ptr,
    r: tl.constexpr,
    TILE_COLS: tl.constexpr,
):
    """Apply Q_k^T row rotations over independent column tiles."""
    pid_col = tl.program_id(0)
    cols = pid_col * TILE_COLS + tl.arange(0, TILE_COLS)
    mask = cols < r
    pad_row = tl.zeros((TILE_COLS,), dtype=tl.float32)

    for k in range(r):
        c = tl.load(cs_ptr + k)
        s = tl.load(ss_ptr + k)

        rk = tl.load(Aw_ptr + k * r + cols, mask=mask, other=0.0)
        rr = pad_row

        tl.store(Aw_mid_ptr + k * r + cols, c * rk + s * rr, mask=mask)
        pad_row = -s * rk + c * rr


@triton.jit
def _right_sweep_rows_kernel(
    Aw_mid_ptr, Aw_out_ptr, cs_ptr, ss_ptr,
    r: tl.constexpr,
    TILE_ROWS: tl.constexpr,
):
    """Apply Q_k column rotations over independent row tiles."""
    pid_row = tl.program_id(0)
    rows = pid_row * TILE_ROWS + tl.arange(0, TILE_ROWS)
    mask = rows < r
    pad_col = tl.zeros((TILE_ROWS,), dtype=tl.float32)

    for k in range(r):
        c = tl.load(cs_ptr + k)
        s = tl.load(ss_ptr + k)

        ck = tl.load(Aw_mid_ptr + rows * r + k, mask=mask, other=0.0)
        cr = pad_col

        tl.store(Aw_out_ptr + rows * r + k, c * ck + s * cr, mask=mask)
        pad_col = -s * ck + c * cr


@triton.jit
def _solve_vw_kernel(
    L_ptr, w_ptr,
    r: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Forward substitution for v_w = L^{-1} w."""
    idx = tl.arange(0, BLOCK)
    mask_r = idx < r

    for k in range(r):
        wk = tl.load(w_ptr + k).to(tl.float32)
        L_row = tl.load(L_ptr + k * r + idx, mask=mask_r, other=0.0)
        vw_prev = tl.load(w_ptr + idx, mask=mask_r, other=0.0)

        contrib = tl.where(idx < k, L_row.to(tl.float32) * vw_prev.to(tl.float32), 0.0)
        acc = tl.sum(contrib, axis=0)

        Lkk = tl.sum(tl.where(idx == k, L_row, 0.0))
        vw_k = ((wk - acc) / Lkk.to(tl.float32)).to(L_row.dtype)
        tl.store(w_ptr + k, vw_k)


@triton.jit
def _rank1_add_kernel(
    Aw_out_ptr, vz_ptr, vw_ptr,
    r: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Add v_z v_w^T to Aw_out over 2D tiles."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < r) & (cols[None, :] < r)

    cur = tl.load(Aw_out_ptr + rows[:, None] * r + cols[None, :], mask=mask, other=0.0)
    vz = tl.load(vz_ptr + rows, mask=rows < r, other=0.0)
    vw = tl.load(vw_ptr + cols, mask=cols < r, other=0.0)

    tl.store(Aw_out_ptr + rows[:, None] * r + cols[None, :],
             cur + vz[:, None] * vw[None, :],
             mask=mask)


def incremental_update_triton(L, Aw, z, w, num_warps=None):
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

    launch_kwargs = {"r": r, "RP": rp, "BLOCK": BLOCK}
    if num_warps is not None:
        launch_kwargs["num_warps"] = num_warps

    _fused_doublesided_kernel[(1,)](
        L, Aw, z_work, w_work, vz, Ah, cs, ss,
        **launch_kwargs,
    )

    return Ah[:r, :r].contiguous(), vz


def incremental_update_triton_tiled(
    L,
    Aw,
    z,
    w,
    tile_rows=64,
    tile_cols=64,
    block_m=16,
    block_n=64,
    num_warps=None,
):
    """Multi-block Triton update.

    This keeps the sequential rank-1 Cholesky update and triangular solve as
    single-block kernels, but parallelizes the double-sided propagation and
    final rank-1 correction over row/column tiles.

    Returns:
        Aw_new: (r, r) updated whitened operator.
        vz: (r,) the vector (L+)^{-1} z.
    """
    r = L.shape[0]

    BLOCK = triton.next_power_of_2(r)
    assert BLOCK <= 1024, f"r={r} too large for single-block setup kernels"
    assert triton.next_power_of_2(tile_rows) == tile_rows, "tile_rows must be a power of two"
    assert triton.next_power_of_2(tile_cols) == tile_cols, "tile_cols must be a power of two"
    assert triton.next_power_of_2(block_m) == block_m, "block_m must be a power of two"
    assert triton.next_power_of_2(block_n) == block_n, "block_n must be a power of two"

    z_work = z.clone()
    vz = torch.empty(r, dtype=L.dtype, device=L.device)
    cs = torch.empty(r, dtype=L.dtype, device=L.device)
    ss = torch.empty(r, dtype=L.dtype, device=L.device)
    Aw_mid = torch.empty_like(Aw)
    Aw_out = torch.empty_like(Aw)
    w_work = w.clone()

    setup_kwargs = {"r": r, "BLOCK": BLOCK}
    sweep_kwargs = {}
    if num_warps is not None:
        setup_kwargs["num_warps"] = num_warps
        sweep_kwargs["num_warps"] = num_warps

    _cholesky_update_kernel[(1,)](
        L, z_work, vz, cs, ss,
        **setup_kwargs,
    )

    col_grid = (triton.cdiv(r, tile_cols),)
    _left_sweep_cols_kernel[col_grid](
        Aw, Aw_mid, cs, ss,
        r=r, TILE_COLS=tile_cols,
        **sweep_kwargs,
    )

    row_grid = (triton.cdiv(r, tile_rows),)
    _right_sweep_rows_kernel[row_grid](
        Aw_mid, Aw_out, cs, ss,
        r=r, TILE_ROWS=tile_rows,
        **sweep_kwargs,
    )

    _solve_vw_kernel[(1,)](
        L, w_work,
        **setup_kwargs,
    )

    rank1_grid = (triton.cdiv(r, block_m), triton.cdiv(r, block_n))
    _rank1_add_kernel[rank1_grid](
        Aw_out, vz, w_work,
        r=r, BLOCK_M=block_m, BLOCK_N=block_n,
        **sweep_kwargs,
    )

    return Aw_out, vz


def incremental_update_triton_batched(L, Aw, z, w):
    """Batched Triton update. Inputs are (T, r, r) / (T, r). L modified in-place.

    Launches one GPU program instance per update so all T updates run in
    parallel, versus T sequential kernel launches with incremental_update_triton.
    Requires r <= 1023 (same constraint as the single path).

    Returns:
        Aw_new: (T, r, r) updated whitened operators.
        vz:     (T, r)    vectors (L+_t)^{-1} z_t for each t.
    """
    T, r = L.shape[0], L.shape[1]
    rp = r + 1

    BLOCK = triton.next_power_of_2(rp)
    assert BLOCK <= 1024, f"r={r} too large for single-block kernel"

    vz     = torch.empty(T, r,     dtype=L.dtype, device=L.device)
    Ah     = torch.empty(T, rp, rp, dtype=L.dtype, device=L.device)
    cs     = torch.empty(T, r,     dtype=L.dtype, device=L.device)
    ss     = torch.empty(T, r,     dtype=L.dtype, device=L.device)
    z_work = z.clone()
    w_work = w.clone()

    _fused_doublesided_kernel[(T,)](
        L, Aw, z_work, w_work, vz, Ah, cs, ss,
        r=r, RP=rp, BLOCK=BLOCK,
    )

    return Ah[:, :r, :r].contiguous(), vz
