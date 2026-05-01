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
     kernel, dispatched with grid=(1,) or grid=(T,). Pad row/column for the
     double-sided propagation stay in registers, so only the returned r x r
     block is materialized. Gated at r <= 1024 so r fits in a single
     power-of-two BLOCK up to 1024.

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
# Phase 2's right sweep updates column vectors directly. That avoids the
# old row-patching O(r^2 * BLOCK) memory traffic without paying for full
# global-memory transposes through a second scratch matrix. The padded
# row and column stay in Triton registers across the rotation loops, so
# only the returned r x r top-left block is materialized in Aw_out.

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
    vz_ptr, Aw_out_ptr, cs_ptr, ss_ptr,
    r: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """SRAM-resident fused kernel. BLOCK must be a power of two >= r.

    Combines the register-virtualized double-sided propagation (no Ah
    scratch buffer; pad row/col live in registers; only the r x r block
    is written) with batching. Launch with grid=(T,) and pass (T,...)
    tensors for batched execution; grid=(1,) gives the single-update
    path (pid=0, offsets degenerate to 0).
    """
    pid     = tl.program_id(0)
    off_mat = pid * r * r      # stride for (r, r) tensors: L, Aw, Aw_out
    off_vec = pid * r          # stride for (r,) tensors: z, w, vz, cs, ss

    # Phase 1: Rank-1 Cholesky update.
    idx = tl.arange(0, BLOCK)

    for k in range(r):
        mask_r = idx < r
        L_col = tl.load(L_ptr + off_mat + idx * r + k, mask=mask_r, other=0.0)
        z_vec = tl.load(z_ptr + off_vec + idx, mask=mask_r, other=0.0)

        Lkk = tl.sum(tl.where(idx == k, L_col, 0.0))
        zk  = tl.sum(tl.where(idx == k, z_vec, 0.0))
        c, s = _givens_params_tri(Lkk, zk)

        tl.store(cs_ptr + off_vec + k, c)
        tl.store(ss_ptr + off_vec + k, s)

        active = mask_r & (idx >= k)
        new_L = c * L_col + s * z_vec
        new_z = -s * L_col + c * z_vec
        tl.store(L_ptr + off_mat + idx * r + k, new_L, mask=active)
        tl.store(z_ptr + off_vec + idx, tl.where(active, new_z, z_vec), mask=mask_r)

    # cs/ss writes from Phase 1 must be visible to all warps before Phase 2
    # reads them. Without this, multi-warp blocks see stale values.
    tl.debug_barrier()

    # Phase 2: Double-sided propagation. The padded row and column start as
    # zeros; only the top-left r x r block is materialized in Aw_out.
    cols_p = tl.arange(0, BLOCK)
    mask_r = cols_p < r
    pad_row = tl.zeros((BLOCK,), dtype=tl.float32)

    # Left sweep: Q_k^T mixes row k with the virtual padded row.
    for k in range(r):
        c = tl.load(cs_ptr + off_vec + k)
        s = tl.load(ss_ptr + off_vec + k)

        rk = tl.load(Aw_ptr + off_mat + k * r + cols_p, mask=mask_r, other=0.0)
        rr = pad_row

        tl.store(Aw_out_ptr + off_mat + k * r + cols_p, c * rk + s * rr, mask=mask_r)
        pad_row = -s * rk + c * rr

    # Left sweep wrote Aw_out by row; right sweep reads Aw_out by column.
    # Different warps own different lanes, so we need a barrier here so
    # column-k reads see all the row-i writes from the left sweep.
    tl.debug_barrier()

    # Right sweep: Q_k mixes columns k and r. Vectorize over rows so each
    # column is loaded and stored once per rotation. The padded column stays
    # in registers and is discarded because only the top-left block is needed.
    rows = tl.arange(0, BLOCK)
    row_mask = rows < r
    pad_col = tl.zeros((BLOCK,), dtype=tl.float32)

    for k in range(r):
        c = tl.load(cs_ptr + off_vec + k)
        s = tl.load(ss_ptr + off_vec + k)

        ck = tl.load(Aw_out_ptr + off_mat + rows * r + k, mask=row_mask, other=0.0)
        cr = pad_col

        tl.store(Aw_out_ptr + off_mat + rows * r + k, c * ck + s * cr, mask=row_mask)
        pad_col = -s * ck + c * cr

    # Phase 3: recover v_z.
    e_last = 1.0
    for k in range(r):
        c = tl.load(cs_ptr + off_vec + k)
        s = tl.load(ss_ptr + off_vec + k)
        tl.store(vz_ptr + off_vec + k, s * e_last)
        e_last = c * e_last

    # Phase 4: forward substitution with f32 accumulation. Overwrites w_ptr
    # with v_w.
    for k in range(r):
        wk = tl.load(w_ptr + off_vec + k).to(tl.float32)
        L_row = tl.load(L_ptr + off_mat + k * r + idx, mask=idx < r, other=0.0)
        vw_prev = tl.load(w_ptr + off_vec + idx, mask=idx < r, other=0.0)

        contrib = tl.where(idx < k, L_row.to(tl.float32) * vw_prev.to(tl.float32), 0.0)
        acc = tl.sum(contrib, axis=0)

        Lkk = tl.sum(tl.where(idx == k, L_row, 0.0))
        vw_k = ((wk - acc) / Lkk.to(tl.float32)).to(L_row.dtype)
        tl.store(w_ptr + off_vec + k, vw_k)

    # Make right-sweep writes to Aw_out, Phase 3 writes to vz, and Phase 4
    # writes to w (now v_w) visible to all warps before the rank-1 correction
    # reads them as vectors.
    tl.debug_barrier()

    # Rank-1 correction: Aw_out += v_z v_w^T.
    for i in range(r):
        vzi = tl.load(vz_ptr + off_vec + i)
        vw_vec = tl.load(w_ptr + off_vec + cols_p, mask=cols_p < r, other=0.0)
        cur = tl.load(Aw_out_ptr + off_mat + i * r + cols_p, mask=cols_p < r, other=0.0)
        tl.store(Aw_out_ptr + off_mat + i * r + cols_p, cur + vzi * vw_vec, mask=cols_p < r)


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

    Requires r <= 1024 so r fits in a single BLOCK of at most 1024.
    For larger r, use the C++ extension or tile the kernel.

    Returns:
        Aw_new: (r, r) updated whitened operator.
        vz: (r,) the vector (L+)^{-1} z.
    """
    r = L.shape[0]

    BLOCK = triton.next_power_of_2(r)
    assert BLOCK <= 1024, f"r={r} too large for single-block kernel"

    vz = torch.empty(r, dtype=L.dtype, device=L.device)
    Aw_out = torch.empty_like(Aw)
    cs = torch.empty(r, dtype=L.dtype, device=L.device)
    ss = torch.empty(r, dtype=L.dtype, device=L.device)

    z_work = z.clone()
    w_work = w.clone()

    launch_kwargs = {"r": r, "BLOCK": BLOCK}
    if num_warps is not None:
        launch_kwargs["num_warps"] = num_warps

    _fused_doublesided_kernel[(1,)](
        L, Aw, z_work, w_work, vz, Aw_out, cs, ss,
        **launch_kwargs,
    )

    return Aw_out, vz


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


def incremental_update_triton_batched(L, Aw, z, w, num_warps=None):
    """Batched Triton update. Inputs are (T, r, r) / (T, r). L modified in-place.

    Launches one GPU program instance per update so all T updates run in
    parallel, versus T sequential kernel launches with incremental_update_triton.
    Requires r <= 1024 (same constraint as the single path).

    Returns:
        Aw_new: (T, r, r) updated whitened operators.
        vz:     (T, r)    vectors (L+_t)^{-1} z_t for each t.
    """
    T, r = L.shape[0], L.shape[1]

    BLOCK = triton.next_power_of_2(r)
    assert BLOCK <= 1024, f"r={r} too large for single-block kernel"

    vz     = torch.empty(T, r,    dtype=L.dtype, device=L.device)
    Aw_out = torch.empty(T, r, r, dtype=L.dtype, device=L.device)
    cs     = torch.empty(T, r,    dtype=L.dtype, device=L.device)
    ss     = torch.empty(T, r,    dtype=L.dtype, device=L.device)
    z_work = z.clone()
    w_work = w.clone()

    launch_kwargs = {"r": r, "BLOCK": BLOCK}
    if num_warps is not None:
        launch_kwargs["num_warps"] = num_warps

    _fused_doublesided_kernel[(T,)](
        L, Aw, z_work, w_work, vz, Aw_out, cs, ss,
        **launch_kwargs,
    )

    return Aw_out, vz


# ===========================================================================
# CHUNK-FUSED KERNEL
#
# Processes C sequential per-token rank-1 updates inside ONE GPU program per
# (b,h) slot, instead of issuing C kernel launches from Python. Replaces the
# per-token `for c in range(C)` loop in
# experiments/backend/ska_recurrence.py:_chunk_forward.
#
# Per token, runs the same five phases as _fused_doublesided_kernel:
#   Phase 1: Givens rank-1 Cholesky update of L (L_curr -> L_new).
#   Phase 2: Double-sided propagation of A_w (left + right sweep).
#   Phase 3: Recover v_z = (L_new)^{-1} z from the saved Givens (c, s).
#   Phase 4: Forward solve v_w = L_new^{-1} w.
#   Phase 7: Rank-1 correction A_w += v_z v_w^T.
#
# Two outputs that the per-token kernel computed via separate launches outside
# the kernel are deliberately moved to a single batched PyTorch call AFTER
# this kernel finishes (so the kernel itself is simpler):
#   k_tilde_t = (L_new_t)^{-T} v_z_t      (one batched back-solve)
#   T_t       = L_new_t^{-1} L_curr_t     (one batched forward-solve)
#
# Memory layout for outputs (matches what
# experiments/backend/ska_recurrence._chunk_forward writes):
#   L_seq:  (BH, C+1, r, r)    L_seq[:, 0] = initial L; the kernel fills 1..C.
#   Aw_seq: (BH, C+1, r, r)    Aw_seq[:, 0] = initial Aw; kernel fills 1..C.
#   vz_seq: (BH, C, r)
#   vw_seq: (BH, C, r)
# ===========================================================================


@triton.jit
def _fused_chunk_kernel(
    L_seq_ptr,       # (BH, C+1, r, r) - slot 0 holds initial L; kernel fills 1..C
    Aw_seq_ptr,      # (BH, C+1, r, r) - slot 0 holds initial Aw; kernel fills 1..C
    K_ptr,           # (BH, C, r) per-token keys
    V_ptr,           # (BH, C, r) per-token values
    vz_seq_ptr,      # (BH, C, r) per-token v_z
    vw_seq_ptr,      # (BH, C, r) per-token v_w
    cs_ptr, ss_ptr,  # (BH, r)    per-program scratch (overwritten each c)
    r: tl.constexpr,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One GPU program per (b,h); processes all C tokens sequentially in SRAM.

    BLOCK must be a power of two with BLOCK >= r and BLOCK <= 1024.
    """
    pid = tl.program_id(0)

    # Per-program base offsets.
    seq_mat_stride = (C + 1) * r * r          # L_seq, Aw_seq stride between BH slots
    seq_vec_stride = C * r                    # vz_seq, vw_seq, K, V stride
    L_seq_base  = L_seq_ptr  + pid * seq_mat_stride
    Aw_seq_base = Aw_seq_ptr + pid * seq_mat_stride
    K_base      = K_ptr      + pid * seq_vec_stride
    V_base      = V_ptr      + pid * seq_vec_stride
    vz_base     = vz_seq_ptr + pid * seq_vec_stride
    vw_base     = vw_seq_ptr + pid * seq_vec_stride
    cs_base     = cs_ptr     + pid * r
    ss_base     = ss_ptr     + pid * r

    idx = tl.arange(0, BLOCK)
    mask_r = idx < r

    # Outer C-token loop. We use plain `range` so r and C are constexpr to
    # the compiler; the inner phase loops also unroll. For r=128, C=64 this
    # is a large IR but well within Triton's compile budget on Triton >= 3.0.
    for c in range(C):
        L_curr_ptr  = L_seq_base  + c * r * r
        L_new_ptr   = L_seq_base  + (c + 1) * r * r
        Aw_curr_ptr = Aw_seq_base + c * r * r
        Aw_new_ptr  = Aw_seq_base + (c + 1) * r * r
        z_in_ptr    = K_base + c * r
        w_in_ptr    = V_base + c * r
        vz_out_ptr  = vz_base + c * r
        vw_out_ptr  = vw_base + c * r

        # ---------------- Phase 1: Givens rank-1 Cholesky update ---------------
        # Read column k from L_curr, write column k to L_new (zeros above
        # diagonal preserved). Track z in registers across the k loop.
        z = tl.load(z_in_ptr + idx, mask=mask_r, other=0.0)
        for k in range(r):
            L_col = tl.load(L_curr_ptr + idx * r + k, mask=mask_r, other=0.0)
            Lkk = tl.sum(tl.where(idx == k, L_col, 0.0))
            zk  = tl.sum(tl.where(idx == k, z,     0.0))
            cval, sval = _givens_params_tri(Lkk, zk)
            tl.store(cs_base + k, cval)
            tl.store(ss_base + k, sval)

            active = mask_r & (idx >= k)
            new_L = cval * L_col + sval * z
            # Lower-triangular structure: idx<k stays 0, idx>=k gets rotated.
            tl.store(L_new_ptr + idx * r + k,
                     tl.where(active, new_L, 0.0),
                     mask=mask_r)

            new_z = -sval * L_col + cval * z
            z = tl.where(active, new_z, z)
        tl.debug_barrier()

        # ---------------- Phase 2 left sweep: Aw_curr -> Aw_new ----------------
        cols_p = tl.arange(0, BLOCK)
        mask_cols = cols_p < r
        pad_row = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in range(r):
            cval = tl.load(cs_base + k)
            sval = tl.load(ss_base + k)
            rk = tl.load(Aw_curr_ptr + k * r + cols_p, mask=mask_cols, other=0.0)
            rr = pad_row
            tl.store(Aw_new_ptr + k * r + cols_p, cval * rk + sval * rr, mask=mask_cols)
            pad_row = -sval * rk + cval * rr
        tl.debug_barrier()

        # ---------------- Phase 2 right sweep: Aw_new in-place ----------------
        rows_p = tl.arange(0, BLOCK)
        mask_rows = rows_p < r
        pad_col = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in range(r):
            cval = tl.load(cs_base + k)
            sval = tl.load(ss_base + k)
            ck = tl.load(Aw_new_ptr + rows_p * r + k, mask=mask_rows, other=0.0)
            cr = pad_col
            tl.store(Aw_new_ptr + rows_p * r + k, cval * ck + sval * cr, mask=mask_rows)
            pad_col = -sval * ck + cval * cr

        # ---------------- Phase 3: recover v_z via Givens ----------------
        e_last = 1.0
        for k in range(r):
            cval = tl.load(cs_base + k)
            sval = tl.load(ss_base + k)
            tl.store(vz_out_ptr + k, sval * e_last)
            e_last = cval * e_last

        # ---------------- Phase 4: forward solve v_w = L_new^{-1} w ----------------
        # Writes vw_out_ptr incrementally; iteration k uses vw_out_ptr at
        # indices < k (already-written) and writes index k.
        for k in range(r):
            wk = tl.load(w_in_ptr + k).to(tl.float32)
            L_row = tl.load(L_new_ptr + k * r + idx, mask=mask_r, other=0.0)
            vw_prev = tl.load(vw_out_ptr + idx, mask=mask_r, other=0.0)

            contrib = tl.where(idx < k,
                               L_row.to(tl.float32) * vw_prev.to(tl.float32),
                               0.0)
            acc = tl.sum(contrib, axis=0)

            Lkk = tl.sum(tl.where(idx == k, L_row, 0.0))
            vw_k = ((wk - acc) / Lkk.to(tl.float32)).to(L_row.dtype)
            tl.store(vw_out_ptr + k, vw_k)
        tl.debug_barrier()

        # ---------------- Phase 7: Aw_new += v_z v_w^T ----------------
        for i in range(r):
            vzi = tl.load(vz_out_ptr + i)
            vw_vec = tl.load(vw_out_ptr + cols_p, mask=mask_cols, other=0.0)
            cur = tl.load(Aw_new_ptr + i * r + cols_p, mask=mask_cols, other=0.0)
            tl.store(Aw_new_ptr + i * r + cols_p,
                     cur + vzi * vw_vec,
                     mask=mask_cols)


def incremental_chunk_update_triton(L, Aw, K, V, num_warps=None):
    """Single-launch C-token SKA chunk update.

    Replaces the per-token Python loop in ska_recurrence._chunk_forward.

    Args:
        L:  (BH, r, r) initial Cholesky factor at chunk start.
        Aw: (BH, r, r) initial whitened operator at chunk start.
        K:  (BH, C, r) keys for the chunk.
        V:  (BH, C, r) values for the chunk.

    Returns:
        L_seq:  (BH, C+1, r, r)  L_t for t=0..C
        Aw_seq: (BH, C+1, r, r)  Aw_t for t=0..C
        vz_seq: (BH, C, r)
        vw_seq: (BH, C, r)
        kt_seq: (BH, C, r)       k_tilde_t = L_new_t^{-T} v_z_t
        T_seq:  (BH, C, r, r)    T_t = L_new_t^{-1} L_curr_t

    Constraint: r <= 128 (so 2 * r*r * 4 bytes = 128KB stays in H200 SRAM
    headroom). Larger r should fall back to per-token kernels.
    """
    BH, r1, r2 = L.shape
    assert r1 == r2, f"L must be square, got {L.shape}"
    BH2, C, r3 = K.shape
    assert BH == BH2 and r1 == r3, (
        f"shape mismatch: L {L.shape} K {K.shape}"
    )
    r = r1
    assert r <= 128, f"chunk kernel requires r<=128 for SRAM budget, got r={r}"

    BLOCK = triton.next_power_of_2(r)
    assert BLOCK <= 1024

    device, dtype = L.device, L.dtype

    # Allocate stacked saves. Use zeros for L_seq so upper-triangular slots
    # stay 0 (the kernel only writes the lower-triangle entries of L_new).
    L_seq  = torch.zeros(BH, C + 1, r, r, device=device, dtype=dtype)
    Aw_seq = torch.empty(BH, C + 1, r, r, device=device, dtype=dtype)
    vz_seq = torch.empty(BH, C, r, device=device, dtype=dtype)
    vw_seq = torch.empty(BH, C, r, device=device, dtype=dtype)

    # Seed slot 0 with the initial state. The kernel will fill slots 1..C.
    L_seq[:, 0]  = L
    Aw_seq[:, 0] = Aw

    cs = torch.empty(BH, r, device=device, dtype=dtype)
    ss = torch.empty(BH, r, device=device, dtype=dtype)

    launch_kwargs = {"r": r, "C": C, "BLOCK": BLOCK}
    if num_warps is not None:
        launch_kwargs["num_warps"] = num_warps

    _fused_chunk_kernel[(BH,)](
        L_seq, Aw_seq, K, V, vz_seq, vw_seq, cs, ss,
        **launch_kwargs,
    )

    # Post-kernel batched solves (each is one PyTorch launch, batched
    # over BH and C). These are the two ops that ska_recurrence used to
    # call once per token, contributing 2C launches per chunk.

    # k_tilde_t = (L_new_t)^{-T} v_z_t
    # L_new_t lives at L_seq[:, 1:]; transpose to upper-triangular.
    L_new_T = L_seq[:, 1:].transpose(-2, -1).contiguous()  # (BH, C, r, r)
    kt_seq = torch.linalg.solve_triangular(
        L_new_T, vz_seq.unsqueeze(-1), upper=True
    ).squeeze(-1)

    # T_t = L_new_t^{-1} L_curr_t  (each token: r columns of forward solve)
    T_seq = torch.linalg.solve_triangular(
        L_seq[:, 1:].contiguous(),
        L_seq[:, :-1].contiguous(),
        upper=False,
    )

    return L_seq, Aw_seq, vz_seq, vw_seq, kt_seq, T_seq


# ===========================================================================
# CHUNK-FUSED BACKWARD KERNEL
#
# One Triton program per (b,h) slot. Iterates the C tokens in REVERSE,
# running the 7-step manual backward (mirroring _per_token_backward_step
# in experiments/backend/ska_recurrence.py) entirely in one kernel launch
# per chunk. Eliminates ~7C launches per chunk in the existing per-token
# Python loop.
#
# Math contract: each iteration consumes saved (L, L_new, Aw, T, vz, vw,
# kt, z, w) and upstream (grad_L_curr, grad_Aw_curr, grad_kt[c]) and
# produces (grad_L_prev, grad_Aw_prev, grad_z, grad_w). After C iterations
# (c = C-1 down to 0), the rolling (grad_L_curr, grad_Aw_curr) state holds
# the final (grad_L_init, grad_Aw_init).
#
# Implementation strategy:
# - All r×r matrices live in pre-allocated global scratch buffers (one
#   slice per program). Triton pipelines reads/writes; bandwidth is not
#   the bottleneck.
# - Triangular solves use the row-by-row sequential pattern:
#     for k in range(r):
#       acc = sum_{i<k} L[k,i] * X[i, :]      (masked sum over loaded buffer)
#       X[k, :] = (B[k, :] - acc) / L[k, k]
#       store X[k, :] back to memory
# - Matrix-matrix matmuls use tl.dot on (BLOCK, BLOCK) tiles.
# - Matrix-vector products use the same row-loop pattern.
#
# Numerical correctness MUST be verified against the eager per-token loop
# via _check_chunk_matches_per_token and _gradcheck_chunk before enabling
# this kernel in production. Default off; opt in via SKA_USE_TRITON_CHUNK_BWD=1.
# ===========================================================================


@triton.jit
def _fused_chunk_backward_kernel(
    # ----- Saved-from-forward inputs -----
    L_seq_ptr,         # (BH, C+1, r, r)
    Aw_seq_ptr,        # (BH, C+1, r, r)
    K_ptr,             # (BH, C, r)         — original z's
    V_ptr,             # (BH, C, r)         — original w's
    vz_seq_ptr,        # (BH, C, r)
    vw_seq_ptr,        # (BH, C, r)
    kt_seq_ptr,        # (BH, C, r)
    T_seq_ptr,         # (BH, C, r, r)
    grad_kt_ptr,       # (BH, C, r)         — upstream grad_K_tilde
    # ----- Rolling state buffers (in/out) -----
    grad_L_curr_ptr,   # (BH, r, r)         — init = grad_L_new from upstream
    grad_Aw_curr_ptr,  # (BH, r, r)         — init = grad_Aw_new from upstream
    # ----- Per-program scratch (reused each iteration) -----
    scratch_X_ptr,     # (BH, r, r)         — primary solve output / matmul scratch
    scratch_Y_ptr,     # (BH, r, r)         — secondary scratch (Y in step 6)
    scratch_M_ptr,     # (BH, r, r)         — grad_T, M_aux, grad_G_new, ...
    scratch_LfT_ptr,   # (BH, r, r)         — grad_L_from_T saved for step 7
    scratch_v1_ptr,    # (BH, r)            — grad_vz scratch
    scratch_v2_ptr,    # (BH, r)            — grad_vw scratch
    scratch_v3_ptr,    # (BH, r)            — Linv_T_grad_vz / vw / grad_z_direct
    # ----- Outputs -----
    grad_K_ptr,        # (BH, C, r)
    grad_V_ptr,        # (BH, C, r)
    r: tl.constexpr,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per (b,h). Sequential C-iteration loop in reverse."""
    pid = tl.program_id(0)

    # Strides
    seq_mat_stride = (C + 1) * r * r
    seq_vec_stride = C * r
    T_seq_stride = C * r * r
    state_stride = r * r

    # Per-program base pointers
    L_seq_base   = L_seq_ptr   + pid * seq_mat_stride
    Aw_seq_base  = Aw_seq_ptr  + pid * seq_mat_stride
    K_base       = K_ptr       + pid * seq_vec_stride
    V_base       = V_ptr       + pid * seq_vec_stride
    vz_base      = vz_seq_ptr  + pid * seq_vec_stride
    vw_base      = vw_seq_ptr  + pid * seq_vec_stride
    kt_base      = kt_seq_ptr  + pid * seq_vec_stride
    T_base       = T_seq_ptr   + pid * T_seq_stride
    grad_kt_base = grad_kt_ptr + pid * seq_vec_stride
    gL_curr_base = grad_L_curr_ptr  + pid * state_stride
    gA_curr_base = grad_Aw_curr_ptr + pid * state_stride
    sX_base      = scratch_X_ptr    + pid * state_stride
    sY_base      = scratch_Y_ptr    + pid * state_stride
    sM_base      = scratch_M_ptr    + pid * state_stride
    sLfT_base    = scratch_LfT_ptr  + pid * state_stride
    sv1_base     = scratch_v1_ptr   + pid * r
    sv2_base     = scratch_v2_ptr   + pid * r
    sv3_base     = scratch_v3_ptr   + pid * r
    gK_base      = grad_K_ptr       + pid * seq_vec_stride
    gV_base      = grad_V_ptr       + pid * seq_vec_stride

    idx = tl.arange(0, BLOCK)
    mask_r = idx < r
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    mask2d = (rows[:, None] < r) & (cols[None, :] < r)

    # ===== Reverse C-iteration loop =====
    for c_inv in range(C):
        c = C - 1 - c_inv

        # Per-token pointer offsets
        L_ptr_c     = L_seq_base   + c * r * r            # L      (= L_seq[c])
        L_new_ptr_c = L_seq_base   + (c + 1) * r * r      # L_new  (= L_seq[c+1])
        Aw_ptr_c    = Aw_seq_base  + c * r * r
        T_ptr_c     = T_base       + c * r * r
        z_ptr_c     = K_base       + c * r
        w_ptr_c     = V_base       + c * r
        vz_ptr_c    = vz_base      + c * r
        vw_ptr_c    = vw_base      + c * r
        kt_ptr_c    = kt_base      + c * r
        gkt_ptr_c   = grad_kt_base + c * r
        gK_ptr_c    = gK_base      + c * r
        gV_ptr_c    = gV_base      + c * r

        # ===================================================================
        # STEP 1: k_tilde = L_new^{-T} v_z
        #   grad_vz = solve(L_new, grad_kt)                  [forward, vector]
        #   outer1 = kt ⊗ grad_kt
        #   update1 = solve(L_new, outer1.T).T               [forward, matrix]
        #   grad_L_curr -= update1
        # ===================================================================

        # Phase 1a: vector forward solve grad_vz = L_new^{-1} grad_kt
        # Stored to scratch_v1
        for k in range(r):
            gkt_k = tl.load(gkt_ptr_c + k).to(tl.float32)
            L_row = tl.load(L_new_ptr_c + k * r + idx, mask=mask_r, other=0.0)
            v1_prev = tl.load(sv1_base + idx, mask=mask_r, other=0.0)
            contrib = tl.where(
                idx < k,
                L_row.to(tl.float32) * v1_prev.to(tl.float32),
                0.0,
            )
            acc = tl.sum(contrib, axis=0)
            Lkk = tl.sum(tl.where(idx == k, L_row, 0.0))
            v1_k = ((gkt_k - acc) / Lkk.to(tl.float32)).to(L_row.dtype)
            tl.store(sv1_base + k, v1_k)
        tl.debug_barrier()

        # Phase 1b: matrix forward solve X = solve(L_new, outer1.T)
        # outer1.T[k, :] = grad_kt[k] * kt
        # Then update1 = X.T, and grad_L_curr -= X.T
        # Compute X into scratch_X.
        for k in range(r):
            gkt_k_scal = tl.load(gkt_ptr_c + k).to(tl.float32)
            kt_vec = tl.load(kt_ptr_c + idx, mask=mask_r, other=0.0).to(tl.float32)
            B_row = gkt_k_scal * kt_vec  # (r,) - row k of outer1.T
            # Load X (rows < k contribute, rest masked to 0).
            X_tile = tl.load(
                sX_base + rows[:, None] * r + cols[None, :],
                mask=mask2d,
                other=0.0,
            ).to(tl.float32)
            X_masked = tl.where(rows[:, None] < k, X_tile, 0.0)
            # Row k of L_new
            L_row_full = tl.load(L_new_ptr_c + k * r + rows, mask=rows < r, other=0.0).to(tl.float32)
            # correction[j] = sum_i L_new[k, i] * X[i, j]
            correction = tl.sum(L_row_full[:, None] * X_masked, axis=0)  # (r,)
            Lkk = tl.load(L_new_ptr_c + k * r + k).to(tl.float32)
            X_k_row = (B_row - correction) / Lkk
            tl.store(sX_base + k * r + idx, X_k_row.to(kt_vec.dtype), mask=mask_r)
        tl.debug_barrier()

        # grad_L_curr -= X.T  (i.e., grad_L_curr[i, j] -= X[j, i])
        for i in range(r):
            X_col_i = tl.load(sX_base + idx * r + i, mask=mask_r, other=0.0).to(tl.float32)
            gL_row_i = tl.load(gL_curr_base + i * r + idx, mask=mask_r, other=0.0).to(tl.float32)
            new_row = gL_row_i - X_col_i
            tl.store(gL_curr_base + i * r + idx, new_row.to(X_col_i.dtype), mask=mask_r)
        tl.debug_barrier()

        # ===================================================================
        # STEP 2: A_w_new = T A_w T^T + v_z v_w^T
        #   grad_T = grad_Aw_curr @ T @ Aw_T + grad_Aw_curr.T @ T @ Aw
        #   grad_Aw_prev = T.T @ grad_Aw_curr @ T
        #   grad_vz += grad_Aw_curr @ vw
        #   grad_vw = grad_Aw_curr.T @ vz
        # ===================================================================

        # Load grad_Aw_curr, T, Aw as 2D tiles.
        gA_tile = tl.load(
            gA_curr_base + rows[:, None] * r + cols[None, :],
            mask=mask2d, other=0.0,
        ).to(tl.float32)
        T_tile = tl.load(
            T_ptr_c + rows[:, None] * r + cols[None, :],
            mask=mask2d, other=0.0,
        ).to(tl.float32)
        Aw_tile = tl.load(
            Aw_ptr_c + rows[:, None] * r + cols[None, :],
            mask=mask2d, other=0.0,
        ).to(tl.float32)

        # gA @ T  → tmp1 (r, r)
        tmp1 = tl.dot(gA_tile, T_tile, allow_tf32=False)
        # tmp1 @ Aw_T = tmp1 @ Aw.T  → first half of grad_T
        # tl.dot doesn't take .T; use trans flag if avail, or load Aw transposed.
        # For simplicity, materialize Aw_T tile by reading transposed.
        Aw_T_tile = tl.load(
            Aw_ptr_c + cols[:, None] * r + rows[None, :],  # transpose access
            mask=(cols[:, None] < r) & (rows[None, :] < r),
            other=0.0,
        ).to(tl.float32)
        gradT_a = tl.dot(tmp1, Aw_T_tile, allow_tf32=False)  # (r, r)

        # gA_T @ T  →  tmp2
        gA_T_tile = tl.load(
            gA_curr_base + cols[:, None] * r + rows[None, :],  # transpose access
            mask=(cols[:, None] < r) & (rows[None, :] < r),
            other=0.0,
        ).to(tl.float32)
        tmp2 = tl.dot(gA_T_tile, T_tile, allow_tf32=False)
        gradT_b = tl.dot(tmp2, Aw_tile, allow_tf32=False)
        grad_T_tile = gradT_a + gradT_b
        tl.store(
            sM_base + rows[:, None] * r + cols[None, :],
            grad_T_tile.to(gA_tile.dtype),
            mask=mask2d,
        )

        # grad_Aw_prev = T.T @ gA @ T
        T_T_tile = tl.load(
            T_ptr_c + cols[:, None] * r + rows[None, :],  # transpose access
            mask=(cols[:, None] < r) & (rows[None, :] < r),
            other=0.0,
        ).to(tl.float32)
        tmp3 = tl.dot(T_T_tile, gA_tile, allow_tf32=False)
        gAw_prev = tl.dot(tmp3, T_tile, allow_tf32=False)
        # We will overwrite grad_Aw_curr at end of iteration; for now stash to scratch_Y
        tl.store(
            sY_base + rows[:, None] * r + cols[None, :],
            gAw_prev.to(gA_tile.dtype),
            mask=mask2d,
        )

        # grad_vz += grad_Aw_curr @ vw  (matvec)
        vw_vec = tl.load(vw_ptr_c + cols, mask=cols < r, other=0.0).to(tl.float32)
        # gA @ vw = sum over j of gA[i, j] * vw[j]; we iterate i
        for i in range(r):
            gA_row_i = tl.load(
                gA_curr_base + i * r + cols, mask=cols < r, other=0.0
            ).to(tl.float32)
            v_add = tl.sum(gA_row_i * vw_vec, axis=0)  # scalar
            v1_old = tl.load(sv1_base + i).to(tl.float32)
            tl.store(sv1_base + i, (v1_old + v_add).to(gA_row_i.dtype))

        # grad_vw = grad_Aw_curr.T @ vz  (matvec)
        vz_vec = tl.load(vz_ptr_c + cols, mask=cols < r, other=0.0).to(tl.float32)
        for i in range(r):
            # Column i of gA: gA[:, i]
            gA_col_i = tl.load(
                gA_curr_base + cols * r + i, mask=cols < r, other=0.0
            ).to(tl.float32)
            v_w_i = tl.sum(gA_col_i * vz_vec, axis=0)
            tl.store(sv2_base + i, v_w_i.to(gA_col_i.dtype))
        tl.debug_barrier()

        # ===================================================================
        # STEP 3: T = L_new^{-1} L
        #   Linv_T_grad_T = solve(L_new.T, grad_T)            [back-solve, matrix]
        #   grad_L_curr -= Linv_T_grad_T @ T.T
        #   grad_L_from_T = Linv_T_grad_T  (saved for step 7)
        # ===================================================================
        # Back solve: solve U X = B where U = L_new^T (upper triangular)
        # Iterate k from r-1 down to 0:
        #   X[k, :] = (B[k, :] - sum_{j > k} U[k, j] * X[j, :]) / U[k, k]
        # U[k, j] = L_new[j, k]  (since U = L_new^T)
        # U[k, k] = L_new[k, k]
        #
        # B = grad_T (in scratch_M). Output X to scratch_X.
        for k_rev in range(r):
            k = r - 1 - k_rev
            B_row = tl.load(sM_base + k * r + idx, mask=mask_r, other=0.0).to(tl.float32)
            # Load X (rows > k contribute)
            X_tile = tl.load(
                sX_base + rows[:, None] * r + cols[None, :],
                mask=mask2d, other=0.0,
            ).to(tl.float32)
            X_masked = tl.where(rows[:, None] > k, X_tile, 0.0)
            # U[k, :] = L_new[:, k]  (column k of L_new)
            U_row = tl.load(
                L_new_ptr_c + rows * r + k, mask=rows < r, other=0.0
            ).to(tl.float32)
            correction = tl.sum(U_row[:, None] * X_masked, axis=0)  # (r,)
            Ukk = tl.load(L_new_ptr_c + k * r + k).to(tl.float32)
            X_k_row = (B_row - correction) / Ukk
            tl.store(sX_base + k * r + idx, X_k_row.to(B_row.dtype), mask=mask_r)
        tl.debug_barrier()

        # Save grad_L_from_T = Linv_T_grad_T = scratch_X to scratch_LfT
        for i in range(r):
            X_row_i = tl.load(sX_base + i * r + idx, mask=mask_r, other=0.0)
            tl.store(sLfT_base + i * r + idx, X_row_i, mask=mask_r)
        tl.debug_barrier()

        # grad_L_curr -= Linv_T_grad_T @ T.T  (i.e., scratch_X @ T.T)
        # Compute as 2D tile op then subtract.
        X_full = tl.load(
            sX_base + rows[:, None] * r + cols[None, :],
            mask=mask2d, other=0.0,
        ).to(tl.float32)
        prod = tl.dot(X_full, T_T_tile, allow_tf32=False)  # T_T_tile from earlier
        gL_full = tl.load(
            gL_curr_base + rows[:, None] * r + cols[None, :],
            mask=mask2d, other=0.0,
        ).to(tl.float32)
        gL_new = gL_full - prod
        tl.store(
            gL_curr_base + rows[:, None] * r + cols[None, :],
            gL_new.to(X_full.dtype),
            mask=mask2d,
        )
        tl.debug_barrier()

        # ===================================================================
        # STEP 4: v_z = L_new^{-1} z
        #   Linv_T_grad_vz = solve(L_new.T, grad_vz)          [back-solve, vector]
        #   grad_L_curr -= Linv_T_grad_vz ⊗ vz
        #   grad_z_direct = Linv_T_grad_vz                    (saved for step 7)
        # ===================================================================
        # Vector back solve: read grad_vz from scratch_v1, write Linv_T_grad_vz to scratch_v3
        for k_rev in range(r):
            k = r - 1 - k_rev
            gvz_k = tl.load(sv1_base + k).to(tl.float32)
            U_col = tl.load(
                L_new_ptr_c + idx * r + k, mask=mask_r, other=0.0
            ).to(tl.float32)  # = U row k = L_new column k
            v3_prev = tl.load(sv3_base + idx, mask=mask_r, other=0.0).to(tl.float32)
            contrib = tl.where(idx > k, U_col * v3_prev, 0.0)
            acc = tl.sum(contrib, axis=0)
            Ukk = tl.load(L_new_ptr_c + k * r + k).to(tl.float32)
            v3_k = (gvz_k - acc) / Ukk
            tl.store(sv3_base + k, v3_k.to(U_col.dtype))
        tl.debug_barrier()

        # grad_L_curr[i, j] -= Linv_T_grad_vz[i] * vz[j]
        for i in range(r):
            v3_i = tl.load(sv3_base + i).to(tl.float32)
            row_old = tl.load(gL_curr_base + i * r + idx, mask=mask_r, other=0.0).to(tl.float32)
            # subtract v3_i * vz[j]
            new_row = row_old - v3_i * vz_vec
            tl.store(gL_curr_base + i * r + idx, new_row.to(row_old.dtype), mask=mask_r)
        tl.debug_barrier()

        # grad_z_direct = scratch_v3 (will be used in step 7). Save copy to scratch_v1
        # (we need to free sv3 for step 5).
        for i in range(r):
            tl.store(sv1_base + i, tl.load(sv3_base + i))
        tl.debug_barrier()

        # ===================================================================
        # STEP 5: v_w = L_new^{-1} w
        #   Linv_T_grad_vw = solve(L_new.T, grad_vw)          [back-solve, vector]
        #   grad_L_curr -= Linv_T_grad_vw ⊗ vw
        #   grad_w = Linv_T_grad_vw                           (output)
        # ===================================================================
        # zero out sv3 first
        tl.store(sv3_base + idx, tl.zeros((BLOCK,), dtype=tl.float32).to(tl.float32), mask=mask_r)
        for k_rev in range(r):
            k = r - 1 - k_rev
            gvw_k = tl.load(sv2_base + k).to(tl.float32)
            U_col = tl.load(
                L_new_ptr_c + idx * r + k, mask=mask_r, other=0.0
            ).to(tl.float32)
            v3_prev = tl.load(sv3_base + idx, mask=mask_r, other=0.0).to(tl.float32)
            contrib = tl.where(idx > k, U_col * v3_prev, 0.0)
            acc = tl.sum(contrib, axis=0)
            Ukk = tl.load(L_new_ptr_c + k * r + k).to(tl.float32)
            v3_k = (gvw_k - acc) / Ukk
            tl.store(sv3_base + k, v3_k.to(U_col.dtype))
        tl.debug_barrier()

        # grad_L_curr[i, j] -= Linv_T_grad_vw[i] * vw[j]
        for i in range(r):
            v3_i = tl.load(sv3_base + i).to(tl.float32)
            row_old = tl.load(gL_curr_base + i * r + idx, mask=mask_r, other=0.0).to(tl.float32)
            new_row = row_old - v3_i * vw_vec
            tl.store(gL_curr_base + i * r + idx, new_row.to(row_old.dtype), mask=mask_r)
        tl.debug_barrier()

        # grad_w = scratch_v3. Save to grad_V[c].
        gw_vec = tl.load(sv3_base + idx, mask=mask_r, other=0.0)
        tl.store(gV_ptr_c + idx, gw_vec, mask=mask_r)
        tl.debug_barrier()

        # ===================================================================
        # STEP 6: G_new = L_new L_new^T (Cholesky adjoint)
        #   M_aux = L_new.T @ grad_L_curr
        #   P = tril(M_aux, -1) + 0.5 * diag(diag(M_aux))
        #   Y = solve(L_new.T, P)                              [back-solve, matrix]
        #   grad_G_new = solve(L_new.T, Y.T).T                 [back-solve, matrix]
        # ===================================================================
        # M_aux = L_new.T @ grad_L_curr
        L_newT_tile = tl.load(
            L_new_ptr_c + cols[:, None] * r + rows[None, :],  # transpose access
            mask=(cols[:, None] < r) & (rows[None, :] < r),
            other=0.0,
        ).to(tl.float32)
        gL_full = tl.load(
            gL_curr_base + rows[:, None] * r + cols[None, :],
            mask=mask2d, other=0.0,
        ).to(tl.float32)
        M_aux = tl.dot(L_newT_tile, gL_full, allow_tf32=False)

        # P = tril(M_aux, diag=-1) + 0.5 * diag(diag(M_aux))
        # i.e. P[i, j] = M_aux[i, j] if i > j, 0.5 * M_aux[i, j] if i == j, 0 otherwise
        P_tile = tl.where(
            rows[:, None] > cols[None, :],
            M_aux,
            tl.where(
                rows[:, None] == cols[None, :],
                0.5 * M_aux,
                0.0,
            ),
        )
        tl.store(
            sM_base + rows[:, None] * r + cols[None, :],
            P_tile.to(gL_full.dtype),
            mask=mask2d,
        )
        tl.debug_barrier()

        # Y = solve(L_new.T, P)  [back-solve, matrix RHS]; output to scratch_Y
        for k_rev in range(r):
            k = r - 1 - k_rev
            B_row = tl.load(sM_base + k * r + idx, mask=mask_r, other=0.0).to(tl.float32)
            Y_tile_load = tl.load(
                sY_base + rows[:, None] * r + cols[None, :],
                mask=mask2d, other=0.0,
            ).to(tl.float32)
            Y_masked = tl.where(rows[:, None] > k, Y_tile_load, 0.0)
            U_row = tl.load(
                L_new_ptr_c + rows * r + k, mask=rows < r, other=0.0
            ).to(tl.float32)
            correction = tl.sum(U_row[:, None] * Y_masked, axis=0)
            Ukk = tl.load(L_new_ptr_c + k * r + k).to(tl.float32)
            Y_k_row = (B_row - correction) / Ukk
            tl.store(sY_base + k * r + idx, Y_k_row.to(B_row.dtype), mask=mask_r)
        tl.debug_barrier()

        # grad_G_new = solve(L_new.T, Y.T).T
        # Equivalent: solve U Z = Y.T (upper-tri back solve), then grad_G_new = Z.T
        # We'll write Z to scratch_M, then store grad_G_new = Z.T to scratch_X.
        # Y.T[k, :] = Y[:, k] (column k of Y).
        for k_rev in range(r):
            k = r - 1 - k_rev
            # Row k of Y.T = column k of Y
            B_row = tl.load(sY_base + idx * r + k, mask=mask_r, other=0.0).to(tl.float32)
            Z_tile_load = tl.load(
                sM_base + rows[:, None] * r + cols[None, :],
                mask=mask2d, other=0.0,
            ).to(tl.float32)
            Z_masked = tl.where(rows[:, None] > k, Z_tile_load, 0.0)
            U_row = tl.load(
                L_new_ptr_c + rows * r + k, mask=rows < r, other=0.0
            ).to(tl.float32)
            correction = tl.sum(U_row[:, None] * Z_masked, axis=0)
            Ukk = tl.load(L_new_ptr_c + k * r + k).to(tl.float32)
            Z_k_row = (B_row - correction) / Ukk
            tl.store(sM_base + k * r + idx, Z_k_row.to(B_row.dtype), mask=mask_r)
        tl.debug_barrier()

        # grad_G_new = Z.T → grad_G_new[i, j] = Z[j, i]
        # Store to scratch_X
        for i in range(r):
            Z_col_i = tl.load(sM_base + idx * r + i, mask=mask_r, other=0.0)
            tl.store(sX_base + i * r + idx, Z_col_i, mask=mask_r)
        tl.debug_barrier()

        # ===================================================================
        # STEP 7: G_new = L L^T + z z^T
        #   H = grad_G_new + grad_G_new.T
        #   grad_L_prev = grad_L_from_T + H @ L
        #   grad_z = grad_z_direct + H @ z
        # ===================================================================
        # H[i, j] = grad_G_new[i, j] + grad_G_new[j, i]
        # We have grad_G_new in scratch_X.
        gG_tile = tl.load(
            sX_base + rows[:, None] * r + cols[None, :],
            mask=mask2d, other=0.0,
        ).to(tl.float32)
        gG_T_tile = tl.load(
            sX_base + cols[:, None] * r + rows[None, :],  # transpose access
            mask=(cols[:, None] < r) & (rows[None, :] < r),
            other=0.0,
        ).to(tl.float32)
        H_tile = gG_tile + gG_T_tile
        # store H to scratch_M for reuse below
        tl.store(
            sM_base + rows[:, None] * r + cols[None, :],
            H_tile.to(gG_tile.dtype),
            mask=mask2d,
        )

        # grad_L_prev = grad_L_from_T + H @ L
        L_tile = tl.load(
            L_ptr_c + rows[:, None] * r + cols[None, :],
            mask=mask2d, other=0.0,
        ).to(tl.float32)
        HL = tl.dot(H_tile, L_tile, allow_tf32=False)
        gLfT_tile = tl.load(
            sLfT_base + rows[:, None] * r + cols[None, :],
            mask=mask2d, other=0.0,
        ).to(tl.float32)
        grad_L_prev = gLfT_tile + HL
        tl.store(
            gL_curr_base + rows[:, None] * r + cols[None, :],
            grad_L_prev.to(gLfT_tile.dtype),
            mask=mask2d,
        )

        # grad_z = grad_z_direct + H @ z
        # grad_z_direct is in scratch_v1; z is at z_ptr_c
        z_vec = tl.load(z_ptr_c + cols, mask=cols < r, other=0.0).to(tl.float32)
        for i in range(r):
            H_row_i = tl.load(sM_base + i * r + cols, mask=cols < r, other=0.0).to(tl.float32)
            Hz_i = tl.sum(H_row_i * z_vec, axis=0)
            v1_i = tl.load(sv1_base + i).to(tl.float32)
            tl.store(gK_ptr_c + i, (v1_i + Hz_i).to(z_vec.dtype))
        tl.debug_barrier()

        # ===================================================================
        # Update grad_Aw_curr = grad_Aw_prev (which is in scratch_Y)
        # ===================================================================
        for i in range(r):
            row_i = tl.load(sY_base + i * r + idx, mask=mask_r, other=0.0)
            tl.store(gA_curr_base + i * r + idx, row_i, mask=mask_r)

        # Reset scratch_v1 for next iteration's Phase 1a
        tl.store(sv1_base + idx, tl.zeros((BLOCK,), dtype=tl.float32).to(tl.float32), mask=mask_r)
        tl.debug_barrier()


def incremental_chunk_backward_triton(
    L_seq, Aw_seq, K, V, vz_seq, vw_seq, kt_seq, T_seq,
    grad_L_new, grad_Aw_new, grad_K_tilde,
):
    """Single-launch C-token SKA chunk backward.

    Counterpart to incremental_chunk_update_triton. Returns grad inputs
    matching what _per_token_backward_step would produce when called C
    times in reverse, i.e. (grad_L_init, grad_Aw_init, grad_K, grad_V).

    Args:
        L_seq, Aw_seq:        (BH, C+1, r, r) - saved from forward
        K, V:                 (BH, C, r)
        vz_seq, vw_seq:       (BH, C, r)
        kt_seq:               (BH, C, r)
        T_seq:                (BH, C, r, r)
        grad_L_new:           (BH, r, r) upstream
        grad_Aw_new:          (BH, r, r) upstream
        grad_K_tilde:         (BH, C, r) upstream

    Returns:
        grad_L_init, grad_Aw_init: (BH, r, r)
        grad_K, grad_V:            (BH, C, r)
    """
    BH, C, r = K.shape
    assert r <= 128, f"chunk backward kernel requires r<=128, got r={r}"
    BLOCK = triton.next_power_of_2(r)
    assert BLOCK <= 1024
    device, dtype = K.device, K.dtype

    # Output / scratch buffers.
    grad_L_curr = grad_L_new.contiguous().clone()           # rolling state
    grad_Aw_curr = grad_Aw_new.contiguous().clone()         # rolling state
    scratch_X = torch.zeros(BH, r, r, device=device, dtype=dtype)
    scratch_Y = torch.zeros(BH, r, r, device=device, dtype=dtype)
    scratch_M = torch.zeros(BH, r, r, device=device, dtype=dtype)
    scratch_LfT = torch.zeros(BH, r, r, device=device, dtype=dtype)
    scratch_v1 = torch.zeros(BH, r, device=device, dtype=dtype)
    scratch_v2 = torch.zeros(BH, r, device=device, dtype=dtype)
    scratch_v3 = torch.zeros(BH, r, device=device, dtype=dtype)

    grad_K = torch.zeros(BH, C, r, device=device, dtype=dtype)
    grad_V = torch.zeros(BH, C, r, device=device, dtype=dtype)

    _fused_chunk_backward_kernel[(BH,)](
        L_seq, Aw_seq, K, V, vz_seq, vw_seq, kt_seq, T_seq, grad_K_tilde,
        grad_L_curr, grad_Aw_curr,
        scratch_X, scratch_Y, scratch_M, scratch_LfT,
        scratch_v1, scratch_v2, scratch_v3,
        grad_K, grad_V,
        r=r, C=C, BLOCK=BLOCK,
    )

    # After the kernel, grad_L_curr/grad_Aw_curr hold grad_L_init/grad_Aw_init.
    return grad_L_curr, grad_Aw_curr, grad_K, grad_V
