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
