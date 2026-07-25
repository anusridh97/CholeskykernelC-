"""
Double-Sided Cholesky Propagation: O(r^2) Whitened Operator Updates

Given:  L, A_w = L^{-1} M L^{-T}, z, w
Update: G+ = G + z z^T,  M+ = M + z w^T
Compute: L+, A_w+ = (L+)^{-1} M+ (L+)^{-T}

This file provides reference, fused, and HBM-resident Triton entry points:

  1. A pure-PyTorch reference implementation (incremental_update_reference)
     used for correctness checking. It is slow and should not be benchmarked.

  2. A fused Triton kernel runnable in single-update (incremental_update_triton)
     and batched (incremental_update_triton_batched) modes — same SRAM-resident
     kernel, dispatched with grid=(1,) or grid=(T,). Pad row/column for the
     double-sided propagation stay in registers, so only the returned r x r
     block is materialized. Gated at r <= 1024 so r fits in a single
     power-of-two BLOCK up to 1024.

  3. The optimized HBM-resident fallback (incremental_update_triton_tiled)
     dispatches to the fully parallel factorization below. It uses reusable
     storage, one two-RHS triangular solve, and parallel prefix scans.

  4. A fully parallel large-rank path (incremental_update_triton_parallel).
     It solves for u=L^{-1}z and y=L^{-1}w once, exploits the diagonal-plus-
     semiseparable Cholesky factor of I+u*u^T, and replaces the serial Givens
     update with matrix-row prefix scans.

  5. Experimental inverse-factor state
     (incremental_update_triton_inverse_factor), which stores L^{-1} instead
     of L between refreshes. This removes the latency-bound triangular solve
     while retaining exact O(r^2) updates and O(r^2) factor storage.
     incremental_update_triton_inverse_factor_large removes the r <= 1024
     single-block restriction with full-axis CUDA scans.

For the fast CPU path, see fast_update.cpp / the fast_cholesky extension.
Intended to be called inside torch.no_grad(); no autograd graph is built.
"""

import torch
import triton
import triton.language as tl


def _validate_cuda_update_inputs(factor, Aw, z, w, factor_name="L"):
    """Validate the layout contract shared by optimized CUDA entry points."""
    tensors = {
        factor_name: factor,
        "Aw": Aw,
        "z": z,
        "w": w,
    }
    if factor.ndim not in (2, 3):
        raise ValueError(f"{factor_name} must have shape (r,r) or (T,r,r)")
    if factor.shape[-2] != factor.shape[-1]:
        raise ValueError(f"{factor_name} must be square")
    if Aw.shape != factor.shape:
        raise ValueError(f"Aw must have the same shape as {factor_name}")
    batch = 1 if factor.ndim == 2 else factor.shape[0]
    r = factor.shape[-1]
    if batch <= 0 or r <= 0:
        raise ValueError("batch size and rank must be positive")
    expected_vector_shape = (r,) if factor.ndim == 2 else (batch, r)
    if z.shape != expected_vector_shape:
        raise ValueError(f"z must have shape {expected_vector_shape}")
    if w.shape != expected_vector_shape:
        raise ValueError(f"w must have shape {expected_vector_shape}")
    for name, tensor in tensors.items():
        if tensor.device != factor.device:
            raise ValueError(f"{name} must be on the same device as {factor_name}")
        if tensor.dtype != factor.dtype:
            raise ValueError(f"{name} must have the same dtype as {factor_name}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if factor.device.type != "cuda":
        raise ValueError("optimized Triton updates require CUDA tensors")
    if factor.dtype != torch.float32:
        raise ValueError("optimized Triton updates currently require FP32")
    return batch, r


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


@triton.jit
def _setup_update_solve_kernel(
    L_ptr, z_ptr, w_ptr, vz_ptr, vw_ptr, cs_ptr, ss_ptr, prefix_c_ptr,
    r: tl.constexpr,
    BLOCK: tl.constexpr,
    STORE_PREFIX: tl.constexpr,
):
    """Rank-one Cholesky update, v_z recovery, and triangular solve.

    Unlike the original tiled setup, z and the forward-substitution state stay
    in registers.  The old implementation reloaded and rewrote an r-vector on
    every one of r rotations/solve steps, adding two avoidable O(r^2) global
    memory round trips to each sequential phase.
    """
    pid = tl.program_id(0)
    off_mat = pid * r * r
    off_vec = pid * r
    idx = tl.arange(0, BLOCK)
    mask_r = idx < r
    z_vec = tl.load(z_ptr + off_vec + idx, mask=mask_r, other=0.0).to(tl.float32)
    w_vec = tl.load(w_ptr + off_vec + idx, mask=mask_r, other=0.0).to(tl.float32)
    e_last = 1.0

    for k in range(r):
        L_col = tl.load(
            L_ptr + off_mat + idx * r + k,
            mask=mask_r,
            other=0.0,
        )
        Lkk = tl.sum(tl.where(idx == k, L_col, 0.0))
        zk = tl.sum(tl.where(idx == k, z_vec, 0.0))
        c, s = _givens_params_tri(Lkk, zk)

        tl.store(cs_ptr + off_vec + k, c)
        tl.store(ss_ptr + off_vec + k, s)
        if STORE_PREFIX:
            tl.store(prefix_c_ptr + off_vec + k, e_last)
        tl.store(vz_ptr + off_vec + k, s * e_last)
        e_last = c * e_last

        active = mask_r & (idx >= k)
        new_L = c * L_col + s * z_vec
        new_z = -s * L_col + c * z_vec
        tl.store(L_ptr + off_mat + idx * r + k, new_L, mask=active)
        z_vec = tl.where(active, new_z, z_vec)

    # The solve reads L entries written by lanes in all participating warps.
    tl.debug_barrier()

    vw_vec = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in range(r):
        L_row = tl.load(
            L_ptr + off_mat + k * r + idx,
            mask=mask_r,
            other=0.0,
        )
        wk = tl.sum(tl.where(idx == k, w_vec, 0.0))
        contrib = tl.where(
            idx < k,
            L_row.to(tl.float32) * vw_vec,
            0.0,
        )
        acc = tl.sum(contrib, axis=0)
        Lkk = tl.sum(tl.where(idx == k, L_row, 0.0)).to(tl.float32)
        vw_k = (wk - acc) / Lkk
        vw_vec = tl.where(idx == k, vw_k, vw_vec)

    tl.store(vw_ptr + off_vec + idx, vw_vec, mask=mask_r)


@triton.jit
def _pack_solve_rhs_kernel(
    z_ptr, w_ptr, rhs_ptr,
    r: tl.constexpr,
    BLOCK: tl.constexpr,
    RHS_WIDTH: tl.constexpr,
):
    """Pack z and w as two right-hand sides for one triangular solve."""
    pid = tl.program_id(0)
    off_vec = pid * r
    off_rhs = pid * r * RHS_WIDTH
    idx = tl.arange(0, BLOCK)
    mask = idx < r
    z = tl.load(z_ptr + off_vec + idx, mask=mask, other=0.0)
    w = tl.load(w_ptr + off_vec + idx, mask=mask, other=0.0)
    tl.store(rhs_ptr + off_rhs + RHS_WIDTH * idx, z, mask=mask)
    tl.store(rhs_ptr + off_rhs + RHS_WIDTH * idx + 1, w, mask=mask)


@triton.jit
def _parallel_update_params_kernel(
    solve_ptr, vz_ptr, vw_ptr, cs_ptr, ss_ptr, prefix_c_ptr,
    r: tl.constexpr,
    BLOCK: tl.constexpr,
    RHS_WIDTH: tl.constexpr,
):
    """Recover every scalar update parameter with parallel prefix sums.

    Let u=L^{-1}z, y=L^{-1}w and S_i=1+sum_{j<=i}u_j^2.  The Cholesky
    factor C of I+u*u^T has

      c_i = sqrt(S_{i-1}/S_i),  s_i = u_i/sqrt(S_i),
      (C^{-1}u)_i = u_i/sqrt(S_{i-1}S_i).

    C^{-1}y also has a prefix-dot closed form, eliminating the second serial
    triangular solve used by the Givens implementation.
    """
    pid = tl.program_id(0)
    off_vec = pid * r
    off_rhs = pid * r * RHS_WIDTH
    idx = tl.arange(0, BLOCK)
    mask = idx < r
    u = tl.load(
        solve_ptr + off_rhs + RHS_WIDTH * idx,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    y = tl.load(
        solve_ptr + off_rhs + RHS_WIDTH * idx + 1,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    u2 = tl.where(mask, u * u, 0.0)
    cumulative_u2 = tl.cumsum(u2, axis=0)
    s_before = 1.0 + cumulative_u2 - u2
    s_after = 1.0 + cumulative_u2
    inv_sqrt_after = tl.rsqrt(s_after)
    prefix_c = tl.rsqrt(s_before)
    c = tl.sqrt(s_before / s_after)
    s = u * inv_sqrt_after
    vz = u * tl.rsqrt(s_before * s_after)

    uy = tl.where(mask, u * y, 0.0)
    dot_before = tl.cumsum(uy, axis=0) - uy
    vw = c * (y - u * dot_before / s_before)

    tl.store(cs_ptr + off_vec + idx, c, mask=mask)
    tl.store(ss_ptr + off_vec + idx, s, mask=mask)
    tl.store(prefix_c_ptr + off_vec + idx, prefix_c, mask=mask)
    tl.store(vz_ptr + off_vec + idx, vz, mask=mask)
    tl.store(vw_ptr + off_vec + idx, vw, mask=mask)


@triton.jit
def _parallel_cholesky_update_kernel(
    L_ptr, solve_ptr, vz_ptr, cs_ptr,
    r: tl.constexpr,
    BLOCK_R: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    RHS_WIDTH: tl.constexpr,
):
    """Form L+ from L and u=L^{-1}z using independent reverse scans.

    If C=chol(I+u*u^T), then L+=L*C and C is diagonal plus a strict-lower
    rank-one matrix.  For i>=j,

      L+_ij = L_ij/c_j + v_zj * sum_{k=j+1}^i L_ik u_k.

    The strict suffix sum is a reverse cumulative sum along each matrix row,
    so all rows and columns are processed in parallel.
    """
    pid_row = tl.program_id(0)
    pid_batch = tl.program_id(1)
    off_mat = pid_batch * r * r
    off_vec = pid_batch * r
    off_rhs = pid_batch * r * RHS_WIDTH

    row_ids = pid_row * TILE_ROWS + tl.arange(0, TILE_ROWS)
    rows = row_ids[:, None]
    reverse_lane = tl.arange(0, BLOCK_R)[None, :]
    cols = (r - 1) - reverse_lane
    mask = (rows < r) & (cols >= 0) & (cols <= rows)
    L_values = tl.load(
        L_ptr + off_mat + rows * r + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    u = tl.load(
        solve_ptr + off_rhs + RHS_WIDTH * cols,
        mask=cols >= 0,
        other=0.0,
    ).to(tl.float32)
    term = tl.where(mask, L_values * u, 0.0)
    reverse_inclusive = tl.cumsum(term, axis=1)
    strict_suffix = reverse_inclusive - term

    c = tl.load(
        cs_ptr + off_vec + cols,
        mask=cols >= 0,
        other=1.0,
    ).to(tl.float32)
    vz = tl.load(
        vz_ptr + off_vec + cols,
        mask=cols >= 0,
        other=0.0,
    ).to(tl.float32)
    updated = L_values / c + vz * strict_suffix
    tl.store(
        L_ptr + off_mat + rows * r + cols,
        updated,
        mask=mask,
    )


@triton.jit
def _inverse_gemv_pair_kernel(
    Linv_ptr, z_ptr, w_ptr, solve_ptr,
    r: tl.constexpr,
    BLOCK_R: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    RHS_WIDTH: tl.constexpr,
):
    """Compute u=L^{-1}z and y=L^{-1}w from a cached inverse factor."""
    pid_row = tl.program_id(0)
    pid_batch = tl.program_id(1)
    off_mat = pid_batch * r * r
    off_vec = pid_batch * r
    off_rhs = pid_batch * r * RHS_WIDTH
    row_ids = pid_row * TILE_ROWS + tl.arange(0, TILE_ROWS)
    rows = row_ids[:, None]
    cols = tl.arange(0, BLOCK_R)[None, :]
    # Linv is stored with an explicitly zero upper triangle.  Keeping this a
    # rectangular tile avoids a masked-reduction code-generation pathology
    # that otherwise retains only the diagonal contribution.
    mask = (rows < r) & (cols < r)
    inverse_values = tl.load(
        Linv_ptr + off_mat + rows * r + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    z = tl.load(
        z_ptr + off_vec + cols,
        mask=cols < r,
        other=0.0,
    ).to(tl.float32)
    w = tl.load(
        w_ptr + off_vec + cols,
        mask=cols < r,
        other=0.0,
    ).to(tl.float32)
    # Triton's 2-D reduction layout can collapse each row to its diagonal
    # lane for these skinny row tiles.  An inclusive scan with a single
    # masked store at the final column is equally parallel and preserves the
    # complete dot product for every row.
    u = tl.cumsum(inverse_values * z, axis=1)
    y = tl.cumsum(inverse_values * w, axis=1)
    store_mask = (rows < r) & (cols == r - 1)
    tl.store(
        solve_ptr + off_rhs + rows * RHS_WIDTH + 0 * cols,
        u,
        mask=store_mask,
    )
    tl.store(
        solve_ptr + off_rhs + rows * RHS_WIDTH + 1 + 0 * cols,
        y,
        mask=store_mask,
    )


@triton.jit
def _parallel_inverse_update_kernel(
    Linv_ptr, solve_ptr, cs_ptr, prefix_c_ptr,
    r: tl.constexpr,
    BLOCK_R: tl.constexpr,
    TILE_COLS: tl.constexpr,
    RHS_WIDTH: tl.constexpr,
):
    """Update cached L^{-1} to (L+)^{-1}=C^{-1}L^{-1} by column scans."""
    pid_col = tl.program_id(0)
    pid_batch = tl.program_id(1)
    off_mat = pid_batch * r * r
    off_vec = pid_batch * r
    off_rhs = pid_batch * r * RHS_WIDTH
    rows = tl.arange(0, BLOCK_R)[:, None]
    cols = pid_col * TILE_COLS + tl.arange(0, TILE_COLS)[None, :]
    mask = (rows < r) & (cols < r)
    inverse_values = tl.load(
        Linv_ptr + off_mat + rows * r + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    u = tl.load(
        solve_ptr + off_rhs + rows * RHS_WIDTH,
        mask=rows < r,
        other=0.0,
    ).to(tl.float32)
    prev_rows = rows - 1
    prev_mask = (
        (prev_rows >= 0)
        & (prev_rows < r)
        & (cols < r)
        & (cols <= prev_rows)
    )
    prev_inverse = tl.load(
        Linv_ptr + off_mat + prev_rows * r + cols,
        mask=prev_mask,
        other=0.0,
    ).to(tl.float32)
    prev_u = tl.load(
        solve_ptr + off_rhs + prev_rows * RHS_WIDTH,
        mask=(prev_rows >= 0) & (prev_rows < r),
        other=0.0,
    ).to(tl.float32)
    prev_term = tl.where(prev_mask, prev_u * prev_inverse, 0.0)
    dot_before = tl.cumsum(prev_term, axis=0)
    c = tl.load(
        cs_ptr + off_vec + rows,
        mask=rows < r,
        other=1.0,
    ).to(tl.float32)
    prefix_c = tl.load(
        prefix_c_ptr + off_vec + rows,
        mask=rows < r,
        other=1.0,
    ).to(tl.float32)
    updated = c * (
        inverse_values - u * dot_before * prefix_c * prefix_c
    )
    tl.store(
        Linv_ptr + off_mat + rows * r + cols,
        updated,
        mask=mask,
    )


@triton.jit
def _inverse_update_and_left_sweep_kernel(
    Linv_ptr, Aw_ptr, Aw_mid_ptr, solve_ptr,
    cs_ptr, ss_ptr, prefix_c_ptr,
    r: tl.constexpr,
    BLOCK_R: tl.constexpr,
    TILE_COLS: tl.constexpr,
    RHS_WIDTH: tl.constexpr,
):
    """Fuse the inverse-factor update with the left operator sweep.

    Both operations are independent prefix scans down the same column tiles
    and use the same c/prefix coefficients, so fusion removes one launch and
    amortizes coefficient traffic.
    """
    pid_col = tl.program_id(0)
    pid_batch = tl.program_id(1)
    off_mat = pid_batch * r * r
    off_vec = pid_batch * r
    off_rhs = pid_batch * r * RHS_WIDTH
    rows = tl.arange(0, BLOCK_R)[:, None]
    cols = pid_col * TILE_COLS + tl.arange(0, TILE_COLS)[None, :]
    matrix_mask = (rows < r) & (cols < r)
    inverse_mask = matrix_mask

    c = tl.load(
        cs_ptr + off_vec + rows,
        mask=rows < r,
        other=1.0,
    ).to(tl.float32)
    s = tl.load(
        ss_ptr + off_vec + rows,
        mask=rows < r,
        other=0.0,
    ).to(tl.float32)
    prefix_c = tl.load(
        prefix_c_ptr + off_vec + rows,
        mask=rows < r,
        other=1.0,
    ).to(tl.float32)

    inverse_values = tl.load(
        Linv_ptr + off_mat + rows * r + cols,
        mask=inverse_mask,
        other=0.0,
    ).to(tl.float32)
    u = tl.load(
        solve_ptr + off_rhs + rows * RHS_WIDTH,
        mask=rows < r,
        other=0.0,
    ).to(tl.float32)
    prev_rows = rows - 1
    prev_inverse_mask = (
        (prev_rows >= 0)
        & (prev_rows < r)
        & (cols < r)
        & (cols <= prev_rows)
    )
    prev_inverse = tl.load(
        Linv_ptr + off_mat + prev_rows * r + cols,
        mask=prev_inverse_mask,
        other=0.0,
    ).to(tl.float32)
    prev_u = tl.load(
        solve_ptr + off_rhs + prev_rows * RHS_WIDTH,
        mask=(prev_rows >= 0) & (prev_rows < r),
        other=0.0,
    ).to(tl.float32)
    prev_inverse_term = tl.where(
        prev_inverse_mask,
        prev_u * prev_inverse,
        0.0,
    )
    inverse_dot_before = tl.cumsum(prev_inverse_term, axis=0)
    inverse_updated = c * (
        inverse_values
        - u * inverse_dot_before * prefix_c * prefix_c
    )
    tl.store(
        Linv_ptr + off_mat + rows * r + cols,
        inverse_updated,
        mask=inverse_mask,
    )

    operator_values = tl.load(
        Aw_ptr + off_mat + rows * r + cols,
        mask=matrix_mask,
        other=0.0,
    ).to(tl.float32)
    operator_term = tl.where(
        matrix_mask,
        -s * operator_values / (prefix_c * c),
        0.0,
    )
    operator_inclusive = tl.cumsum(operator_term, axis=0)
    operator_pad_before = prefix_c * (
        operator_inclusive - operator_term
    )
    operator_updated = c * operator_values + s * operator_pad_before
    tl.store(
        Aw_mid_ptr + off_mat + rows * r + cols,
        operator_updated,
        mask=matrix_mask,
    )


@triton.jit
def _inverse_update_left_params_kernel(
    Linv_ptr, Aw_ptr, Aw_mid_ptr, solve_ptr,
    cs_ptr, ss_ptr, prefix_c_ptr, vz_ptr, vw_ptr,
    r: tl.constexpr,
    BLOCK_R: tl.constexpr,
    TILE_COLS: tl.constexpr,
    RHS_WIDTH: tl.constexpr,
):
    """Three-way fusion for inverse-factor state.

    Each column program reconstructs the O(r) update parameters from u and y,
    then applies both column scans.  Program zero publishes the parameter
    vectors for the subsequent right-sweep launch.
    """
    pid_col = tl.program_id(0)
    pid_batch = tl.program_id(1)
    off_mat = pid_batch * r * r
    off_vec = pid_batch * r
    off_rhs = pid_batch * r * RHS_WIDTH
    row_ids = tl.arange(0, BLOCK_R)
    rows = row_ids[:, None]
    cols = pid_col * TILE_COLS + tl.arange(0, TILE_COLS)[None, :]
    row_mask = row_ids < r
    matrix_mask = row_mask[:, None] & (cols < r)
    inverse_mask = matrix_mask

    u_vec = tl.load(
        solve_ptr + off_rhs + row_ids * RHS_WIDTH,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    y_vec = tl.load(
        solve_ptr + off_rhs + row_ids * RHS_WIDTH + 1,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    u2 = tl.where(row_mask, u_vec * u_vec, 0.0)
    cumulative_u2 = tl.cumsum(u2, axis=0)
    s_before = 1.0 + cumulative_u2 - u2
    s_after = 1.0 + cumulative_u2
    prefix_c_vec = tl.rsqrt(s_before)
    c_vec = tl.sqrt(s_before / s_after)
    s_vec = u_vec * tl.rsqrt(s_after)
    vz_vec = u_vec * tl.rsqrt(s_before * s_after)
    uy = tl.where(row_mask, u_vec * y_vec, 0.0)
    dot_before = tl.cumsum(uy, axis=0) - uy
    vw_vec = c_vec * (y_vec - u_vec * dot_before / s_before)

    publish_vector_mask = row_mask & (pid_col == 0)
    tl.store(cs_ptr + off_vec + row_ids, c_vec, mask=publish_vector_mask)
    tl.store(ss_ptr + off_vec + row_ids, s_vec, mask=publish_vector_mask)
    tl.store(
        prefix_c_ptr + off_vec + row_ids,
        prefix_c_vec,
        mask=publish_vector_mask,
    )
    tl.store(vz_ptr + off_vec + row_ids, vz_vec, mask=publish_vector_mask)
    tl.store(vw_ptr + off_vec + row_ids, vw_vec, mask=publish_vector_mask)

    inverse_values = tl.load(
        Linv_ptr + off_mat + rows * r + cols,
        mask=inverse_mask,
        other=0.0,
    ).to(tl.float32)
    u = u_vec[:, None]
    c = c_vec[:, None]
    s = s_vec[:, None]
    prefix_c = prefix_c_vec[:, None]
    prev_rows = rows - 1
    prev_inverse_mask = (
        (prev_rows >= 0)
        & (prev_rows < r)
        & (cols < r)
        & (cols <= prev_rows)
    )
    prev_inverse = tl.load(
        Linv_ptr + off_mat + prev_rows * r + cols,
        mask=prev_inverse_mask,
        other=0.0,
    ).to(tl.float32)
    prev_u = tl.load(
        solve_ptr + off_rhs + prev_rows * RHS_WIDTH,
        mask=(prev_rows >= 0) & (prev_rows < r),
        other=0.0,
    ).to(tl.float32)
    prev_inverse_term = tl.where(
        prev_inverse_mask,
        prev_u * prev_inverse,
        0.0,
    )
    inverse_dot_before = tl.cumsum(prev_inverse_term, axis=0)
    inverse_updated = c * (
        inverse_values
        - u * inverse_dot_before * prefix_c * prefix_c
    )
    tl.store(
        Linv_ptr + off_mat + rows * r + cols,
        inverse_updated,
        mask=inverse_mask,
    )

    operator_values = tl.load(
        Aw_ptr + off_mat + rows * r + cols,
        mask=matrix_mask,
        other=0.0,
    ).to(tl.float32)
    operator_term = tl.where(
        matrix_mask,
        -s * operator_values / (prefix_c * c),
        0.0,
    )
    operator_inclusive = tl.cumsum(operator_term, axis=0)
    operator_pad_before = prefix_c * (
        operator_inclusive - operator_term
    )
    operator_updated = c * operator_values + s * operator_pad_before
    tl.store(
        Aw_mid_ptr + off_mat + rows * r + cols,
        operator_updated,
        mask=matrix_mask,
    )


@triton.jit
def _large_pack_rhs_kernel(
    z_ptr, w_ptr, rhs_ptr,
    total, r,
    RHS_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Pack z and w for a two-column GEMM at arbitrary rank."""
    offsets = (
        tl.program_id(0).to(tl.int64) * BLOCK
        + tl.arange(0, BLOCK).to(tl.int64)
    )
    mask = offsets < total
    batch = offsets // r
    index = offsets - batch * r
    vector_offset = batch * r + index
    rhs_offset = batch * r * RHS_WIDTH + index * RHS_WIDTH
    z = tl.load(z_ptr + vector_offset, mask=mask, other=0.0)
    w = tl.load(w_ptr + vector_offset, mask=mask, other=0.0)
    tl.store(rhs_ptr + rhs_offset, z, mask=mask)
    tl.store(rhs_ptr + rhs_offset + 1, w, mask=mask)


@triton.jit
def _large_vector_scan_terms_kernel(
    solve_ptr, u2_ptr, uy_ptr,
    total, r,
    RHS_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Form u^2 and u*y before their one-dimensional prefix scans."""
    offsets = (
        tl.program_id(0).to(tl.int64) * BLOCK
        + tl.arange(0, BLOCK).to(tl.int64)
    )
    mask = offsets < total
    batch = offsets // r
    index = offsets - batch * r
    rhs_offset = batch * r * RHS_WIDTH + index * RHS_WIDTH
    u = tl.load(solve_ptr + rhs_offset, mask=mask, other=0.0)
    y = tl.load(solve_ptr + rhs_offset + 1, mask=mask, other=0.0)
    tl.store(u2_ptr + offsets, u * u, mask=mask)
    tl.store(uy_ptr + offsets, u * y, mask=mask)


@triton.jit
def _large_parameter_finalize_kernel(
    solve_ptr, cumulative_u2_ptr, cumulative_uy_ptr,
    cs_ptr, ss_ptr, prefix_c_ptr, vz_ptr, vw_ptr,
    total, r,
    RHS_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = (
        tl.program_id(0).to(tl.int64) * BLOCK
        + tl.arange(0, BLOCK).to(tl.int64)
    )
    mask = offsets < total
    batch = offsets // r
    index = offsets - batch * r
    vector_offset = batch * r + index
    rhs_offset = batch * r * RHS_WIDTH + index * RHS_WIDTH
    u = tl.load(solve_ptr + rhs_offset, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(
        solve_ptr + rhs_offset + 1,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    cumulative_u2 = tl.load(
        cumulative_u2_ptr + vector_offset,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    cumulative_uy = tl.load(
        cumulative_uy_ptr + vector_offset,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    u2 = u * u
    s_before = 1.0 + cumulative_u2 - u2
    s_after = 1.0 + cumulative_u2
    prefix_c = tl.rsqrt(s_before)
    c = tl.sqrt(s_before / s_after)
    s = u * tl.rsqrt(s_after)
    vz = u * tl.rsqrt(s_before * s_after)
    dot_before = cumulative_uy - u * y
    vw = c * (y - u * dot_before / s_before)
    tl.store(cs_ptr + vector_offset, c, mask=mask)
    tl.store(ss_ptr + vector_offset, s, mask=mask)
    tl.store(prefix_c_ptr + vector_offset, prefix_c, mask=mask)
    tl.store(vz_ptr + vector_offset, vz, mask=mask)
    tl.store(vw_ptr + vector_offset, vw, mask=mask)


@triton.jit
def _large_matrix_scan_term_kernel(
    matrix_ptr, scan_ptr, solve_ptr, cs_ptr, ss_ptr, prefix_c_ptr,
    total, matrix_size, r,
    MODE: tl.constexpr,
    RHS_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Materialize one additive scan term for an arbitrary dense rank."""
    offsets = (
        tl.program_id(0).to(tl.int64) * BLOCK
        + tl.arange(0, BLOCK).to(tl.int64)
    )
    mask = offsets < total
    batch = offsets // matrix_size
    within = offsets - batch * matrix_size
    row = within // r
    col = within - row * r
    vector_base = batch * r
    value = tl.load(matrix_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if MODE == 0:
        u = tl.load(
            solve_ptr + batch * r * RHS_WIDTH + row * RHS_WIDTH,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        value = tl.where(col <= row, u * value, 0.0)
    elif MODE == 1:
        c = tl.load(cs_ptr + vector_base + row, mask=mask, other=1.0)
        s = tl.load(ss_ptr + vector_base + row, mask=mask, other=0.0)
        prefix_c = tl.load(
            prefix_c_ptr + vector_base + row,
            mask=mask,
            other=1.0,
        )
        value = -s * value / (prefix_c * c)
    else:
        c = tl.load(cs_ptr + vector_base + col, mask=mask, other=1.0)
        s = tl.load(ss_ptr + vector_base + col, mask=mask, other=0.0)
        prefix_c = tl.load(
            prefix_c_ptr + vector_base + col,
            mask=mask,
            other=1.0,
        )
        value = -s * value / (prefix_c * c)
    tl.store(scan_ptr + offsets, value, mask=mask)


@triton.jit
def _large_matrix_scan_finalize_kernel(
    matrix_ptr, scan_ptr, output_ptr, solve_ptr,
    cs_ptr, ss_ptr, prefix_c_ptr, vz_ptr, vw_ptr,
    total, matrix_size, r,
    MODE: tl.constexpr,
    RHS_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Apply an exclusive prefix and finalize one arbitrary-rank sweep."""
    offsets = (
        tl.program_id(0).to(tl.int64) * BLOCK
        + tl.arange(0, BLOCK).to(tl.int64)
    )
    mask = offsets < total
    batch = offsets // matrix_size
    within = offsets - batch * matrix_size
    row = within // r
    col = within - row * r
    vector_base = batch * r
    value = tl.load(matrix_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inclusive = tl.load(scan_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if MODE == 0:
        u = tl.load(
            solve_ptr + batch * r * RHS_WIDTH + row * RHS_WIDTH,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        term = tl.where(col <= row, u * value, 0.0)
        c = tl.load(cs_ptr + vector_base + row, mask=mask, other=1.0)
        prefix_c = tl.load(
            prefix_c_ptr + vector_base + row,
            mask=mask,
            other=1.0,
        )
        updated = c * (
            value - u * (inclusive - term) * prefix_c * prefix_c
        )
        updated = tl.where(col <= row, updated, 0.0)
    elif MODE == 1:
        c = tl.load(cs_ptr + vector_base + row, mask=mask, other=1.0)
        s = tl.load(ss_ptr + vector_base + row, mask=mask, other=0.0)
        prefix_c = tl.load(
            prefix_c_ptr + vector_base + row,
            mask=mask,
            other=1.0,
        )
        term = -s * value / (prefix_c * c)
        updated = c * value + s * prefix_c * (inclusive - term)
    else:
        c = tl.load(cs_ptr + vector_base + col, mask=mask, other=1.0)
        s = tl.load(ss_ptr + vector_base + col, mask=mask, other=0.0)
        prefix_c = tl.load(
            prefix_c_ptr + vector_base + col,
            mask=mask,
            other=1.0,
        )
        term = -s * value / (prefix_c * c)
        vz = tl.load(vz_ptr + vector_base + row, mask=mask, other=0.0)
        vw = tl.load(vw_ptr + vector_base + col, mask=mask, other=0.0)
        updated = (
            c * value
            + s * prefix_c * (inclusive - term)
            + vz * vw
        )
    tl.store(output_ptr + offsets, updated, mask=mask)


@triton.jit
def _inplace_inverse_left_sweep_kernel(
    Linv_ptr,
    Aw_ptr,
    solve_ptr,
    cs_ptr,
    ss_ptr,
    prefix_c_ptr,
    r,
    matrix_size,
    RHS_WIDTH: tl.constexpr,
    TILE_COLS: tl.constexpr,
):
    """Update L^{-1} and apply the left sweep with no dense workspace.

    Each program owns independent columns and walks down them once.  The
    recurrence is serial within a column but all columns run concurrently.
    Keeping the two loop-carried scalars in registers permits ranks whose two
    state matrices occupy most device memory.
    """
    pid_col = tl.program_id(0)
    pid_batch = tl.program_id(1)
    cols = (
        pid_col.to(tl.int64) * TILE_COLS
        + tl.arange(0, TILE_COLS).to(tl.int64)
    )
    col_mask = cols < r
    matrix_base = pid_batch.to(tl.int64) * matrix_size
    vector_base = pid_batch.to(tl.int64) * r
    rhs_base = vector_base * RHS_WIDTH
    inverse_dot = tl.zeros((TILE_COLS,), dtype=tl.float32)
    operator_pad = tl.zeros((TILE_COLS,), dtype=tl.float32)

    for row in tl.range(0, r, num_stages=1):
        row64 = row.to(tl.int64)
        offsets = matrix_base + row64 * r + cols
        inverse_mask = col_mask & (cols <= row64)
        inverse_value = tl.load(
            Linv_ptr + offsets,
            mask=inverse_mask,
            other=0.0,
        ).to(tl.float32)
        u = tl.load(solve_ptr + rhs_base + row64 * RHS_WIDTH).to(tl.float32)
        c = tl.load(cs_ptr + vector_base + row64).to(tl.float32)
        s = tl.load(ss_ptr + vector_base + row64).to(tl.float32)
        prefix_c = tl.load(
            prefix_c_ptr + vector_base + row64
        ).to(tl.float32)
        inverse_updated = c * (
            inverse_value
            - u * inverse_dot * prefix_c * prefix_c
        )
        tl.store(
            Linv_ptr + offsets,
            inverse_updated,
            mask=inverse_mask,
        )
        inverse_dot += tl.where(
            inverse_mask,
            u * inverse_value,
            0.0,
        )

        operator_value = tl.load(
            Aw_ptr + offsets,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        operator_updated = c * operator_value + s * operator_pad
        operator_pad = -s * operator_value + c * operator_pad
        tl.store(
            Aw_ptr + offsets,
            operator_updated,
            mask=col_mask,
        )


@triton.jit
def _inplace_right_sweep_rank1_kernel(
    Aw_ptr,
    cs_ptr,
    ss_ptr,
    vz_ptr,
    vw_ptr,
    r,
    matrix_size,
    TILE_ROWS: tl.constexpr,
):
    """Apply the right sweep and rank-one correction in place."""
    pid_row = tl.program_id(0)
    pid_batch = tl.program_id(1)
    rows = (
        pid_row.to(tl.int64) * TILE_ROWS
        + tl.arange(0, TILE_ROWS).to(tl.int64)
    )
    row_mask = rows < r
    matrix_base = pid_batch.to(tl.int64) * matrix_size
    vector_base = pid_batch.to(tl.int64) * r
    operator_pad = tl.zeros((TILE_ROWS,), dtype=tl.float32)
    vz = tl.load(
        vz_ptr + vector_base + rows,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)

    for col in tl.range(0, r, num_stages=1):
        col64 = col.to(tl.int64)
        offsets = matrix_base + rows * r + col64
        operator_value = tl.load(
            Aw_ptr + offsets,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        c = tl.load(cs_ptr + vector_base + col64).to(tl.float32)
        s = tl.load(ss_ptr + vector_base + col64).to(tl.float32)
        vw = tl.load(vw_ptr + vector_base + col64).to(tl.float32)
        operator_updated = c * operator_value + s * operator_pad
        operator_pad = -s * operator_value + c * operator_pad
        tl.store(
            Aw_ptr + offsets,
            operator_updated + vz * vw,
            mask=row_mask,
        )


@triton.jit
def _left_sweep_cols_batched_kernel(
    Aw_ptr, Aw_mid_ptr, cs_ptr, ss_ptr,
    r: tl.constexpr,
    TILE_COLS: tl.constexpr,
):
    """Apply left sweeps over column tiles and independent batch items."""
    pid_col = tl.program_id(0)
    pid_batch = tl.program_id(1)
    off_mat = pid_batch * r * r
    off_vec = pid_batch * r
    cols = pid_col * TILE_COLS + tl.arange(0, TILE_COLS)
    mask = cols < r
    pad_row = tl.zeros((TILE_COLS,), dtype=tl.float32)

    for k in range(r):
        c = tl.load(cs_ptr + off_vec + k)
        s = tl.load(ss_ptr + off_vec + k)
        rk = tl.load(
            Aw_ptr + off_mat + k * r + cols,
            mask=mask,
            other=0.0,
        )
        out = c * rk + s * pad_row
        pad_row = -s * rk + c * pad_row
        tl.store(Aw_mid_ptr + off_mat + k * r + cols, out, mask=mask)


@triton.jit
def _right_sweep_rows_rank1_kernel(
    Aw_mid_ptr, Aw_out_ptr, cs_ptr, ss_ptr, vz_ptr, vw_ptr,
    r: tl.constexpr,
    TILE_ROWS: tl.constexpr,
):
    """Right sweep with the final v_z v_w^T correction fused into stores."""
    pid_row = tl.program_id(0)
    pid_batch = tl.program_id(1)
    off_mat = pid_batch * r * r
    off_vec = pid_batch * r
    rows = pid_row * TILE_ROWS + tl.arange(0, TILE_ROWS)
    mask = rows < r
    pad_col = tl.zeros((TILE_ROWS,), dtype=tl.float32)
    vz = tl.load(vz_ptr + off_vec + rows, mask=mask, other=0.0)

    for k in range(r):
        c = tl.load(cs_ptr + off_vec + k)
        s = tl.load(ss_ptr + off_vec + k)
        ck = tl.load(
            Aw_mid_ptr + off_mat + rows * r + k,
            mask=mask,
            other=0.0,
        )
        out = c * ck + s * pad_col
        pad_col = -s * ck + c * pad_col
        vw = tl.load(vw_ptr + off_vec + k)
        tl.store(
            Aw_out_ptr + off_mat + rows * r + k,
            out + vz * vw,
            mask=mask,
        )


@triton.jit
def _compose_affine(a_left, b_left, a_right, b_right):
    """Compose p -> a*p+b transforms in left-to-right sequence order."""
    return a_left * a_right, b_left * a_right + b_right


@triton.jit
def _left_sweep_scan_kernel(
    Aw_ptr, Aw_mid_ptr, cs_ptr, ss_ptr,
    r: tl.constexpr,
    BLOCK_R: tl.constexpr,
    TILE_COLS: tl.constexpr,
):
    """Parallel-scan form of the left Givens sweep.

    For a column value x_k, the virtual padded-row recurrence is
        p_k = c_k p_{k-1} - s_k x_k.
    Each recurrence step is an affine map.  Associative composition reduces
    the r-step serial chain to a parallel prefix scan.
    """
    pid_col = tl.program_id(0)
    pid_batch = tl.program_id(1)
    off_mat = pid_batch * r * r
    off_vec = pid_batch * r
    rows = tl.arange(0, BLOCK_R)[:, None]
    cols = pid_col * TILE_COLS + tl.arange(0, TILE_COLS)[None, :]
    mask = (rows < r) & (cols < r)
    x = tl.load(Aw_ptr + off_mat + rows * r + cols, mask=mask, other=0.0)

    prev_rows = rows - 1
    prev_mask = (prev_rows >= 0) & (prev_rows < r) & (cols < r)
    x_prev = tl.load(
        Aw_ptr + off_mat + prev_rows * r + cols,
        mask=prev_mask,
        other=0.0,
    )
    c_prev = tl.load(
        cs_ptr + off_vec + prev_rows,
        mask=(prev_rows >= 0) & (prev_rows < r),
        other=1.0,
    )
    s_prev = tl.load(
        ss_ptr + off_vec + prev_rows,
        mask=(prev_rows >= 0) & (prev_rows < r),
        other=0.0,
    )
    # Broadcast the affine coefficients to the complete 2D tile.
    a = c_prev + tl.zeros((BLOCK_R, TILE_COLS), dtype=tl.float32)
    b = -s_prev * x_prev
    _, pad_before = tl.associative_scan(
        (a, b),
        axis=0,
        combine_fn=_compose_affine,
    )

    c = tl.load(cs_ptr + off_vec + rows, mask=rows < r, other=1.0)
    s = tl.load(ss_ptr + off_vec + rows, mask=rows < r, other=0.0)
    out = c * x + s * pad_before
    tl.store(Aw_mid_ptr + off_mat + rows * r + cols, out, mask=mask)


@triton.jit
def _right_sweep_scan_rank1_kernel(
    Aw_mid_ptr, Aw_out_ptr, cs_ptr, ss_ptr, vz_ptr, vw_ptr,
    r: tl.constexpr,
    BLOCK_R: tl.constexpr,
    TILE_ROWS: tl.constexpr,
):
    """Parallel-scan right sweep with fused rank-one correction."""
    pid_row = tl.program_id(0)
    pid_batch = tl.program_id(1)
    off_mat = pid_batch * r * r
    off_vec = pid_batch * r
    rows = pid_row * TILE_ROWS + tl.arange(0, TILE_ROWS)[:, None]
    cols = tl.arange(0, BLOCK_R)[None, :]
    mask = (rows < r) & (cols < r)
    x = tl.load(
        Aw_mid_ptr + off_mat + rows * r + cols,
        mask=mask,
        other=0.0,
    )

    prev_cols = cols - 1
    prev_mask = (rows < r) & (prev_cols >= 0) & (prev_cols < r)
    x_prev = tl.load(
        Aw_mid_ptr + off_mat + rows * r + prev_cols,
        mask=prev_mask,
        other=0.0,
    )
    c_prev = tl.load(
        cs_ptr + off_vec + prev_cols,
        mask=(prev_cols >= 0) & (prev_cols < r),
        other=1.0,
    )
    s_prev = tl.load(
        ss_ptr + off_vec + prev_cols,
        mask=(prev_cols >= 0) & (prev_cols < r),
        other=0.0,
    )
    a = c_prev + tl.zeros((TILE_ROWS, BLOCK_R), dtype=tl.float32)
    b = -s_prev * x_prev
    _, pad_before = tl.associative_scan(
        (a, b),
        axis=1,
        combine_fn=_compose_affine,
    )

    c = tl.load(cs_ptr + off_vec + cols, mask=cols < r, other=1.0)
    s = tl.load(ss_ptr + off_vec + cols, mask=cols < r, other=0.0)
    vz = tl.load(vz_ptr + off_vec + rows, mask=rows < r, other=0.0)
    vw = tl.load(vw_ptr + off_vec + cols, mask=cols < r, other=0.0)
    out = c * x + s * pad_before + vz * vw
    tl.store(Aw_out_ptr + off_mat + rows * r + cols, out, mask=mask)


@triton.jit
def _left_sweep_cumsum_kernel(
    Aw_ptr, Aw_mid_ptr, cs_ptr, ss_ptr, prefix_c_ptr,
    r: tl.constexpr,
    BLOCK_R: tl.constexpr,
    TILE_COLS: tl.constexpr,
):
    """Left sweep reduced to a weighted prefix sum.

    With P_k = prod_{j<k} c_j, the padded recurrence has the closed form
        p_{k-1} = P_k sum_{j<k} (-s_j x_j / (P_j c_j)).
    This replaces generic affine composition with Triton's specialized
    parallel cumulative sum.
    """
    pid_col = tl.program_id(0)
    pid_batch = tl.program_id(1)
    off_mat = pid_batch * r * r
    off_vec = pid_batch * r
    rows = tl.arange(0, BLOCK_R)[:, None]
    cols = pid_col * TILE_COLS + tl.arange(0, TILE_COLS)[None, :]
    mask = (rows < r) & (cols < r)
    x = tl.load(Aw_ptr + off_mat + rows * r + cols, mask=mask, other=0.0)
    c = tl.load(cs_ptr + off_vec + rows, mask=rows < r, other=1.0)
    s = tl.load(ss_ptr + off_vec + rows, mask=rows < r, other=0.0)
    prefix_c = tl.load(
        prefix_c_ptr + off_vec + rows,
        mask=rows < r,
        other=1.0,
    )
    denominator = prefix_c * c
    term = tl.where(mask, -s * x / denominator, 0.0)
    inclusive = tl.cumsum(term, axis=0)
    pad_before = prefix_c * (inclusive - term)
    out = c * x + s * pad_before
    tl.store(Aw_mid_ptr + off_mat + rows * r + cols, out, mask=mask)


@triton.jit
def _right_sweep_cumsum_rank1_kernel(
    Aw_mid_ptr, Aw_out_ptr, cs_ptr, ss_ptr,
    prefix_c_ptr, vz_ptr, vw_ptr,
    r: tl.constexpr,
    BLOCK_R: tl.constexpr,
    TILE_ROWS: tl.constexpr,
):
    """Weighted-prefix right sweep with fused rank-one correction."""
    pid_row = tl.program_id(0)
    pid_batch = tl.program_id(1)
    off_mat = pid_batch * r * r
    off_vec = pid_batch * r
    rows = pid_row * TILE_ROWS + tl.arange(0, TILE_ROWS)[:, None]
    cols = tl.arange(0, BLOCK_R)[None, :]
    mask = (rows < r) & (cols < r)
    x = tl.load(
        Aw_mid_ptr + off_mat + rows * r + cols,
        mask=mask,
        other=0.0,
    )
    c = tl.load(cs_ptr + off_vec + cols, mask=cols < r, other=1.0)
    s = tl.load(ss_ptr + off_vec + cols, mask=cols < r, other=0.0)
    prefix_c = tl.load(
        prefix_c_ptr + off_vec + cols,
        mask=cols < r,
        other=1.0,
    )
    denominator = prefix_c * c
    term = tl.where(mask, -s * x / denominator, 0.0)
    inclusive = tl.cumsum(term, axis=1)
    pad_before = prefix_c * (inclusive - term)
    vz = tl.load(vz_ptr + off_vec + rows, mask=rows < r, other=0.0)
    vw = tl.load(vw_ptr + off_vec + cols, mask=cols < r, other=0.0)
    out = c * x + s * pad_before + vz * vw
    tl.store(Aw_out_ptr + off_mat + rows * r + cols, out, mask=mask)


class TritonTiledWorkspace:
    """Reusable buffers for the optimized HBM-resident CUDA path.

    Returned ``Aw`` and ``vz`` tensors alias ``aw_out`` and ``vz`` in this
    workspace and are overwritten by the next call that reuses it.
    """

    def __init__(self, like, solve_width=2):
        if like.ndim not in (2, 3) or like.shape[-2] != like.shape[-1]:
            raise ValueError("workspace prototype must contain square matrices")
        r = like.shape[-1]
        batch = 1 if like.ndim == 2 else like.shape[0]
        self.rank = r
        self.batch = batch
        self.shape = like.shape
        self.dtype = like.dtype
        self.device = like.device
        if solve_width < 2:
            raise ValueError("solve_width must be at least two")
        self.solve_width = solve_width
        vector_shape = (r,) if like.ndim == 2 else (batch, r)
        self.vz = torch.empty(vector_shape, dtype=like.dtype, device=like.device)
        self.vw = torch.empty(vector_shape, dtype=like.dtype, device=like.device)
        self.cs = torch.empty(vector_shape, dtype=like.dtype, device=like.device)
        self.ss = torch.empty(vector_shape, dtype=like.dtype, device=like.device)
        self.prefix_c = torch.empty(
            vector_shape,
            dtype=like.dtype,
            device=like.device,
        )
        rhs_shape = (
            (r, solve_width)
            if like.ndim == 2
            else (batch, r, solve_width)
        )
        self.rhs = torch.empty(rhs_shape, dtype=like.dtype, device=like.device)
        self.solve = torch.empty(rhs_shape, dtype=like.dtype, device=like.device)
        self.aw_mid = torch.empty_like(like)
        self.aw_out = torch.empty_like(like)

    @property
    def nbytes(self):
        buffers = (
            self.vz,
            self.vw,
            self.cs,
            self.ss,
            self.prefix_c,
            self.rhs,
            self.solve,
            self.aw_mid,
            self.aw_out,
        )
        return sum(buffer.numel() * buffer.element_size() for buffer in buffers)

    def validate(self, like, solve_width=None):
        if (
            like.shape != self.shape
            or like.dtype != self.dtype
            or like.device != self.device
        ):
            raise ValueError("workspace shape, dtype, or device does not match inputs")
        if solve_width is not None and solve_width != self.solve_width:
            raise ValueError("workspace solve width does not match requested width")


class TritonInverseInplaceWorkspace:
    """Vector-only workspace for the two-matrix inverse-factor path.

    The dense state consists only of the caller-owned ``Linv`` and ``Aw``.
    Returned ``vz`` aliases this workspace and is overwritten on reuse.
    """

    def __init__(self, like):
        if like.ndim not in (2, 3) or like.shape[-2] != like.shape[-1]:
            raise ValueError("workspace prototype must contain square matrices")
        r = like.shape[-1]
        batch = 1 if like.ndim == 2 else like.shape[0]
        self.rank = r
        self.batch = batch
        self.shape = like.shape
        self.dtype = like.dtype
        self.device = like.device
        vector_shape = (r,) if like.ndim == 2 else (batch, r)
        rhs_shape = (r, 2) if like.ndim == 2 else (batch, r, 2)
        self.vz = torch.empty(vector_shape, dtype=like.dtype, device=like.device)
        self.vw = torch.empty(vector_shape, dtype=like.dtype, device=like.device)
        self.cs = torch.empty(vector_shape, dtype=like.dtype, device=like.device)
        self.ss = torch.empty(vector_shape, dtype=like.dtype, device=like.device)
        self.prefix_c = torch.empty(
            vector_shape,
            dtype=like.dtype,
            device=like.device,
        )
        self.rhs = torch.empty(rhs_shape, dtype=like.dtype, device=like.device)
        self.solve = torch.empty(rhs_shape, dtype=like.dtype, device=like.device)

    @property
    def nbytes(self):
        buffers = (
            self.vz,
            self.vw,
            self.cs,
            self.ss,
            self.prefix_c,
            self.rhs,
            self.solve,
        )
        return sum(buffer.numel() * buffer.element_size() for buffer in buffers)

    def validate(self, like):
        if (
            like.shape != self.shape
            or like.dtype != self.dtype
            or like.device != self.device
        ):
            raise ValueError("workspace shape, dtype, or device does not match inputs")


def _default_serial_tile(rank):
    # About 128 independent recurrence chains gives the H100 enough CTAs to
    # hide the long per-chain dependency latency.
    return max(1, min(8, rank // 128))


def _default_scan_tile(rank):
    # Keep roughly 2K values per program; scan temporaries otherwise create
    # excessive register pressure at r=1024.
    target = max(1, min(8, 2048 // rank))
    return triton.next_power_of_2(target)


def _default_parallel_config(rank):
    """H100-confirmed defaults for the public rank-256--1024 fallback."""
    if rank <= 256:
        return 2, 8, 8, 8
    if rank <= 512:
        return 1, 8, 4, 8
    return 4, 8, 4, 8


def _validate_prefix_kernel_config(rank, tile_rows, tile_cols):
    block = triton.next_power_of_2(rank)
    if block > 1024:
        raise ValueError(f"r={rank} too large for prefix-scan kernel")
    for name, value in (("tile_rows", tile_rows), ("tile_cols", tile_cols)):
        if value <= 0 or triton.next_power_of_2(value) != value:
            raise ValueError(f"{name} must be a positive power of two")
    return block


def incremental_update_triton_tiled_v2(
    L,
    Aw,
    z,
    w,
    tile_rows=None,
    tile_cols=None,
    setup_num_warps=4,
    sweep_num_warps=1,
    workspace=None,
):
    """Optimized three-launch HBM-resident update.

    The setup kernel keeps z and v_w in registers, the right sweep includes
    the rank-one correction, and a reusable workspace removes allocator work
    from steady-state calls.
    """
    if L.ndim not in (2, 3):
        raise ValueError("L must have shape (r,r) or (T,r,r)")
    batch = 1 if L.ndim == 2 else L.shape[0]
    r = L.shape[-1]
    BLOCK = triton.next_power_of_2(r)
    assert BLOCK <= 1024, f"r={r} too large for single-block setup kernel"
    tile_rows = tile_rows or _default_serial_tile(r)
    tile_cols = tile_cols or _default_serial_tile(r)
    assert triton.next_power_of_2(tile_rows) == tile_rows
    assert triton.next_power_of_2(tile_cols) == tile_cols

    if workspace is None:
        workspace = TritonTiledWorkspace(Aw)
    else:
        workspace.validate(Aw)

    _setup_update_solve_kernel[(batch,)](
        L, z, w,
        workspace.vz, workspace.vw, workspace.cs, workspace.ss,
        workspace.prefix_c,
        r=r, BLOCK=BLOCK, STORE_PREFIX=False,
        num_warps=setup_num_warps,
    )
    _left_sweep_cols_batched_kernel[(triton.cdiv(r, tile_cols), batch)](
        Aw, workspace.aw_mid, workspace.cs, workspace.ss,
        r=r, TILE_COLS=tile_cols, num_warps=sweep_num_warps,
    )
    _right_sweep_rows_rank1_kernel[(triton.cdiv(r, tile_rows), batch)](
        workspace.aw_mid, workspace.aw_out,
        workspace.cs, workspace.ss, workspace.vz, workspace.vw,
        r=r, TILE_ROWS=tile_rows, num_warps=sweep_num_warps,
    )
    return workspace.aw_out, workspace.vz


def incremental_update_triton_scan(
    L,
    Aw,
    z,
    w,
    tile_rows=None,
    tile_cols=None,
    setup_num_warps=4,
    sweep_num_warps=8,
    workspace=None,
):
    """Three-launch update with parallel-prefix double-sided sweeps."""
    if L.ndim not in (2, 3):
        raise ValueError("L must have shape (r,r) or (T,r,r)")
    batch = 1 if L.ndim == 2 else L.shape[0]
    r = L.shape[-1]
    BLOCK = triton.next_power_of_2(r)
    assert BLOCK <= 1024, f"r={r} too large for single-block setup kernel"
    tile_rows = tile_rows or _default_scan_tile(r)
    tile_cols = tile_cols or _default_scan_tile(r)
    assert triton.next_power_of_2(tile_rows) == tile_rows
    assert triton.next_power_of_2(tile_cols) == tile_cols

    if workspace is None:
        workspace = TritonTiledWorkspace(Aw)
    else:
        workspace.validate(Aw)

    _setup_update_solve_kernel[(batch,)](
        L, z, w,
        workspace.vz, workspace.vw, workspace.cs, workspace.ss,
        workspace.prefix_c,
        r=r, BLOCK=BLOCK, STORE_PREFIX=False,
        num_warps=setup_num_warps,
    )
    _left_sweep_scan_kernel[(triton.cdiv(r, tile_cols), batch)](
        Aw, workspace.aw_mid, workspace.cs, workspace.ss,
        r=r, BLOCK_R=BLOCK, TILE_COLS=tile_cols,
        num_warps=sweep_num_warps,
    )
    _right_sweep_scan_rank1_kernel[(triton.cdiv(r, tile_rows), batch)](
        workspace.aw_mid, workspace.aw_out,
        workspace.cs, workspace.ss, workspace.vz, workspace.vw,
        r=r, BLOCK_R=BLOCK, TILE_ROWS=tile_rows,
        num_warps=sweep_num_warps,
    )
    return workspace.aw_out, workspace.vz


def incremental_update_triton_cumsum(
    L,
    Aw,
    z,
    w,
    tile_rows=None,
    tile_cols=None,
    setup_num_warps=4,
    sweep_num_warps=8,
    workspace=None,
):
    """Three-launch update using weighted cumulative-sum sweeps."""
    if L.ndim not in (2, 3):
        raise ValueError("L must have shape (r,r) or (T,r,r)")
    batch = 1 if L.ndim == 2 else L.shape[0]
    r = L.shape[-1]
    BLOCK = triton.next_power_of_2(r)
    assert BLOCK <= 1024, f"r={r} too large for single-block setup kernel"
    tile_rows = tile_rows or _default_scan_tile(r)
    tile_cols = tile_cols or _default_scan_tile(r)
    assert triton.next_power_of_2(tile_rows) == tile_rows
    assert triton.next_power_of_2(tile_cols) == tile_cols

    if workspace is None:
        workspace = TritonTiledWorkspace(Aw)
    else:
        workspace.validate(Aw)

    _setup_update_solve_kernel[(batch,)](
        L, z, w,
        workspace.vz, workspace.vw, workspace.cs, workspace.ss,
        workspace.prefix_c,
        r=r, BLOCK=BLOCK, STORE_PREFIX=True,
        num_warps=setup_num_warps,
    )
    _left_sweep_cumsum_kernel[(triton.cdiv(r, tile_cols), batch)](
        Aw, workspace.aw_mid, workspace.cs, workspace.ss,
        workspace.prefix_c,
        r=r, BLOCK_R=BLOCK, TILE_COLS=tile_cols,
        num_warps=sweep_num_warps,
    )
    _right_sweep_cumsum_rank1_kernel[
        (triton.cdiv(r, tile_rows), batch)
    ](
        workspace.aw_mid, workspace.aw_out,
        workspace.cs, workspace.ss, workspace.prefix_c,
        workspace.vz, workspace.vw,
        r=r, BLOCK_R=BLOCK, TILE_ROWS=tile_rows,
        num_warps=sweep_num_warps,
    )
    return workspace.aw_out, workspace.vz


def incremental_update_triton_parallel(
    L,
    Aw,
    z,
    w,
    tile_rows=None,
    tile_cols=None,
    setup_num_warps=4,
    sweep_num_warps=8,
    solve_width=2,
    solve_right=False,
    workspace=None,
):
    """Fully parallel large-rank update based on the factor of I+u*u^T.

    A single two-RHS triangular solve computes u=L^{-1}z and y=L^{-1}w.
    Closed-form prefix sums then update L, recover v_z and v_w, and feed the
    same parallel double-sided sweeps as ``incremental_update_triton_cumsum``.
    This removes the serial r-step Givens and forward-substitution loops that
    dominate the large-rank HBM fallback.
    """
    batch, r = _validate_cuda_update_inputs(L, Aw, z, w)
    tile_rows = tile_rows or _default_scan_tile(r)
    tile_cols = tile_cols or _default_scan_tile(r)
    BLOCK = _validate_prefix_kernel_config(r, tile_rows, tile_cols)

    if workspace is None:
        workspace = TritonTiledWorkspace(Aw, solve_width=solve_width)
    else:
        workspace.validate(Aw, solve_width=solve_width)

    _pack_solve_rhs_kernel[(batch,)](
        z,
        w,
        workspace.rhs,
        r=r,
        BLOCK=BLOCK,
        RHS_WIDTH=solve_width,
        num_warps=setup_num_warps,
    )
    if solve_right:
        torch.linalg.solve_triangular(
            L.mT,
            workspace.rhs.mT,
            upper=True,
            left=False,
            out=workspace.solve.mT,
        )
    else:
        torch.linalg.solve_triangular(
            L,
            workspace.rhs,
            upper=False,
            out=workspace.solve,
        )
    _parallel_update_params_kernel[(batch,)](
        workspace.solve,
        workspace.vz,
        workspace.vw,
        workspace.cs,
        workspace.ss,
        workspace.prefix_c,
        r=r,
        BLOCK=BLOCK,
        RHS_WIDTH=solve_width,
        num_warps=setup_num_warps,
    )
    _parallel_cholesky_update_kernel[
        (triton.cdiv(r, tile_rows), batch)
    ](
        L,
        workspace.solve,
        workspace.vz,
        workspace.cs,
        r=r,
        BLOCK_R=BLOCK,
        TILE_ROWS=tile_rows,
        RHS_WIDTH=solve_width,
        num_warps=sweep_num_warps,
    )
    _left_sweep_cumsum_kernel[(triton.cdiv(r, tile_cols), batch)](
        Aw,
        workspace.aw_mid,
        workspace.cs,
        workspace.ss,
        workspace.prefix_c,
        r=r,
        BLOCK_R=BLOCK,
        TILE_COLS=tile_cols,
        num_warps=sweep_num_warps,
    )
    _right_sweep_cumsum_rank1_kernel[
        (triton.cdiv(r, tile_rows), batch)
    ](
        workspace.aw_mid,
        workspace.aw_out,
        workspace.cs,
        workspace.ss,
        workspace.prefix_c,
        workspace.vz,
        workspace.vw,
        r=r,
        BLOCK_R=BLOCK,
        TILE_ROWS=tile_rows,
        num_warps=sweep_num_warps,
    )
    return workspace.aw_out, workspace.vz


def incremental_update_triton_inverse_state(
    L,
    Linv,
    Aw,
    z,
    w,
    tile_rows=None,
    tile_cols=None,
    setup_num_warps=4,
    sweep_num_warps=8,
    gemv_backend="matmul",
    workspace=None,
):
    """Parallel update that maintains a cached inverse Cholesky factor.

    The additional ``Linv`` state is modified in place to ``(L+)^{-1}``.
    Maintaining it replaces the latency-bound two-RHS triangular solve with
    two fused GEMVs plus one O(r^2) prefix-scan update.  Total complexity and
    storage remain O(r^2), with one additional triangular matrix of state.
    """
    batch, r = _validate_cuda_update_inputs(L, Aw, z, w)
    if Linv.shape != L.shape:
        raise ValueError("Linv must have the same shape as L")
    if (
        Linv.device != L.device
        or Linv.dtype != L.dtype
        or not Linv.is_contiguous()
    ):
        raise ValueError("Linv must match L's device, dtype, and layout")
    tile_rows = tile_rows or _default_scan_tile(r)
    tile_cols = tile_cols or _default_scan_tile(r)
    block = _validate_prefix_kernel_config(r, tile_rows, tile_cols)

    if workspace is None:
        workspace = TritonTiledWorkspace(Aw, solve_width=2)
    else:
        workspace.validate(Aw, solve_width=2)

    if gemv_backend == "triton":
        _inverse_gemv_pair_kernel[
            (triton.cdiv(r, tile_rows), batch)
        ](
            Linv,
            z,
            w,
            workspace.solve,
            r=r,
            BLOCK_R=block,
            TILE_ROWS=tile_rows,
            RHS_WIDTH=2,
            num_warps=sweep_num_warps,
        )
    elif gemv_backend == "matmul":
        _pack_solve_rhs_kernel[(batch,)](
            z,
            w,
            workspace.rhs,
            r=r,
            BLOCK=block,
            RHS_WIDTH=2,
            num_warps=setup_num_warps,
        )
        torch.matmul(Linv, workspace.rhs, out=workspace.solve)
    else:
        raise ValueError(f"unknown inverse GEMV backend: {gemv_backend}")
    _parallel_update_params_kernel[(batch,)](
        workspace.solve,
        workspace.vz,
        workspace.vw,
        workspace.cs,
        workspace.ss,
        workspace.prefix_c,
        r=r,
        BLOCK=block,
        RHS_WIDTH=2,
        num_warps=setup_num_warps,
    )
    _parallel_cholesky_update_kernel[
        (triton.cdiv(r, tile_rows), batch)
    ](
        L,
        workspace.solve,
        workspace.vz,
        workspace.cs,
        r=r,
        BLOCK_R=block,
        TILE_ROWS=tile_rows,
        RHS_WIDTH=2,
        num_warps=sweep_num_warps,
    )
    _inverse_update_and_left_sweep_kernel[
        (triton.cdiv(r, tile_cols), batch)
    ](
        Linv,
        Aw,
        workspace.aw_mid,
        workspace.solve,
        workspace.cs,
        workspace.ss,
        workspace.prefix_c,
        r=r,
        BLOCK_R=block,
        TILE_COLS=tile_cols,
        RHS_WIDTH=2,
        num_warps=sweep_num_warps,
    )
    _right_sweep_cumsum_rank1_kernel[
        (triton.cdiv(r, tile_rows), batch)
    ](
        workspace.aw_mid,
        workspace.aw_out,
        workspace.cs,
        workspace.ss,
        workspace.prefix_c,
        workspace.vz,
        workspace.vw,
        r=r,
        BLOCK_R=block,
        TILE_ROWS=tile_rows,
        num_warps=sweep_num_warps,
    )
    return workspace.aw_out, workspace.vz


def incremental_update_triton_inverse_factor(
    Linv,
    Aw,
    z,
    w,
    tile_rows=None,
    tile_cols=None,
    setup_num_warps=4,
    sweep_num_warps=8,
    gemv_backend="matmul",
    workspace=None,
):
    """Experimental low-latency state: maintain L^{-1} instead of L.

    ``Linv`` is modified in place to ``(L+)^{-1}``. It must be a row-major,
    contiguous lower-triangular matrix with an explicitly zero upper
    triangle; initialize it as ``torch.linalg.inv(L).tril().contiguous()``.
    Since future rank-one
    updates require only L^{-1}z and L^{-1}w, explicitly materializing L+ is
    unnecessary between refreshes.  This keeps O(r^2) factor storage while
    removing the separate L+ matrix scan from the compatibility path.

    The single-block inverse scan is optimized for latency and accumulates
    more FP32 state error than the full-axis implementation below.  Use it
    only with periodic exact refreshes; use
    ``incremental_update_triton_inverse_factor_large`` when the cached inverse
    itself must remain tightly matched over long update sequences.
    """
    batch, r = _validate_cuda_update_inputs(
        Linv,
        Aw,
        z,
        w,
        factor_name="Linv",
    )
    tile_rows = tile_rows or _default_scan_tile(r)
    tile_cols = tile_cols or _default_scan_tile(r)
    block = _validate_prefix_kernel_config(r, tile_rows, tile_cols)

    if workspace is None:
        workspace = TritonTiledWorkspace(Aw, solve_width=2)
    else:
        workspace.validate(Aw, solve_width=2)

    if gemv_backend == "triton":
        _inverse_gemv_pair_kernel[
            (triton.cdiv(r, tile_rows), batch)
        ](
            Linv,
            z,
            w,
            workspace.solve,
            r=r,
            BLOCK_R=block,
            TILE_ROWS=tile_rows,
            RHS_WIDTH=2,
            num_warps=sweep_num_warps,
        )
    elif gemv_backend == "matmul":
        _pack_solve_rhs_kernel[(batch,)](
            z,
            w,
            workspace.rhs,
            r=r,
            BLOCK=block,
            RHS_WIDTH=2,
            num_warps=setup_num_warps,
        )
        torch.matmul(Linv, workspace.rhs, out=workspace.solve)
    else:
        raise ValueError(f"unknown inverse GEMV backend: {gemv_backend}")
    _inverse_update_left_params_kernel[
        (triton.cdiv(r, tile_cols), batch)
    ](
        Linv,
        Aw,
        workspace.aw_mid,
        workspace.solve,
        workspace.cs,
        workspace.ss,
        workspace.prefix_c,
        workspace.vz,
        workspace.vw,
        r=r,
        BLOCK_R=block,
        TILE_COLS=tile_cols,
        RHS_WIDTH=2,
        num_warps=sweep_num_warps,
    )
    _right_sweep_cumsum_rank1_kernel[
        (triton.cdiv(r, tile_rows), batch)
    ](
        workspace.aw_mid,
        workspace.aw_out,
        workspace.cs,
        workspace.ss,
        workspace.prefix_c,
        workspace.vz,
        workspace.vw,
        r=r,
        BLOCK_R=block,
        TILE_ROWS=tile_rows,
        num_warps=sweep_num_warps,
    )
    return workspace.aw_out, workspace.vz


def incremental_update_triton_inverse_factor_large(
    Linv,
    Aw,
    z,
    w,
    workspace=None,
    block_size=256,
):
    """Arbitrary-rank inverse-factor update using full-axis CUDA scans.

    This path removes the single-Triton-block ``r <= 1024`` restriction.
    It uses the same exact identities as
    :func:`incremental_update_triton_inverse_factor`, but delegates the long
    prefix sums to ``torch.cumsum`` and fuses all surrounding elementwise
    work in Triton kernels.

    To hold peak storage to the factor, the operator, and two workspace
    matrices, the final right sweep overwrites ``Aw`` and returns it.  Callers
    must therefore treat ``Aw`` as mutable state.
    """
    batch, r = _validate_cuda_update_inputs(
        Linv,
        Aw,
        z,
        w,
        factor_name="Linv",
    )
    if (
        block_size <= 0
        or block_size > 1024
        or triton.next_power_of_2(block_size) != block_size
    ):
        raise ValueError("block_size must be a power of two in [1, 1024]")

    if workspace is None:
        workspace = TritonTiledWorkspace(Aw, solve_width=2)
    else:
        workspace.validate(Aw, solve_width=2)

    vector_total = batch * r
    matrix_total = batch * r * r
    vector_grid = (triton.cdiv(vector_total, block_size),)
    matrix_grid = (triton.cdiv(matrix_total, block_size),)

    _large_pack_rhs_kernel[vector_grid](
        z,
        w,
        workspace.rhs,
        total=vector_total,
        r=r,
        RHS_WIDTH=2,
        BLOCK=block_size,
        num_warps=4,
    )
    torch.matmul(Linv, workspace.rhs, out=workspace.solve)
    _large_vector_scan_terms_kernel[vector_grid](
        workspace.solve,
        workspace.cs,
        workspace.ss,
        total=vector_total,
        r=r,
        RHS_WIDTH=2,
        BLOCK=block_size,
        num_warps=4,
    )
    torch.cumsum(workspace.cs, dim=-1, out=workspace.cs)
    torch.cumsum(workspace.ss, dim=-1, out=workspace.ss)
    _large_parameter_finalize_kernel[vector_grid](
        workspace.solve,
        workspace.cs,
        workspace.ss,
        workspace.cs,
        workspace.ss,
        workspace.prefix_c,
        workspace.vz,
        workspace.vw,
        total=vector_total,
        r=r,
        RHS_WIDTH=2,
        BLOCK=block_size,
        num_warps=4,
    )

    # Update the cached inverse factor: scan down each column.
    _large_matrix_scan_term_kernel[matrix_grid](
        Linv,
        workspace.aw_mid,
        workspace.solve,
        workspace.cs,
        workspace.ss,
        workspace.prefix_c,
        total=matrix_total,
        matrix_size=r * r,
        r=r,
        MODE=0,
        RHS_WIDTH=2,
        BLOCK=block_size,
        num_warps=4,
    )
    torch.cumsum(workspace.aw_mid, dim=-2, out=workspace.aw_mid)
    _large_matrix_scan_finalize_kernel[matrix_grid](
        Linv,
        workspace.aw_mid,
        Linv,
        workspace.solve,
        workspace.cs,
        workspace.ss,
        workspace.prefix_c,
        workspace.vz,
        workspace.vw,
        total=matrix_total,
        matrix_size=r * r,
        r=r,
        MODE=0,
        RHS_WIDTH=2,
        BLOCK=block_size,
        num_warps=4,
    )

    # Left weighted sweep.
    _large_matrix_scan_term_kernel[matrix_grid](
        Aw,
        workspace.aw_mid,
        workspace.solve,
        workspace.cs,
        workspace.ss,
        workspace.prefix_c,
        total=matrix_total,
        matrix_size=r * r,
        r=r,
        MODE=1,
        RHS_WIDTH=2,
        BLOCK=block_size,
        num_warps=4,
    )
    torch.cumsum(workspace.aw_mid, dim=-2, out=workspace.aw_mid)
    _large_matrix_scan_finalize_kernel[matrix_grid](
        Aw,
        workspace.aw_mid,
        workspace.aw_out,
        workspace.solve,
        workspace.cs,
        workspace.ss,
        workspace.prefix_c,
        workspace.vz,
        workspace.vw,
        total=matrix_total,
        matrix_size=r * r,
        r=r,
        MODE=1,
        RHS_WIDTH=2,
        BLOCK=block_size,
        num_warps=4,
    )

    # Right weighted sweep plus rank-one correction.  Aw is no longer read
    # after this point, so it is the final output buffer.
    _large_matrix_scan_term_kernel[matrix_grid](
        workspace.aw_out,
        workspace.aw_mid,
        workspace.solve,
        workspace.cs,
        workspace.ss,
        workspace.prefix_c,
        total=matrix_total,
        matrix_size=r * r,
        r=r,
        MODE=2,
        RHS_WIDTH=2,
        BLOCK=block_size,
        num_warps=4,
    )
    torch.cumsum(workspace.aw_mid, dim=-1, out=workspace.aw_mid)
    _large_matrix_scan_finalize_kernel[matrix_grid](
        workspace.aw_out,
        workspace.aw_mid,
        Aw,
        workspace.solve,
        workspace.cs,
        workspace.ss,
        workspace.prefix_c,
        workspace.vz,
        workspace.vw,
        total=matrix_total,
        matrix_size=r * r,
        r=r,
        MODE=2,
        RHS_WIDTH=2,
        BLOCK=block_size,
        num_warps=4,
    )
    return Aw, workspace.vz


def incremental_update_triton_inverse_factor_inplace(
    Linv,
    Aw,
    z,
    w,
    tile_rows=8,
    tile_cols=8,
    setup_num_warps=4,
    sweep_num_warps=4,
    workspace=None,
    block_size=256,
):
    """Arbitrary-rank inverse-factor update with only two dense matrices.

    ``Linv`` and ``Aw`` are both updated in place.  Unlike
    :func:`incremental_update_triton_inverse_factor_large`, this path avoids
    dense prefix-scan buffers by assigning one serial recurrence to each
    independent row or column.  The recurrences run concurrently across all
    rows/columns, retaining O(r^2) work while reducing dense storage to the
    two caller-owned state matrices.

    ``Linv`` must be row-major, contiguous, explicitly lower triangular, and
    have a zero upper triangle.  A valid initialization is
    ``torch.linalg.inv(L).tril().contiguous()``.
    """
    batch, r = _validate_cuda_update_inputs(
        Linv,
        Aw,
        z,
        w,
        factor_name="Linv",
    )
    if (
        block_size <= 0
        or block_size > 1024
        or triton.next_power_of_2(block_size) != block_size
    ):
        raise ValueError("block_size must be a power of two in [1, 1024]")
    for name, value in (("tile_rows", tile_rows), ("tile_cols", tile_cols)):
        if (
            value <= 0
            or value > 64
            or triton.next_power_of_2(value) != value
        ):
            raise ValueError(f"{name} must be a power of two in [1, 64]")
    if setup_num_warps not in (1, 2, 4, 8):
        raise ValueError("setup_num_warps must be one of 1, 2, 4, or 8")
    if sweep_num_warps not in (1, 2, 4, 8):
        raise ValueError("sweep_num_warps must be one of 1, 2, 4, or 8")

    if workspace is None:
        workspace = TritonInverseInplaceWorkspace(Aw)
    else:
        workspace.validate(Aw)

    vector_total = batch * r
    vector_grid = (triton.cdiv(vector_total, block_size),)
    _large_pack_rhs_kernel[vector_grid](
        z,
        w,
        workspace.rhs,
        total=vector_total,
        r=r,
        RHS_WIDTH=2,
        BLOCK=block_size,
        num_warps=setup_num_warps,
    )
    torch.matmul(Linv, workspace.rhs, out=workspace.solve)
    _large_vector_scan_terms_kernel[vector_grid](
        workspace.solve,
        workspace.cs,
        workspace.ss,
        total=vector_total,
        r=r,
        RHS_WIDTH=2,
        BLOCK=block_size,
        num_warps=setup_num_warps,
    )
    torch.cumsum(workspace.cs, dim=-1, out=workspace.cs)
    torch.cumsum(workspace.ss, dim=-1, out=workspace.ss)
    _large_parameter_finalize_kernel[vector_grid](
        workspace.solve,
        workspace.cs,
        workspace.ss,
        workspace.cs,
        workspace.ss,
        workspace.prefix_c,
        workspace.vz,
        workspace.vw,
        total=vector_total,
        r=r,
        RHS_WIDTH=2,
        BLOCK=block_size,
        num_warps=setup_num_warps,
    )

    matrix_size = r * r
    _inplace_inverse_left_sweep_kernel[
        (triton.cdiv(r, tile_cols), batch)
    ](
        Linv,
        Aw,
        workspace.solve,
        workspace.cs,
        workspace.ss,
        workspace.prefix_c,
        r=r,
        matrix_size=matrix_size,
        RHS_WIDTH=2,
        TILE_COLS=tile_cols,
        num_warps=sweep_num_warps,
    )
    _inplace_right_sweep_rank1_kernel[
        (triton.cdiv(r, tile_rows), batch)
    ](
        Aw,
        workspace.cs,
        workspace.ss,
        workspace.vz,
        workspace.vw,
        r=r,
        matrix_size=matrix_size,
        TILE_ROWS=tile_rows,
        num_warps=sweep_num_warps,
    )
    return Aw, workspace.vz


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


def incremental_update_triton_tiled_legacy(
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
    """Original five-launch multi-block Triton update kept for benchmarking.

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


def incremental_update_triton_tiled(
    L,
    Aw,
    z,
    w,
    block_m=16,
    block_n=64,
    num_warps=None,
    *,
    tile_rows=None,
    tile_cols=None,
    sweep_num_warps=None,
    workspace=None,
):
    """Fully parallel HBM-resident fallback under the original public name.

    ``block_m`` and ``block_n`` remain accepted for source compatibility but
    are obsolete.  The implementation is algebraically equivalent to the
    serial Givens path, but computes all update coefficients and matrix sweeps
    with parallel prefix sums.  Use ``workspace`` to eliminate steady-state
    allocations; returned tensors then alias that workspace and must be cloned
    if they need to outlive the next update.
    """
    del block_m, block_n
    _validate_cuda_update_inputs(L, Aw, z, w)
    (
        default_rows,
        default_cols,
        default_setup_warps,
        default_sweep_warps,
    ) = _default_parallel_config(L.shape[-1])
    return incremental_update_triton_parallel(
        L,
        Aw,
        z,
        w,
        tile_rows=default_rows if tile_rows is None else tile_rows,
        tile_cols=default_cols if tile_cols is None else tile_cols,
        setup_num_warps=(
            default_setup_warps if num_warps is None else num_warps
        ),
        sweep_num_warps=(
            default_sweep_warps
            if sweep_num_warps is None
            else sweep_num_warps
        ),
        workspace=workspace,
    )


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
