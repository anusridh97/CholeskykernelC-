"""Rotation parameters for dense two-column or single-RHS vectors."""
import triton
import triton.language as tl
@triton.jit
def _pack_rhs2(
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
def _scan_terms2(
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
def _finalize_params2(
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
def _u2_terms1(
    solve_ptr,
    u2_ptr,
    total,
    BLOCK: tl.constexpr,
):
    offsets = (
        tl.program_id(0).to(tl.int64) * BLOCK
        + tl.arange(0, BLOCK).to(tl.int64)
    )
    mask = offsets < total
    u = tl.load(solve_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(u2_ptr + offsets, u * u, mask=mask)


@triton.jit
def _finalize_params1(
    solve_ptr,
    cumulative_u2_ptr,
    cs_ptr,
    ss_ptr,
    prefix_c_ptr,
    vz_ptr,
    total,
    BLOCK: tl.constexpr,
):
    offsets = (
        tl.program_id(0).to(tl.int64) * BLOCK
        + tl.arange(0, BLOCK).to(tl.int64)
    )
    mask = offsets < total
    u = tl.load(solve_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    cumulative_u2 = tl.load(
        cumulative_u2_ptr + offsets,
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
    tl.store(cs_ptr + offsets, c, mask=mask)
    tl.store(ss_ptr + offsets, s, mask=mask)
    tl.store(prefix_c_ptr + offsets, prefix_c, mask=mask)
    tl.store(vz_ptr + offsets, vz, mask=mask)

