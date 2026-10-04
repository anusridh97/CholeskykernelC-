"""Original arbitrary-rank inverse-gauge GPU path (including batches)."""
from dataclasses import dataclass
import torch
import triton
import triton.language as tl
from .._validation import check_inverse_state
from ._vector_kernels import _pack_rhs2, _scan_terms2, _finalize_params2
@dataclass(frozen=True)
class InplaceConfig:
    """Serial recurrence tile width and Triton launch warp counts."""
    tile: int = 8
    setup_num_warps: int = 4
    sweep_num_warps: int = 4
    @classmethod
    def for_rank(cls, rank: int):
        """Return the archived rank-ladder configuration."""
        if rank < 1: raise ValueError("rank must be positive")
        return cls(8 if rank <= 8192 else 16 if rank <= 16384 else 32, 4, 1)
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


class InplaceWorkspace:
    """Vector-only workspace for the two-matrix inverse-factor path.

    The dense state consists only of the caller-owned ``Linv`` and ``Aw``.
    Returned ``vz`` aliases this workspace and is overwritten on reuse.
    """

    def __init__(self, like):
        if like.ndim not in (2, 3) or like.shape[-2] != like.shape[-1]:
            raise ValueError("workspace prototype must contain square matrices")
        if not like.is_cuda or like.dtype != torch.float32 or not like.is_contiguous():
            raise ValueError("workspace requires contiguous CUDA FP32")
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


def _update_inplace(
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
    batch, r = check_inverse_state(
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
        workspace = InplaceWorkspace(Aw)
    else:
        workspace.validate(Aw)

    vector_total = batch * r
    vector_grid = (triton.cdiv(vector_total, block_size),)
    _pack_rhs2[vector_grid](
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
    _scan_terms2[vector_grid](
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
    _finalize_params2[vector_grid](
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


@torch.no_grad()
def update_inplace(Linv: torch.Tensor, Aw: torch.Tensor, z: torch.Tensor, w: torch.Tensor, *, workspace: InplaceWorkspace | None = None, config: InplaceConfig | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Update dense (r,r)/(T,r,r) Linv/Aw in place; vz aliases workspace.

    Linv must have zero upper triangle. Disable TF32. Vectors are (r,)/(T,r).
    """
    config = config or InplaceConfig()
    return _update_inplace(Linv, Aw, z, w, tile_rows=config.tile, tile_cols=config.tile, setup_num_warps=config.setup_num_warps, sweep_num_warps=config.sweep_num_warps, workspace=workspace)
