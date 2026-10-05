"""B200 full-matrix asymmetric inverse-factor update.

The geometry update remains FP32 and dense-compatible.  A_w uses the exact
descriptor transport with 64x65 shared-memory tiles to avoid bank conflicts.
"""
import torch
import triton
import triton.language as tl
from .._validation import check_inverse_state, check_cuda_tensors
from ._vector_kernels import _pack_rhs2, _finalize_params2, _scan_terms2

@triton.jit
def _inverse_only_update(p, solve, c, prefix_c, r, matrix_size, RHS_WIDTH: tl.constexpr, TILE_COLS: tl.constexpr):
    """Update dense row-major inverse factor (r,r) from two-RHS solve (r,2)."""
    pid = tl.program_id(0)
    cols = pid.to(tl.int64) * TILE_COLS + tl.arange(0, TILE_COLS).to(tl.int64)
    col_mask = cols < r
    dot = tl.zeros((TILE_COLS,), tl.float32)
    for row in tl.range(0, r, num_stages=1):
        row64 = row.to(tl.int64)
        offsets = row64 * r + cols
        mask = col_mask & (cols <= row64)
        value = tl.load(p + offsets, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(solve + row64 * RHS_WIDTH).to(tl.float32)
        cr = tl.load(c + row64).to(tl.float32)
        pc = tl.load(prefix_c + row64).to(tl.float32)
        tl.store(p + offsets, cr * (value - u * dot * pc * pc), mask=mask)
        dot += tl.where(mask, u * value, 0.0)

class TiledWorkspace:
    """Two (r,2) buffers, seven vectors, and four (n*n,64) descriptor arrays."""

    def __init__(self, like: torch.Tensor, tile=64) -> None:
        """Initialize workspace buffers matching the documented state shape."""
        if like.ndim != 2 or like.shape[0] != like.shape[1]:
            raise ValueError('B200 asymmetric workspace requires one square matrix')
        if like.dtype != torch.float32 or not like.is_cuda or (not like.is_contiguous()):
            raise ValueError('A_w must be contiguous CUDA FP32')
        if tile != 64:
            raise ValueError('tile size is fixed at 64')
        r = like.shape[0]
        if r < 1:
            raise ValueError('rank must be positive')
        n = (r + tile - 1) // tile
        nt = n * n
        self.rank, self.device, self.dtype = (r, like.device, like.dtype)
        self.rhs = torch.empty((r, 2), device=like.device, dtype=like.dtype)
        self.solve = torch.empty_like(self.rhs)
        self.cs = torch.empty(r, device=like.device, dtype=like.dtype)
        self.ss = torch.empty_like(self.cs)
        self.prefix_c = torch.empty_like(self.cs)
        self.vz = torch.empty_like(self.cs)
        self.vw = torch.empty_like(self.cs)
        self.h = torch.empty_like(self.cs)
        self.d = torch.empty_like(self.cs)
        self.south = torch.empty(nt * tile, device=like.device, dtype=like.dtype)
        self.east0 = torch.empty_like(self.south)
        self.north = torch.empty_like(self.south)
        self.west = torch.empty_like(self.south)

    @property
    def nbytes(self) -> int:
        """Return exact allocated workspace bytes."""
        return sum((v.numel() * v.element_size() for v in self.__dict__.values() if isinstance(v, torch.Tensor)))

    def validate(self, like: torch.Tensor):
        """Validate workspace shape, dtype and device."""
        if like.ndim != 2 or like.shape != (self.rank, self.rank) or like.device != self.device or (like.dtype != self.dtype) or (not like.is_contiguous()):
            raise ValueError('workspace does not match A_w')
        n = (self.rank + 63) // 64
        for name in ('rhs', 'solve', 'cs', 'ss', 'prefix_c', 'vz', 'vw', 'h', 'd', 'south', 'east0', 'north', 'west'):
            tensor = getattr(self, name)
            shape = (self.rank, 2) if name in ('rhs', 'solve') else (n * n * 64,) if name in ('south', 'east0', 'north', 'west') else (self.rank,)
            check_cuda_tensors(like, tensor)
            if tensor.shape != shape:
                raise ValueError('workspace buffer shape mismatch')

def _update_tiled(Linv: torch.Tensor, Aw: torch.Tensor, z: torch.Tensor, w: torch.Tensor, tile_rows=8, tile_cols=8, setup_num_warps=4, sweep_num_warps=4, workspace=None, block_size=256):
    """Update dense row-major ``Linv`` and asymmetric ``Aw`` in place.

    Extra compatibility keywords match the frozen black-hole driver.  The
    A_w tile size is fixed at 64; ``tile_cols`` still controls the independent
    inverse-factor column group.
    """
    from .. import _C
    check_inverse_state(Linv, Aw, z, w, factor_name='Linv')
    if Linv.ndim != 2 or Linv.shape != Aw.shape or Aw.ndim != 2 or (Linv.shape[0] != Linv.shape[1]):
        raise ValueError('Linv and Aw must be matching square matrices')
    if Linv.dtype != torch.float32 or Aw.dtype != torch.float32 or (not Linv.is_cuda) or (not Aw.is_cuda) or (not Linv.is_contiguous()) or (not Aw.is_contiguous()):
        raise ValueError('Linv and Aw must be contiguous CUDA FP32')
    r = Linv.shape[0]
    if z.shape != (r,) or w.shape != (r,):
        raise ValueError('z and w must be rank-length vectors')
    workspace = workspace or TiledWorkspace(Aw)
    workspace.validate(Aw)
    total = r
    grid = (triton.cdiv(total, block_size),)
    _pack_rhs2[grid](z, w, workspace.rhs, total=total, r=r, RHS_WIDTH=2, BLOCK=block_size, num_warps=setup_num_warps)
    torch.matmul(Linv, workspace.rhs, out=workspace.solve)
    _scan_terms2[grid](workspace.solve, workspace.cs, workspace.ss, total=total, r=r, RHS_WIDTH=2, BLOCK=block_size, num_warps=setup_num_warps)
    torch.cumsum(workspace.cs, dim=-1, out=workspace.cs)
    torch.cumsum(workspace.ss, dim=-1, out=workspace.ss)
    _finalize_params2[grid](workspace.solve, workspace.cs, workspace.ss, workspace.cs, workspace.ss, workspace.prefix_c, workspace.vz, workspace.vw, total=total, r=r, RHS_WIDTH=2, BLOCK=block_size, num_warps=setup_num_warps)
    _inverse_only_update[triton.cdiv(r, tile_cols),](Linv, workspace.solve, workspace.cs, workspace.prefix_c, r, r * r, RHS_WIDTH=2, TILE_COLS=tile_cols, num_warps=sweep_num_warps)
    torch.mul(workspace.ss, workspace.prefix_c, out=workspace.h)
    torch.mul(workspace.prefix_c, workspace.cs, out=workspace.d)
    torch.div(workspace.ss, workspace.d, out=workspace.d)
    workspace.d.neg_()
    _C.asym_descriptor(Aw, workspace.cs, workspace.h, workspace.d, workspace.vz, workspace.vw, workspace.south, workspace.east0, workspace.north, workspace.west, r)
    return (Aw, workspace.vz)

@torch.no_grad()
def update_tiled(Linv: torch.Tensor, Aw: torch.Tensor, z: torch.Tensor, w: torch.Tensor, *, workspace: TiledWorkspace | None=None, p_tile_cols: int=8, p_num_warps: int=4) -> tuple[torch.Tensor, torch.Tensor]:
    """Update dense (r,r) inverse-gauge state in place; z/w are (r,); vz aliases workspace.

    CUDA FP32 only, zero upper triangle in Linv, TF32 disabled; no batching.
    """
    if p_tile_cols not in (1, 2, 4, 8, 16, 32, 64) or p_num_warps not in (1, 2, 4, 8):
        raise ValueError('invalid inverse-factor tile width or warp count')
    return _update_tiled(Linv, Aw, z, w, workspace=workspace, tile_cols=p_tile_cols, sweep_num_warps=p_num_warps)
