"""64x64 lower-tile and row-packed storage with bounded-memory torch actions."""
import torch
TILE = 64

def tile_count(rank: int) -> int:
    """Number of padded 64-element blocks."""
    if not isinstance(rank, int) or rank < 1:
        raise ValueError('rank must be positive')
    return (rank + 63) // 64

def tile_elements(rank: int) -> int:
    """FP32 element count including padding in lower-tile storage."""
    n = tile_count(rank)
    return n * (n + 1) // 2 * 4096

def tile_diagonal_offsets(rank: int, device='cuda') -> torch.Tensor:
    """Int64 offsets of the logical diagonal."""
    i = torch.arange(rank, device=device, dtype=torch.int64)
    t = i // 64
    q = i % 64
    return (t * (t + 1) // 2 + t) * 4096 + q * 64 + q

def packed_size(rank: int) -> int:
    """Element count of row-packed lower storage."""
    tile_count(rank)
    return rank * (rank + 1) // 2

def _check_dense(matrix: torch.Tensor) -> int:
    """Validate one square contiguous CUDA FP32 matrix."""
    from ._validation import check_cuda_tensors
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError('expected nonempty square matrix')
    check_cuda_tensors(matrix)
    return matrix.shape[0]

def dense_to_tile_lower(matrix: torch.Tensor, lower_triangular: bool=False) -> torch.Tensor:
    """Pack (r,r) CUDA FP32 into tile-lower storage; padding is zero."""
    r = _check_dense(matrix)
    from .cuda._layout_kernels import _dense_to_tile
    source = matrix.tril().contiguous() if lower_triangular else matrix
    out = torch.zeros(tile_elements(r), device=matrix.device, dtype=matrix.dtype)
    _dense_to_tile[tile_count(r), tile_count(r), 8](source, out, r, tile_count(r), r * r, B=64, STRIPE=8, num_warps=8)
    return out

def tile_lower_to_dense(tiles: torch.Tensor, rank: int, lower_triangular: bool=False) -> torch.Tensor:
    """Unpack lower tiles to symmetric (r,r), or zero-upper-triangle factor."""
    from ._validation import check_cuda_tensors
    if tiles.shape != (tile_elements(rank),):
        raise ValueError('incorrect tile storage length')
    check_cuda_tensors(tiles)
    from .cuda._layout_kernels import _tile_to_dense
    out = torch.empty((rank, rank), device=tiles.device, dtype=tiles.dtype)
    _tile_to_dense[tile_count(rank), tile_count(rank), 8](tiles, out, rank, tile_count(rank), rank * rank, B=64, STRIPE=8, num_warps=8)
    return out.tril_() if lower_triangular else out

def pack_symmetric_lower(matrix: torch.Tensor) -> torch.Tensor:
    """Pack symmetric CUDA FP32 (r,r) into row-packed lower storage."""
    r = _check_dense(matrix)
    from .cuda._layout_kernels import _pack_lower
    out = torch.empty(packed_size(r), device=matrix.device, dtype=matrix.dtype)
    _pack_lower[r, (r + 255) // 256, 1](matrix, out, r=r, matrix_size=r * r, pn=packed_size(r), BLOCK=256, num_warps=4)
    return out

def unpack_symmetric_lower(packed: torch.Tensor, rank: int) -> torch.Tensor:
    """Unpack row-packed CUDA FP32 into symmetric (r,r)."""
    from ._validation import check_cuda_tensors
    if packed.shape != (packed_size(rank),):
        raise ValueError('incorrect packed storage length')
    check_cuda_tensors(packed)
    from .cuda._layout_kernels import _unpack_lower
    out = torch.empty((rank, rank), device=packed.device, dtype=packed.dtype)
    _unpack_lower[rank, (rank + 255) // 256, 1](packed, out, r=rank, matrix_size=rank * rank, pn=packed_size(rank), BLOCK=256, num_warps=4)
    return out

def tile_lower_matmul(tiles: torch.Tensor, X: torch.Tensor, rank: int, symmetric: bool, *, transpose: bool=False) -> torch.Tensor:
    """Apply tile state to (r,k) probes using O(64*r) scratch on CPU/CUDA.

    For a factor, diagonal tiles are treated as lower triangular. transpose=True
    applies the factor transpose without allocating a dense matrix.
    """
    if tiles.shape != (tile_elements(rank),) or X.ndim != 2 or X.shape[0] != rank:
        raise ValueError('incorrect tile/probe shapes')
    out = torch.zeros_like(X)
    for i in range(tile_count(rank)):
        start = i * 64
        end = min(rank, start + 64)
        first = i * (i + 1) // 2 * 4096
        strip = tiles[first:first + (i + 1) * 4096].view(i + 1, 64, 64).permute(1, 0, 2).reshape(64, -1)[:end - start, :end].clone()
        if not symmetric:
            strip[:, start:end] = strip[:, start:end].tril()
        if transpose and (not symmetric):
            out[:end] += strip.T @ X[start:end]
        else:
            out[start:end] += strip @ X[:end]
            if symmetric and start:
                out[:start] += strip[:, :start].T @ X[start:end]
    return out

def packed_symmetric_matmul(packed: torch.Tensor, X: torch.Tensor, rank: int, block_rows: int=1024) -> torch.Tensor:
    """Apply row-packed symmetric state to (r,k) probes using row-block scratch."""
    if packed.shape != (packed_size(rank),) or X.ndim != 2 or X.shape[0] != rank or (block_rows < 1):
        raise ValueError('incorrect packed/probe shapes or block_rows')
    out = torch.zeros_like(X)
    for start in range(0, rank, block_rows):
        end = min(rank, start + block_rows)
        rows = torch.arange(start, end, device=packed.device, dtype=torch.int64)[:, None]
        cols = torch.arange(end, device=packed.device, dtype=torch.int64)[None, :]
        mask = cols <= rows
        offsets = torch.where(mask, rows * (rows + 1) // 2 + cols, 0)
        block = torch.where(mask, packed[offsets], 0)
        out[start:end] += block @ X[:end]
        out[:start] += block[:, :start].T @ X[start:end]
        out[start:end] += block[:, start:end].tril(-1).T @ X[start:end]
    return out
