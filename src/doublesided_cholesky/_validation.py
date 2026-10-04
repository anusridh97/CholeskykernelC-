"""Validation before kernel launches."""
import torch

def check_inverse_state(factor: torch.Tensor, Aw: torch.Tensor, z: torch.Tensor, w: torch.Tensor, factor_name='L'):
    """Validate the layout contract shared by optimized CUDA entry points."""
    tensors = {factor_name: factor, 'Aw': Aw, 'z': z, 'w': w}
    if factor.ndim not in (2, 3):
        raise ValueError(f'{factor_name} must have shape (r,r) or (T,r,r)')
    if factor.shape[-2] != factor.shape[-1]:
        raise ValueError(f'{factor_name} must be square')
    if Aw.shape != factor.shape:
        raise ValueError(f'Aw must have the same shape as {factor_name}')
    batch = 1 if factor.ndim == 2 else factor.shape[0]
    r = factor.shape[-1]
    if batch <= 0 or r <= 0:
        raise ValueError('batch size and rank must be positive')
    expected_vector_shape = (r,) if factor.ndim == 2 else (batch, r)
    if z.shape != expected_vector_shape:
        raise ValueError(f'z must have shape {expected_vector_shape}')
    if w.shape != expected_vector_shape:
        raise ValueError(f'w must have shape {expected_vector_shape}')
    for name, tensor in tensors.items():
        if tensor.device != factor.device:
            raise ValueError(f'{name} must be on the same device as {factor_name}')
        if tensor.dtype != factor.dtype:
            raise ValueError(f'{name} must have the same dtype as {factor_name}')
        if not tensor.is_contiguous():
            raise ValueError(f'{name} must be contiguous')
    if factor.device.type != 'cuda':
        raise ValueError('optimized Triton updates require CUDA tensors')
    if factor.dtype != torch.float32:
        raise ValueError('optimized Triton updates currently require FP32')
    return (batch, r)

def check_cuda_tensors(*tensors: torch.Tensor) -> None:
    """Require contiguous FP32 CUDA tensors on one device."""
    for tensor in tensors:
        if not tensor.is_cuda or tensor.dtype != torch.float32 or (not tensor.is_contiguous()) or (tensor.device != tensors[0].device):
            raise ValueError('all tensors must be contiguous FP32 CUDA on the same device')

def check_tiled_symmetric(P: torch.Tensor, A: torch.Tensor, z: torch.Tensor, rank: int) -> None:
    """Validate single-state tile-lower P/A and dense z (rank,)."""
    from .layouts import tile_elements
    if not isinstance(rank, int) or rank < 1:
        raise ValueError('rank must be a positive integer')
    if P.shape != (tile_elements(rank),) or A.shape != P.shape or z.shape != (rank,):
        raise ValueError('tile buffers or z have incorrect shape')
    check_cuda_tensors(P, A, z)
