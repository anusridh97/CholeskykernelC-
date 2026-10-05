"""CPU factor-gauge updates. Inputs are CPU float32/float64 of one dtype."""
import torch

def update(L: torch.Tensor, Aw: torch.Tensor, z: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Update row-major L (r,r) in place; return new Aw/vz; preserve Aw, z and w."""
    from . import _C_cpu
    return _C_cpu.fast_incremental_update(L, Aw, z, w)

def update_batched(L: torch.Tensor, Aw: torch.Tensor, z: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Update row-major L (T,r,r) in place with z/w (T,r); preserve Aw, z and w."""
    from . import _C_cpu
    return _C_cpu.fast_incremental_update_batched(L, Aw, z, w)
