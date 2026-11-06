"""Double-sided Cholesky rank-one updates; GPU modules load on demand."""
from importlib import import_module
from . import reference, cpu
__version__ = '0.1.0'
__all__ = ['reference', 'cpu', 'cuda', 'mlx', 'layouts', '__version__']

def __getattr__(name: str):
    """Load optional backends and layouts only when requested."""
    if name in ('cuda', 'mlx', 'layouts'):
        module = import_module(f'{__name__}.{name}')
        globals()[name] = module
        return module
    raise AttributeError(name)
