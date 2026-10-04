"""Double-sided Cholesky rank-one updates; GPU modules load on demand."""
from importlib import import_module
from . import reference, cpu
__version__ = "0.1.0"
__all__ = ["reference", "cpu", "cuda", "layouts", "__version__"]
def __getattr__(name: str):
    """Load optional CUDA and layout modules without requiring Triton on CPU."""
    if name in ("cuda", "layouts"):
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(name)
