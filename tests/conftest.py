"""Shared inputs and explicit hardware gates."""
import importlib.util
import os
import pytest
import torch

def pytest_collection_modifyitems(items):
    """Skip GPU tests before loading GPU modules on CPU installations."""
    for item in items:
        if 'cuda' in item.keywords and (not torch.cuda.is_available() or importlib.util.find_spec('triton') is None):
            item.add_marker(pytest.mark.skip(reason='requires CUDA and Triton'))
        if 'large' in item.keywords:
            need = item.get_closest_marker('large').kwargs.get('gib', 40)
            if os.environ.get('DSC_RUN_LARGE') != '1' or not torch.cuda.is_available():
                item.add_marker(pytest.mark.skip(reason='set DSC_RUN_LARGE=1 on a suitable GPU'))
            elif torch.cuda.mem_get_info()[0] < need * 2 ** 30:
                item.add_marker(pytest.mark.skip(reason=f'requires {need} GiB free GPU memory'))
        if 'needs_ext' in item.keywords:
            name = item.get_closest_marker('needs_ext').kwargs.get('module', '_C_cpu')
            if importlib.util.find_spec('doublesided_cholesky.' + name) is None:
                item.add_marker(pytest.mark.skip(reason=f'extension {name} not built'))

def ground_truth(G, M, z, w):
    """Full re-factorization: L+ = chol(G + zz^T), Aw+ = L+^{-1} (M + zw^T) L+^{-T}."""
    G_new = G + torch.outer(z, z)
    M_new = M + torch.outer(z, w)
    L_new = torch.linalg.cholesky(G_new)
    Linv_new = torch.linalg.inv(L_new)
    Aw_new = Linv_new @ M_new @ Linv_new.T
    return (L_new, Aw_new)

def random_problem(r, dtype=torch.float64, seed=0):
    """Generate a well-conditioned test problem."""
    torch.manual_seed(seed)
    A = torch.randn(r, r, dtype=dtype)
    G = A @ A.T + r * torch.eye(r, dtype=dtype)
    L = torch.linalg.cholesky(G).contiguous()
    M = torch.randn(r, r, dtype=dtype)
    Linv = torch.linalg.inv(L)
    Aw = Linv @ M @ Linv.T
    z = torch.randn(r, dtype=dtype) * 0.1
    w = torch.randn(r, dtype=dtype) * 0.1
    return (G, L, M, Aw, z, w)

@pytest.fixture(autouse=True, scope="session")
def highest_precision():
    """Keep GPU correctness checks in the same FP32 mode as the benchmarks."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
