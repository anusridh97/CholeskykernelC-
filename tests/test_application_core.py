from __future__ import annotations

import pytest
import torch

from benchmarks.applications._corrected.core import (
    cyclic_shift_state,
    double_whiten,
    extension_available,
    incremental_update,
    relative_error,
    whiten_vector,
)


BACKENDS = ["torch"] + (["extension"] if extension_available() else [])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("metric_sign", [1, -1])
@pytest.mark.parametrize("operator_sign", [-1, 0, 1])
def test_signed_update_matches_fresh_factorization(
    backend: str,
    metric_sign: int,
    operator_sign: int,
) -> None:
    torch.manual_seed(4)
    r = 24
    dtype = torch.float64
    A = torch.randn(r, r, dtype=dtype)
    G = A @ A.mT + r * torch.eye(r, dtype=dtype)
    L = torch.linalg.cholesky(G)
    M = torch.randn(r, r, dtype=dtype)
    Aw = double_whiten(L, M)
    z = 0.05 * torch.randn(r, dtype=dtype)
    w = torch.randn(r, dtype=dtype)

    result = incremental_update(
        L,
        Aw,
        z,
        w,
        metric_sign=metric_sign,
        operator_sign=operator_sign,
        backend=backend,
    )
    G_new = G + metric_sign * torch.outer(z, z)
    M_new = M + operator_sign * torch.outer(z, w)
    L_exact = torch.linalg.cholesky(G_new)
    Aw_exact = double_whiten(L_exact, M_new)
    vz_exact = whiten_vector(L_exact, z)

    assert relative_error(result.L, L_exact) < 2e-12
    assert relative_error(result.Aw, Aw_exact) < 2e-11
    assert relative_error(result.vz, vz_exact) < 2e-12


@pytest.mark.parametrize("backend", BACKENDS)
def test_sequential_state_and_noncontiguous_cholesky_layout(backend: str) -> None:
    torch.manual_seed(9)
    r = 32
    dtype = torch.float64
    A = torch.randn(r, r, dtype=dtype)
    G = A @ A.mT + r * torch.eye(r, dtype=dtype)
    L = torch.linalg.cholesky(G)  # commonly Fortran-strided
    M = torch.randn(r, r, dtype=dtype)
    Aw = double_whiten(L, M)

    for _ in range(100):
        z = 0.01 * torch.randn(r, dtype=dtype)
        w = 0.01 * torch.randn(r, dtype=dtype)
        result = incremental_update(L, Aw, z, w, backend=backend)
        L, Aw = result.L, result.Aw
        G = G + torch.outer(z, z)
        M = M + torch.outer(z, w)

    L_exact = torch.linalg.cholesky(G)
    Aw_exact = double_whiten(L_exact, M)
    assert relative_error(L, L_exact) < 5e-12
    assert relative_error(Aw, Aw_exact) < 5e-11


@pytest.mark.parametrize("backend", BACKENDS)
def test_cyclic_shift_transport(backend: str) -> None:
    torch.manual_seed(17)
    r = 20
    dtype = torch.float64
    A = torch.randn(r, r, dtype=dtype)
    G = A @ A.mT + r * torch.eye(r, dtype=dtype)
    L = torch.linalg.cholesky(G)
    M = torch.randn(r, r, dtype=dtype)
    m = torch.randn(r, dtype=dtype)
    Aw = double_whiten(L, M)
    mw = whiten_vector(L, m)

    L_new, Aw_new, mw_new = cyclic_shift_state(L, Aw, mw, backend=backend)
    permutation = torch.tensor(list(range(1, r)) + [0])
    G_exact = G[permutation][:, permutation]
    M_exact = M[permutation][:, permutation]
    m_exact = m[permutation]
    L_exact = torch.linalg.cholesky(G_exact)

    assert relative_error(L_new, L_exact) < 2e-12
    assert relative_error(Aw_new, double_whiten(L_exact, M_exact)) < 2e-11
    assert relative_error(mw_new, whiten_vector(L_exact, m_exact)) < 2e-12
