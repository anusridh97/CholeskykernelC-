import pytest
import torch
from doublesided_cholesky import reference
from conftest import random_problem, ground_truth

@pytest.mark.parametrize("rank", [1, 8, 16, 64, 128])
def test_reference(rank):
    G, L, M, Aw, z, w = random_problem(rank)
    exact_L, exact_A = ground_truth(G, M, z, w)
    actual, vz = reference.update(L, Aw, z, w)
    torch.testing.assert_close(L, exact_L, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(actual, exact_A, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(vz, torch.linalg.solve_triangular(exact_L, z[:,None], upper=False)[:,0], rtol=1e-10, atol=1e-12)
    P, A = reference.inverse_state(G, M)
    assert P.is_contiguous() and torch.count_nonzero(P.triu(1)) == 0
    torch.testing.assert_close(P @ G @ P.T, torch.eye(rank, dtype=P.dtype))
    torch.testing.assert_close(A, Aw)
