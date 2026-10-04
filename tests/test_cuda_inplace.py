import pytest
import torch
from doublesided_cholesky import reference
pytestmark = pytest.mark.cuda

def problem(rank, symmetric=False, seed=17):
    torch.manual_seed(seed)
    q = torch.randn(rank, min(rank, 32), dtype=torch.float64) / rank ** 0.5
    G = torch.eye(rank, dtype=torch.float64) + q @ q.T
    M = torch.randn(rank, rank, dtype=torch.float64) / rank ** 0.5
    if symmetric:
        M = (M + M.T) / 2
    z = torch.randn(rank, dtype=torch.float64) / rank ** 0.5
    w = z if symmetric else torch.randn(rank, dtype=torch.float64) / rank ** 0.5
    P, A = reference.inverse_state(G, M)
    return (G, M, P, A, z, w)

def relative(actual, expected):
    return ((actual.double() - expected.double()).norm() / expected.double().norm().clamp_min(1e-30)).item()

@pytest.mark.parametrize('rank', [129, 257, 1025])
def test_inplace_ground_truth(rank):
    from doublesided_cholesky.cuda import update_inplace
    G, M, P, A, z, w = problem(rank)
    expected_P, expected_A = reference.inverse_state(G + z[:, None] * z, M + z[:, None] * w)
    P, A, z, w = [t.float().cuda().contiguous() for t in (P, A, z, w)]
    update_inplace(P, A, z, w)
    assert relative(P.cpu(), expected_P) < 5e-06
    assert relative(A.cpu(), expected_A) < 5e-06

def test_batch_and_validation():
    from doublesided_cholesky.cuda import update_inplace
    states = [problem(129, seed=i) for i in range(3)]
    P, A, z, w = [torch.stack([s[index] for s in states]).float().cuda().contiguous() for index in [2, 3, 4, 5]]
    p, a = (P.clone(), A.clone())
    update_inplace(P, A, z, w)
    for i in range(3):
        update_inplace(p[i], a[i], z[i], w[i])
    assert relative(P, p) < 2e-06 and relative(A, a) < 2e-06
    for args in [(P.double(), A, z, w), (P.transpose(1, 2), A, z, w), (P.cpu(), A, z, w)]:
        with pytest.raises(ValueError):
            update_inplace(*args)
