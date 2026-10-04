import pytest
import torch
from doublesided_cholesky import reference
from test_cuda_inplace import problem, relative
pytestmark = [pytest.mark.cuda, pytest.mark.needs_ext(module='_C')]

@pytest.mark.parametrize('rank', [1, 63, 64, 65, 129, 257, 1025, 4097])
def test_tiled(rank):
    from doublesided_cholesky.cuda import update_inplace, update_tiled
    G, M, P, A, z, w = problem(rank)
    expected_P, expected_A = reference.inverse_state(G + z[:, None] * z, M + z[:, None] * w)
    P, A, z, w = [t.float().cuda().contiguous() for t in (P, A, z, w)]
    baseline_P, baseline_A = (P.clone(), A.clone())
    _, baseline_v = update_inplace(baseline_P, baseline_A, z, w)
    _, v = update_tiled(P, A, z, w)
    for a, b in [(P, baseline_P), (A, baseline_A), (v, baseline_v)]:
        assert relative(a, b) < 1e-06
    assert relative(P.cpu(), expected_P) < 5e-06 and relative(A.cpu(), expected_A) < 5e-06

def test_trajectory():
    from doublesided_cholesky.cuda import update_inplace, update_tiled, InplaceWorkspace, TiledWorkspace
    _, _, P, A, _, _ = problem(129, seed=31)
    P, A = (P.float().cuda(), A.float().cuda())
    p, a = (P.clone(), A.clone())
    iw, tw = (InplaceWorkspace(A), TiledWorkspace(A))
    torch.manual_seed(33)
    for _ in range(100):
        z = torch.randn(129, device='cuda') / 129 ** 0.5
        w = torch.randn_like(z) / 129 ** 0.5
        update_inplace(p, a, z, w, workspace=iw)
        update_tiled(P, A, z, w, workspace=tw)
    assert torch.isfinite(A).all() and relative(A, a) < 1e-05

def test_native_checks():
    from doublesided_cholesky import _C
    A = torch.zeros(65, 65, device='cuda')
    v = torch.zeros(65, device='cuda')
    scratch = torch.zeros(256, device='cuda')
    for bad in [A.double(), A[:, 0], A.T]:
        with pytest.raises(RuntimeError):
            _C.asym_descriptor(bad, v, v, v, v, v, scratch, scratch, scratch, scratch, 65)
