import pytest
import torch
from doublesided_cholesky import layouts, reference
from test_cuda_inplace import problem, relative
pytestmark = [pytest.mark.cuda, pytest.mark.needs_ext(module='_C')]

@pytest.mark.parametrize('rank', [1, 2, 63, 64, 65, 127, 128, 129, 257, 1025, 4097])
def test_symmetric_ground_truth_and_tiled(rank):
    from doublesided_cholesky.cuda import update_symmetric_tiled, update_tiled
    G, M, P, A, z, _ = problem(rank, symmetric=True)
    expected_P, expected_A = reference.inverse_state(G + z[:, None] * z, M + z[:, None] * z)
    P, A, z = [t.float().cuda().contiguous() for t in (P, A, z)]
    packed_P = layouts.dense_to_tile_lower(P, lower_triangular=True)
    packed_A = layouts.dense_to_tile_lower(A)
    _, v = update_symmetric_tiled(packed_P, packed_A, z, rank=rank)
    actual_P = layouts.tile_lower_to_dense(packed_P, rank, lower_triangular=True)
    actual_A = layouts.tile_lower_to_dense(packed_A, rank)
    for actual, expected in [(actual_P, expected_P), (actual_A, expected_A), (v, expected_P @ z.cpu().double())]:
        assert relative(actual.cpu(), expected) < 5e-06
    update_tiled(P, A, z, z)
    assert relative(actual_P, P) < 1e-06 and relative(actual_A, A) < 1e-06
    for i in range(layouts.tile_count(rank)):
        for j in range(i + 1):
            for packed in (packed_P, packed_A):
                tile = packed[(i * (i + 1) // 2 + j) * 4096:][:4096].view(64, 64)
                assert torch.count_nonzero(tile[max(0, rank - i * 64):, :]) == 0
                assert torch.count_nonzero(tile[:, max(0, rank - j * 64):]) == 0

@pytest.mark.parametrize('rank', [257, 4097])
def test_split_k_and_workspace(rank):
    from doublesided_cholesky.cuda import update_symmetric_tiled, SymmetricTiledWorkspace
    _, _, P, A, z, _ = problem(rank, True)
    P, A, z = [t.float().cuda().contiguous() for t in (P, A, z)]
    pt = layouts.dense_to_tile_lower(P, True)
    at = layouts.dense_to_tile_lower(A)
    outputs = []
    for split in [1, 4, 16, 64]:
        ws = SymmetricTiledWorkspace(rank, split_k=split)
        p, a = (pt.clone(), at.clone())
        update_symmetric_tiled(p, a, z, rank=rank, workspace=ws)
        outputs.append((p, a))
        p2, a2 = (pt.clone(), at.clone())
        update_symmetric_tiled(p2, a2, z, rank=rank, workspace=ws)
        assert torch.equal(p, p2) and torch.equal(a, a2)
        n = layouts.tile_count(rank)
        assert ws.nbytes == (5 * n * (n + 1) // 2 * 64 + (7 + split) * rank) * 4
    for p, a in outputs[1:]:
        assert relative(p, outputs[0][0]) < 1e-06 and relative(a, outputs[0][1]) < 1e-06
    with pytest.raises(ValueError):
        update_symmetric_tiled(pt, at, z, rank=rank, workspace=SymmetricTiledWorkspace(rank + 1))
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        assert SymmetricTiledWorkspace(rank).solve.dtype == torch.float32
    finally:
        torch.set_default_dtype(old)

def test_symmetric_trajectory():
    from doublesided_cholesky.cuda import update_symmetric_tiled, SymmetricTiledWorkspace
    rank = 257
    G, M, P, A, _, _ = problem(rank, True)
    pt = layouts.dense_to_tile_lower(P.float().cuda(), True)
    at = layouts.dense_to_tile_lower(A.float().cuda())
    ws = SymmetricTiledWorkspace(rank)
    torch.manual_seed(33)
    for _ in range(100):
        z = torch.randn(rank, device='cuda') / rank ** 0.5
        zd = z.cpu().double()
        update_symmetric_tiled(pt, at, z, rank=rank, workspace=ws)
        G += zd[:, None] * zd
        M += zd[:, None] * zd
        P, A = reference.inverse_state(G, M)
    assert relative(layouts.tile_lower_to_dense(pt, rank, True).cpu(), P) < 0.0001
    assert relative(layouts.tile_lower_to_dense(at, rank).cpu(), A) < 0.0001
    assert torch.isfinite(pt).all() and torch.isfinite(at).all()

@pytest.mark.parametrize('rank', [1, 65, 300])
def test_packed_outer(rank):
    from doublesided_cholesky.cuda import packed_symmetric_rank1_update_
    torch.manual_seed(3)
    G = torch.randn(rank, rank, device='cuda')
    G = ((G + G.T) / 2).contiguous()
    M = G.clone()
    z = torch.randn(rank, device='cuda')
    gp = layouts.pack_symmetric_lower(G)
    mp = layouts.pack_symmetric_lower(M)
    packed_symmetric_rank1_update_(gp, mp, z, rank=rank)
    expected = torch.addr(G, z, z)
    assert torch.equal(layouts.unpack_symmetric_lower(gp, rank), expected)
    assert torch.equal(gp, mp)

def test_native_and_python_rejections():
    from doublesided_cholesky import _C
    from doublesided_cholesky.cuda import update_symmetric_tiled, SymmetricTiledWorkspace
    rank = 65
    p = torch.zeros(layouts.tile_elements(rank), device='cuda')
    z = torch.zeros(rank, device='cuda')
    ws = SymmetricTiledWorkspace(rank)
    for bad in [p.double(), p[:-1], p.cpu()]:
        with pytest.raises(ValueError):
            update_symmetric_tiled(bad, p, z, rank=rank)
    with pytest.raises(RuntimeError):
        _C.pz_split_k(p, z, ws.partial, ws.solve, rank, 0)
    with pytest.raises(RuntimeError):
        _C.pz_split_k(p, z, ws.partial[:1], ws.solve, rank, 1)
    with pytest.raises(RuntimeError):
        _C.sym_descriptor(p, z, z, z, z, z, z, z, z, z, rank)
