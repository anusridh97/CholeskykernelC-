import pytest
import torch
from doublesided_cholesky import cpu
from conftest import random_problem, ground_truth
pytestmark = pytest.mark.needs_ext

@pytest.mark.parametrize("rank", [1, 8, 33, 64, 128, 257])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_ground_truth(rank, dtype):
    G,L,M,A,z,w = random_problem(rank, dtype=dtype)
    expected_L,expected_A = ground_truth(G.double(), M.double(), z.double(), w.double())
    before = [t.clone() for t in (A,z,w)]
    ptr = L.data_ptr()
    out,vz = cpu.update(L,A,z,w)
    tolerance = 1e-4 if dtype == torch.float32 else 1e-10
    for actual, expected in [(L, expected_L), (out, expected_A), (vz, torch.linalg.solve_triangular(expected_L,z.double()[:,None],upper=False)[:,0])]:
        assert ((actual.double()-expected).norm()/expected.norm()).item() <= tolerance
    assert L.data_ptr() == ptr
    assert all(torch.equal(a,b) for a,b in zip(before,(A,z,w)))

def test_batched():
    states=[random_problem(33, seed=i) for i in range(3)]
    L=torch.stack([s[1] for s in states]); A=torch.stack([s[3] for s in states]); z=torch.stack([s[4] for s in states]); w=torch.stack([s[5] for s in states])
    out,vz=cpu.update_batched(L,A,z,w)
    for i,state in enumerate(states):
        li=state[1].clone(); ai,vi=cpu.update(li,state[3],state[4],state[5])
        torch.testing.assert_close(L[i],li,rtol=1e-10,atol=1e-12)
        torch.testing.assert_close(out[i],ai,rtol=1e-10,atol=1e-12)
        torch.testing.assert_close(vz[i],vi,rtol=1e-10,atol=1e-12)

def test_reject_bad_inputs():
    _,L,_,A,z,w=random_problem(17)
    for args in [(L.T,A,z,w),(L,A.float(),z,w),(L,A,z[:-1],w),(L.long(),A.long(),z.long(),w.long())]:
        with pytest.raises(RuntimeError): cpu.update(*args)
