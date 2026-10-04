"""Hardware release gates: explicitly opt in with DSC_RUN_LARGE=1."""
import numpy as np
import pytest
import torch
from doublesided_cholesky import layouts
from test_cuda_inplace import relative
pytestmark=[pytest.mark.cuda,pytest.mark.needs_ext(module='_C')]

def probes(rank):
    rng=np.random.Generator(np.random.PCG64(29421))
    return torch.from_numpy(((2*rng.integers(0,2,size=(rank,16))-1)/np.sqrt(rank)).astype(np.float32)).cuda()

@pytest.mark.large(gib=36)
def test_f1_rank_65536():
    from doublesided_cholesky.cuda import SymmetricTiledWorkspace,update_symmetric_tiled,packed_symmetric_rank1_update_
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    rank=65536;torch.manual_seed(0)
    z=.05*torch.randn(rank,device='cuda');indices=torch.arange(rank,device='cuda',dtype=torch.int64)
    G=torch.zeros(layouts.packed_size(rank),device='cuda');M=torch.zeros_like(G)
    G[indices*(indices+3)//2]=rank;M[indices*(indices+3)//2]=1
    P=torch.zeros(layouts.tile_elements(rank),device='cuda');A=torch.zeros_like(P)
    diagonal=layouts.tile_diagonal_offsets(rank);P[diagonal]=rank**-.5;A[diagonal]=1/rank
    ws=SymmetricTiledWorkspace(rank)
    packed_symmetric_rank1_update_(G,M,z,rank=rank);update_symmetric_tiled(P,A,z,rank=rank,workspace=ws)
    Q=probes(rank);PTQ=layouts.tile_lower_matmul(P,Q,rank,False,transpose=True)
    factor=layouts.tile_lower_matmul(P,layouts.packed_symmetric_matmul(G,PTQ,rank),rank,False)
    operator=layouts.tile_lower_matmul(P,layouts.packed_symmetric_matmul(M,PTQ,rank),rank,False)
    assert relative(factor,Q)<1e-3
    assert relative(layouts.tile_lower_matmul(A,Q,rank,True),operator)<1e-3
    # Independently evaluate the formerly overflowing last row of tiles in float64.
    u=z.double()/rank**.5;cum=torch.cumsum(u*u,0);before=1+cum-u*u;c=torch.sqrt(before/(1+cum))
    rows=torch.arange(65472,65536,device='cuda')[:,None];cols=torch.arange(32768,65536,device='cuda')[None,:]
    expected=c[rows]*((rows==cols).double()/rank**.5-u[rows]/before[rows]*u[cols]/rank**.5*(cols<rows))
    first=(1023*1024//2+512)*4096
    actual=P[first:].view(512,64,64).permute(1,0,2).reshape(64,32768)
    assert relative(actual,expected)<1e-5

@pytest.mark.large(gib=72)
def test_asymmetric_rank_65536():
    from doublesided_cholesky.cuda import update_inplace,update_tiled,InplaceConfig
    rank=65536;torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    P=torch.zeros((rank,rank),device='cuda');P.diagonal().fill_(rank**-.5)
    A=torch.zeros_like(P);p=P.clone();a=A.clone()
    z=torch.randn(rank,device='cuda')/rank**.5;w=torch.randn_like(z)/rank**.5
    cfg=InplaceConfig.for_rank(rank)
    update_inplace(p,a,z,w,config=cfg);update_tiled(P,A,z,w,p_tile_cols=cfg.tile,p_num_warps=cfg.sweep_num_warps)
    Q=probes(rank)
    assert relative(P@Q,p@Q)<1e-5 and relative(A@Q,a@Q)<1e-5
