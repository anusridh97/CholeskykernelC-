import ast
from pathlib import Path
import pytest
import torch
from doublesided_cholesky import layouts

def tile_on_cpu(matrix):
    rank=matrix.shape[0];out=torch.zeros(layouts.tile_elements(rank),dtype=matrix.dtype)
    for i in range(layouts.tile_count(rank)):
        for j in range(i+1):
            tile=out[(i*(i+1)//2+j)*4096:][:4096].view(64,64)
            block=matrix[i*64:min(rank,(i+1)*64),j*64:min(rank,(j+1)*64)]
            tile[:block.shape[0],:block.shape[1]]=block
    return out

def test_f1_large_offsets():
    offset=(1023*(1023+1)//2+1023)*4096+63*64+63
    assert offset==layouts.tile_elements(65536)-1 and offset>2**31
    assert ((1023*1024//2+512)*4096)==2**31
    assert sum(i*4096>=2**31 for i in range(1024*1025//2))==512

@pytest.mark.parametrize('rank',[1,65,300])
def test_action_helpers_cpu(rank):
    generator=torch.Generator().manual_seed(7)
    matrix=torch.randn(rank,rank,generator=generator,dtype=torch.float64); matrix=(matrix+matrix.T)/2
    probes=torch.randn(rank,16,generator=generator,dtype=torch.float64)
    packed=matrix[torch.tril_indices(rank,rank).unbind()]
    torch.testing.assert_close(layouts.packed_symmetric_matmul(packed,probes,rank,block_rows=37),matrix@probes)
    torch.testing.assert_close(layouts.tile_lower_matmul(tile_on_cpu(matrix),probes,rank,True),matrix@probes)
    factor=matrix.tril()
    torch.testing.assert_close(layouts.tile_lower_matmul(tile_on_cpu(factor),probes,rank,False),factor@probes)
    torch.testing.assert_close(layouts.tile_lower_matmul(tile_on_cpu(factor),probes,rank,False,transpose=True),factor.T@probes)

@pytest.mark.cuda
@pytest.mark.parametrize('rank',[1,63,64,65,129,1000])
def test_roundtrip_cuda(rank):
    matrix=torch.randn(rank,rank,device='cuda');matrix=((matrix+matrix.T)/2).contiguous()
    for lower in [False,True]:
        source=matrix.tril() if lower else matrix
        tiles=layouts.dense_to_tile_lower(source,lower_triangular=lower)
        out=layouts.tile_lower_to_dense(tiles,rank,lower_triangular=lower)
        assert torch.equal(source,out)
    packed=layouts.pack_symmetric_lower(matrix)
    assert torch.equal(layouts.unpack_symmetric_lower(packed,rank),matrix)
