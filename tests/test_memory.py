from benchmarks.synthetic_symmetric import memory_bytes
from doublesided_cholesky import layouts

def test_rank_65536_memory_accounting():
    budget=memory_bytes(65536,'both')
    assert layouts.tile_elements(65536)==2149580800
    assert budget['workspace_bytes']/2**30==0.6429443359375
    assert budget['incremental_state_bytes']==(2*(65536*65537//2)+2*2149580800)*4
    assert budget['fresh_state_bytes']==4*65536**2*4+4
    assert memory_bytes(65536,'fresh')['incremental_state_bytes']==0
    assert memory_bytes(65536,'fresh')['workspace_bytes']==0
    assert memory_bytes(65536,'incremental')['fresh_state_bytes']==0
