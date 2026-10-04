"""Symmetric FP32 inverse-gauge update in lower 64x64 tile storage."""
import torch
import triton
import triton.language as tl
from ..layouts import tile_elements, tile_count, packed_size
from .._validation import check_tiled_symmetric, check_cuda_tensors
from ._vector_kernels import _u2_terms1, _finalize_params1
@triton.jit
def _p_update_tilemajor(p,solve,cs,pc,r,n,total,B:tl.constexpr,STRIPE:tl.constexpr):
 tj,sg=tl.program_id(0),tl.program_id(1);cols=tj*B+sg*STRIPE+tl.arange(0,STRIPE).to(tl.int64);cm=cols<r;carry=tl.zeros((STRIPE,),tl.float32);lr=tl.arange(0,B).to(tl.int64)
 for ti in tl.range(tj,n,1,num_stages=2):
  rows=ti*B+lr[:,None];mask=(rows<r)&cm[None,:]&(cols[None,:]<=rows);off=(ti.to(tl.int64)*(ti+1)//2+tj)*B*B+lr[:,None]*B+(sg*STRIPE+tl.arange(0,STRIPE))[None,:]
  u=tl.load(solve+rows,mask=rows<r,other=0.0).to(tl.float32);c=tl.load(cs+rows,mask=rows<r,other=1.0).to(tl.float32);q=tl.load(pc+rows,mask=rows<r,other=1.0).to(tl.float32);x=tl.load(p+off,mask=mask,other=0.0).to(tl.float32)
  term=tl.where(mask,u*x,0.0);inc=tl.cumsum(term,axis=0);tl.store(p+off,c*(x-u*(carry[None,:]+inc-term)*q*q),mask=mask);carry+=tl.sum(term,axis=0)

@triton.jit
def _packed_symmetric_outer2(g, m, z, r, pn, BLOCK: tl.constexpr):
    row=tl.program_id(0).to(tl.int64); cols=tl.program_id(1).to(tl.int64)*BLOCK+tl.arange(0,BLOCK).to(tl.int64)
    batch=tl.program_id(2).to(tl.int64); mask=(row<r)&(cols<=row); vb=batch*r; pb=batch*pn
    zr=tl.load(z+vb+row,mask=row<r,other=0.0).to(tl.float32)
    zc=tl.load(z+vb+cols,mask=mask,other=0.0).to(tl.float32); off=pb+row*(row+1)//2+cols
    gv=tl.load(g+off,mask=mask,other=0.0).to(tl.float32); mv=tl.load(m+off,mask=mask,other=0.0).to(tl.float32); add=zr*zc
    tl.store(g+off,gv+add,mask=mask); tl.store(m+off,mv+add,mask=mask)


def _packed_update(g,m,z,r:int):
    batch=1 if z.ndim==1 else z.shape[0]; pn=packed_size(r)
    if g.shape[-1]!=pn or m.shape!=g.shape: raise ValueError('packed G/M shape mismatch')
    _packed_symmetric_outer2[(r,triton.cdiv(r,256),batch)](g,m,z,r=r,pn=pn,BLOCK=256,num_warps=4)
    return g,m


class SymmetricTiledWorkspace:
    """FP32 vectors plus five (nt,64) tile descriptor arrays; returned vz aliases this state."""
    def __init__(self, rank: int, device="cuda", split_k: int | None=None):
        count=tile_count(rank)
        self.rank=rank
        self.split_k=split_k if split_k is not None else (1 if rank<=8192 else 16 if rank<=32768 else 64)
        if not isinstance(self.split_k,int) or not 1<=self.split_k<=65535:
            raise ValueError("split_k must be an integer in [1,65535]")
        if torch.device(device).type!='cuda':raise ValueError("workspace requires CUDA")
        for name in ('solve','cs','ss','pc','vz','h','d'):
            setattr(self,name,torch.empty(rank,device=device,dtype=torch.float32))
        self.device=self.solve.device
        self.partial=torch.empty(self.split_k*rank,device=device,dtype=torch.float32)
        for name in ('south','east0','upper','north','west'):
            setattr(self,name,torch.empty(count*(count+1)//2*64,device=device,dtype=torch.float32))

    @property
    def nbytes(self) -> int:
        """Exact allocated bytes, including split-K partials and descriptors."""
        return sum(t.numel()*t.element_size() for t in vars(self).values() if isinstance(t,torch.Tensor))

    def validate(self, rank: int, like: torch.Tensor) -> None:
        """Reject incompatible rank, device, dtype, layout or buffer sizes before a launch."""
        if rank!=self.rank or like.device!=self.device:raise ValueError("workspace rank/device mismatch")
        count=tile_count(rank)
        for name in ('solve','cs','ss','pc','vz','h','d','partial','south','east0','upper','north','west'):
            tensor=getattr(self,name)
            size=self.split_k*rank if name=='partial' else count*(count+1)//2*64 if name in ('south','east0','upper','north','west') else rank
            check_cuda_tensors(like,tensor)
            if tensor.shape!=(size,):raise ValueError("workspace buffer shape mismatch")

@torch.no_grad()
def update_symmetric_tiled(P: torch.Tensor, A: torch.Tensor, z: torch.Tensor, *, rank: int, workspace: SymmetricTiledWorkspace | None=None) -> tuple[torch.Tensor,torch.Tensor]:
    """Update tile-lower P/A in place for w=z; returned (A,vz) aliases state/workspace.

    P/A have length tile_elements(rank); z is (rank,). CUDA FP32, contiguous,
    single-state only; factor diagonal tiles have zero upper triangles and padding.
    """
    check_tiled_symmetric(P,A,z,rank)
    workspace=workspace or SymmetricTiledWorkspace(rank,z.device)
    workspace.validate(rank,A)
    from .. import _C
    n=tile_count(rank);total=tile_elements(rank)
    _C.pz_split_k(P,z,workspace.partial,workspace.solve,rank,workspace.split_k)
    grid=(triton.cdiv(rank,256),)
    _u2_terms1[grid](workspace.solve,workspace.cs,total=rank,BLOCK=256,num_warps=4)
    torch.cumsum(workspace.cs,dim=-1,out=workspace.cs)
    _finalize_params1[grid](workspace.solve,workspace.cs,workspace.cs,workspace.ss,workspace.pc,workspace.vz,total=rank,BLOCK=256,num_warps=4)
    _p_update_tilemajor[(n,8)](P,workspace.solve,workspace.cs,workspace.pc,r=rank,n=n,total=total,B=64,STRIPE=8,num_warps=8)
    torch.mul(workspace.ss,workspace.pc,out=workspace.h)
    torch.mul(workspace.pc,workspace.cs,out=workspace.d)
    torch.div(workspace.ss,workspace.d,out=workspace.d)
    workspace.d.neg_()
    _C.sym_descriptor(A,workspace.cs,workspace.h,workspace.d,workspace.vz,workspace.south,workspace.east0,workspace.upper,workspace.north,workspace.west,rank)
    return A,workspace.vz

@torch.no_grad()
def packed_symmetric_rank1_update_(G: torch.Tensor, M: torch.Tensor, z: torch.Tensor, *, rank: int) -> None:
    """Add zz^T to row-packed G/M in place; G/M are (r*(r+1)//2,), z is (r,)."""
    if G.shape!=(packed_size(rank),) or M.shape!=G.shape or z.shape!=(rank,):raise ValueError("incorrect packed/vector shapes")
    check_cuda_tensors(G,M,z)
    _packed_update(G,M,z,rank)
