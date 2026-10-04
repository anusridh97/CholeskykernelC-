"""Float64 descriptor-transport math check, with configurable tiles and no GPU.

The same zero-carry descriptors and prefix corrections as the CUDA implementations
are checked against both a dense two-pass recurrence and fresh factorization.
"""
import numpy as np

def params(P,z,w):
    u=P@z;y=P@w;cum=np.cumsum(u*u);cuy=np.cumsum(u*y)
    before=1+cum-u*u;after=1+cum;q=1/np.sqrt(before)
    c=np.sqrt(before/after);s=u/np.sqrt(after)
    return u,c,s,q,u/np.sqrt(before*after),c*(y-u*(cuy-u*y)/before)

def p_update(P,u,c,q):
    out=P.copy();acc=np.zeros(P.shape[0])
    for i in range(P.shape[0]):
        out[i]=c[i]*(P[i]-u[i]*q[i]**2*acc)
        acc+=u[i]*P[i]
    return np.tril(out)

def dense_two_pass(A,c,h,d,vz,vw):
    out=A.copy();rank=A.shape[0]
    for col in range(rank):
        carry=0.
        for row in range(rank):
            value=out[row,col];out[row,col]=c[row]*value+h[row]*carry;carry+=d[row]*value
    for row in range(rank):
        carry=0.
        for col in range(rank):
            value=out[row,col];out[row,col]=c[col]*value+h[col]*carry+vz[row]*vw[col];carry+=d[col]*value
    return out

def descriptor(A,c,h,d,vz,vw,B,symmetric):
    """Emulate lower-tile mirrored transport or full-matrix transport."""
    rank=A.shape[0];n=(rank+B-1)//B
    pad=np.zeros((n*B,n*B));pad[:rank,:rank]=A
    c,h,d,vz,vw=[np.pad(v,(0,n*B-rank)) for v in (c,h,d,vz,vw)]
    indices=[(i,j) for i in range(n) for j in range(i+1 if symmetric else n)]
    tiles={(i,j):pad[i*B:(i+1)*B,j*B:(j+1)*B].copy() for i,j in indices}
    south={};east={};upper={}
    for (i,j),original in tiles.items():
        x=original.copy();upper[i,j]=np.zeros(B)
        if symmetric and i>j:
            for b in range(B):upper[i,j][b]=sum(d[j*B+a]*x[b,a] for a in range(B))
        south[i,j]=np.zeros(B);east[i,j]=np.zeros(B)
        for b in range(B):
            carry=0.
            for a in range(B):
                row=i*B+a;value=x[a,b];x[a,b]=c[row]*value+h[row]*carry;carry+=d[row]*value
            south[i,j][b]=carry
        for a in range(B):
            last=a if symmetric and i==j else B-1
            east[i,j][a]=sum(d[j*B+b]*x[a,b] for b in range(last+1))
    north={};west={}
    for j in range(n):
        carry=sum((upper[j,k] for k in range(j)),start=np.zeros(B)) if symmetric else np.zeros(B)
        for i in range(j if symmetric else 0,n):
            north[i,j]=carry.copy();carry=carry+south[i,j]
    for i in range(n):
        carry=np.zeros(B)
        for j in range(i+1 if symmetric else n):
            west[i,j]=carry.copy()
            dot=sum(d[j*B+b]*north[i,j][b] for b in range(B))
            carry=carry+east[i,j]+h[i*B:(i+1)*B]*dot
    out=np.zeros_like(pad)
    for (i,j),original in tiles.items():
        x=original.copy()
        for b in range(B):
            carry=north[i,j][b]
            for a in range(B):
                row=i*B+a;value=x[a,b];x[a,b]=c[row]*value+h[row]*carry;carry+=d[row]*value
        for a in range(B):
            carry=west[i,j][a];row=i*B+a;last=a if symmetric and i==j else B-1
            for b in range(last+1):
                col=j*B+b;value=x[a,b];x[a,b]=c[col]*value+h[col]*carry+vz[row]*vw[col];carry+=d[col]*value
        if symmetric and i==j:x=np.tril(x)+np.tril(x,-1).T
        out[i*B:(i+1)*B,j*B:(j+1)*B]=x
        if symmetric and i!=j:out[j*B:(j+1)*B,i*B:(i+1)*B]=x.T
    return out[:rank,:rank]

def rel(a,b):return np.linalg.norm(a-b)/np.linalg.norm(b)

def test_descriptor_transport_matches_refactorization():
    rng=np.random.default_rng(0)
    for rank,tile in [(1,4),(7,4),(8,4),(13,4),(37,8),(70,16),(130,64)]:
        Q=rng.standard_normal((rank,max(1,rank//2)))/np.sqrt(rank)
        G=np.eye(rank)+Q@Q.T;Ms=rng.standard_normal((rank,rank))/np.sqrt(rank);Msym=(Ms+Ms.T)/2
        z=rng.standard_normal(rank)/np.sqrt(rank);w=rng.standard_normal(rank)/np.sqrt(rank)
        P=np.linalg.inv(np.linalg.cholesky(G));Pnew=np.linalg.inv(np.linalg.cholesky(G+np.outer(z,z)))
        for symmetric,M,rhs in [(True,Msym,z),(False,rng.standard_normal((rank,rank))/np.sqrt(rank),w)]:
            A=P@M@P.T;u,c,s,q,vz,vw=params(P,z,rhs);h=s*q;d=-s/(q*c)
            if symmetric:assert np.allclose(vz,vw,rtol=1e-12,atol=1e-15);vw=vz
            exact=Pnew@(M+np.outer(z,rhs))@Pnew.T
            errors=[rel(p_update(P,u,c,q),Pnew),rel(dense_two_pass(A,c,h,d,vz,vw),exact),rel(descriptor(A,c,h,d,vz,vw,tile,symmetric),exact),rel(vz,Pnew@z)]
            assert max(errors)<1e-11,(rank,tile,symmetric,errors)
