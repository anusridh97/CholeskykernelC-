"""Factor-gauge reference: L (r,r), Aw (r,r), z/w (r,)."""
import torch

def givens_params(a: float, b: float):
    """Givens (c, s) such that c*a + s*b = rho, -s*a + c*b = 0."""
    import math
    rho = math.sqrt(a * a + b * b)
    if rho == 0:
        return (1.0, 0.0)
    return (a / rho, b / rho)

def cholesky_rank1_update(L: torch.Tensor, z: torch.Tensor):
    """Rank-1 Cholesky update via augmented Givens rotations.

    Operates on [L | z] by right-multiplying Givens rotations that zero
    each entry of the last column. Modifies L in place to L+. Returns
    (cs, ss) rotation parameters, which Phase 2 re-uses.
    """
    r = L.shape[0]
    cs = torch.zeros(r, dtype=L.dtype, device=L.device)
    ss = torch.zeros(r, dtype=L.dtype, device=L.device)
    for k in range(r):
        c, s = givens_params(L[k, k].item(), z[k].item())
        cs[k], ss[k] = (c, s)
        col_k = L[:, k].clone()
        col_z = z.clone()
        L[:, k] = c * col_k + s * col_z
        z[:] = -s * col_k + c * col_z
    return (cs, ss)

def doublesided_propagate(Aw: torch.Tensor, cs: torch.Tensor, ss: torch.Tensor):
    """Apply Q^T [Aw 0; 0 0] Q, return the top-left r x r block."""
    r = Aw.shape[0]
    Ah = torch.zeros(r + 1, r + 1, dtype=Aw.dtype, device=Aw.device)
    Ah[:r, :r] = Aw
    for k in range(r):
        c, s = (cs[k].item(), ss[k].item())
        rk, rr = (Ah[k].clone(), Ah[r].clone())
        Ah[k] = c * rk + s * rr
        Ah[r] = -s * rk + c * rr
    for k in range(r):
        c, s = (cs[k].item(), ss[k].item())
        ck, cr = (Ah[:, k].clone(), Ah[:, r].clone())
        Ah[:, k] = c * ck + s * cr
        Ah[:, r] = -s * ck + c * cr
    return Ah[:r, :r].clone()

def recover_vz(cs: torch.Tensor, ss: torch.Tensor, r):
    """Top r entries of Q^T e_{r+1} = (L+)^{-1} z."""
    vz = torch.zeros(r, dtype=cs.dtype, device=cs.device)
    e_last = 1.0
    for k in range(r):
        c, s = (cs[k].item(), ss[k].item())
        vz[k] = s * e_last
        e_last = c * e_last
    return vz

@torch.no_grad()
def update(L: torch.Tensor, Aw: torch.Tensor, z: torch.Tensor, w: torch.Tensor):
    """Full O(r^2) update. Modifies L in place to L+. Slow, for correctness."""
    r = L.shape[0]
    z_work = z.clone()
    cs, ss = cholesky_rank1_update(L, z_work)
    Aw_core = doublesided_propagate(Aw, cs, ss)
    vz = recover_vz(cs, ss, r)
    vw = torch.linalg.solve_triangular(L, w.unsqueeze(1), upper=False).squeeze(1)
    return (Aw_core + torch.outer(vz, vw), vz)

@torch.no_grad()
def refactor_dense(G: torch.Tensor, M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return row-major L and Aw = solve(L, M) solve(L.T) for square G/M."""
    if G.ndim != 2 or G.shape[0] != G.shape[1] or M.shape != G.shape:
        raise ValueError('G and M must be matching square matrices')
    L = torch.linalg.cholesky(G).contiguous()
    Aw = torch.linalg.solve_triangular(L, M, upper=False)
    Aw = torch.linalg.solve_triangular(L.mT, Aw, upper=True, left=False).contiguous()
    return (L, Aw)

@torch.no_grad()
def inverse_state(G: torch.Tensor, M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct dense inverse-gauge (Linv, Aw); Linv has an exactly zero upper triangle."""
    L, Aw = refactor_dense(G, M)
    Linv = torch.linalg.solve_triangular(L, torch.eye(L.shape[0], device=L.device, dtype=L.dtype), upper=False).tril_().contiguous()
    return (Linv, Aw)
