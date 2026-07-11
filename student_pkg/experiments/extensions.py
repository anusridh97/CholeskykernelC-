"""Rank-k updates (Appendix C.3) and leverage-guarded hyperbolic downdates (C.4).

Two extensions to the double-sided primitive that widen the Thompson-sampling
design space:

  * ``rank_k_update`` — for Z, W ∈ R^{r×k}, apply G += Z Zᵀ, M += Z Wᵀ and
    transport A_w in O(r²k). A batched TS acquisition of k arms is then ONE
    rank-k update instead of k sequential sweeps. Since the columns compose
    exactly (G += Σ zᵢzᵢᵀ, M += Σ zᵢwᵢᵀ), this is k rank-1 positive updates —
    and each of those runs on the fast batched Triton kernel.

  * ``downdate`` — a hyperbolic Cholesky downdate G⁻ = G − z zᵀ with the
    double-sided transport A⁻ = (L⁻)⁻¹ M (L⁻)⁻ᵀ − v_z v_wᵀ. Admissible iff the
    leverage τ = ‖L⁻¹ z‖² < 1; we reject/skip downdates with τ above a guard
    threshold, following the leverage-controlled stability of Appendix C.4. This
    enables sliding-window forgetting for nonstationary problems.

The hyperbolic rotation for active entries (ℓ, ζ) with ℓ² > ζ² is
ρ = √(ℓ²−ζ²), c = ℓ/ρ, s = −ζ/ρ (c²−s²=1); the operator is transported by the
inverse blocks [[c,−s],[−s,c]] applied double-sidedly to the padded core.

The downdate is a CPU reference (there is no hyperbolic Triton kernel yet); the
rank-k *positive* path uses the existing batched kernel. Both are verified
against a full re-factorization to machine precision.
"""

from __future__ import annotations

import torch

from .charts import refresh_bucket


def rank_k_update(L, Aw, Z, W, force_reference=False):
    """Batched rank-k positive update. Z, W: (T, r, k). L, Aw: (T, r, r).

    Applies G += Z Zᵀ, M += Z Wᵀ and returns (Aw_new, VZ) with VZ the stacked
    (L⁺)⁻¹ zᵢ. L is updated in place. Runs k rank-1 kernel calls (each batched
    over T), so on GPU this is k launches of the fast primitive.
    """
    k = Z.shape[2]
    VZ = []
    Aw_cur = Aw
    for j in range(k):
        Aw_cur, vz = refresh_bucket(L, Aw_cur, Z[:, :, j].contiguous(),
                                    W[:, :, j].contiguous(),
                                    force_reference=force_reference)
        VZ.append(vz)
    return Aw_cur, torch.stack(VZ, dim=2)


def _downdate_one(L, Aw, z, w):
    """Hyperbolic downdate of a single chart (r×r). Modifies L in place to L⁻.

    Returns (Aw_new, vz, leverage). Assumes admissibility already checked.
    """
    r = L.shape[0]
    dtype = L.dtype
    zc = z.clone()
    cs = torch.zeros(r, dtype=dtype, device=L.device)
    ss = torch.zeros(r, dtype=dtype, device=L.device)
    for k in range(r):
        l = L[k, k]
        ze = zc[k]
        rho = torch.sqrt((l * l - ze * ze).clamp_min(1e-30))
        c = l / rho
        s = -ze / rho
        cs[k], ss[k] = c, s
        col = L[:, k].clone()
        zz = zc.clone()
        L[:, k] = c * col + s * zz
        zc = s * col + c * zz
    # Double-sided transport with inverse hyperbolic blocks [[c,-s],[-s,c]].
    Ah = torch.zeros(r + 1, r + 1, dtype=dtype, device=L.device)
    Ah[:r, :r] = Aw
    for k in range(r):
        c, s = cs[k], ss[k]
        rk, rr = Ah[k].clone(), Ah[r].clone()
        Ah[k] = c * rk - s * rr
        Ah[r] = -s * rk + c * rr
    for k in range(r):
        c, s = cs[k], ss[k]
        ck, cr = Ah[:, k].clone(), Ah[:, r].clone()
        Ah[:, k] = c * ck - s * cr
        Ah[:, r] = -s * ck + c * cr
    core = Ah[:r, :r].clone()
    vz = torch.linalg.solve_triangular(L, z.unsqueeze(1), upper=False).squeeze(1)
    vw = torch.linalg.solve_triangular(L, w.unsqueeze(1), upper=False).squeeze(1)
    return core - torch.outer(vz, vw), vz


def downdate(L, Aw, z, w, max_leverage=0.98):
    """Batched leverage-guarded hyperbolic downdate. L, Aw: (T, r, r); z, w: (T, r).

    Applies G⁻ = G − z zᵀ, M⁻ = M − z wᵀ per chart where admissible. Charts whose
    leverage τ = ‖L⁻¹ z‖² ≥ ``max_leverage`` are LEFT UNCHANGED (the guard) so a
    near-singular downdate never blows up. L is updated in place for admitted
    charts.

    Returns (Aw_new, admitted) where admitted is a (T,) bool mask.
    """
    T, r = L.shape[0], L.shape[1]
    y = torch.linalg.solve_triangular(L, z.unsqueeze(-1), upper=False).squeeze(-1)
    tau = (y * y).sum(dim=1)                                  # (T,)
    admitted = tau < max_leverage
    Aw_out = Aw.clone()
    for t in range(T):
        if not admitted[t]:
            continue
        Aw_out[t], _ = _downdate_one(L[t], Aw[t], z[t].clone(), w[t].clone())
    return Aw_out, admitted


def leverage(L, z):
    """τ = ‖L⁻¹ z‖² per chart (downdate admissible iff τ < 1). L:(T,r,r), z:(T,r)."""
    y = torch.linalg.solve_triangular(L, z.unsqueeze(-1), upper=False).squeeze(-1)
    return (y * y).sum(dim=1)
