"""Massively parallel GP Thompson sampling via the same double-sided primitive.

Thousands of independent conjugate-GP arms on one GPU, each conditioned by one
rank-1 whitened update per round, all fused into a single batched kernel launch.
This is the "T is where we win" story on real hardware, scaling the batched
primitive far past the MLX section's T = 16.

Faithful streaming SVGP conditioning (paper §4.4/§4.5), using the SAME primitive
as the natural-gradient optimizer — z = w = a whitened vector, not an ad-hoc
precision filter. Fixed inducing set Z of size r with kernel K_uu = k(Z,Z) and
factor L_uu = chol(K_uu). For an input x the whitened cross-covariance is

    φ(x) = L_uu^{-1} k_u(x),      k_u(x) = k(Z, x) ∈ R^r.

When the candidate set equals the inducing set, φ(x_j) = L_uu^⊤ e_j (the j-th row
of L_uu), so Φ := L_uu^⊤ has φ_j as its j-th column.

Whitened variational posterior q(v)=N(m,S) over v = L_uu^{-1}u, prior N(0,I).
Streaming a noisy observation of candidate j (Gaussian noise σ) is the conjugate
natural-gradient update (dual parameterization, Adam et al. 2021):

    precision  G = S^{-1} ← G + (1/σ^2) φ_j φ_j^⊤        (rank-1: z = φ_j/σ)
    info vec   b = G m   ← b + (y/σ^2) φ_j

The double-sided primitive maintains, per arm, the Cholesky factor L=chol(G) and
the whitened operator A_w = L^{-1} M L^{-⊤} = I − L^{-1}L^{-⊤} (M = G − I, so
z = w = φ_j/σ) — exactly the object the paper accelerates. Thompson sampling
draws v = m + L^{-⊤}ξ and predicts f = Φ^⊤ v over all candidates from the same
maintained factor. Ground truth is a full re-cholesky of G.
"""

from __future__ import annotations

import torch

from .charts import refresh_bucket, gpu_kernel_available  # noqa: F401


def rbf_kernel(X, lengthscale=0.1, jitter=1e-4):
    """RBF/Gaussian kernel matrix for points X (r, dims)."""
    d2 = torch.cdist(X, X) ** 2
    K = torch.exp(-0.5 * d2 / lengthscale ** 2)
    K = K + jitter * torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
    return K


class BatchedGPThompson:
    """T independent conjugate-GP arms over a shared inducing/candidate set.

    Each round conditions every arm on one Thompson-selected candidate via a
    single batched rank-1 whitened update. Arms optimize independent (per-arm
    shifted) objectives so they are genuinely different problems.

    Args:
        X: (r, dims) inducing = candidate points (shared across arms).
        objective: callable X (r, dims) -> (r,) true values; each arm gets a
            shifted copy.
        T: number of parallel arms.
        noise: observation noise std σ.
        lengthscale, jitter: RBF hyperparameters.
        seed: RNG seed.
        force_reference: force the CPU reference refresh (for cross-checks).
    """

    def __init__(self, X, objective, T=256, noise=0.1, lengthscale=0.1,
                 jitter=1e-4, seed=0, force_reference=False):
        self.X = X
        self.r = X.shape[0]
        self.T = T
        self.noise = noise
        self.device = X.device
        self.dtype = X.dtype
        self.force_reference = force_reference
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

        # Fixed inducing kernel and whitened cross-covariance columns Φ = L_uu^⊤.
        K = rbf_kernel(X, lengthscale, jitter)
        self.K = K
        self.L_uu = torch.linalg.cholesky(K)
        self.Phi = self.L_uu.mT.contiguous()        # (r, r): φ_j = Phi[:, j]

        # Per-arm state: precision factor L (init chol(I)=I), whitened core A_w=0,
        # information vector b=0.
        eye = torch.eye(self.r, device=self.device, dtype=self.dtype)
        self.L = eye.unsqueeze(0).repeat(T, 1, 1).contiguous()
        self.Aw = torch.zeros(T, self.r, self.r, device=self.device, dtype=self.dtype)
        self.b = torch.zeros(T, self.r, device=self.device, dtype=self.dtype)
        self.counts = torch.zeros(T, self.r, device=self.device, dtype=self.dtype)

        # Per-arm objective realizations (shifted) on the grid.
        base = objective(X)
        shifts = torch.arange(T, device=self.device, dtype=self.dtype).reshape(T, 1)
        grid = torch.arange(self.r, device=self.device, dtype=self.dtype)
        self.f_true = base.unsqueeze(0) + 0.05 * torch.sin(shifts + grid)
        self.f_max = self.f_true.max(dim=1).values
        self.regret_history = []

    # -- posterior helpers (from the maintained precision factor L) -----------

    def posterior_mean_v(self):
        """m = G^{-1} b via two triangular solves with L (whitened coords)."""
        y = torch.linalg.solve_triangular(self.L, self.b.unsqueeze(-1), upper=False)
        return torch.linalg.solve_triangular(self.L.mT, y, upper=True).squeeze(-1)

    def thompson_predict(self):
        """Draw v ~ N(m, G^{-1}) per arm and return predictions f = Φ^⊤ v (T, r)."""
        m = self.posterior_mean_v()
        xi = torch.randn(self.T, self.r, generator=self.gen).to(self.device, self.dtype)
        v = m + torch.linalg.solve_triangular(
            self.L.mT, xi.unsqueeze(-1), upper=True).squeeze(-1)
        return torch.einsum("ij,tj->ti", self.Phi.mT, v)  # φ_j^⊤ v for each j

    def step(self):
        """One batched Thompson round for all T arms; returns mean simple regret."""
        # 1. Thompson-select a candidate per arm.
        f = self.thompson_predict()
        idx = f.argmax(dim=1)                                    # (T,)
        phi = self.Phi[:, idx].mT.contiguous()                  # (T, r): φ_{idx}

        # 2. Observe (noisy) and accumulate the information vector.
        y = self.f_true.gather(1, idx.unsqueeze(1)).squeeze(1)
        y = y + self.noise * torch.randn(self.T, generator=self.gen).to(self.device, self.dtype)
        self.b = self.b + (y / self.noise ** 2).unsqueeze(1) * phi
        self.counts.scatter_add_(
            1, idx.unsqueeze(1), torch.ones(self.T, 1, device=self.device, dtype=self.dtype))

        # 3. Rank-1 whitened precision update, batched: z = w = φ_{idx}/σ.
        z = phi / self.noise
        Aw_new, _ = refresh_bucket(self.L, self.Aw, z, z,
                                   force_reference=self.force_reference)
        self.Aw = Aw_new

        # 4. Simple regret of the current posterior-mean optimum.
        mean_f = torch.einsum("ij,tj->ti", self.Phi.mT, self.posterior_mean_v())
        best = mean_f.argmax(dim=1)
        achieved = self.f_true.gather(1, best.unsqueeze(1)).squeeze(1)
        regret = (self.f_max - achieved).clamp_min(0).mean().item()
        self.regret_history.append(regret)
        return regret

    # -- diagnostics ----------------------------------------------------------

    @torch.no_grad()
    def refactor_ground_truth(self, counts):
        """L, A_w from a full re-cholesky of G = I + (1/σ^2) Φ diag(counts) Φ^⊤."""
        eye = torch.eye(self.r, device=self.device, dtype=self.dtype)
        G = eye.unsqueeze(0) + (1.0 / self.noise ** 2) * (
            self.Phi.unsqueeze(0) * counts.unsqueeze(1)) @ self.Phi.mT.unsqueeze(0)
        L = torch.linalg.cholesky(G)
        Linv = torch.linalg.inv(L)
        Aw = eye - Linv @ Linv.mT
        return L, Aw
