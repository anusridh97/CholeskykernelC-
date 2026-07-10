"""Massively parallel GP Thompson sampling via batched rank-1 conditioning.

Hundreds to thousands of independent GP surrogates on one GPU — parallel bandit
arms, TuRBO-style trust regions, or many BO seeds/restarts — each taking a
rank-1 conditioning update per round, all fused into a single kernel launch.
This is the purest expression of "T is where we win": the MLX section only
reached T = 16, and this scales the batched kernel to large T on real hardware.

Why normal BoTorch/GPyTorch has a hard time with it: the idiomatic path is one
model object per surrogate, each re-conditioned (or refit) independently and
sequentially, at O(r^3) per model. Here every arm shares one launch and pays
O(r^2) per rank-1 conditioning step, with all state device-resident between
rounds.

Setup (exact recursive Gaussian conditioning / information filter):

  * Each arm has the SAME fixed candidate set X (r points) and RBF prior
    f ~ N(0, K). We maintain, per arm, the Cholesky factor L of the posterior
    *precision* Lambda = K^{-1} + (1/sigma^2) sum_k e_{i_k} e_{i_k}^T and the
    whitened operator A_w = L^{-1} K L^{-T}.
  * Observing candidate i is a rank-1 precision bump  Lambda += (1/sigma^2) e_i e_i^T,
    i.e. z = (1/sigma) e_i, w = 0 (K, hence M, stays fixed while L changes, so
    the double-sided propagation of A_w is exercised every round).
  * The information vector b = (1/sigma^2) sum_k y_k e_{i_k} is accumulated
    on-device; the posterior mean mu = Lambda^{-1} b and Thompson draws
    f = mu + L^{-T} xi come from two triangular solves with the maintained L.

Ground truth for correctness is the full refactorization L = chol(Lambda),
reusing the same equivalence the kernel's correctness tests establish.
"""

from __future__ import annotations

import torch

from .charts import refresh_bucket, gpu_kernel_available  # noqa: F401


def rbf_kernel(X, lengthscale=0.1, jitter=1e-4):
    """RBF/Gaussian kernel matrix for candidate points X (r, dims)."""
    d2 = torch.cdist(X, X) ** 2
    K = torch.exp(-0.5 * d2 / lengthscale ** 2)
    K = K + jitter * torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
    return K


class BatchedGPThompson:
    """T independent GP arms over a shared candidate set, conditioned in batch.

    Args:
        X: (r, dims) candidate points (shared across arms).
        objective: callable X (r, dims) -> (r,) true function values; each arm
            optimizes a shifted copy so arms are genuinely independent problems.
        T: number of parallel arms.
        noise: observation noise std sigma.
        lengthscale, jitter: RBF hyperparameters.
        seed: RNG seed.
        force_reference: force the CPU reference refresh (for cross-checks).
    """

    def __init__(
        self,
        X,
        objective,
        T=256,
        noise=0.1,
        lengthscale=0.1,
        jitter=1e-4,
        seed=0,
        force_reference=False,
    ):
        self.X = X
        self.r = X.shape[0]
        self.T = T
        self.noise = noise
        self.device = X.device
        self.dtype = X.dtype
        self.force_reference = force_reference
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

        # Shared prior; each arm gets an independent objective realization by a
        # per-arm phase shift, so they explore different landscapes.
        K = rbf_kernel(X, lengthscale, jitter)
        self.K = K
        Kinv = torch.linalg.inv(K)
        # Prior precision factor, replicated across arms.
        L0 = torch.linalg.cholesky(Kinv).contiguous()
        self.L = L0.unsqueeze(0).repeat(T, 1, 1).contiguous()
        # M = K (fixed); A_w = L^{-1} K L^{-T} refreshed each round.
        self.Aw = self._init_aw(L0, K, T)

        # Per-arm true objective values on the grid (shifted per arm).
        base = objective(X)  # (r,)
        shifts = torch.arange(T, device=self.device, dtype=self.dtype).reshape(T, 1)
        self.f_true = base.unsqueeze(0) + 0.05 * torch.sin(
            shifts + torch.arange(self.r, device=self.device, dtype=self.dtype)
        )
        self.f_max = self.f_true.max(dim=1).values  # (T,)

        # Information vector b, accumulated on device.
        self.b = torch.zeros(T, self.r, device=self.device, dtype=self.dtype)
        # Per-candidate observation counts, for the refactorization ground truth.
        self.counts = torch.zeros(T, self.r, device=self.device, dtype=self.dtype)
        self.regret_history = []

    def _init_aw(self, L0, K, T):
        Aw0 = torch.linalg.solve_triangular(
            L0, K, upper=False
        )
        Aw0 = torch.linalg.solve_triangular(L0, Aw0.mT, upper=False).mT
        return Aw0.unsqueeze(0).repeat(T, 1, 1).contiguous()

    def posterior_mean(self):
        """mu = Lambda^{-1} b via two triangular solves with L (per arm)."""
        y = torch.linalg.solve_triangular(self.L, self.b.unsqueeze(-1), upper=False)
        mu = torch.linalg.solve_triangular(self.L.mT, y, upper=True).squeeze(-1)
        return mu

    def thompson_sample(self):
        """Draw f ~ N(mu, Lambda^{-1}) per arm: f = mu + L^{-T} xi."""
        mu = self.posterior_mean()
        xi = torch.randn(self.T, self.r, generator=self.gen).to(self.device, self.dtype)
        sample = torch.linalg.solve_triangular(
            self.L.mT, xi.unsqueeze(-1), upper=True
        ).squeeze(-1)
        return mu + sample

    def step(self):
        """One Thompson round for all T arms in a single batched conditioning.

        Returns the mean instantaneous simple regret across arms this round.
        """
        # 1. Thompson-select a candidate per arm.
        f = self.thompson_sample()
        idx = f.argmax(dim=1)  # (T,)

        # 2. Observe (noisy) and accumulate the information vector.
        y = self.f_true.gather(1, idx.unsqueeze(1)).squeeze(1)
        y = y + self.noise * torch.randn(self.T, generator=self.gen).to(self.device, self.dtype)
        self.b.scatter_add_(1, idx.unsqueeze(1), (y / self.noise ** 2).unsqueeze(1))
        self.counts.scatter_add_(
            1, idx.unsqueeze(1), torch.ones(self.T, 1, device=self.device, dtype=self.dtype)
        )

        # 3. Rank-1 precision update, batched: z = (1/sigma) e_i, w = 0.
        z = torch.zeros(self.T, self.r, device=self.device, dtype=self.dtype)
        z.scatter_(1, idx.unsqueeze(1), 1.0 / self.noise)
        w = torch.zeros_like(z)
        Aw_new, _ = refresh_bucket(
            self.L, self.Aw, z, w, force_reference=self.force_reference
        )
        self.Aw = Aw_new

        # 4. Track simple regret (best-so-far predicted optimum vs true max).
        mu = self.posterior_mean()
        best_pred_idx = mu.argmax(dim=1)
        achieved = self.f_true.gather(1, best_pred_idx.unsqueeze(1)).squeeze(1)
        regret = (self.f_max - achieved).clamp_min(0).mean().item()
        self.regret_history.append(regret)
        return regret

    # -- diagnostics ---------------------------------------------------------

    @torch.no_grad()
    def refactor_ground_truth(self, counts):
        """L, A_w from a full re-cholesky of Lambda = K^{-1} + (1/s^2)diag(counts)."""
        Kinv = torch.linalg.inv(self.K)
        Lam = Kinv.unsqueeze(0) + (1.0 / self.noise ** 2) * torch.diag_embed(counts)
        L = torch.linalg.cholesky(Lam)
        Linv_K = torch.linalg.solve_triangular(L, self.K.unsqueeze(0).expand_as(Lam), upper=False)
        Aw = torch.linalg.solve_triangular(L, Linv_K.mT, upper=False).mT
        return L, Aw
