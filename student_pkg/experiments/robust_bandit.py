"""Robust / sandwich Thompson sampling — double-sided under misspecification.

A contextual bandit whose posterior covariance is the heteroskedasticity-
consistent (Huber–White) *sandwich*

    Σ_t = G_t^{-1} M_t G_t^{-1},
    G_t = λ I + Σ_i x_i x_i^⊤        (the "bread": design Gram, z = x_i)
    M_t = λ_M I + Σ_i e_i^2 x_i x_i^⊤ (the "meat": score covariance, w = e_i^2 x_i)

with whitened sandwich operator A_t = L_t^{-1} M_t L_t^{-⊤} (L_t = chol G_t), so
Σ_t = L_t^{-⊤} A_t L_t^{-1} and Thompson draws are

    θ^{(s)} = θ̂_t + L_t^{-⊤} R_t ε,   R_t R_t^⊤ = A_t.

The key structural fact: under a *correctly specified*, homoskedastic model
e_i^2 ≈ σ^2 is constant, so M_t ≈ σ^2 G_t and A_t ≈ σ^2 I — the degenerate,
single-sided negative control. Under **heteroskedastic or heavy-tailed** noise
e_i^2 varies with x_i, so M_t ≠ σ^2 G_t and A_t separates from I: the sandwich is
real and the double-sided operator carries information a root-scaled identity
cannot.

Each observation is one rank-1 double-sided update (z = x_i, w = e_i^2 x_i, so
G += x x^⊤ and M += e^2 x x^⊤ share z). The learning step also *reads* A_t via a
covariance-level calibration (shrinkage toward a trace-scaled identity — a real
robustness regularizer), which a sampling root cannot do without reconstructing
A = R R^⊤. That calibration, not the (symmetric) structure, is what makes this
double-sided rather than a one-sided control.
"""

from __future__ import annotations

import torch

from .charts import refresh_bucket


class BatchedRobustSandwichTS:
    """T independent robust linear contextual bandits with sandwich TS.

    Args:
        r: context/feature dimension.
        T: number of parallel bandit tasks.
        n_arms: candidate contexts per round.
        ridge: λ on the design Gram G (and λ_M on M).
        noise_mode: 'homo' (constant σ — the negative control), 'hetero'
            (σ(x) grows with a fixed direction), or 'heavy' (Student-t, ν d.o.f.).
        noise_scale: base σ0.
        calibrate_every: rounds between covariance-level shrinkage calibrations.
        calibrate_rho: shrinkage strength ρ toward the trace-scaled identity.
        seed, force_reference, dtype, device: as elsewhere.
    """

    def __init__(self, r=32, T=256, n_arms=20, ridge=1.0, noise_mode="hetero",
                 noise_scale=0.3, hetero_beta=3.0, t_dof=2.5, calibrate_every=10,
                 calibrate_rho=0.1, seed=0, force_reference=False,
                 dtype=torch.float32, device=None):
        self.r = r
        self.T = T
        self.n_arms = n_arms
        self.lam = ridge
        self.noise_mode = noise_mode
        self.sigma0 = noise_scale
        self.beta = hetero_beta
        self.dof = t_dof
        self.calibrate_every = calibrate_every
        self.rho = calibrate_rho
        self.force_reference = force_reference
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.step_count = 0

        d = self.device
        eye = torch.eye(r, device=d, dtype=dtype)
        self._eye = eye
        self.G = (eye * self.lam).unsqueeze(0).repeat(T, 1, 1).contiguous()
        self.M = (eye * self.lam).unsqueeze(0).repeat(T, 1, 1).contiguous()
        self.L = torch.linalg.cholesky(self.G).contiguous()
        self.A = self._whiten(self.L, self.M)            # = (λ_M/λ) I
        self.b = torch.zeros(T, r, device=d, dtype=dtype)
        self.theta_hat = torch.zeros(T, r, device=d, dtype=dtype)

        # True parameter and the heteroskedasticity direction.
        self.theta_star = torch.randn(T, r, generator=self.gen).to(d, dtype)
        self.w_noise = torch.randn(T, r, generator=self.gen).to(d, dtype)
        self.w_noise = self.w_noise / self.w_noise.norm(dim=1, keepdim=True).clamp_min(1e-6)
        self.regret_history = []
        self.coverage_history = []

    @staticmethod
    def _whiten(L, M):
        Y = torch.linalg.solve_triangular(L, M, upper=False)
        return torch.linalg.solve_triangular(L, Y.mT, upper=False).mT

    # -- noise model ---------------------------------------------------------

    def _noise(self, x):
        """Per-task observation noise given the chosen context x (T, r)."""
        base = torch.randn(self.T, generator=self.gen).to(self.device, self.dtype)
        if self.noise_mode == "homo":
            return self.sigma0 * base
        if self.noise_mode == "heavy":
            # Student-t via normal / sqrt(chi2/ν): heavy-tailed, x-independent.
            g = torch.randn(self.T, self.dof_int(), generator=self.gen).to(self.device, self.dtype)
            chi2 = (g ** 2).sum(dim=1) / self.dof_int()
            return self.sigma0 * base / chi2.clamp_min(1e-6).sqrt()
        # hetero: σ(x) grows with |w_noise·x|.
        proj = (x * self.w_noise).sum(dim=1).abs()
        sigma = self.sigma0 * (1.0 + self.beta * proj)
        return sigma * base

    def dof_int(self):
        return max(1, int(round(self.dof)))

    # -- one round -----------------------------------------------------------

    def step(self):
        self.step_count += 1
        d = self.device
        # Candidate contexts for this round.
        X = torch.randn(self.T, self.n_arms, self.r, generator=self.gen).to(d, self.dtype)

        # Thompson: θ^(s) = θ̂ + L^{-⊤} R ε,  R = chol(A).
        R = torch.linalg.cholesky(self.A + 1e-5 * self._eye)
        eps = torch.randn(self.T, self.r, generator=self.gen).to(d, self.dtype)
        Reps = torch.einsum("tij,tj->ti", R, eps)
        samp = torch.linalg.solve_triangular(
            self.L.mT, Reps.unsqueeze(-1), upper=True).squeeze(-1)   # L^{-⊤} R ε
        theta_s = self.theta_hat + samp
        scores = torch.einsum("tkr,tr->tk", X, theta_s)
        k = scores.argmax(dim=1)                                     # (T,)
        x = X[torch.arange(self.T, device=d), k]                     # chosen context (T, r)

        # Observe reward with (mis)specified noise.
        mean = (x * self.theta_star).sum(dim=1)
        y = mean + self._noise(x)
        # Residual under the current estimate (the sandwich "score").
        e = y - (x * self.theta_hat).sum(dim=1)

        # Rank-1 double-sided update: z = x, w = e^2 x  (G += x x^⊤, M += e^2 x x^⊤).
        z = x
        w = (e ** 2).unsqueeze(1) * x
        self.G = self.G + z.unsqueeze(2) * z.unsqueeze(1)
        self.M = self.M + z.unsqueeze(2) * w.unsqueeze(1)
        A_new, _ = refresh_bucket(self.L, self.A, z, w,
                                  force_reference=self.force_reference)
        self.A = A_new

        # Point estimate θ̂ = G^{-1} b (ridge regression).
        self.b = self.b + y.unsqueeze(1) * x
        yb = torch.linalg.solve_triangular(self.L, self.b.unsqueeze(-1), upper=False)
        self.theta_hat = torch.linalg.solve_triangular(
            self.L.mT, yb, upper=True).squeeze(-1)

        # Covariance-level calibration (reads full A; root cannot do this):
        # shrink toward a trace-scaled identity for robustness.
        if self.step_count % self.calibrate_every == 0:
            tau = (self.A.diagonal(dim1=1, dim2=2).mean(dim=1)).reshape(-1, 1, 1)
            self.A = (1 - self.rho) * self.A + self.rho * tau * self._eye
            self.M = (1 - self.rho) * self.M + self.rho * tau * self.G   # keep exact in sync

        # Metrics: instantaneous regret + posterior coverage of θ*.
        best = (X * self.theta_star.unsqueeze(1)).sum(dim=2).max(dim=1).values
        regret = (best - mean).clamp_min(0).mean().item()
        self.regret_history.append(regret)
        self.coverage_history.append(self._coverage())
        return regret

    @torch.no_grad()
    def _coverage(self, level=2.0):
        """Fraction of tasks whose θ* lies within the sandwich credible ellipsoid.

        Mahalanobis (θ*−θ̂)^⊤ Σ^{-1} (θ*−θ̂) with Σ = L^{-⊤} A L^{-1}, so
        Σ^{-1} = L A^{-1} L^⊤ and the quadratic form = ‖R^{-1} L^⊤ δ‖^2.
        """
        delta = self.theta_star - self.theta_hat
        Ltd = torch.einsum("tji,tj->ti", self.L, delta)     # L^⊤ δ
        R = torch.linalg.cholesky(self.A + 1e-5 * self._eye)
        u = torch.linalg.solve_triangular(R, Ltd.unsqueeze(-1), upper=False).squeeze(-1)
        maha = (u ** 2).sum(dim=1) / self.r                 # normalized
        return (maha <= level).float().mean().item()

    # -- diagnostics ---------------------------------------------------------

    @torch.no_grad()
    def refactor_ground_truth(self):
        L = torch.linalg.cholesky(self.G)
        return L, self._whiten(L, self.M)

    @torch.no_grad()
    def separation_from_identity(self):
        """Mean ‖A − (tr A/r) I‖/‖A‖ — how far the sandwich is from a scaled I.

        ≈ 0 for a correctly specified homoskedastic model (negative control);
        O(1) under heteroskedastic / heavy-tailed misspecification.
        """
        tau = self.A.diagonal(dim1=1, dim2=2).mean(dim=1).reshape(-1, 1, 1)
        num = torch.linalg.norm((self.A - tau * self._eye).reshape(self.T, -1), dim=1)
        den = torch.linalg.norm(self.A.reshape(self.T, -1), dim=1).clamp_min(1e-12)
        return (num / den).mean().item()


class RealDataSandwichTS(BatchedRobustSandwichTS):
    """Robust sandwich TS driven by a real dataset (off-data replay).

    Each round presents ``n_arms`` real rows as candidate arms; the learner
    Thompson-picks one and observes that row's REAL (heavy-tailed) reward, so the
    misspecification that separates A from I comes from the data, not injected
    noise. T parallel replays use independent row subsamples.

    Set ``gaussianize=True`` for the matched control: replace the real reward by
    θ_OLS·x + homoskedastic Gaussian noise of the fitted residual variance — same
    linear signal, tails removed — so the drop in separation isolates the effect
    of the real heavy tail.

    Args:
        contexts: (N, d) real contexts.
        rewards: (N,) real (standardized) rewards.
        T, n_arms, ridge, calibrate_every, calibrate_rho, seed, force_reference,
        dtype, device: as in the base class (r is inferred from contexts).
    """

    def __init__(self, contexts, rewards, T=64, n_arms=16, ridge=1.0,
                 gaussianize=False, calibrate_every=10 ** 9, calibrate_rho=0.1,
                 seed=0, force_reference=False, dtype=torch.float32, device=None):
        device = device or contexts.device
        contexts = contexts.to(device, dtype)
        rewards = rewards.to(device, dtype)
        r = contexts.shape[1]
        # Base init sets up G, M, L, A, b, theta_hat; we override the data hooks.
        super().__init__(r=r, T=T, n_arms=n_arms, ridge=ridge, noise_mode="homo",
                         calibrate_every=calibrate_every, calibrate_rho=calibrate_rho,
                         seed=seed, force_reference=force_reference,
                         dtype=dtype, device=device)
        self.C = contexts
        self.rewards = rewards
        self.N = contexts.shape[0]
        self.gaussianize = gaussianize
        # OLS fit on the full data, for the matched-Gaussian control and residual σ.
        G0 = contexts.T @ contexts + ridge * self._eye
        theta_ols = torch.linalg.solve(G0, contexts.T @ rewards)
        resid = rewards - contexts @ theta_ols
        self.theta_ols = theta_ols
        self.sigma_ols = resid.std().clamp_min(1e-6)

    def step(self):
        self.step_count += 1
        d = self.device
        # Present n_arms real rows per task as candidate arms.
        idx = torch.randint(0, self.N, (self.T, self.n_arms), generator=self.gen).to(d)
        X = self.C[idx]                                   # (T, n_arms, r)

        R = torch.linalg.cholesky(self.A + 1e-5 * self._eye)
        eps = torch.randn(self.T, self.r, generator=self.gen).to(d, self.dtype)
        Reps = torch.einsum("tij,tj->ti", R, eps)
        samp = torch.linalg.solve_triangular(
            self.L.mT, Reps.unsqueeze(-1), upper=True).squeeze(-1)
        theta_s = self.theta_hat + samp
        scores = torch.einsum("tkr,tr->tk", X, theta_s)
        k = scores.argmax(dim=1)
        chosen = idx[torch.arange(self.T, device=d), k]   # chosen row per task
        x = self.C[chosen]                                # (T, r)

        if self.gaussianize:
            y = (x * self.theta_ols).sum(dim=1) + self.sigma_ols * torch.randn(
                self.T, generator=self.gen).to(d, self.dtype)
        else:
            y = self.rewards[chosen]                      # REAL heavy-tailed reward
        e = y - (x * self.theta_hat).sum(dim=1)

        z = x
        w = (e ** 2).unsqueeze(1) * x
        self.G = self.G + z.unsqueeze(2) * z.unsqueeze(1)
        self.M = self.M + z.unsqueeze(2) * w.unsqueeze(1)
        A_new, _ = refresh_bucket(self.L, self.A, z, w, force_reference=self.force_reference)
        self.A = A_new

        self.b = self.b + y.unsqueeze(1) * x
        yb = torch.linalg.solve_triangular(self.L, self.b.unsqueeze(-1), upper=False)
        self.theta_hat = torch.linalg.solve_triangular(self.L.mT, yb, upper=True).squeeze(-1)

        if self.step_count % self.calibrate_every == 0:
            tau = (self.A.diagonal(dim1=1, dim2=2).mean(dim=1)).reshape(-1, 1, 1)
            self.A = (1 - self.rho) * self.A + self.rho * tau * self._eye
            self.M = (1 - self.rho) * self.M + self.rho * tau * self.G

        # Regret vs the best real reward among the presented arms.
        arm_rewards = (self.rewards[idx] if not self.gaussianize
                       else (X * self.theta_ols).sum(dim=2))
        best = arm_rewards.max(dim=1).values
        got = arm_rewards[torch.arange(self.T, device=d), k]
        regret = (best - got).clamp_min(0).mean().item()
        self.regret_history.append(regret)
        return regret
