"""Nonstationary sliding-window sandwich Thompson sampling.

Showcases both paper extensions on a genuinely double-sided workload:

  * batched TS acquisition — each round acquires a batch of k arms and folds
    them in as ONE rank-k positive update (Appendix C.3), G += ZZᵀ, M += ZWᵀ;

  * sliding-window forgetting — when the window is full the oldest batch is
    removed by a leverage-guarded hyperbolic downdate (Appendix C.4), G −= ZZᵀ,
    M −= ZWᵀ, so the whitened sandwich A_t = L⁻¹ M_t L⁻ᵀ tracks only the last W
    rounds and adapts when the environment changes.

The reward parameter θ⋆ switches at the midpoint (an abrupt regime change). The
sliding window forgets pre-switch data and recovers; a no-forget baseline that
accumulates everything stays biased by stale observations. The reward noise is
heteroskedastic, so the sandwich A_t ≠ I and the double-sided maintenance is
real (not a covariance root).

Verified: the windowed A_t matches a full re-cholesky of the current window to
machine precision; the guard skips inadmissible downdates (τ ≥ threshold).
"""

from __future__ import annotations

import torch

from .extensions import rank_k_update, downdate


class SlidingWindowSandwichTS:
    """T nonstationary robust bandits over a sliding window of the last W rounds.

    Args:
        r: feature dimension.
        T: parallel tasks.
        n_arms: candidate contexts per round.
        batch_k: arms acquired per round (the rank-k batch).
        window: number of rounds retained before forgetting begins.
        ridge, hetero_beta, noise_scale: model + noise settings.
        forget: if False, never downdate (the no-forget baseline).
        max_leverage: downdate guard threshold.
        seed, force_reference, dtype, device: as elsewhere.
    """

    def __init__(self, r=24, T=128, n_arms=16, batch_k=4, window=25, ridge=1.0,
                 hetero_beta=3.0, noise_scale=0.3, forget=True, max_leverage=0.98,
                 seed=0, force_reference=False, dtype=torch.float32, device=None):
        self.r, self.T, self.n_arms, self.k = r, T, n_arms, batch_k
        self.window = window
        self.lam = ridge
        self.beta = hetero_beta
        self.sigma0 = noise_scale
        self.forget = forget
        self.max_leverage = max_leverage
        self.force_reference = force_reference
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.step_count = 0
        self.n_skipped = 0

        d = self.device
        eye = torch.eye(r, device=d, dtype=dtype)
        self._eye = eye
        self.G = (eye * self.lam).unsqueeze(0).repeat(T, 1, 1).contiguous()
        self.M = (eye * self.lam).unsqueeze(0).repeat(T, 1, 1).contiguous()
        self.L = torch.linalg.cholesky(self.G).contiguous()
        self.A = eye.unsqueeze(0).repeat(T, 1, 1).contiguous()   # A = (λ_M/λ)I = I
        self.b = torch.zeros(T, r, device=d, dtype=dtype)
        self.theta_hat = torch.zeros(T, r, device=d, dtype=dtype)

        # Two regimes; θ⋆ switches at the midpoint.
        self.theta_a = torch.randn(T, r, generator=self.gen).to(d, dtype)
        self.theta_b = torch.randn(T, r, generator=self.gen).to(d, dtype)
        self.w_noise = torch.randn(T, r, generator=self.gen).to(d, dtype)
        self.w_noise = self.w_noise / self.w_noise.norm(dim=1, keepdim=True).clamp_min(1e-6)

        self.buffer = []   # deque of (Z, W, db) per round, for exact forgetting
        self.regret_history = []
        self.switch_at = None   # set by run loop

    def theta_star(self):
        sw = self.switch_at if self.switch_at is not None else 10 ** 9
        return self.theta_b if self.step_count > sw else self.theta_a

    def step(self):
        self.step_count += 1
        d = self.device
        X = torch.randn(self.T, self.n_arms, self.r, generator=self.gen).to(d, self.dtype)

        # Batched Thompson acquisition: sample θ^(s), take the top-k arms.
        R = torch.linalg.cholesky(self.A + 1e-5 * self._eye)
        eps = torch.randn(self.T, self.r, generator=self.gen).to(d, self.dtype)
        Reps = torch.einsum("tij,tj->ti", R, eps)
        samp = torch.linalg.solve_triangular(
            self.L.mT, Reps.unsqueeze(-1), upper=True).squeeze(-1)
        theta_s = self.theta_hat + samp
        scores = torch.einsum("tkr,tr->tk", X, theta_s)
        topk = scores.topk(self.k, dim=1).indices                # (T, k)
        Xa = torch.gather(X, 1, topk.unsqueeze(-1).expand(-1, -1, self.r))  # (T,k,r)

        # Observe heteroskedastic rewards for the batch.
        theta = self.theta_star()
        mean = torch.einsum("tkr,tr->tk", Xa, theta)             # (T, k)
        proj = torch.einsum("tkr,tr->tk", Xa, self.w_noise).abs()
        sigma = self.sigma0 * (1.0 + self.beta * proj)
        y = mean + sigma * torch.randn(self.T, self.k, generator=self.gen).to(d, self.dtype)
        e = y - torch.einsum("tkr,tr->tk", Xa, self.theta_hat)   # residuals (T, k)

        # Rank-k add: Z = x's, W = e²·x's  (G += ZZᵀ, M += ZWᵀ).
        Z = Xa.transpose(1, 2).contiguous()                      # (T, r, k)
        Wm = (e.unsqueeze(2) ** 2 * Xa).transpose(1, 2).contiguous()
        self.A, _ = rank_k_update(self.L, self.A, Z, Wm, force_reference=self.force_reference)
        self.G = self.G + Z @ Z.mT
        self.M = self.M + Z @ Wm.mT
        db = torch.einsum("tk,tkr->tr", y, Xa)
        self.b = self.b + db
        self.buffer.append((Z, Wm, db))

        # Sliding-window forget: hyperbolic downdate of the oldest batch.
        if self.forget and len(self.buffer) > self.window:
            Zo, Wo, dbo = self.buffer.pop(0)
            for j in range(self.k):
                A_new, admitted = downdate(self.L, self.A, Zo[:, :, j].contiguous(),
                                           Wo[:, :, j].contiguous(),
                                           max_leverage=self.max_leverage)
                self.A = A_new
                # Keep exact G, M in sync only for admitted charts.
                m = admitted.reshape(-1, 1, 1).to(self.dtype)
                zj = Zo[:, :, j]
                wj = Wo[:, :, j]
                self.G = self.G - m * (zj.unsqueeze(2) * zj.unsqueeze(1))
                self.M = self.M - m * (zj.unsqueeze(2) * wj.unsqueeze(1))
                self.n_skipped += int((~admitted).sum())
            self.b = self.b - dbo

        # Refresh θ̂ = G⁻¹ b from the current window.
        yb = torch.linalg.solve_triangular(self.L, self.b.unsqueeze(-1), upper=False)
        self.theta_hat = torch.linalg.solve_triangular(self.L.mT, yb, upper=True).squeeze(-1)

        # Regret of the acquired batch under the current θ⋆.
        best = mean.max(dim=1).values                            # best presented arm (approx)
        best_all = torch.einsum("tkr,tr->tk", X, theta).max(dim=1).values
        chosen = mean.mean(dim=1)
        regret = (best_all - chosen).clamp_min(0).mean().item()
        self.regret_history.append(regret)
        return regret

    @torch.no_grad()
    def refactor_ground_truth(self):
        """A from a full re-cholesky of the CURRENT window's exact (G, M)."""
        L = torch.linalg.cholesky(self.G)
        Y = torch.linalg.solve_triangular(L, self.M, upper=False)
        A = torch.linalg.solve_triangular(L, Y.mT, upper=False).mT
        return L, A
