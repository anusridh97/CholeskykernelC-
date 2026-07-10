"""Non-symmetric whitened dynamics operator — a decisively double-sided workload.

Massive parallelism alone does not make Thompson sampling double-sided. For a
Gaussian posterior, ordinary parallel TS only needs a *sampling root* A^{1/2}ε,
which can be transported one-sidedly (R_t ← P_t^⊤ R_t) followed by a Cholesky
update. Scaling workers from 1 to 10,000 just enlarges a matmul; it never forces
you to maintain A_t = L^{-1} M_t L^{-⊤} explicitly. A convincing double-sided
experiment needs the posterior-learning step to *read or modify A_t itself*.

The cleanest algebraically unambiguous case is a **non-symmetric** operator,
where a single-sided covariance root does not exist at all. For an online
Bayesian linear dynamical system we maintain

    G_t = λ I + Σ_{i<t} φ_i φ_i^⊤          (SPD metric, z = φ_i)
    M_t = Σ_{i<t} φ_i (Θ⋆ φ_i)^⊤          (lag-one cross-moment, w = response)
    A_t = L_t^{-1} M_t L_t^{-⊤}            (whitened dynamics operator)

with L_t = chol(G_t). Because w ≠ z, generally A_t ≠ A_t^⊤, so A_t is NOT
representable as R_t R_t^⊤ — the root-only baseline is mathematically
inapplicable, not merely slower. Each transition is one rank-1 double-sided
update G += z z^⊤, M += z w^⊤ (the kernel's genuine z≠w path), and A_t is
transported in O(r^2); the baseline recomputes L, A from scratch in O(r^3).

Model-based Thompson sampling for adaptive control *reads A_t*: each round we
draw candidate dynamics, roll them out K steps in whitened coordinates
(ξ_{k+1} = A^{(s)} ξ_k — the paper's operator recurrence, Appendix C.7), score
candidate actions by predicted cost, act, observe, and update. The rollout
consumes the full non-symmetric A_t, which no covariance root can supply.

Batched over T independent systems (one Givens sweep each, one launch for all),
this is also the massive-parallelism story — but now the parallelism sits on top
of a genuinely double-sided primitive.
"""

from __future__ import annotations

import torch

from .charts import refresh_bucket


def _stable_operator(T, r, device, dtype, gen, radius=0.9):
    """Random per-system dynamics Θ⋆ with spectral radius < 1 (stable)."""
    A = torch.randn(T, r, r, device=device, dtype=dtype, generator=gen)
    # Scale rows so the operator norm is comfortably below 1.
    n = torch.linalg.matrix_norm(A, ord=2, dim=(1, 2)).clamp_min(1e-6)
    return radius * A / n.reshape(T, 1, 1)


class BatchedDynamicsThompson:
    """T independent online Bayesian linear systems with double-sided A_t.

    Each system estimates its dynamics operator Θ⋆ from transitions, maintains
    the whitened (non-symmetric) operator A_t via the shared primitive, and runs
    model-based Thompson sampling that reads A_t through K-step rollouts.

    Args:
        r: feature/state dimension (chart rank).
        T: number of parallel systems.
        damping: λ ridge on the metric G.
        noise: process noise std.
        horizon: rollout length K used when scoring candidate actions.
        n_actions: number of candidate inputs per round.
        seed: RNG seed.
        force_reference: force the CPU reference refresh (for cross-checks).
        dtype: tensor dtype.
        device: torch device.
    """

    def __init__(self, r=64, T=256, damping=1.0, noise=0.05, horizon=1,
                 n_actions=8, seed=0, force_reference=False,
                 dtype=torch.float32, device=None):
        self.r = r
        self.T = T
        self.lam = damping
        self.noise = noise
        self.K = horizon
        self.n_actions = n_actions
        self.force_reference = force_reference
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

        d = self.device
        eye = torch.eye(r, device=d, dtype=dtype)
        # Metric factor L = chol(G), G init λI; whitened operator A_t init 0 (M=0).
        self.L = (eye * self.lam ** 0.5).unsqueeze(0).repeat(T, 1, 1).contiguous()
        self.A = torch.zeros(T, r, r, device=d, dtype=dtype)
        # Exact G, M for the refactorization ground truth / drift diagnostic.
        self.G = (eye * self.lam).unsqueeze(0).repeat(T, 1, 1).contiguous()
        self.M = torch.zeros(T, r, r, device=d, dtype=dtype)

        # True dynamics + fixed candidate input effects per system.
        self.Theta = _stable_operator(T, r, d, dtype, self._cudagen(seed))
        self.U = 0.3 * torch.randn(T, self.n_actions, r, device=d, dtype=dtype,
                                   generator=self._cudagen(seed + 1))
        # Current state and a per-system target.
        self.phi = 0.1 * torch.randn(T, r, device=d, dtype=dtype, generator=self._cudagen(seed + 2))
        self.target = torch.randn(T, r, device=d, dtype=dtype, generator=self._cudagen(seed + 3))
        self.regret_history = []

    def _cudagen(self, s):
        g = torch.Generator(device="cpu").manual_seed(s)
        return g

    # -- posterior helpers ---------------------------------------------------

    def theta_hat(self):
        """Posterior-mean dynamics Θ̂ = M^⊤ G^{-1} (feature→feature), per system."""
        # G^{-1} M via two triangular solves, then transpose.
        y = torch.linalg.solve_triangular(self.L, self.M, upper=False)
        GinvM = torch.linalg.solve_triangular(self.L.mT, y, upper=True)  # (T,r,r)
        return GinvM.mT  # Θ̂ = M^⊤ G^{-1} = (G^{-1} M)^⊤

    def sample_operator(self):
        """Thompson draw of a candidate whitened operator: A^{(s)} = A + scale·Ω.

        Reads A_t as the posterior-mean whitened operator and perturbs it by
        posterior-scale noise. Non-symmetric, so no root representation exists.
        """
        scale = self.noise / (self.G.diagonal(dim1=1, dim2=2).mean(dim=1).sqrt() + 1e-6)
        Omega = torch.randn(self.T, self.r, self.r, device=self.device,
                            dtype=self.dtype, generator=self.gen)
        return self.A + scale.reshape(self.T, 1, 1) * Omega

    def step(self):
        """One model-based Thompson round for all T systems; returns mean regret."""
        # 1. Draw a candidate whitened operator A^{(s)} (reads A_t) and use it to
        #    predict the dynamics response in feature space. Since Θ̂ = M^⊤ G^{-1}
        #    and A = L^{-1} M L^{-⊤}, one has Θ̂ φ = L A^⊤ L^{-1} φ — one whitened
        #    solve, one read of the (non-symmetric) operator, one lift:
        #        ζ = A^{(s)⊤} (L^{-1} φ),   response = L ζ.
        As = self.sample_operator()                            # (T, r, r)
        xi0 = torch.linalg.solve_triangular(
            self.L, self.phi.unsqueeze(-1), upper=False).squeeze(-1)  # ξ0 = L^{-1} φ
        # K-step model-predictive rollout in feature space (reads A each step);
        # actions enter at the first step only.
        zeta = torch.einsum("tji,tj->ti", As, xi0)             # ζ = A^{(s)⊤} ξ0
        response = torch.einsum("tij,tj->ti", self.L, zeta)    # Θ̂ φ (feature space)

        best_cost = None
        best_action = None
        for a in range(self.n_actions):
            pred = response + self.U[:, a]                     # predicted φ_next
            for _ in range(self.K - 1):                        # further free rollout
                xi = torch.linalg.solve_triangular(
                    self.L, pred.unsqueeze(-1), upper=False).squeeze(-1)
                z2 = torch.einsum("tji,tj->ti", As, xi)
                pred = torch.einsum("tij,tj->ti", self.L, z2)
            cost = ((pred - self.target) ** 2).sum(dim=1)      # (T,)
            if best_cost is None:
                best_cost, best_action = cost, torch.full((self.T,), a, device=self.device)
            else:
                take = cost < best_cost
                best_cost = torch.where(take, cost, best_cost)
                best_action = torch.where(take, a, best_action)

        # 3. Apply best action, observe the true transition.
        Ua = self.U[torch.arange(self.T), best_action]         # (T, r)
        drive = torch.einsum("tij,tj->ti", self.Theta, self.phi)  # Θ⋆ φ
        phi_next = drive + Ua + self.noise * torch.randn(
            self.T, self.r, device=self.device, dtype=self.dtype, generator=self.gen)

        # 4. Double-sided rank-1 update: z = φ, w = dynamics response (Θ⋆φ).
        z = self.phi
        w = phi_next - Ua                                       # observed Θ⋆φ (+noise)
        self.G = self.G + z.unsqueeze(2) * z.unsqueeze(1)
        self.M = self.M + z.unsqueeze(2) * w.unsqueeze(1)
        A_new, _ = refresh_bucket(self.L, self.A, z, w,
                                  force_reference=self.force_reference)
        self.A = A_new

        # 5. Regret: distance to target under the chosen action vs the best action
        #    an oracle (knowing Θ⋆) would have taken.
        with torch.no_grad():
            reg = self._oracle_regret(Ua)
        self.phi = phi_next
        self.regret_history.append(reg)
        return reg

    @torch.no_grad()
    def _oracle_regret(self, chosen_Ua):
        drive = torch.einsum("tij,tj->ti", self.Theta, self.phi)
        # cost of each action under true dynamics (one-step, deterministic part).
        costs = []
        for a in range(self.n_actions):
            nxt = drive + self.U[:, a]
            costs.append(((nxt - self.target) ** 2).sum(dim=1))
        costs = torch.stack(costs, dim=1)                      # (T, A)
        chosen = ((drive + chosen_Ua - self.target) ** 2).sum(dim=1)
        return (chosen - costs.min(dim=1).values).clamp_min(0).mean().item()

    # -- diagnostics ---------------------------------------------------------

    @torch.no_grad()
    def refactor_ground_truth(self):
        """A from a full re-cholesky: L = chol(G), A = L^{-1} M L^{-⊤}."""
        L = torch.linalg.cholesky(self.G)
        Y = torch.linalg.solve_triangular(L, self.M, upper=False)          # L^{-1} M
        A = torch.linalg.solve_triangular(L, Y.mT, upper=False).mT         # (L^{-1} M) L^{-⊤}
        return L, A

    @torch.no_grad()
    def symmetry_gap(self):
        """Mean ‖A − A^⊤‖/‖A‖ — how far A is from any R R^⊤ representation."""
        num = torch.linalg.norm((self.A - self.A.mT).reshape(self.T, -1), dim=1)
        den = torch.linalg.norm(self.A.reshape(self.T, -1), dim=1).clamp_min(1e-12)
        return (num / den).mean().item()
