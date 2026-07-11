"""Adaptive-inducing full-covariance SVGP with parallel Thompson sampling.

The flagship double-sided variational experiment. Unlike the one-sided negative
control (parallel_gp_thompson.py), here the whitened variational covariance

    A_t = L_t^{-1} S_t L_t^{-⊤},     K_{uu,t} = L_t L_t^⊤,   q_t(u) = N(m_t, S_t)

is the *authoritative* variational state, and the posterior-learning step reads
and modifies A_t directly — so a sampling root A_t = R_t R_t^⊤ is not a
sufficient representation.

Each round does three things to A_t:

  1. Inducing-basis change. Moving/replacing inducing points changes K_{uu} by a
     rank-1 update. The double-sided primitive transports A_t into the new
     whitened basis in O(r^2): with M = S_t and w = 0,
     A_t ← TL_r(Q^⊤ Â_t Q) = L_{t+1}^{-1} S_t L_{t+1}^{-⊤}. This is genuinely
     double-sided (both L^{-1} and L^{-⊤} change) even though w = 0.

  2. Covariance-form conditioning. Observing an inducing value is a rank-1
     Kalman downdate that READS the full A_t on both sides:
     A ← A − (A φ̃)(A φ̃)^⊤ / (φ̃^⊤ A φ̃ + σ^2),  φ̃ = L^⊤ e_j.

  3. Trust-region damping toward the prior (a covariance-LEVEL operation a root
     cannot do without reconstructing A = R R^⊤): A ← (1−ρ)A + ρ I.

Then S independent function samples are drawn per task (u = m + L R ξ, R a root
of A used only for sampling) and Thompson-acquired in parallel.

Baselines (see tests/bench_svgp_pdts.py): full re-factorization (recompute L, A
each round, O(r^3)); root-only (maintain R with A = R R^⊤ — transports and
conditions via Cholesky updates, but must reconstruct A at O(r^3) to damp);
growing one-sided factor (exact but O(r t) memory).

Correctness: the maintained A_t matches a full re-cholesky of the exact (S_t,
G_t) to machine precision after transport + conditioning + damping.
"""

from __future__ import annotations

import torch

from .charts import refresh_bucket


def rbf_kernel(X, lengthscale=0.2, jitter=1e-3):
    d2 = torch.cdist(X, X) ** 2
    K = torch.exp(-0.5 * d2 / lengthscale ** 2)
    return K + jitter * torch.eye(X.shape[0], device=X.device, dtype=X.dtype)


class BatchedAdaptiveSVGP:
    """T independent SVGP tasks with full-covariance double-sided maintenance.

    Args:
        X: (r, dims) inducing = candidate points (shared across tasks).
        objective: callable X -> (r,) values; each task gets a shifted copy.
        T: number of parallel tasks.
        noise: observation noise std σ.
        damp_rho: trust-region damping strength ρ (covariance-level op).
        move_scale: rank-1 inducing-basis perturbation scale per round.
        n_samples: Thompson function samples per task per round.
        seed, force_reference, dtype, device: as elsewhere.
    """

    def __init__(self, X, objective, T=256, noise=0.05, damp_rho=0.02,
                 move_scale=0.1, n_samples=16, seed=0, force_reference=False,
                 transport_mode="incremental", dtype=torch.float32, device=None):
        self.r = X.shape[0]
        self.T = T
        self.noise = noise
        self.rho = damp_rho
        self.move_scale = move_scale
        self.n_samples = n_samples
        self.force_reference = force_reference
        self.transport_mode = transport_mode
        self.dtype = dtype
        self.device = device or X.device
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

        d = self.device
        K0 = rbf_kernel(X.to(d, dtype))
        eye = torch.eye(self.r, device=d, dtype=dtype)
        # Metric G = K_uu (init K0); whitened covariance A = L^{-1} S L^{-⊤}.
        # Prior S = K_uu ⇒ A = I.
        self.G = K0.unsqueeze(0).repeat(T, 1, 1).contiguous()
        self.S = K0.unsqueeze(0).repeat(T, 1, 1).contiguous()
        self.L = torch.linalg.cholesky(self.G).contiguous()
        self.A = eye.unsqueeze(0).repeat(T, 1, 1).contiguous()
        self.mu = torch.zeros(T, self.r, device=d, dtype=dtype)      # un-whitened mean

        base = objective(X.to(d, dtype))
        shifts = torch.arange(T, device=d, dtype=dtype).reshape(T, 1)
        self.f_true = base.unsqueeze(0) + 0.05 * torch.sin(
            shifts + torch.arange(self.r, device=d, dtype=dtype))
        self.f_max = self.f_true.max(dim=1).values
        self.regret_history = []
        self._eye = eye

    # -- the three double-sided covariance operations ------------------------

    def transport(self, z):
        """Inducing-basis change K_uu += z z^⊤: transport A into the new basis.

        ``incremental`` uses the O(r^2) double-sided primitive; ``refactor``
        recomputes L=chol(G) and A=L^{-1}S L^{-⊤} from scratch at O(r^3) — the
        baseline that does not maintain A double-sidedly.
        """
        self.G = self.G + z.unsqueeze(2) * z.unsqueeze(1)   # exact metric (ground truth)
        if self.transport_mode == "refactor":
            self.L = torch.linalg.cholesky(self.G).contiguous()
            Y = torch.linalg.solve_triangular(self.L, self.S, upper=False)
            self.A = torch.linalg.solve_triangular(self.L, Y.mT, upper=False).mT
            return
        w = torch.zeros_like(z)                      # M = S unchanged; only basis moves
        A_new, _ = refresh_bucket(self.L, self.A, z, w,
                                  force_reference=self.force_reference)
        self.A = A_new

    def condition(self, j):
        """Covariance-form Kalman downdate on candidate j (per task). Reads full A."""
        ar = torch.arange(self.T, device=self.device)
        # Un-whitened update to S, mu (ground truth), observing u_j directly.
        s_col = self.S[ar, :, j]                     # (T, r) = S[:, j]
        sjj = self.S[ar, j, j]                       # (T,)
        denom = (sjj + self.noise ** 2).clamp_min(1e-12)
        y = self.f_true[ar, j] + self.noise * torch.randn(
            self.T, generator=self.gen).to(self.device, self.dtype)
        innov = y - self.mu[ar, j]
        self.mu = self.mu + s_col * (innov / denom).unsqueeze(1)
        self.S = self.S - (s_col.unsqueeze(2) * s_col.unsqueeze(1)) / denom.reshape(-1, 1, 1)
        # Whitened covariance-form downdate on A directly (φ̃ = L^⊤ e_j = row j of L).
        phit = self.L[ar, j, :]                      # (T, r)
        Aphi = torch.einsum("tij,tj->ti", self.A, phit)      # A φ̃
        d = (torch.einsum("ti,ti->t", phit, Aphi) + self.noise ** 2).clamp_min(1e-12)
        self.A = self.A - (Aphi.unsqueeze(2) * Aphi.unsqueeze(1)) / d.reshape(-1, 1, 1)

    def damp(self):
        """Trust-region damping A ← (1−ρ)A + ρI (covariance-level; root can't)."""
        self.A = (1 - self.rho) * self.A + self.rho * self._eye
        self.S = (1 - self.rho) * self.S + self.rho * self.G

    # -- Thompson sampling ---------------------------------------------------

    def sample_and_pick(self):
        """Draw n_samples function samples per task and Thompson-pick a candidate."""
        # A root R (chol) is auxiliary, used only to draw samples: u = mu + L R ξ.
        R = torch.linalg.cholesky(self.A + 1e-5 * self._eye)
        xi = torch.randn(self.T, self.r, self.n_samples, generator=self.gen).to(
            self.device, self.dtype)
        v = torch.einsum("tij,tjs->tis", R, xi)              # R ξ  (T, r, S)
        u = self.mu.unsqueeze(2) + torch.einsum("tij,tjs->tis", self.L, v)
        # Thompson: for each sample pick argmax candidate, then majority/first.
        picks = u.argmax(dim=1)                              # (T, S)
        return picks[:, 0]                                   # one draw per task

    def step(self, move=True):
        """One BO round: (optional) basis move, TS acquire+observe+condition, damp."""
        if move:
            z = self.move_scale * torch.randn(
                self.T, self.r, generator=self.gen).to(self.device, self.dtype)
            self.transport(z)
        j = self.sample_and_pick()                           # (T,)
        self.condition(j)
        self.damp()
        # Simple regret of the current posterior-mean optimum.
        best = self.mu.argmax(dim=1)
        achieved = self.f_true[torch.arange(self.T, device=self.device), best]
        regret = (self.f_max - achieved).clamp_min(0).mean().item()
        self.regret_history.append(regret)
        return regret

    # -- diagnostics ---------------------------------------------------------

    @torch.no_grad()
    def refactor_ground_truth(self):
        """A from a full re-cholesky of the exact state: L=chol(G), A=L^{-1}S L^{-⊤}."""
        L = torch.linalg.cholesky(self.G)
        Y = torch.linalg.solve_triangular(L, self.S, upper=False)
        A = torch.linalg.solve_triangular(L, Y.mT, upper=False).mT
        return L, A


class BatchedFantasySVGP:
    """Fantasy-conditioned asynchronous Thompson sampling — batched double-sided.

    J fantasy branches per task each carry a whitened covariance A^{(j)} = L^{-1}
    S^{(j)} L^{-⊤}. Conditioning on a branch's pending batch (features H_j) is the
    covariance-form (rank-k) Kalman downdate that reads the full A^{(j)}:

        A^{(j)} ← A^{(j)} − A^{(j)} H_j^⊤ (H_j A^{(j)} H_j^⊤ + σ^2 I)^{-1} H_j A^{(j)}.

    If the inducing basis also changes, each branch first requires the double-sided
    transport A^{(j)} ← L_{+}^{-1} S^{(j)} L_{+}^{-⊤} (the primitive, w = 0). All
    T·J branches are updated in one batched launch — one Givens sweep per branch,
    covariance conditioning consuming the full matrix. The mandatory baseline is
    square-root conditioning; this experiment tests whether batched *explicit-
    covariance* maintenance is competitive at scale.

    Args:
        X: (r, dims) inducing points.
        T: tasks; J: fantasy branches per task; k: pending batch size.
        noise, seed, force_reference, dtype, device: as elsewhere.
    """

    def __init__(self, X, T=64, J=64, k=4, noise=0.05, seed=0,
                 force_reference=False, dtype=torch.float32, device=None):
        self.r = X.shape[0]
        self.T, self.J, self.k = T, J, k
        self.B = T * J                                   # total branches
        self.noise = noise
        self.force_reference = force_reference
        self.dtype = dtype
        self.device = device or X.device
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

        d = self.device
        K0 = rbf_kernel(X.to(d, dtype))
        eye = torch.eye(self.r, device=d, dtype=dtype)
        self._eye = eye
        # Shared inducing factor per branch (replicated so the batched primitive
        # transports every branch in one launch); whitened covariance per branch.
        self.L = torch.linalg.cholesky(K0).unsqueeze(0).repeat(self.B, 1, 1).contiguous()
        self.G = K0.unsqueeze(0).repeat(self.B, 1, 1).contiguous()
        self.A = eye.unsqueeze(0).repeat(self.B, 1, 1).contiguous()
        self.S = K0.unsqueeze(0).repeat(self.B, 1, 1).contiguous()

    def transport(self, z):
        """Double-sided transport of every branch under K_uu += z z^⊤ (z: (B, r))."""
        w = torch.zeros_like(z)
        A_new, _ = refresh_bucket(self.L, self.A, z, w,
                                  force_reference=self.force_reference)
        self.A = A_new
        self.G = self.G + z.unsqueeze(2) * z.unsqueeze(1)

    def fantasy_condition(self, H):
        """Rank-k covariance-form conditioning per branch. H: (B, k, r). Reads A."""
        # Whitened pending features Φ = H L  (so the downdate matches S-conditioning:
        # L^{-1}SH^⊤ = A Φ^⊤ and HSH^⊤ = Φ A Φ^⊤).
        Ht = torch.einsum("bkr,brj->bkj", H, self.L)         # H L  (B, k, r)
        AHt = torch.einsum("bij,bkj->bik", self.A, Ht)       # A φ̃^⊤  (B, r, k)
        gram = torch.einsum("bki,bil->bkl", Ht, AHt)         # φ̃ A φ̃^⊤ (B, k, k)
        gram = gram + self.noise ** 2 * torch.eye(self.k, device=self.device, dtype=self.dtype)
        sol = torch.linalg.solve(gram, AHt.mT)               # (B, k, r)
        self.A = self.A - torch.einsum("bik,bkl->bil", AHt, sol)
        # Mirror on exact S for the ground truth (features H directly).
        SHt = torch.einsum("bij,bkj->bik", self.S, H)        # S H^⊤ (B, r, k)
        gramS = torch.einsum("bki,bil->bkl", H, SHt) + \
            self.noise ** 2 * torch.eye(self.k, device=self.device, dtype=self.dtype)
        solS = torch.linalg.solve(gramS, SHt.mT)
        self.S = self.S - torch.einsum("bik,bkl->bil", SHt, solS)

    def random_pending(self):
        """Distinct pending feature batch per branch (asynchronous fantasies)."""
        return torch.randn(self.B, self.k, self.r, generator=self.gen).to(
            self.device, self.dtype) * 0.5

    def round(self, move=True):
        """One fantasy BO round over all T·J branches (transport + condition)."""
        if move:
            z = 0.05 * torch.randn(self.B, self.r, generator=self.gen).to(
                self.device, self.dtype)
            self.transport(z)
        self.fantasy_condition(self.random_pending())

    @torch.no_grad()
    def refactor_ground_truth(self):
        L = torch.linalg.cholesky(self.G)
        Y = torch.linalg.solve_triangular(L, self.S, upper=False)
        A = torch.linalg.solve_triangular(L, Y.mT, upper=False).mT
        return L, A
