"""Transported-subspace natural gradient optimizer (paper §3.5), GPU-batched.

Faithful implementation of the natural-gradient preconditioner from the paper,
with the one addition that makes it a GPU experiment: many independent charts
(one per parameter block) whose steady-state whitened-core updates are refreshed
in a single batched kernel launch.

Per preconditioned block, we maintain a low-rank curvature chart

    F_t ≈ λ I + U_t C_t U_t^⊤,     U_t ∈ R^{d×r} orthonormal,  λ > 0 damping.

Following the paper, the chart is stored in *whitened* form. With metric
G_t = λ I + Σ_k a_k a_k^⊤ (the damped accumulated subspace Fisher, a_k = U^⊤ g_k),
Cholesky L_t = chol(G_t), and operator M_t = G_t − λ I, the double-sided
primitive maintains the whitened core

    C_{w,t} = L_t^{-1} M_t L_t^{-⊤} = I − λ G_t^{-1}

under each rank-1 Fisher contribution by calling the kernel with z = w = a_t
(so both G and M receive a_t a_t^⊤). This is the O(r^2) steady-state path.

Preconditioning is Woodbury on F (U orthonormal ⇒ F acts as G on range(U) and
as λI off it):

    a = U^⊤ g,   r_⊥ = g − U a,
    p = F^{-1} g = λ^{-1} r_⊥ + U (G^{-1} a),

where G^{-1} a = L^{-⊤}(L^{-1} a) uses the maintained factor — no root, no
eigendecomposition (O(dr + r^2) total). The kernel returns v_z = L^{-1} a for
free, so G^{-1} a costs just one extra back-solve L^{-⊤} v_z. (The whitened core
C_w = I − λ L^{-1} L^{-⊤} that the primitive maintains is the paper's tracked
object and the drift metric; note it is not G^{-1}, since L^{-1}L^{-⊤} ≠
L^{-⊤}L^{-1}, so preconditioning goes through the solve, not C_w.)

Chart refresh (paper Fig. 2). When the residual energy ρ_t^2/‖g_t‖^2 > τ, the
chart is missing curvature. We enlarge the basis by the unit residual direction,
form the augmented (r+1)×(r+1) core, eigendecompose it, keep the top-r
eigenvectors R, and rotate U_{t+1} = [U | u_⊥] R (O(dr^2) basis rotation, O(r^3)
eigendecomposition — rare, since most steps stay on the steady path).

Baselines for comparison: SOAP (experiments/soap.py), SGD, and AdamW, plus the
built-in Refact / periodic_refact ablation modes below (full re-cholesky of the
same chart every step / every N steps).

The ``refresh_mode`` argument switches the *core maintenance* only, holding the
chart machinery fixed, so the ablation is exactly "our O(r^2) primitive" vs.
"O(r^3) refactorization" on identical charts:
  * ``"incremental"``     — batched double-sided primitive every step.
  * ``"refact"``          — full chol(G)+re-whiten every step (the paper's Refact).
  * ``"periodic_refact"`` — refactorize every ``refact_every`` steps, stale between.
"""

from __future__ import annotations

import torch

from .charts import refresh_bucket


def _orthonormal(d, r, gen, device, dtype):
    """(d, r) matrix with orthonormal columns (r ≤ d), from QR of a Gaussian."""
    g = torch.randn(d, r, device=device, dtype=dtype, generator=gen)
    q, _ = torch.linalg.qr(g)
    return q[:, :r].contiguous()


class IncrementalCurvatureOptimizer:
    """Transported-subspace natural gradient with batched incremental charts.

    Args:
        params: iterable of parameters (or a single tensor).
        lr: learning rate on the preconditioned direction.
        rank: maximum chart rank r per block; a block of width d uses
            r_b = min(rank, d).
        damping: λ, the scalar damping in F = λI + U C U^⊤.
        momentum: heavy-ball momentum on the preconditioned direction.
        tau: refresh threshold on residual energy ρ^2/‖g‖^2 (paper τ=0.1).
        refresh_mode: core-maintenance path (see module docstring).
        refact_every: refactorization period for ``periodic_refact``.
        seed: seeds the initial chart bases.
        force_reference: force the CPU reference core update (for cross-checks).
    """

    def __init__(
        self,
        params,
        lr=0.1,
        rank=32,
        damping=1.0,
        momentum=0.9,
        tau=0.1,
        refresh_mode="incremental",
        refact_every=50,
        seed=0,
        force_reference=False,
    ):
        if isinstance(params, torch.Tensor):
            params = [params]
        self.params = [p for p in params if p.requires_grad]
        assert self.params, "no parameters with requires_grad=True"
        self.lr = lr
        self.rank = rank
        self.lam = damping
        self.momentum = momentum
        self.tau = tau
        self.refresh_mode = refresh_mode
        self.refact_every = refact_every
        self.force_reference = force_reference
        self.step_count = 0
        self.n_refresh = 0

        device = self.params[0].device
        dtype = self.params[0].dtype
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        # One block per parameter tensor. r_b = min(rank, numel).
        self.blocks = []
        for p in self.params:
            d = p.numel()
            r = min(rank, d)
            self.blocks.append({
                "param": p, "d": d, "r": r,
                "U": _orthonormal(d, r, gen, device, dtype),
                "buf": torch.zeros_like(p),
                "can_refresh": r < d,
            })

        # Bucket blocks by rank r for dense batched core updates. Each bucket
        # stacks L, Aw(=C_w), and the exact metric G for its blocks.
        self.buckets = {}
        by_rank = {}
        for i, blk in enumerate(self.blocks):
            by_rank.setdefault(blk["r"], []).append(i)
        for r, idx in by_rank.items():
            T = len(idx)
            eye = torch.eye(r, device=device, dtype=dtype)
            self.buckets[r] = {
                "idx": idx,
                "L": (eye * self.lam ** 0.5).unsqueeze(0).repeat(T, 1, 1).contiguous(),
                "Aw": torch.zeros(T, r, r, device=device, dtype=dtype),  # C_w = 0 at init
                "G": (eye * self.lam).unsqueeze(0).repeat(T, 1, 1).contiguous(),
            }
        self._eye_cache = {}

    def _eye(self, r, device, dtype):
        key = (r, device, dtype)
        if key not in self._eye_cache:
            self._eye_cache[key] = torch.eye(r, device=device, dtype=dtype)
        return self._eye_cache[key]

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    # -- core maintenance ----------------------------------------------------

    def _core_update(self, bucket, a):
        """Advance (L, Aw=C_w) for every chart under G,M += a a^⊤; return G^{-1}a.

        The double-sided primitive maintains the whitened core
        C_w = L^{-1} M L^{-⊤} = I − λ L^{-1} L^{-⊤} (the paper's object, tracked
        by chart_drift) and returns v_z = L^{-1} a for free. The Woodbury
        preconditioner needs G^{-1} a = L^{-⊤} L^{-1} a = L^{-⊤} v_z, one extra
        back-solve. (Note G^{-1}=L^{-⊤}L^{-1} ≠ L^{-1}L^{-⊤}, so the core cannot
        substitute for the solve.) Returns G^{-1}a (T, r). Dispatches on mode.
        """
        r = a.shape[1]
        eye = self._eye(r, a.device, a.dtype)
        if self.refresh_mode == "incremental":
            Aw_new, vz = refresh_bucket(bucket["L"], bucket["Aw"], a, a,
                                        force_reference=self.force_reference)
            bucket["Aw"] = Aw_new
            bucket["G"] = bucket["G"] + a.unsqueeze(2) * a.unsqueeze(1)
            return torch.linalg.solve_triangular(
                bucket["L"].mT, vz.unsqueeze(-1), upper=True).squeeze(-1)
        # Refactorization baselines: accumulate exact G, (periodically) re-chol
        # and re-whiten the core from scratch at O(r^3).
        bucket["G"] = bucket["G"] + a.unsqueeze(2) * a.unsqueeze(1)
        do = (self.refresh_mode == "refact") or (self.step_count % self.refact_every == 1)
        if do:
            bucket["L"] = torch.linalg.cholesky(bucket["G"]).contiguous()
            Linv = torch.linalg.inv(bucket["L"])
            bucket["Aw"] = eye - self.lam * (Linv @ Linv.mT)
        y = torch.linalg.solve_triangular(bucket["L"], a.unsqueeze(-1), upper=False)
        return torch.linalg.solve_triangular(bucket["L"].mT, y, upper=True).squeeze(-1)

    @torch.no_grad()
    def _refresh_block(self, bucket, j, blk, g, a, rv, rho):
        """Enlarge/eigendecompose/truncate one chart to capture the residual."""
        U = blk["U"]
        r = blk["r"]
        device, dtype = g.device, g.dtype
        u_new = rv / rho                                   # (d,), ⟂ range(U)
        b = torch.cat([a, rho.reshape(1)])                 # (r+1,) coeffs of g
        Gpre = bucket["G"][j]                              # pre-step metric (r,r)
        eye1 = self._eye(r + 1, device, dtype)
        Gbar = torch.zeros(r + 1, r + 1, device=device, dtype=dtype)
        Gbar[:r, :r] = Gpre
        Gbar[r, r] = self.lam                              # new direction damping
        Gbar = Gbar + b.unsqueeze(1) * b.unsqueeze(0)      # + this sample's aa^⊤
        fisher = Gbar - self.lam * eye1
        evals, evecs = torch.linalg.eigh(fisher)           # ascending
        R = evecs[:, -r:]                                  # top-r eigenvectors (r+1, r)
        Ubar = torch.cat([U, u_new.unsqueeze(1)], dim=1)   # (d, r+1)
        U_new = (Ubar @ R).contiguous()                    # (d, r) orthonormal
        G_new = self.lam * self._eye(r, device, dtype) + (R.mT @ fisher @ R)
        L_new = torch.linalg.cholesky(G_new)
        Linv = torch.linalg.inv(L_new)
        Aw_new = self._eye(r, device, dtype) - self.lam * (Linv @ Linv.mT)
        # Write back into the block + bucket stacks.
        blk["U"] = U_new
        bucket["L"][j] = L_new
        bucket["Aw"][j] = Aw_new
        bucket["G"][j] = G_new
        a_new = U_new.mT @ g
        rv_new = g - U_new @ a_new
        y = torch.linalg.solve_triangular(L_new, a_new.unsqueeze(-1), upper=False)
        x_new = torch.linalg.solve_triangular(L_new.mT, y, upper=True).squeeze(-1)
        self.n_refresh += 1
        return U_new, a_new, x_new, rv_new

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        for r, bucket in self.buckets.items():
            idx = bucket["idx"]
            T = len(idx)
            device = self.params[0].device
            dtype = self.params[0].dtype

            # 1. Project each block's gradient into its chart: a_j = U_j^⊤ g_j.
            a = torch.zeros(T, r, device=device, dtype=dtype)
            gflat, present = [], []
            for jloc, bi in enumerate(idx):
                blk = self.blocks[bi]
                gp = blk["param"].grad
                if gp is None:
                    gflat.append(None)
                    continue
                g = gp.reshape(-1)
                gflat.append(g)
                a[jloc] = blk["U"].mT @ g

            # 2. Batched steady-state core update (advances L, C_w; returns G^{-1}a).
            x = self._core_update(bucket, a)

            # 3. Per block: refresh if residual energy too high, then precondition.
            for jloc, bi in enumerate(idx):
                blk = self.blocks[bi]
                g = gflat[jloc]
                if g is None:
                    continue
                U = blk["U"]
                aj = a[jloc]
                xj = x[jloc]
                rv = g - U @ aj
                rho2 = (rv @ rv)
                gnorm2 = (g @ g).clamp_min(1e-30)
                # Chart refresh is identical across modes (it is an O(r^3)
                # eigendecomposition regardless); only the steady-state core
                # maintenance differs between incremental and the refact baselines.
                if blk["can_refresh"] and (rho2 / gnorm2) > self.tau:
                    rho = rho2.clamp_min(1e-30).sqrt()
                    U, aj, xj, rv = self._refresh_block(bucket, jloc, blk, g, aj, rv, rho)

                # Woodbury preconditioned direction F^{-1}g = λ^{-1} r_⊥ + U G^{-1}a
                #   = λ^{-1}(g − U a) + U x,   x = G^{-1}a = L^{-⊤} v_z.
                p = rv / self.lam + U @ xj
                buf = blk["buf"].reshape(-1)
                buf.mul_(self.momentum).add_(p)
                blk["param"].add_(buf.reshape(blk["param"].shape), alpha=-self.lr)

    # -- diagnostics ---------------------------------------------------------

    @torch.no_grad()
    def chart_drift(self):
        """Mean ‖C_w,inc − C_w,exact‖ / ‖C_w,exact‖ over charts (Table-4 drift)."""
        drifts = []
        for r, bucket in self.buckets.items():
            eye = self._eye(r, bucket["G"].device, bucket["G"].dtype)
            L = torch.linalg.cholesky(bucket["G"])
            Linv = torch.linalg.inv(L)
            Aw_ref = eye - self.lam * (Linv @ Linv.mT)
            T = len(bucket["idx"])
            num = torch.linalg.norm((bucket["Aw"] - Aw_ref).reshape(T, -1), dim=1)
            den = torch.linalg.norm(Aw_ref.reshape(T, -1), dim=1).clamp_min(1e-12)
            drifts.append((num / den).mean().item())
        return sum(drifts) / len(drifts) if drifts else 0.0

    def num_blocks(self):
        return len(self.blocks)

    def bucket_summary(self):
        return {r: len(b["idx"]) for r, b in self.buckets.items()}
