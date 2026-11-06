"""Adaptive-inducing full-covariance sparse GP state.

This is a correctness-oriented application of the double-sided transport.  It
uses a Gaussian posterior q(u)=N(m,S) over inducing values.  Predictions consume

    m_w = L^{-1}m,      S_w = L^{-1} S L^{-T},      K_uu = L L^T.

When the oldest inducing point is replaced, the surviving marginal of q(u) is
kept and the new inducing variable is attached through the GP prior conditional.
Both K_uu and S then change only in their final row/column, hence by signed rank
at most two.  The proposed path transports S_w in O(r^2); the refactor baseline
recomputes the Cholesky factor and double whitening from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .core import (
    Backend,
    apply_arrowhead_metric_delta,
    apply_arrowhead_symmetric_operator_delta,
    cyclic_shift_state,
    double_whiten,
    relative_error,
    symmetrize,
    whiten_vector,
)

Method = Literal["ours", "refact"]


def rbf_kernel(
    X1: torch.Tensor,
    X2: torch.Tensor,
    *,
    lengthscale: float | torch.Tensor = 1.0,
    variance: float = 1.0,
) -> torch.Tensor:
    scale = torch.as_tensor(lengthscale, dtype=X1.dtype, device=X1.device)
    distances = torch.cdist(X1 / scale, X2 / scale, p=2)
    return variance * torch.exp(-0.5 * distances.square())


@dataclass
class GPDiagnostics:
    replacement_count: int = 0
    fallback_refactors: int = 0
    max_gram_error: float = 0.0
    max_whitened_error: float = 0.0


class AdaptiveInducingGaussian:
    """Full-covariance online sparse GP with an adaptive inducing queue."""

    def __init__(
        self,
        Z: torch.Tensor,
        *,
        method: Method = "ours",
        backend: Backend = "auto",
        lengthscale: float = 1.0,
        variance: float = 1.0,
        noise_std: float = 0.1,
        jitter: float = 1e-5,
        refactor_every: int = 50,
        downdate_tol: float | None = None,
    ) -> None:
        if Z.ndim != 2:
            raise ValueError("Z must have shape (r,d)")
        if Z.shape[0] < 2:
            raise ValueError("at least two inducing points are required")
        if Z.dtype not in (torch.float32, torch.float64):
            raise TypeError("Z must be float32 or float64")
        if method not in ("ours", "refact"):
            raise ValueError("method must be 'ours' or 'refact'")

        self.Z = Z.clone()
        self.r = Z.shape[0]
        self.method = method
        self.backend = backend
        self.lengthscale = float(lengthscale)
        self.variance = float(variance)
        self.noise_std = float(noise_std)
        self.jitter = float(jitter)
        self.refactor_every = int(refactor_every)
        self.downdate_tol = (
            float(downdate_tol)
            if downdate_tol is not None
            else (1e-5 if Z.dtype == torch.float32 else 1e-10)
        )

        eye = torch.eye(self.r, dtype=Z.dtype, device=Z.device)
        self.G = self._kernel(self.Z, self.Z) + self.jitter * eye
        self.M = self.G.clone()  # q(u) starts at the GP prior.
        self.m = torch.zeros(self.r, dtype=Z.dtype, device=Z.device)
        self.L = torch.linalg.cholesky(self.G).contiguous()
        self.Aw = torch.eye(self.r, dtype=Z.dtype, device=Z.device)
        self.mw = torch.zeros_like(self.m)
        self.diagnostics = GPDiagnostics()

    def _kernel(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        return rbf_kernel(
            X1,
            X2,
            lengthscale=self.lengthscale,
            variance=self.variance,
        )

    @torch.no_grad()
    def refactor(self) -> None:
        self.L = torch.linalg.cholesky(symmetrize(self.G)).contiguous()
        self.Aw = symmetrize(double_whiten(self.L, self.M))
        self.mw = whiten_vector(self.L, self.m)

    @torch.no_grad()
    def _roll_oldest_to_last(self) -> None:
        self.L, self.Aw, self.mw = cyclic_shift_state(
            self.L,
            self.Aw,
            self.mw,
            backend=self.backend,
        )
        self.Z = torch.roll(self.Z, shifts=-1, dims=0)
        self.G = torch.roll(torch.roll(self.G, shifts=-1, dims=0), shifts=-1, dims=1)
        self.M = torch.roll(torch.roll(self.M, shifts=-1, dims=0), shifts=-1, dims=1)
        self.m = torch.roll(self.m, shifts=-1, dims=0)

    @torch.no_grad()
    def replace_oldest(self, x_new: torch.Tensor) -> None:
        """Replace the oldest inducing point by ``x_new`` in O(r^2) for ours."""
        if x_new.shape != (self.Z.shape[1],):
            raise ValueError(
                f"x_new must have shape {(self.Z.shape[1],)}; got {tuple(x_new.shape)}"
            )
        x_new = x_new.to(dtype=self.Z.dtype, device=self.Z.device)
        self._roll_oldest_to_last()

        keep = self.r - 1
        L_keep = self.L[:keep, :keep]
        k_keep_new = self._kernel(self.Z[:keep], x_new.unsqueeze(0)).squeeze(1)
        conditional_weights = torch.cholesky_solve(
            k_keep_new.unsqueeze(1), L_keep
        ).squeeze(1)
        k_new_new = x_new.new_tensor(self.variance + self.jitter)
        conditional_variance = (
            k_new_new - torch.dot(k_keep_new, conditional_weights)
        ).clamp_min(self.jitter)

        m_new = torch.empty_like(self.m)
        m_new[:keep] = self.m[:keep]
        m_new[-1] = torch.dot(conditional_weights, self.m[:keep])

        M_new = torch.zeros_like(self.M)
        M_new[:keep, :keep] = self.M[:keep, :keep]
        cross = self.M[:keep, :keep] @ conditional_weights
        M_new[:keep, -1] = cross
        M_new[-1, :keep] = cross
        M_new[-1, -1] = conditional_variance + torch.dot(conditional_weights, cross)
        M_new = symmetrize(M_new)

        G_new = torch.zeros_like(self.G)
        G_new[:keep, :keep] = self.G[:keep, :keep]
        G_new[:keep, -1] = k_keep_new
        G_new[-1, :keep] = k_keep_new
        G_new[-1, -1] = k_new_new
        G_new = symmetrize(G_new)

        delta_G = G_new - self.G
        delta_M = M_new - self.M

        if self.method == "ours":
            try:
                self.L, self.Aw = apply_arrowhead_metric_delta(
                    self.L,
                    self.Aw,
                    delta_G,
                    backend=self.backend,
                    downdate_tol=self.downdate_tol,
                )
                self.Aw = apply_arrowhead_symmetric_operator_delta(
                    self.L, self.Aw, delta_M
                )
                self.mw = whiten_vector(self.L, m_new)
            except (RuntimeError, ValueError):
                # A leverage guard or a near-degenerate replacement should not
                # silently corrupt the state.  A fresh factorization is the
                # explicit, recorded fallback.
                self.G = G_new
                self.M = M_new
                self.m = m_new
                self.Z[-1] = x_new
                self.refactor()
                self.diagnostics.fallback_refactors += 1
                self.diagnostics.replacement_count += 1
                return
        else:
            self.G = G_new
            self.M = M_new
            self.m = m_new
            self.Z[-1] = x_new
            self.refactor()
            self.diagnostics.replacement_count += 1
            return

        self.G = G_new
        self.M = M_new
        self.m = m_new
        self.Z[-1] = x_new
        self.diagnostics.replacement_count += 1

        if self.refactor_every > 0 and (
            self.diagnostics.replacement_count % self.refactor_every == 0
        ):
            self.refactor()

    @torch.no_grad()
    def condition(self, x: torch.Tensor, y: torch.Tensor | float) -> None:
        """Apply one Gaussian observation update to q(u)."""
        x = x.to(dtype=self.Z.dtype, device=self.Z.device)
        y_t = torch.as_tensor(y, dtype=self.Z.dtype, device=self.Z.device)
        k_u = self._kernel(self.Z, x.unsqueeze(0)).squeeze(1)
        h = torch.cholesky_solve(k_u.unsqueeze(1), self.L).squeeze(1)

        conditional_noise = (
            self.variance - torch.dot(k_u, h)
        ).clamp_min(0.0) + self.noise_std**2
        s = self.M @ h
        innovation_variance = (conditional_noise + torch.dot(h, s)).clamp_min(
            torch.finfo(self.Z.dtype).eps
        )
        innovation = y_t - torch.dot(h, self.m)

        delta_m = s * (innovation / innovation_variance)
        downdate_vector = s / torch.sqrt(innovation_variance)

        self.m = self.m + delta_m
        self.M = symmetrize(self.M - torch.outer(downdate_vector, downdate_vector))

        whitened_delta_m = whiten_vector(self.L, delta_m)
        whitened_downdate = whiten_vector(self.L, downdate_vector)
        self.mw = self.mw + whitened_delta_m
        self.Aw = symmetrize(
            self.Aw - torch.outer(whitened_downdate, whitened_downdate)
        )

    @torch.no_grad()
    def replace_and_condition(
        self,
        x: torch.Tensor,
        y: torch.Tensor | float,
    ) -> None:
        self.replace_oldest(x)
        self.condition(x, y)

    @torch.no_grad()
    def predict(
        self,
        X: torch.Tensor,
        *,
        include_noise: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        X = X.to(dtype=self.Z.dtype, device=self.Z.device)
        K_u_x = self._kernel(self.Z, X)
        phi = torch.linalg.solve_triangular(
            self.L, K_u_x, upper=False
        ).mT  # K_xu L^{-T}

        mean = phi @ self.mw
        covariance_correction = symmetrize(self.Aw) - torch.eye(
            self.r, dtype=self.Z.dtype, device=self.Z.device
        )
        variance = self.variance + torch.sum(
            (phi @ covariance_correction) * phi, dim=1
        )
        if include_noise:
            variance = variance + self.noise_std**2
        return mean, variance.clamp_min(torch.finfo(self.Z.dtype).eps)

    @torch.no_grad()
    def sample_marginals(
        self,
        X: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        mean, variance = self.predict(X)
        noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=generator,
        )
        return mean + variance.sqrt() * noise

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        L_exact = torch.linalg.cholesky(symmetrize(self.G))
        Aw_exact = double_whiten(L_exact, self.M)
        mw_exact = whiten_vector(L_exact, self.m)
        metrics = {
            "L_relative": relative_error(self.L, L_exact),
            "Aw_relative": relative_error(self.Aw, Aw_exact),
            "mw_relative": relative_error(self.mw, mw_exact),
            "M_symmetry": float(torch.linalg.vector_norm(self.M - self.M.mT)),
        }
        self.diagnostics.max_gram_error = max(
            self.diagnostics.max_gram_error, metrics["L_relative"]
        )
        self.diagnostics.max_whitened_error = max(
            self.diagnostics.max_whitened_error, metrics["Aw_relative"]
        )
        return metrics


__all__ = [
    "AdaptiveInducingGaussian",
    "GPDiagnostics",
    "Method",
    "rbf_kernel",
]
