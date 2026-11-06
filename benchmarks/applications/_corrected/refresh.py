"""Transported low-rank curvature charts.

For ordinary low-rank natural gradient, the corrected state is

    G = lambda I + C,        G = L L^T,
    C_w = L^{-1} C L^{-T}.

The exact Woodbury inverse action uses two triangular solves with the
incrementally updated L; the uploaded ``/lambda**2`` expression was incorrect.
The transported C_w is retained as an explicit consistency target, but ordinary
F^{-1}g does not mathematically require a second dense operator.

To test the genuinely double-sided path downstream, this module also provides a
``TransportedSandwichChart``.  It applies the positive sandwich operator
G^{-1} M G^{-1} in chart coordinates, which necessarily consumes
A_w=L^{-1} M L^{-T}.  Setting its operator scale to one collapses to ordinary
natural gradient and makes A_w=I; choosing a distinct scale gives the
nontrivial transported-operator experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Literal

import torch

from .core import (
    Backend,
    double_whiten,
    incremental_update,
    relative_error,
    symmetrize,
)


@dataclass
class ChartDiagnostics:
    steady_updates: int = 0
    refreshes: int = 0
    refactors: int = 0
    max_drift: float = 0.0
    core_seconds: float = 0.0


class TransportedChart:
    """Low-rank curvature chart with a genuinely transported damped core."""

    def __init__(
        self,
        d: int,
        r: int,
        *,
        lam: float = 1.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        backend: Backend = "auto",
        seed: int = 0,
    ) -> None:
        if not 0 < r <= d:
            raise ValueError(f"require 0 < r <= d; got r={r}, d={d}")
        if lam <= 0:
            raise ValueError("lam must be positive")
        self.d = int(d)
        self.r = int(r)
        self.lam = float(lam)
        self.device = torch.device(device)
        self.dtype = dtype
        self.backend = backend

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        random_matrix = torch.randn(
            d, r, dtype=dtype, device=self.device, generator=generator
        )
        self.U = torch.linalg.qr(random_matrix, mode="reduced").Q

        self.C = torch.zeros(r, r, dtype=dtype, device=self.device)
        self.L = math.sqrt(self.lam) * torch.eye(
            r, dtype=dtype, device=self.device
        )
        self.Cw = torch.zeros_like(self.C)
        self.diagnostics = ChartDiagnostics()

    @torch.no_grad()
    def project(self, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if g.shape != (self.d,):
            raise ValueError(f"g must have shape {(self.d,)}; got {tuple(g.shape)}")
        a = self.U.mT @ g
        residual = g - self.U @ a
        rho = torch.linalg.vector_norm(residual)
        return a, residual, rho

    @torch.no_grad()
    def refactor(self) -> None:
        eye = torch.eye(self.r, dtype=self.dtype, device=self.device)
        self.L = torch.linalg.cholesky(symmetrize(self.C) + self.lam * eye).contiguous()
        self.Cw = symmetrize(double_whiten(self.L, self.C))
        self.diagnostics.refactors += 1

    @torch.no_grad()
    def steady_state_update(self, g: torch.Tensor, *, refact: bool = False) -> None:
        a, _, _ = self.project(g)
        start = time.perf_counter()
        if refact:
            self.C = symmetrize(self.C + torch.outer(a, a))
            self.refactor()
        else:
            result = incremental_update(
                self.L,
                self.Cw,
                a,
                a,
                metric_sign=1,
                operator_sign=1,
                backend=self.backend,
            )
            self.L, self.Cw = result.L, symmetrize(result.Aw)
            self.C = symmetrize(self.C + torch.outer(a, a))
        self.diagnostics.core_seconds += time.perf_counter() - start
        self.diagnostics.steady_updates += 1

    @torch.no_grad()
    def refresh(self, g: torch.Tensor) -> None:
        a, residual, rho = self.project(g)
        if float(rho) <= 10 * torch.finfo(self.dtype).eps:
            self.steady_state_update(g)
            return

        u_new = residual / rho
        U_bar = torch.cat([self.U, u_new.unsqueeze(1)], dim=1)
        c_bar = torch.cat([a, rho.reshape(1)])

        C_bar = torch.zeros(
            self.r + 1,
            self.r + 1,
            dtype=self.dtype,
            device=self.device,
        )
        C_bar[: self.r, : self.r] = self.C
        C_bar = symmetrize(C_bar + torch.outer(c_bar, c_bar))

        eigenvalues, eigenvectors = torch.linalg.eigh(C_bar)
        selected = torch.argsort(eigenvalues, descending=True)[: self.r]
        rotation = eigenvectors[:, selected]
        kept = eigenvalues[selected].clamp_min(0.0)

        self.U = U_bar @ rotation
        self.C = torch.diag(kept)
        self.refactor()
        self.diagnostics.refreshes += 1

    @torch.no_grad()
    def update(
        self,
        g: torch.Tensor,
        *,
        refresh_threshold: float = 0.1,
        refact: bool = False,
    ) -> None:
        _, _, rho = self.project(g)
        norm = torch.linalg.vector_norm(g)
        if float(norm) <= torch.finfo(self.dtype).tiny:
            return
        unexplained = float((rho / norm).square())
        if unexplained > refresh_threshold:
            self.refresh(g)
        else:
            self.steady_state_update(g, refact=refact)

    @torch.no_grad()
    def precondition(self, g: torch.Tensor) -> torch.Tensor:
        """Apply the exact Woodbury inverse in O(dr+r^2)."""
        a = self.U.mT @ g
        perpendicular = g - self.U @ a
        inside = torch.cholesky_solve(a.unsqueeze(1), self.L).squeeze(1)
        return perpendicular / self.lam + self.U @ inside

    @torch.no_grad()
    def precondition_via_factor(self, g: torch.Tensor) -> torch.Tensor:
        """Independent dense oracle used only by validation tests."""
        identity = torch.eye(self.d, dtype=self.dtype, device=self.device)
        full = self.lam * identity + self.U @ self.C @ self.U.mT
        return torch.linalg.solve(full, g)

    @torch.no_grad()
    def drift(self) -> float:
        exact = double_whiten(self.L, self.C)
        value = relative_error(self.Cw, exact)
        self.diagnostics.max_drift = max(self.diagnostics.max_drift, value)
        return value

    @torch.no_grad()
    def validate(self, g: torch.Tensor | None = None) -> dict[str, float]:
        eye_d = torch.eye(self.r, dtype=self.dtype, device=self.device)
        metrics = {
            "orthogonality": float(torch.linalg.vector_norm(self.U.mT @ self.U - eye_d)),
            "core_drift": self.drift(),
            "minimum_core_eigenvalue": float(torch.linalg.eigvalsh(self.C).min()),
        }
        if g is not None:
            metrics["preconditioner_relative"] = relative_error(
                self.precondition(g), self.precondition_via_factor(g)
            )
        return metrics



class TransportedSandwichChart(TransportedChart):
    """Chart whose update direction explicitly uses a transported operator.

    The metric core is ``G=lambda I+C`` and the numerator is
    ``M=lambda I+beta C``.  The chart action is

        p_parallel = G^{-1} M G^{-1} a.

    ``beta=1`` recovers the ordinary inverse and yields the degenerate
    transported state ``Aw=I``.  A distinct beta creates the nontrivial
    two-sided experiment while preserving a positive preconditioner.
    """

    def __init__(self, *args, beta: float = 0.9, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.beta = float(beta)
        self.M = self.lam * torch.eye(
            self.r, dtype=self.dtype, device=self.device
        )
        self.Cw = torch.eye(self.r, dtype=self.dtype, device=self.device)

    @torch.no_grad()
    def refactor(self) -> None:
        eye = torch.eye(self.r, dtype=self.dtype, device=self.device)
        self.L = torch.linalg.cholesky(symmetrize(self.C) + self.lam * eye).contiguous()
        self.M = self.lam * eye + self.beta * self.C
        self.Cw = symmetrize(double_whiten(self.L, self.M))
        self.diagnostics.refactors += 1

    @torch.no_grad()
    def steady_state_update(self, g: torch.Tensor, *, refact: bool = False) -> None:
        a, _, _ = self.project(g)
        start = time.perf_counter()
        if refact:
            self.C = symmetrize(self.C + torch.outer(a, a))
            self.refactor()
        else:
            result = incremental_update(
                self.L,
                self.Cw,
                a,
                self.beta * a,
                metric_sign=1,
                operator_sign=1,
                backend=self.backend,
            )
            self.L, self.Cw = result.L, symmetrize(result.Aw)
            self.C = symmetrize(self.C + torch.outer(a, a))
            self.M = self.lam * torch.eye(
                self.r, dtype=self.dtype, device=self.device
            ) + self.beta * self.C
        self.diagnostics.core_seconds += time.perf_counter() - start
        self.diagnostics.steady_updates += 1

    @torch.no_grad()
    def precondition(self, g: torch.Tensor) -> torch.Tensor:
        a = self.U.mT @ g
        perpendicular = g - self.U @ a
        q = torch.linalg.solve_triangular(
            self.L, a.unsqueeze(1), upper=False
        ).squeeze(1)
        weighted = self.Cw @ q
        inside = torch.linalg.solve_triangular(
            self.L.mT, weighted.unsqueeze(1), upper=True
        ).squeeze(1)
        return perpendicular / self.lam + self.U @ inside

    @torch.no_grad()
    def precondition_via_factor(self, g: torch.Tensor) -> torch.Tensor:
        a = self.U.mT @ g
        perpendicular = g - self.U @ a
        G = self.lam * torch.eye(
            self.r, dtype=self.dtype, device=self.device
        ) + self.C
        inside = torch.linalg.solve(G, self.M @ torch.linalg.solve(G, a))
        return perpendicular / self.lam + self.U @ inside

    @torch.no_grad()
    def drift(self) -> float:
        exact = double_whiten(self.L, self.M)
        value = relative_error(self.Cw, exact)
        self.diagnostics.max_drift = max(self.diagnostics.max_drift, value)
        return value

def _flatten_first_layer(model: torch.nn.Sequential) -> torch.Tensor:
    return torch.cat([model[0].weight.flatten(), model[0].bias.flatten()])


def _first_layer_gradient(model: torch.nn.Sequential) -> torch.Tensor:
    return torch.cat(
        [model[0].weight.grad.flatten(), model[0].bias.grad.flatten()]
    ).detach()


@torch.no_grad()
def _apply_first_layer_step(
    model: torch.nn.Sequential,
    direction: torch.Tensor,
    learning_rate: float,
) -> None:
    weight_count = model[0].weight.numel()
    current = _flatten_first_layer(model)
    updated = current - learning_rate * direction
    model[0].weight.copy_(updated[:weight_count].reshape_as(model[0].weight))
    model[0].bias.copy_(updated[weight_count:])


@torch.no_grad()
def _sgd_other_layers(
    model: torch.nn.Sequential,
    learning_rate: float,
) -> None:
    for index, parameter in enumerate(model.parameters()):
        if index < 2:  # first-layer weight and bias were handled above.
            continue
        parameter.add_(parameter.grad, alpha=-learning_rate)


def make_mlp(input_dim: int, hidden_dims: list[int]) -> torch.nn.Sequential:
    layers: list[torch.nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend([torch.nn.Linear(previous, width), torch.nn.ReLU()])
        previous = width
    layers.append(torch.nn.Linear(previous, 1))
    return torch.nn.Sequential(*layers)


def transported_ng_experiment(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    hidden_dims: list[int] | None = None,
    chart_rank: int = 32,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-2,
    lam: float = 1.0,
    refresh_threshold: float = 0.1,
    refactor_every: int = 50,
    seed: int = 42,
    backend: Backend = "auto",
    chart_mode: Literal["inverse", "sandwich"] = "inverse",
    sandwich_beta: float = 0.9,
    methods: tuple[str, ...] = ("ours", "refact", "sgd"),
) -> dict[str, dict[str, object]]:
    """Train matched models with transported, refactored, and SGD paths."""
    hidden_dims = hidden_dims or [128]
    device = X_train.device
    dtype = X_train.dtype
    input_dim = X_train.shape[1]

    # Precompute epoch permutations so all methods see the same batches.
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1234)
    permutations = [
        torch.randperm(X_train.shape[0], generator=generator, device=device)
        for _ in range(epochs)
    ]

    results: dict[str, dict[str, object]] = {}
    for method in methods:
        torch.manual_seed(seed)
        model = make_mlp(input_dim, hidden_dims).to(device=device, dtype=dtype)
        first_width = hidden_dims[0]
        parameter_dim = input_dim * first_width + first_width
        chart = None
        if method in ("ours", "refact"):
            chart_class = (
                TransportedChart if chart_mode == "inverse" else TransportedSandwichChart
            )
            chart_kwargs = dict(
                lam=lam,
                device=device,
                dtype=dtype,
                backend=backend,
                seed=seed + 77,
            )
            if chart_mode == "sandwich":
                chart_kwargs["beta"] = sandwich_beta
            chart = chart_class(parameter_dim, chart_rank, **chart_kwargs)

        losses: list[float] = []
        wall_times: list[float] = []
        core_times: list[float] = []
        step = 0
        start = time.perf_counter()

        for epoch in range(epochs):
            permutation = permutations[epoch]
            for begin in range(0, X_train.shape[0] - batch_size + 1, batch_size):
                index = permutation[begin : begin + batch_size]
                xb, yb = X_train[index], y_train[index]

                prediction = model(xb).squeeze(-1)
                loss = torch.nn.functional.mse_loss(prediction, yb)
                model.zero_grad(set_to_none=True)
                loss.backward()

                if method == "sgd":
                    with torch.no_grad():
                        for parameter in model.parameters():
                            parameter.add_(parameter.grad, alpha=-learning_rate)
                else:
                    assert chart is not None
                    gradient = _first_layer_gradient(model)
                    # Batch-mean gradients are already normalized by MSELoss.
                    chart.update(
                        gradient,
                        refresh_threshold=refresh_threshold,
                        refact=(method == "refact"),
                    )
                    if (
                        method == "ours"
                        and refactor_every > 0
                        and (step + 1) % refactor_every == 0
                    ):
                        chart.refactor()
                    direction = chart.precondition(gradient)
                    _apply_first_layer_step(model, direction, learning_rate)
                    _sgd_other_layers(model, learning_rate)

                step += 1
                losses.append(float(loss.detach()))
                wall_times.append(time.perf_counter() - start)
                core_times.append(chart.diagnostics.core_seconds if chart else 0.0)

        with torch.no_grad():
            test_mse = float(
                torch.nn.functional.mse_loss(
                    model(X_test).squeeze(-1), y_test
                )
            )

        results[method] = {
            "losses": losses,
            "wall_times": wall_times,
            "core_times": core_times,
            "test_mse": test_mse,
            "steps": step,
            "core_fraction": (
                core_times[-1] / wall_times[-1]
                if wall_times and wall_times[-1] > 0
                else 0.0
            ),
            "chart_diagnostics": (
                chart.diagnostics.__dict__.copy() if chart is not None else None
            ),
        }

    return results


__all__ = [
    "ChartDiagnostics",
    "TransportedChart",
    "TransportedSandwichChart",
    "make_mlp",
    "transported_ng_experiment",
]
