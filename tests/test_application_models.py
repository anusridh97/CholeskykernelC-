from __future__ import annotations

import pytest
import torch

from benchmarks.applications._corrected.adaptive_svgp import AdaptiveInducingGaussian
from benchmarks.applications._corrected.core import extension_available, relative_error
from benchmarks.applications._corrected.refresh import TransportedChart, TransportedSandwichChart


BACKENDS = ["torch"] + (["extension"] if extension_available() else [])


@pytest.mark.parametrize("backend", BACKENDS)
def test_adaptive_gp_matches_refactor(backend: str) -> None:
    torch.manual_seed(0)
    dtype = torch.float64
    Z = torch.randn(16, 3, dtype=dtype)
    ours = AdaptiveInducingGaussian(
        Z,
        method="ours",
        backend=backend,
        lengthscale=1.3,
        noise_std=0.15,
        refactor_every=0,
    )
    refact = AdaptiveInducingGaussian(
        Z,
        method="refact",
        backend=backend,
        lengthscale=1.3,
        noise_std=0.15,
        refactor_every=0,
    )

    for _ in range(30):
        x = torch.randn(3, dtype=dtype)
        y = torch.sin(x.sum())
        ours.replace_and_condition(x, y)
        refact.replace_and_condition(x, y)

    X_test = torch.randn(64, 3, dtype=dtype)
    mean_ours, var_ours = ours.predict(X_test)
    mean_refact, var_refact = refact.predict(X_test)
    assert relative_error(mean_ours, mean_refact) < 2e-10
    assert relative_error(var_ours, var_refact) < 2e-10
    assert ours.diagnostics.fallback_refactors == 0
    validation = ours.validate()
    assert validation["L_relative"] < 2e-10
    assert validation["Aw_relative"] < 2e-9


@pytest.mark.parametrize("chart_type", [TransportedChart, TransportedSandwichChart])
@pytest.mark.parametrize("backend", BACKENDS)
def test_chart_preconditioner_matches_oracle(chart_type, backend: str) -> None:
    torch.manual_seed(3)
    kwargs = {"beta": 0.8} if chart_type is TransportedSandwichChart else {}
    chart = chart_type(
        96,
        16,
        lam=0.5,
        dtype=torch.float64,
        backend=backend,
        seed=11,
        **kwargs,
    )
    for _ in range(40):
        g = torch.randn(96, dtype=torch.float64)
        chart.update(g, refresh_threshold=2.0)
        assert relative_error(chart.precondition(g), chart.precondition_via_factor(g)) < 2e-10
    assert chart.drift() < 2e-10
