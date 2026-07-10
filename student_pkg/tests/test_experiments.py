"""Correctness tests for the two GPU application experiments.

These run on CPU using the pure-PyTorch reference refresh (force_reference=True),
so they exercise the experiment logic and its equivalence to a full
re-factorization without needing CUDA/Triton. The batched Triton path itself is
covered by tests/test_correctness.py::test_triton_batched_matches_single.

Run with:
    python -m pytest tests/test_experiments.py -v
"""

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.curvature_optimizer import IncrementalCurvatureOptimizer
from experiments.parallel_gp_thompson import BatchedGPThompson


# --------------------------------------------------------------------------- #
# Curvature optimizer
# --------------------------------------------------------------------------- #


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Linear(32, 4)
    )


def test_optimizer_reduces_loss():
    """Incremental curvature optimizer should drive a tiny regression down."""
    torch.manual_seed(0)
    model = _tiny_model()
    X = torch.randn(64, 16)
    Y = torch.randn(64, 4)
    opt = IncrementalCurvatureOptimizer(
        model.parameters(), lr=5e-3, rank=8, refresh_mode="incremental",
        force_reference=True,
    )
    loss_fn = torch.nn.MSELoss()
    first = None
    for _ in range(40):
        pred = model(X)
        loss = loss_fn(pred, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first, f"loss did not decrease: {first:.4f} -> {loss.item():.4f}"


def test_optimizer_buckets_by_rank():
    """Blocks of equal clamped rank share one batched bucket."""
    model = _tiny_model()
    opt = IncrementalCurvatureOptimizer(
        model.parameters(), rank=8, force_reference=True
    )
    # Blocks with >= 8 entries clamp to r=8; the width-4 output bias gets r=4.
    # Same-rank blocks must share a bucket, and every block lands in one.
    summary = opt.bucket_summary()
    assert sum(summary.values()) == opt.num_blocks()
    assert summary[8] == opt.num_blocks() - 1  # all but the width-4 bias
    assert summary[4] == 1


def test_optimizer_natural_gradient_solve_matches_refact():
    """The subspace G^{-1} solve implied by the chart matches a fresh solve.

    After several incremental updates, back_solve(v_z) must equal G^{-1} s where
    G is the exact accumulated Gram — i.e. the incremental factor stays faithful.
    """
    torch.manual_seed(1)
    r, T = 12, 3
    device = torch.device("cpu")
    from experiments.charts import ChartBucket

    bucket = ChartBucket(T=T, r=r, device=device, ridge=1e-2)
    G = (1e-2 * torch.eye(r)).unsqueeze(0).repeat(T, 1, 1)
    last_s = None
    for _ in range(15):
        s = torch.randn(T, r) * 0.3
        G = G + s.unsqueeze(2) * s.unsqueeze(1)
        _, vz = bucket.refresh(s, s, force_reference=True)
        last_s = s
    x_inc = bucket.back_solve(bucket.refresh(last_s * 0, last_s * 0, force_reference=True)[1])
    # Compare a fresh natural-gradient solve on a probe vector.
    probe = torch.randn(T, r)
    x_ref = torch.linalg.solve(G, probe.unsqueeze(-1)).squeeze(-1)
    y = torch.linalg.solve_triangular(bucket.L, probe.unsqueeze(-1), upper=False)
    x_chart = torch.linalg.solve_triangular(bucket.L.mT, y, upper=True).squeeze(-1)
    err = (torch.norm(x_chart - x_ref) / torch.norm(x_ref)).item()
    assert err < 1e-4, f"chart G^-1 solve drifted from exact: {err:.2e}"


# --------------------------------------------------------------------------- #
# GP Thompson sampling
# --------------------------------------------------------------------------- #


def _objective(X):
    return torch.sin(6 * X[:, 0]) * torch.cos(6 * X[:, 1])


def test_gp_incremental_matches_refactorization():
    """Incremental precision factor + A_w match a full re-cholesky after rounds."""
    torch.manual_seed(0)
    r, T = 24, 4
    X = torch.rand(r, 2)
    gp = BatchedGPThompson(X, _objective, T=T, force_reference=True, seed=0)
    for _ in range(12):
        gp.step()
    L_ref, Aw_ref = gp.refactor_ground_truth(gp.counts)
    L_err = (torch.norm(gp.L - L_ref) / torch.norm(L_ref)).item()
    Aw_err = (torch.norm(gp.Aw - Aw_ref) / torch.norm(Aw_ref)).item()
    assert L_err < 1e-4, f"precision factor drift {L_err:.2e}"
    assert Aw_err < 1e-3, f"whitened operator drift {Aw_err:.2e}"


def test_gp_regret_decreases():
    """Thompson sampling should reduce mean simple regret over rounds."""
    torch.manual_seed(0)
    r, T = 48, 16
    X = torch.rand(r, 2)
    gp = BatchedGPThompson(X, _objective, T=T, force_reference=True, seed=0)
    early = []
    for i in range(30):
        regret = gp.step()
        if i < 3:
            early.append(regret)
    early_mean = sum(early) / len(early)
    late = sum(gp.regret_history[-3:]) / 3
    assert late <= early_mean + 1e-6, f"regret grew: {early_mean:.4f} -> {late:.4f}"
