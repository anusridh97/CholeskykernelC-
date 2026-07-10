"""Correctness tests for the two GPU application experiments.

These run on CPU using the pure-PyTorch reference refresh (force_reference=True),
so they exercise the experiment math and its equivalence to a full
re-factorization without needing CUDA/Triton. The batched Triton path itself is
covered by tests/test_correctness.py::test_triton_batched_matches_single.

Run with:
    python -m pytest tests/test_experiments.py -v
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.curvature_optimizer import IncrementalCurvatureOptimizer, _orthonormal
from experiments.parallel_gp_thompson import BatchedGPThompson


# --------------------------------------------------------------------------- #
# Curvature optimizer — transported-subspace natural gradient (§3.5)
# --------------------------------------------------------------------------- #


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Linear(32, 4)
    )


def test_optimizer_reduces_loss():
    torch.manual_seed(0)
    model = _tiny_model()
    X = torch.randn(64, 16)
    Y = torch.randn(64, 4)
    opt = IncrementalCurvatureOptimizer(
        model.parameters(), lr=0.05, rank=8, damping=1.0, force_reference=True)
    loss_fn = torch.nn.MSELoss()
    first = None
    for _ in range(40):
        loss = loss_fn(model(X), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first, f"loss did not decrease: {first:.4f} -> {loss.item():.4f}"


def test_optimizer_buckets_by_rank():
    model = _tiny_model()
    opt = IncrementalCurvatureOptimizer(model.parameters(), rank=8, force_reference=True)
    summary = opt.bucket_summary()
    assert sum(summary.values()) == opt.num_blocks()
    assert summary[8] == opt.num_blocks() - 1  # all but the width-4 output bias
    assert summary[4] == 1


def test_woodbury_preconditioner_is_exact():
    """p = λ⁻¹ r_⊥ + U G⁻¹a must equal the dense solve (λI + U(G−λI)Uᵀ)⁻¹ g."""
    torch.manual_seed(0)
    d, r, lam = 40, 8, 0.7
    gen = torch.Generator().manual_seed(1)
    U = _orthonormal(d, r, gen, torch.device("cpu"), torch.float64)
    G = lam * torch.eye(r, dtype=torch.float64)
    for _ in range(20):
        a = torch.randn(r, dtype=torch.float64)
        G = G + torch.outer(a, a)
    L = torch.linalg.cholesky(G)
    g = torch.randn(d, dtype=torch.float64)
    a = U.mT @ g
    rv = g - U @ a
    y = torch.linalg.solve_triangular(L, a.unsqueeze(1), upper=False)
    x = torch.linalg.solve_triangular(L.mT, y, upper=True).squeeze(1)  # G^{-1} a
    p = rv / lam + U @ x
    F = lam * torch.eye(d, dtype=torch.float64) + U @ (G - lam * torch.eye(r, dtype=torch.float64)) @ U.mT
    p_dense = torch.linalg.solve(F, g)
    err = (torch.norm(p - p_dense) / torch.norm(p_dense)).item()
    assert err < 1e-10, f"Woodbury preconditioner not exact: {err:.2e}"


def test_incremental_core_matches_refactorization():
    """Incremental whitened core C_w must match a fresh re-cholesky (drift ~0)."""
    torch.manual_seed(0)
    m = torch.nn.Linear(16, 16, bias=False).double()
    opt = IncrementalCurvatureOptimizer(
        [m.weight], lr=0.0, rank=8, damping=0.7, tau=1e9,  # tau=inf → no refresh
        refresh_mode="incremental", force_reference=True)
    X = torch.randn(64, 16, dtype=torch.float64)
    for _ in range(40):
        loss = (m(X) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert opt.chart_drift() < 1e-10, f"core drift too high: {opt.chart_drift():.2e}"


def test_refresh_keeps_chart_orthonormal():
    torch.manual_seed(3)
    m = torch.nn.Linear(50, 50, bias=False).double()
    opt = IncrementalCurvatureOptimizer(
        [m.weight], lr=0.05, rank=8, damping=0.7, tau=0.1, force_reference=True)
    X = torch.randn(128, 50, dtype=torch.float64)
    for _ in range(60):
        loss = ((m(X) - 1.0) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    U = opt.blocks[0]["U"]
    ortho = torch.norm(U.mT @ U - torch.eye(U.shape[1], dtype=torch.float64)).item()
    assert ortho < 1e-10, f"chart lost orthonormality: {ortho:.2e}"
    assert opt.n_refresh > 0, "refresh never triggered"


def test_incremental_tracks_refact_optimizer():
    """Incremental and full-refact optimizers (identical charts) should agree."""
    def run(mode):
        torch.manual_seed(7)
        m = torch.nn.Sequential(
            torch.nn.Linear(20, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)).double()
        o = IncrementalCurvatureOptimizer(
            m.parameters(), lr=0.1, rank=8, damping=0.7, momentum=0.0, tau=0.1,
            refresh_mode=mode, force_reference=True)
        X = torch.randn(128, 20, dtype=torch.float64)
        Y = torch.randn(128, 1, dtype=torch.float64)
        for _ in range(60):
            l = ((m(X) - Y) ** 2).mean()
            o.zero_grad()
            l.backward()
            o.step()
        return l.item(), [p.detach().clone() for p in m.parameters()]

    li, pi = run("incremental")
    lr, pr = run("refact")
    assert abs(li - lr) < 1e-3, f"final loss differs: {li:.5f} vs {lr:.5f}"
    dmax = max((torch.norm(a - b) / torch.norm(b).clamp_min(1e-12)).item()
               for a, b in zip(pi, pr))
    assert dmax < 5e-3, f"params diverged: {dmax:.2e}"


# --------------------------------------------------------------------------- #
# GP Thompson sampling — whitened SVGP streaming conditioning (§4.4/§4.5)
# --------------------------------------------------------------------------- #


def _objective(X):
    return torch.sin(6 * X[:, 0]) * torch.cos(6 * X[:, 1])


def test_gp_incremental_matches_refactorization():
    """Incremental precision factor + A_w match a full re-cholesky after rounds."""
    torch.manual_seed(0)
    r, T = 24, 4
    X = torch.rand(r, 2, dtype=torch.float64)
    gp = BatchedGPThompson(X, _objective, T=T, force_reference=True, seed=0)
    for _ in range(12):
        gp.step()
    L_ref, Aw_ref = gp.refactor_ground_truth(gp.counts)
    L_err = (torch.norm(gp.L - L_ref) / torch.norm(L_ref)).item()
    Aw_err = (torch.norm(gp.Aw - Aw_ref) / torch.norm(Aw_ref)).item()
    assert L_err < 1e-10, f"precision factor drift {L_err:.2e}"
    assert Aw_err < 1e-9, f"whitened operator drift {Aw_err:.2e}"


def test_gp_regret_decreases():
    torch.manual_seed(0)
    r, T = 48, 16
    X = torch.rand(r, 2, dtype=torch.float64)
    gp = BatchedGPThompson(X, _objective, T=T, force_reference=True, seed=0)
    early = []
    for i in range(30):
        regret = gp.step()
        if i < 3:
            early.append(regret)
    early_mean = sum(early) / len(early)
    late = sum(gp.regret_history[-3:]) / 3
    assert late <= early_mean + 1e-6, f"regret grew: {early_mean:.4f} -> {late:.4f}"


# --------------------------------------------------------------------------- #
# SOAP baseline
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Non-symmetric dynamics Thompson sampling — the decisive double-sided workload
# --------------------------------------------------------------------------- #


def test_dynamics_incremental_matches_refactorization():
    """Incremental non-symmetric A_t must match a full re-cholesky (machine prec)."""
    from experiments.dynamics_thompson import BatchedDynamicsThompson
    torch.manual_seed(0)
    gp = BatchedDynamicsThompson(
        r=24, T=6, horizon=1, noise=0.03, force_reference=True,
        dtype=torch.float64, seed=0)
    for _ in range(30):
        gp.step()
    _, A_ref = gp.refactor_ground_truth()
    err = (torch.norm(gp.A - A_ref) / torch.norm(A_ref)).item()
    assert err < 1e-10, f"non-symmetric A drift too high: {err:.2e}"


def test_dynamics_operator_is_non_symmetric():
    """A_t ≠ A_t^⊤ (so no root A = R R^⊤ exists — double-sided is unavoidable)."""
    from experiments.dynamics_thompson import BatchedDynamicsThompson
    torch.manual_seed(0)
    gp = BatchedDynamicsThompson(
        r=24, T=6, horizon=1, force_reference=True, dtype=torch.float64, seed=0)
    for _ in range(20):
        gp.step()
    assert gp.symmetry_gap() > 0.2, f"A unexpectedly near-symmetric: {gp.symmetry_gap():.3f}"


def test_dynamics_regret_decreases():
    from experiments.dynamics_thompson import BatchedDynamicsThompson
    torch.manual_seed(0)
    gp = BatchedDynamicsThompson(
        r=24, T=32, horizon=1, noise=0.03, force_reference=True,
        dtype=torch.float64, seed=1)
    regs = [gp.step() for _ in range(50)]
    early = sum(regs[:5]) / 5
    late = sum(regs[-5:]) / 5
    assert late <= early + 1e-6, f"regret grew: {early:.4f} -> {late:.4f}"


def test_soap_reduces_loss():
    from experiments.soap import SOAP
    torch.manual_seed(0)
    m = torch.nn.Sequential(
        torch.nn.Linear(20, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
    opt = SOAP(m.parameters(), lr=3e-3)
    X = torch.randn(128, 20)
    Y = torch.randn(128, 1)
    first = None
    for _ in range(80):
        l = ((m(X) - Y) ** 2).mean()
        opt.zero_grad()
        l.backward()
        opt.step()
        if first is None:
            first = l.item()
    assert l.item() < first, f"SOAP did not reduce loss: {first:.4f} -> {l.item():.4f}"
