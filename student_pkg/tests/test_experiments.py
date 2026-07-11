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


# --------------------------------------------------------------------------- #
# Adaptive-inducing full-covariance SVGP-PDTS (#2) and fantasy branches (#3)
# --------------------------------------------------------------------------- #


def _obj2(X):
    return torch.sin(6 * X[:, 0]) * torch.cos(6 * X[:, 1])


def test_svgp_covariance_matches_refactorization():
    """A_t (transport + covariance conditioning + damping) matches re-cholesky."""
    from experiments.svgp_pdts import BatchedAdaptiveSVGP
    torch.manual_seed(0)
    X = torch.rand(20, 2, dtype=torch.float64)
    gp = BatchedAdaptiveSVGP(X, _obj2, T=6, noise=0.05, damp_rho=0.03, move_scale=0.05,
                             force_reference=True, dtype=torch.float64, seed=0)
    for _ in range(25):
        gp.step()
    _, A_ref = gp.refactor_ground_truth()
    err = (torch.norm(gp.A - A_ref) / torch.norm(A_ref)).item()
    assert err < 1e-10, f"variational covariance drift too high: {err:.2e}"
    # Covariance stays SPD.
    assert torch.linalg.eigvalsh(gp.A).min().item() > 0, "A lost positive-definiteness"


def test_svgp_regret_decreases():
    from experiments.svgp_pdts import BatchedAdaptiveSVGP
    torch.manual_seed(0)
    X = torch.rand(24, 2, dtype=torch.float64)
    gp = BatchedAdaptiveSVGP(X, _obj2, T=16, noise=0.05, force_reference=True,
                             dtype=torch.float64, seed=1)
    regs = [gp.step() for _ in range(30)]
    assert sum(regs[-3:]) / 3 <= sum(regs[:3]) / 3 + 1e-6, "regret grew"


def test_svgp_incremental_matches_refactor_transport():
    """Incremental and refactor transport modes give the same covariance."""
    from experiments.svgp_pdts import BatchedAdaptiveSVGP
    X = torch.rand(20, 2, dtype=torch.float64)
    a = BatchedAdaptiveSVGP(X, _obj2, T=4, transport_mode="incremental",
                            force_reference=True, dtype=torch.float64, seed=3)
    b = BatchedAdaptiveSVGP(X, _obj2, T=4, transport_mode="refactor",
                            force_reference=True, dtype=torch.float64, seed=3)
    for _ in range(15):
        a.step(); b.step()
    err = (torch.norm(a.A - b.A) / torch.norm(b.A)).item()
    assert err < 1e-9, f"incremental vs refactor transport disagree: {err:.2e}"


def test_fantasy_covariance_matches_refactorization():
    """Per-branch fantasy covariance matches a full re-cholesky."""
    from experiments.svgp_pdts import BatchedFantasySVGP
    torch.manual_seed(0)
    X = torch.rand(20, 2, dtype=torch.float64)
    fp = BatchedFantasySVGP(X, T=4, J=8, k=3, noise=0.05, force_reference=True,
                            dtype=torch.float64, seed=0)
    for _ in range(15):
        fp.round()
    _, A_ref = fp.refactor_ground_truth()
    err = (torch.norm(fp.A - A_ref) / torch.norm(A_ref)).item()
    assert err < 1e-9, f"fantasy covariance drift too high: {err:.2e}"


def test_robust_sandwich_matches_refactorization():
    """The whitened sandwich A_t matches a full re-cholesky to machine precision."""
    from experiments.robust_bandit import BatchedRobustSandwichTS
    torch.manual_seed(0)
    b = BatchedRobustSandwichTS(r=16, T=16, noise_mode="hetero", force_reference=True,
                                calibrate_every=1000, dtype=torch.float64, seed=0)
    for _ in range(40):
        b.step()
    _, A_ref = b.refactor_ground_truth()
    err = (torch.norm(b.A - A_ref) / torch.norm(A_ref)).item()
    assert err < 1e-10, f"sandwich drift too high: {err:.2e}"


def test_robust_sandwich_separates_from_identity_under_misspec():
    """Heavy-tailed misspecification separates A from a scaled identity more than
    the homoskedastic (negative-control) case."""
    from experiments.robust_bandit import BatchedRobustSandwichTS

    def sep(mode):
        torch.manual_seed(0)
        b = BatchedRobustSandwichTS(r=8, T=64, n_arms=12, noise_mode=mode,
                                    noise_scale=0.3, hetero_beta=4.0,
                                    calibrate_every=10**9, force_reference=True,
                                    dtype=torch.float64, seed=0)
        b.theta_star = b.theta_star * 0.5
        for _ in range(250):
            b.step()
        return b.separation_from_identity()

    assert sep("heavy") > sep("homo") + 0.05, "heavy tails should separate A from cI more"


def test_real_data_sandwich_matches_refactorization():
    """On a real dataset, the whitened sandwich A_t matches a full re-cholesky.

    Skips if the small real dataset cannot be fetched (offline / proxy)."""
    import pytest
    from experiments.bandit_data import load_charges
    from experiments.robust_bandit import RealDataSandwichTS
    try:
        C, r, _ = load_charges(dtype=torch.float64)
    except RuntimeError as exc:
        pytest.skip(f"real dataset unavailable: {exc}")
    torch.manual_seed(0)
    b = RealDataSandwichTS(C, r, T=32, n_arms=16, force_reference=True,
                           dtype=torch.float64, seed=0)
    for _ in range(60):
        b.step()
    _, A_ref = b.refactor_ground_truth()
    err = (torch.norm(b.A - A_ref) / torch.norm(A_ref)).item()
    assert err < 1e-10, f"real-data sandwich drift too high: {err:.2e}"


def test_real_tails_drive_separation():
    """Real heavy-tailed reward separates A from cI more than the Gaussian control."""
    import pytest
    from experiments.bandit_data import load_charges
    from experiments.robust_bandit import RealDataSandwichTS
    try:
        C, r, _ = load_charges(dtype=torch.float64)
    except RuntimeError as exc:
        pytest.skip(f"real dataset unavailable: {exc}")

    def sep(gauss):
        torch.manual_seed(0)
        b = RealDataSandwichTS(C, r, T=64, n_arms=16, gaussianize=gauss,
                               force_reference=True, dtype=torch.float64, seed=0)
        for _ in range(250):
            b.step()
        return b.separation_from_identity()

    assert sep(False) > sep(True) + 0.03, "real tails should separate A from cI more"


# --------------------------------------------------------------------------- #
# Rank-k updates (C.3) and hyperbolic downdates (C.4)
# --------------------------------------------------------------------------- #


def _spd_chart(r, T, seed):
    torch.manual_seed(seed)
    Araw = torch.randn(T, r, r, dtype=torch.float64)
    G = Araw @ Araw.mT + r * torch.eye(r, dtype=torch.float64)
    L = torch.linalg.cholesky(G).contiguous()
    M = torch.randn(T, r, r, dtype=torch.float64)
    Linv = torch.linalg.inv(L)
    Aw = Linv @ M @ Linv.mT
    return G, L, M, Aw


def test_rank_k_matches_refactorization():
    from experiments.extensions import rank_k_update
    r, T, k = 16, 4, 3
    G, L, M, Aw = _spd_chart(r, T, 0)
    Z = torch.randn(T, r, k, dtype=torch.float64) * 0.2
    W = torch.randn(T, r, k, dtype=torch.float64) * 0.2
    L_test = L.clone()
    Aw_new, _ = rank_k_update(L_test, Aw.clone(), Z, W, force_reference=True)
    G_new = G + Z @ Z.mT
    M_new = M + Z @ W.mT
    L_ex = torch.linalg.cholesky(G_new)
    Linv = torch.linalg.inv(L_ex)
    Aw_ex = Linv @ M_new @ Linv.mT
    err = (torch.norm(Aw_new - Aw_ex) / torch.norm(Aw_ex)).item()
    assert err < 1e-10, f"rank-k update mismatch: {err:.2e}"


def test_downdate_matches_refactorization():
    from experiments.extensions import downdate
    r, T = 16, 4
    G, L, M, Aw = _spd_chart(r, T, 1)
    z = torch.randn(T, r, dtype=torch.float64) * 0.1
    w = torch.randn(T, r, dtype=torch.float64) * 0.1
    L_test = L.clone()
    Aw_new, admitted = downdate(L_test, Aw.clone(), z, w, max_leverage=0.999)
    assert admitted.all(), "small downdates should be admissible"
    G_new = G - z.unsqueeze(2) * z.unsqueeze(1)
    M_new = M - z.unsqueeze(2) * w.unsqueeze(1)
    L_ex = torch.linalg.cholesky(G_new)
    Linv = torch.linalg.inv(L_ex)
    Aw_ex = Linv @ M_new @ Linv.mT
    L_err = (torch.norm(L_test - L_ex) / torch.norm(L_ex)).item()
    A_err = (torch.norm(Aw_new - Aw_ex) / torch.norm(Aw_ex)).item()
    assert L_err < 1e-10 and A_err < 1e-9, f"downdate mismatch: L {L_err:.2e}, A {A_err:.2e}"


def test_downdate_leverage_guard():
    """A downdate with leverage τ ≥ threshold is skipped, not blown up."""
    from experiments.extensions import downdate, leverage
    r = 12
    G, L, M, Aw = _spd_chart(r, 1, 2)
    # Build a z with large leverage: z aligned with a low-energy direction, scaled up.
    z = (L @ torch.ones(1, r, 1, dtype=torch.float64)).squeeze(-1) * 0.9  # ‖L^{-1}z‖ ~ 0.9·√r
    tau = leverage(L, z).item()
    L_test = L.clone()
    _, admitted = downdate(L_test, Aw.clone(), z, z, max_leverage=min(tau, 1.0) - 1e-6)
    assert not admitted.all(), f"guard should reject τ={tau:.2f} downdate"
    assert torch.allclose(L_test, L), "rejected downdate must leave L unchanged"


def test_add_then_downdate_is_identity():
    """Adding a rank-1 then downdating the same vector returns to the start."""
    from experiments.extensions import downdate
    from experiments.charts import refresh_bucket
    r, T = 12, 3
    G, L, M, Aw = _spd_chart(r, T, 3)
    z = torch.randn(T, r, dtype=torch.float64) * 0.1
    w = torch.randn(T, r, dtype=torch.float64) * 0.1
    L2 = L.clone()
    Aw2, _ = refresh_bucket(L2, Aw.clone(), z, w, force_reference=True)
    Aw3, admitted = downdate(L2, Aw2, z, w, max_leverage=0.999)
    assert admitted.all()
    assert (torch.norm(L2 - L) / torch.norm(L)).item() < 1e-9
    assert (torch.norm(Aw3 - Aw) / torch.norm(Aw)).item() < 1e-8


def test_sliding_window_matches_refactor_and_forgets():
    from experiments.nonstationary_bandit import SlidingWindowSandwichTS
    torch.manual_seed(0)
    b = SlidingWindowSandwichTS(r=16, T=16, n_arms=12, batch_k=3, window=8,
                                forget=True, force_reference=True,
                                dtype=torch.float64, seed=0)
    for _ in range(40):
        b.step()
    _, A_ref = b.refactor_ground_truth()
    err = (torch.norm(b.A - A_ref) / torch.norm(A_ref)).item()
    assert err < 1e-9, f"windowed sandwich drift: {err:.2e}"
    assert len(b.buffer) == b.window, "window did not stay bounded"


def test_forgetting_beats_no_forget_under_drift():
    from experiments.nonstationary_bandit import SlidingWindowSandwichTS

    def post_switch_regret(forget):
        torch.manual_seed(1)
        g = SlidingWindowSandwichTS(r=16, T=64, n_arms=16, batch_k=4, window=15,
                                    forget=forget, noise_scale=0.2,
                                    force_reference=True, dtype=torch.float64, seed=1)
        g.switch_at = 40
        regs = [g.step() for _ in range(80)]
        return sum(regs[45:]) / len(regs[45:])

    assert post_switch_regret(True) < post_switch_regret(False), \
        "sliding window should adapt to the switch better than no-forget"


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
