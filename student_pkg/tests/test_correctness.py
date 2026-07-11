"""
Correctness tests for the double-sided Cholesky propagation.

Every implementation (Python reference, C++ extension, Triton kernel) must
produce the same L+ and Aw+ as a ground-truth full re-factorization, up to
floating-point roundoff.

Run with:
    python -m pytest tests/test_correctness.py -v

The C++ tests are skipped if fast_cholesky is not built. The Triton tests
are skipped if CUDA is not available.
"""
import pytest
import torch

from doublesided_cholesky import incremental_update_reference


def ground_truth(G, M, z, w):
    """Full re-factorization: L+ = chol(G + zz^T), Aw+ = L+^{-1} (M + zw^T) L+^{-T}."""
    G_new = G + torch.outer(z, z)
    M_new = M + torch.outer(z, w)
    L_new = torch.linalg.cholesky(G_new)
    Linv_new = torch.linalg.inv(L_new)
    Aw_new = Linv_new @ M_new @ Linv_new.T
    return L_new, Aw_new


def random_problem(r, dtype=torch.float64, seed=0):
    """Generate a well-conditioned test problem."""
    torch.manual_seed(seed)
    A = torch.randn(r, r, dtype=dtype)
    G = A @ A.T + r * torch.eye(r, dtype=dtype)
    L = torch.linalg.cholesky(G).contiguous()
    M = torch.randn(r, r, dtype=dtype)
    Linv = torch.linalg.inv(L)
    Aw = Linv @ M @ Linv.T
    z = torch.randn(r, dtype=dtype) * 0.1
    w = torch.randn(r, dtype=dtype) * 0.1
    return G, L, M, Aw, z, w


@pytest.mark.parametrize("r", [8, 16, 64, 128])
def test_reference_matches_ground_truth(r):
    """Python reference should match full re-factorization to float64 precision."""
    G, L, M, Aw, z, w = random_problem(r, dtype=torch.float64)
    L_exact, Aw_exact = ground_truth(G, M, z, w)

    L_ref = L.clone()
    Aw_ref = incremental_update_reference(L_ref, Aw.clone(), z.clone(), w.clone())

    L_err = (torch.norm(L_ref - L_exact) / torch.norm(L_exact)).item()
    Aw_err = (torch.norm(Aw_ref - Aw_exact) / torch.norm(Aw_exact)).item()

    assert L_err < 1e-10, f"L+ error {L_err:.2e} exceeds tolerance"
    assert Aw_err < 1e-10, f"Aw+ error {Aw_err:.2e} exceeds tolerance"


@pytest.mark.parametrize("r", [8, 16, 64, 128])
def test_cpp_matches_ground_truth(r):
    """C++ extension should match full re-factorization."""
    try:
        import fast_cholesky
    except ImportError:
        pytest.skip("fast_cholesky extension not built")

    G, L, M, Aw, z, w = random_problem(r, dtype=torch.float64)
    L_exact, Aw_exact = ground_truth(G, M, z, w)

    L_test = L.clone()
    Aw_test, _ = fast_cholesky.fast_incremental_update(
        L_test, Aw.clone(), z.clone(), w.clone()
    )

    L_err = (torch.norm(L_test - L_exact) / torch.norm(L_exact)).item()
    Aw_err = (torch.norm(Aw_test - Aw_exact) / torch.norm(Aw_exact)).item()

    assert L_err < 1e-10, f"L+ error {L_err:.2e}"
    assert Aw_err < 1e-10, f"Aw+ error {Aw_err:.2e}"


def test_cpp_batched_matches_single():
    """Batched C++ path should match T independent single calls."""
    try:
        import fast_cholesky
    except ImportError:
        pytest.skip("fast_cholesky extension not built")

    r = 32
    T = 5
    torch.manual_seed(7)
    L_stack = torch.zeros(T, r, r, dtype=torch.float64)
    Aw_stack = torch.zeros(T, r, r, dtype=torch.float64)
    z_stack = torch.zeros(T, r, dtype=torch.float64)
    w_stack = torch.zeros(T, r, dtype=torch.float64)

    single_outputs = []
    for t in range(T):
        _, L, _, Aw, z, w = random_problem(r, dtype=torch.float64, seed=t)
        L_stack[t] = L
        Aw_stack[t] = Aw
        z_stack[t] = z
        w_stack[t] = w
        L_copy = L.clone()
        Aw_out, _ = fast_cholesky.fast_incremental_update(
            L_copy, Aw.clone(), z.clone(), w.clone()
        )
        single_outputs.append((L_copy, Aw_out))

    Aw_batched, _ = fast_cholesky.fast_incremental_update_batched(
        L_stack.clone(), Aw_stack.clone(), z_stack.clone(), w_stack.clone()
    )

    for t in range(T):
        err = (torch.norm(Aw_batched[t] - single_outputs[t][1])
               / torch.norm(single_outputs[t][1])).item()
        assert err < 1e-10, f"batched slice {t}: err={err:.2e}"


@pytest.mark.parametrize("r", [16, 64, 256])
def test_triton_matches_ground_truth(r):
    """Triton kernel should match full re-factorization on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from doublesided_cholesky import incremental_update_triton

    G, L, M, Aw, z, w = random_problem(r, dtype=torch.float32)
    G, L, M, Aw, z, w = [t.cuda() for t in (G, L, M, Aw, z, w)]
    L_exact, Aw_exact = ground_truth(G, M, z, w)

    L_test = L.clone()
    Aw_test, _ = incremental_update_triton(L_test, Aw.clone(), z.clone(), w.clone())

    L_err = (torch.norm(L_test - L_exact) / torch.norm(L_exact)).item()
    Aw_err = (torch.norm(Aw_test - Aw_exact) / torch.norm(Aw_exact)).item()

    # Looser tolerance for float32.
    assert L_err < 1e-4, f"L+ error {L_err:.2e}"
    assert Aw_err < 1e-4, f"Aw+ error {Aw_err:.2e}"


@pytest.mark.parametrize("r", [16, 64, 256])
def test_tiled_triton_matches_ground_truth(r):
    """Tiled Triton kernel should match full re-factorization on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from doublesided_cholesky import incremental_update_triton_tiled

    G, L, M, Aw, z, w = random_problem(r, dtype=torch.float32)
    G, L, M, Aw, z, w = [t.cuda() for t in (G, L, M, Aw, z, w)]
    L_exact, Aw_exact = ground_truth(G, M, z, w)

    L_test = L.clone()
    Aw_test, _ = incremental_update_triton_tiled(
        L_test, Aw.clone(), z.clone(), w.clone()
    )

    L_err = (torch.norm(L_test - L_exact) / torch.norm(L_exact)).item()
    Aw_err = (torch.norm(Aw_test - Aw_exact) / torch.norm(Aw_exact)).item()

    assert L_err < 1e-4, f"L+ error {L_err:.2e}"
    assert Aw_err < 1e-4, f"Aw+ error {Aw_err:.2e}"


@pytest.mark.parametrize("r", [16, 64, 256])
def test_tiled_triton_matches_fused_triton(r):
    """Tiled Triton path should agree with the single-kernel baseline."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from doublesided_cholesky import (
        incremental_update_triton,
        incremental_update_triton_tiled,
    )

    _, L, _, Aw, z, w = random_problem(r, dtype=torch.float32)
    L, Aw, z, w = [t.cuda() for t in (L, Aw, z, w)]

    L_fused = L.clone()
    Aw_fused, vz_fused = incremental_update_triton(
        L_fused, Aw.clone(), z.clone(), w.clone()
    )

    L_tiled = L.clone()
    Aw_tiled, vz_tiled = incremental_update_triton_tiled(
        L_tiled, Aw.clone(), z.clone(), w.clone()
    )

    L_err = (torch.norm(L_tiled - L_fused) / torch.norm(L_fused)).item()
    Aw_err = (torch.norm(Aw_tiled - Aw_fused) / torch.norm(Aw_fused)).item()
    vz_err = (torch.norm(vz_tiled - vz_fused) / torch.norm(vz_fused)).item()

    assert L_err < 1e-6, f"L+ disagreement {L_err:.2e}"
    assert Aw_err < 1e-5, f"Aw+ disagreement {Aw_err:.2e}"
    assert vz_err < 1e-6, f"vz disagreement {vz_err:.2e}"


def test_triton_downdate_matches_reference():
    """Batched Triton hyperbolic downdate should match the CPU reference downdate."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from doublesided_cholesky import downdate_triton_batched
    from experiments.extensions import downdate as downdate_reference

    r, T = 64, 5
    torch.manual_seed(0)
    A = torch.randn(T, r, r).cuda()
    G = A @ A.mT + r * torch.eye(r, device="cuda").unsqueeze(0)
    L = torch.linalg.cholesky(G).contiguous()
    M = torch.randn(T, r, r).cuda()
    Linv = torch.linalg.inv(L)
    Aw = Linv @ M @ Linv.mT
    z = torch.randn(T, r, device="cuda") * 0.1
    w = torch.randn(T, r, device="cuda") * 0.1

    # Reference (forced CPU-style loop on GPU tensors) vs the fused kernel.
    L_ref = L.clone()
    Aw_ref, adm_ref = downdate_reference(L_ref, Aw.clone(), z, w,
                                         max_leverage=0.999, force_reference=True)
    L_tri = L.clone()
    Aw_tri, _, adm_tri = downdate_triton_batched(L_tri, Aw.clone(), z, w,
                                                 max_leverage=0.999)

    assert torch.equal(adm_ref, adm_tri)
    L_err = (torch.norm(L_tri - L_ref) / torch.norm(L_ref)).item()
    Aw_err = (torch.norm(Aw_tri - Aw_ref) / torch.norm(Aw_ref)).item()
    assert L_err < 1e-4, f"L- kernel vs reference {L_err:.2e}"
    assert Aw_err < 1e-4, f"A- kernel vs reference {Aw_err:.2e}"

    # And against a full re-factorization ground truth.
    G_new = G - z.unsqueeze(2) * z.unsqueeze(1)
    M_new = M - z.unsqueeze(2) * w.unsqueeze(1)
    L_ex = torch.linalg.cholesky(G_new)
    Linv_ex = torch.linalg.inv(L_ex)
    Aw_ex = Linv_ex @ M_new @ Linv_ex.mT
    assert (torch.norm(Aw_tri - Aw_ex) / torch.norm(Aw_ex)).item() < 1e-3


def test_triton_downdate_guard_skips():
    """Inadmissible charts (leverage ≥ threshold) are left unchanged by the kernel."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from doublesided_cholesky import downdate_triton_batched

    r = 32
    torch.manual_seed(1)
    A = torch.randn(1, r, r).cuda()
    G = A @ A.mT + r * torch.eye(r, device="cuda").unsqueeze(0)
    L = torch.linalg.cholesky(G).contiguous()
    Aw = torch.zeros(1, r, r, device="cuda")
    # z with large leverage (aligned with L's columns, scaled up).
    z = (L @ torch.ones(1, r, 1, device="cuda")).squeeze(-1) * 0.9
    admitted = torch.zeros(1, dtype=torch.bool, device="cuda")  # force skip
    L_in = L.clone()
    Aw_out, _, _ = downdate_triton_batched(L_in, Aw.clone(), z, z, admitted=admitted)
    assert torch.equal(L_in, L), "skipped downdate must leave L unchanged"
    assert torch.equal(Aw_out, Aw), "skipped downdate must leave A_w unchanged"


def test_triton_batched_matches_single():
    """Batched Triton path should match T independent single calls."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from doublesided_cholesky import incremental_update_triton, incremental_update_triton_batched

    r = 32
    T = 5
    L_stack  = torch.zeros(T, r, r, dtype=torch.float32).cuda()
    Aw_stack = torch.zeros(T, r, r, dtype=torch.float32).cuda()
    z_stack  = torch.zeros(T, r,    dtype=torch.float32).cuda()
    w_stack  = torch.zeros(T, r,    dtype=torch.float32).cuda()

    single_aw = []
    for t in range(T):
        _, L, _, Aw, z, w = random_problem(r, dtype=torch.float32, seed=t)
        L, Aw, z, w = L.cuda(), Aw.cuda(), z.cuda(), w.cuda()
        L_stack[t]  = L
        Aw_stack[t] = Aw
        z_stack[t]  = z
        w_stack[t]  = w
        Aw_out, _ = incremental_update_triton(L.clone(), Aw.clone(), z.clone(), w.clone())
        single_aw.append(Aw_out)

    Aw_batched, _ = incremental_update_triton_batched(
        L_stack.clone(), Aw_stack.clone(), z_stack.clone(), w_stack.clone()
    )

    for t in range(T):
        err = (torch.norm(Aw_batched[t] - single_aw[t]) / torch.norm(single_aw[t])).item()
        assert err < 1e-5, f"batched slice {t}: err={err:.2e}"
