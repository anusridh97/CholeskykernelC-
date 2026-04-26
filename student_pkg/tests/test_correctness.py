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
