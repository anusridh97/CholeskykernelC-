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


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(torch.float64, 2e-15), (torch.float32, 3e-6)],
)
@pytest.mark.parametrize("k", [1, 2, 5])
def test_cpp_native_multi_matches_sequential(dtype, tolerance, k):
    """Allocation-free native rank-k path matches k legacy updates."""
    try:
        import fast_cholesky
    except ImportError:
        pytest.skip("fast_cholesky extension not built")

    r = 33  # Exercise the scalar tails of every AVX-512 sweep.
    _, L, _, Aw, _, _ = random_problem(r, dtype=dtype, seed=100 + k)
    generator = torch.Generator().manual_seed(200 + k)
    Z = torch.randn(r, k, dtype=dtype, generator=generator) * 0.05
    W = torch.randn(r, k, dtype=dtype, generator=generator) * 0.05

    L_legacy = L.clone()
    Aw_legacy = Aw.clone()
    vz_legacy = []
    for column in range(k):
        Aw_legacy, vz = fast_cholesky.fast_incremental_update(
            L_legacy,
            Aw_legacy,
            Z[:, column].clone(),
            W[:, column],
        )
        vz_legacy.append(vz)

    L_native = L.clone()
    Aw_native = Aw.clone()
    workspace = torch.empty(7 * r, dtype=dtype)
    vz_native = torch.empty(k, r, dtype=dtype)
    Aw_pointer = Aw_native.data_ptr()
    vz_pointer = vz_native.data_ptr()
    Aw_result, vz_result = fast_cholesky.fast_incremental_update_multi_out(
        L_native,
        Aw_native,
        Z.contiguous(),
        W.contiguous(),
        workspace,
        vz_native,
    )

    assert Aw_result.data_ptr() == Aw_pointer
    assert vz_result.data_ptr() == vz_pointer
    expected_vz = torch.stack(vz_legacy)
    for actual, expected in (
        (L_native, L_legacy),
        (Aw_native, Aw_legacy),
        (vz_native, expected_vz),
    ):
        error = (torch.norm(actual - expected) / torch.norm(expected)).item()
        assert error < tolerance


def test_cpp_native_multi_convenience_preserves_operator_input():
    """Convenience rank-k API retains legacy Aw ownership semantics."""
    try:
        import fast_cholesky
    except ImportError:
        pytest.skip("fast_cholesky extension not built")

    r, k = 17, 3
    _, L, _, Aw, _, _ = random_problem(r, dtype=torch.float64, seed=301)
    generator = torch.Generator().manual_seed(302)
    Z = torch.randn(r, k, dtype=torch.float64, generator=generator) * 0.05
    W = torch.randn(r, k, dtype=torch.float64, generator=generator) * 0.05
    Aw_before = Aw.clone()

    Aw_result, vz_result = fast_cholesky.fast_incremental_update_multi(
        L.clone(),
        Aw,
        Z,
        W,
    )
    assert torch.equal(Aw, Aw_before)
    assert Aw_result.shape == Aw.shape
    assert vz_result.shape == (k, r)


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(torch.float64, 2e-14), (torch.float32, 3e-6)],
)
@pytest.mark.parametrize(("r", "k"), [(17, 1), (33, 4), (64, 8), (65, 16)])
def test_cpp_compact_factor_matches_fresh(dtype, tolerance, r, k):
    """Compact rank-k C and C^-1 representations reproduce fresh Cholesky."""
    try:
        import fast_cholesky
    except ImportError:
        pytest.skip("fast_cholesky extension not built")

    G, L, _, _, _, _ = random_problem(r, dtype=dtype, seed=400 + r + k)
    generator = torch.Generator().manual_seed(500 + r + k)
    Z = torch.randn(r, k, dtype=dtype, generator=generator) * 0.05
    U = torch.linalg.solve_triangular(L, Z, upper=False).contiguous()
    d = torch.empty(r, dtype=dtype)
    Q = torch.empty_like(U)
    workspace = torch.empty(k * k + 2 * k, dtype=dtype)
    phase_seconds = torch.empty(2, dtype=torch.float64)

    L_compact = L.clone()
    pointer = L_compact.data_ptr()
    result = fast_cholesky.fast_compact_factor_profile_out(
        L_compact,
        U,
        d,
        Q,
        workspace,
        phase_seconds,
    )
    assert result.data_ptr() == pointer
    assert bool(torch.all(phase_seconds >= 0.0))

    C = torch.zeros(r, r, dtype=dtype)
    C.diagonal().copy_(d)
    P = torch.zeros_like(C)
    P.diagonal().copy_(d.reciprocal())
    for row in range(r):
        if row:
            C[row, :row] = U[row] @ Q[:row].mT
            P[row, :row] = -(Q[row] @ U[:row].mT)

    X = torch.randn(
        r,
        k + 3,
        dtype=dtype,
        generator=generator,
    ).contiguous()
    compact_product = torch.empty_like(X)
    apply_workspace = torch.empty(k * X.shape[1], dtype=dtype)
    product_pointer = compact_product.data_ptr()
    product_result = fast_cholesky.fast_compact_apply_left_out(
        X,
        U,
        Q,
        d,
        compact_product,
        apply_workspace,
    )
    assert product_result.data_ptr() == product_pointer

    identity = torch.eye(r, dtype=dtype)
    L_fresh = torch.linalg.cholesky(G + Z @ Z.mT)
    comparisons = (
        (C @ C.mT, identity + U @ U.mT),
        (P @ C, identity),
        (compact_product, P @ X),
        (L_compact, L @ C),
        (L_compact, L_fresh),
    )
    for actual, expected in comparisons:
        error = (torch.norm(actual - expected) / torch.norm(expected)).item()
        assert error < tolerance


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


@pytest.mark.parametrize("r", [16, 64, 256, 512, 1024])
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


def test_tiled_triton_preserves_legacy_positional_arguments():
    """The optimized public wrapper retains the old positional call shape."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from doublesided_cholesky import incremental_update_triton_tiled

    _, L, _, Aw, z, w = random_problem(64, dtype=torch.float32)
    L, Aw, z, w = [tensor.cuda() for tensor in (L, Aw, z, w)]
    L_keyword = L.clone()
    L_positional = L.clone()
    Aw_keyword, vz_keyword = incremental_update_triton_tiled(
        L_keyword, Aw.clone(), z, w, block_m=16, block_n=64, num_warps=4
    )
    Aw_positional, vz_positional = incremental_update_triton_tiled(
        L_positional, Aw.clone(), z, w, 16, 64, 4
    )

    assert torch.equal(L_keyword, L_positional)
    assert torch.equal(Aw_keyword, Aw_positional)
    assert torch.equal(vz_keyword, vz_positional)


@pytest.mark.parametrize("r", [256, 512, 1024])
@pytest.mark.parametrize(
    "implementation_name",
    (
        "incremental_update_triton_tiled_v2",
        "incremental_update_triton_scan",
        "incremental_update_triton_cumsum",
        "incremental_update_triton_parallel",
    ),
)
def test_optimized_large_rank_triton_matches_ground_truth(r, implementation_name):
    """Optimized HBM-resident paths must remain correct through rank 1024."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    import doublesided_cholesky

    implementation = getattr(doublesided_cholesky, implementation_name)
    G, L, M, Aw, z, w = random_problem(r, dtype=torch.float32)
    G, L, M, Aw, z, w = [tensor.cuda() for tensor in (G, L, M, Aw, z, w)]
    L_exact, Aw_exact = ground_truth(G, M, z, w)

    L_test = L.clone()
    Aw_test, _ = implementation(L_test, Aw, z, w)

    L_err = (torch.norm(L_test - L_exact) / torch.norm(L_exact)).item()
    Aw_err = (torch.norm(Aw_test - Aw_exact) / torch.norm(Aw_exact)).item()

    assert L_err < 1e-4, f"{implementation_name}: L+ error {L_err:.2e}"
    assert Aw_err < 1e-4, f"{implementation_name}: Aw+ error {Aw_err:.2e}"


@pytest.mark.parametrize(
    "implementation_name",
    (
        "incremental_update_triton_tiled_v2",
        "incremental_update_triton_scan",
        "incremental_update_triton_cumsum",
        "incremental_update_triton_parallel",
    ),
)
def test_optimized_large_rank_triton_batched_matches_single(implementation_name):
    """Large-rank optimized paths must process batch members independently."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    import doublesided_cholesky

    implementation = getattr(doublesided_cholesky, implementation_name)
    rank = 256
    batch = 4
    problems = [
        random_problem(rank, dtype=torch.float32, seed=seed)
        for seed in range(batch)
    ]
    L = torch.stack([problem[1] for problem in problems]).cuda()
    Aw = torch.stack([problem[3] for problem in problems]).cuda()
    z = torch.stack([problem[4] for problem in problems]).cuda()
    w = torch.stack([problem[5] for problem in problems]).cuda()

    expected_L = []
    expected_Aw = []
    for index in range(batch):
        L_single = L[index].clone()
        Aw_single, _ = implementation(
            L_single,
            Aw[index],
            z[index],
            w[index],
        )
        expected_L.append(L_single)
        expected_Aw.append(Aw_single.clone())

    L_batched = L.clone()
    Aw_batched, _ = implementation(L_batched, Aw, z, w)
    expected_L = torch.stack(expected_L)
    expected_Aw = torch.stack(expected_Aw)

    L_err = (torch.norm(L_batched - expected_L) / torch.norm(expected_L)).item()
    Aw_err = (
        torch.norm(Aw_batched - expected_Aw) / torch.norm(expected_Aw)
    ).item()
    assert L_err < 1e-6, f"{implementation_name}: batched L+ error {L_err:.2e}"
    assert Aw_err < 1e-5, f"{implementation_name}: batched Aw+ error {Aw_err:.2e}"


@pytest.mark.parametrize("r", [255, 257, 513, 1000])
def test_parallel_triton_non_power_rank(r):
    """Padding and reverse scans must work away from powers of two."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from doublesided_cholesky import incremental_update_triton_parallel

    G, L, M, Aw, z, w = random_problem(r, dtype=torch.float32, seed=r)
    G, L, M, Aw, z, w = [
        tensor.cuda() for tensor in (G, L, M, Aw, z, w)
    ]
    L_exact, Aw_exact = ground_truth(G, M, z, w)
    L_test = L.clone()
    Aw_test, _ = incremental_update_triton_parallel(L_test, Aw, z, w)
    L_err = (torch.norm(L_test - L_exact) / torch.norm(L_exact)).item()
    Aw_err = (torch.norm(Aw_test - Aw_exact) / torch.norm(Aw_exact)).item()
    assert L_err < 1e-4, f"non-power r={r}: L+ error {L_err:.2e}"
    assert Aw_err < 1e-4, f"non-power r={r}: Aw+ error {Aw_err:.2e}"


@pytest.mark.parametrize("r", [257, 1025])
def test_inverse_factor_large_non_power_rank(r):
    """The full-axis inverse path must cross the old r=1024 boundary."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from doublesided_cholesky import (
        TritonTiledWorkspace,
        incremental_update_triton_inverse_factor_large,
    )

    G, L, M, Aw, z, w = random_problem(r, dtype=torch.float32, seed=r)
    G, L, M, Aw, z, w = [
        tensor.cuda() for tensor in (G, L, M, Aw, z, w)
    ]
    L_exact, Aw_exact = ground_truth(G, M, z, w)
    Linv_exact = torch.linalg.inv(L_exact)
    Linv_test = torch.linalg.inv(L).tril().contiguous()
    Aw_test = Aw.clone()
    workspace = TritonTiledWorkspace(Aw_test, solve_width=2)
    output, _ = incremental_update_triton_inverse_factor_large(
        Linv_test,
        Aw_test,
        z,
        w,
        workspace=workspace,
    )
    assert output.data_ptr() == Aw_test.data_ptr()
    Linv_err = (
        torch.norm(Linv_test - Linv_exact) / torch.norm(Linv_exact)
    ).item()
    Aw_err = (torch.norm(output - Aw_exact) / torch.norm(Aw_exact)).item()
    assert Linv_err < 2e-5, f"large inverse r={r}: error {Linv_err:.2e}"
    assert Aw_err < 2e-5, f"large Aw r={r}: error {Aw_err:.2e}"


@pytest.mark.parametrize("r", [129, 257, 1025])
def test_inverse_factor_inplace_non_power_rank(r):
    """The two-matrix path must remain exact across the old rank boundary."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from doublesided_cholesky import (
        TritonInverseInplaceWorkspace,
        incremental_update_triton_inverse_factor_inplace,
    )

    G, L, M, Aw, z, w = random_problem(r, dtype=torch.float32, seed=r + 1)
    G, L, M, Aw, z, w = [
        tensor.cuda() for tensor in (G, L, M, Aw, z, w)
    ]
    L_exact, Aw_exact = ground_truth(G, M, z, w)
    Linv_exact = torch.linalg.inv(L_exact)
    Linv_test = torch.linalg.inv(L).tril().contiguous()
    Aw_test = Aw.clone()
    workspace = TritonInverseInplaceWorkspace(Aw_test)
    output, _ = incremental_update_triton_inverse_factor_inplace(
        Linv_test,
        Aw_test,
        z,
        w,
        workspace=workspace,
    )
    assert output.data_ptr() == Aw_test.data_ptr()
    Linv_err = (
        torch.norm(Linv_test - Linv_exact) / torch.norm(Linv_exact)
    ).item()
    Aw_err = (torch.norm(output - Aw_exact) / torch.norm(Aw_exact)).item()
    assert Linv_err < 5e-6, f"in-place inverse r={r}: error {Linv_err:.2e}"
    assert Aw_err < 5e-6, f"in-place Aw r={r}: error {Aw_err:.2e}"


def test_inverse_factor_inplace_batched_matches_single():
    """Dynamic row/column recurrences must reset at batch boundaries."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from doublesided_cholesky import (
        incremental_update_triton_inverse_factor_inplace,
    )

    rank = 129
    batch = 3
    problems = [
        random_problem(rank, dtype=torch.float32, seed=20 + index)
        for index in range(batch)
    ]
    Linv = torch.stack(
        [torch.linalg.inv(problem[1]).tril() for problem in problems]
    ).contiguous().cuda()
    Aw = torch.stack([problem[3] for problem in problems]).cuda()
    z = torch.stack([problem[4] for problem in problems]).cuda()
    w = torch.stack([problem[5] for problem in problems]).cuda()

    expected_inverse = []
    expected_aw = []
    for index in range(batch):
        inverse_single = Linv[index].clone()
        aw_single = Aw[index].clone()
        incremental_update_triton_inverse_factor_inplace(
            inverse_single,
            aw_single,
            z[index],
            w[index],
        )
        expected_inverse.append(inverse_single)
        expected_aw.append(aw_single)

    incremental_update_triton_inverse_factor_inplace(Linv, Aw, z, w)
    expected_inverse = torch.stack(expected_inverse)
    expected_aw = torch.stack(expected_aw)
    inverse_err = (
        torch.norm(Linv - expected_inverse) / torch.norm(expected_inverse)
    ).item()
    aw_err = (torch.norm(Aw - expected_aw) / torch.norm(expected_aw)).item()
    assert inverse_err < 2e-6
    assert aw_err < 2e-6


def test_optimized_cuda_input_validation():
    """Optimized entry points reject unsupported layouts before launching."""
    from doublesided_cholesky import incremental_update_triton_parallel

    rank = 8
    L = torch.eye(rank)
    Aw = torch.eye(rank)
    z = torch.ones(rank)
    w = torch.ones(rank)
    with pytest.raises(ValueError, match="CUDA"):
        incremental_update_triton_parallel(L, Aw, z, w)
    with pytest.raises(ValueError, match="same shape"):
        incremental_update_triton_parallel(
            L,
            Aw[:-1],
            z,
            w,
        )
    with pytest.raises(ValueError, match="z must have shape"):
        incremental_update_triton_parallel(
            L,
            Aw,
            z[:-1],
            w,
        )


@pytest.mark.parametrize("r", [256, 512, 1024])
def test_inverse_state_triton_matches_ground_truth(r):
    """The cached inverse-factor path must update L, L^{-1}, and Aw."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from doublesided_cholesky import incremental_update_triton_inverse_state

    G, L, M, Aw, z, w = random_problem(r, dtype=torch.float32)
    G, L, M, Aw, z, w = [tensor.cuda() for tensor in (G, L, M, Aw, z, w)]
    L_exact, Aw_exact = ground_truth(G, M, z, w)
    Linv_exact = torch.linalg.inv(L_exact)

    L_test = L.clone()
    Linv_test = torch.linalg.inv(L_test).tril().contiguous()
    Aw_test, _ = incremental_update_triton_inverse_state(
        L_test,
        Linv_test,
        Aw,
        z,
        w,
    )
    L_err = (torch.norm(L_test - L_exact) / torch.norm(L_exact)).item()
    Linv_err = (
        torch.norm(Linv_test - Linv_exact) / torch.norm(Linv_exact)
    ).item()
    Aw_err = (torch.norm(Aw_test - Aw_exact) / torch.norm(Aw_exact)).item()
    assert L_err < 1e-4, f"inverse-state L+ error {L_err:.2e}"
    assert Linv_err < 1e-4, f"inverse-state inverse error {Linv_err:.2e}"
    assert Aw_err < 1e-4, f"inverse-state Aw+ error {Aw_err:.2e}"


@pytest.mark.parametrize("r", [256, 512, 1024])
def test_inverse_factor_triton_matches_ground_truth(r):
    """Inverse-factor state must remain exact without materializing L+."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from doublesided_cholesky import incremental_update_triton_inverse_factor

    G, L, M, Aw, z, w = random_problem(r, dtype=torch.float32)
    G, L, M, Aw, z, w = [tensor.cuda() for tensor in (G, L, M, Aw, z, w)]
    L_exact, Aw_exact = ground_truth(G, M, z, w)
    Linv_exact = torch.linalg.inv(L_exact)

    Linv_test = torch.linalg.inv(L).tril().contiguous()
    Aw_test, _ = incremental_update_triton_inverse_factor(
        Linv_test,
        Aw,
        z,
        w,
    )
    Linv_err = (
        torch.norm(Linv_test - Linv_exact) / torch.norm(Linv_exact)
    ).item()
    Aw_err = (torch.norm(Aw_test - Aw_exact) / torch.norm(Aw_exact)).item()
    assert Linv_err < 5e-6, f"inverse-factor inverse error {Linv_err:.2e}"
    assert Aw_err < 5e-6, f"inverse-factor Aw+ error {Aw_err:.2e}"


def test_inverse_factor_rejects_column_major_inverse():
    """Raw torch.linalg.inv output must not enter row-major Triton indexing."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from doublesided_cholesky import incremental_update_triton_inverse_factor

    _, L, _, Aw, z, w = random_problem(64, dtype=torch.float32)
    L, Aw, z, w = [tensor.cuda() for tensor in (L, Aw, z, w)]
    inverse_column_major = torch.linalg.inv(L)
    assert not inverse_column_major.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        incremental_update_triton_inverse_factor(
            inverse_column_major,
            Aw,
            z,
            w,
        )


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
    # The parallel prefix formulation reassociates FP32 sums relative to the
    # serial fused kernel; both independently match the exact solve.
    assert vz_err < 1e-5, f"vz disagreement {vz_err:.2e}"


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
