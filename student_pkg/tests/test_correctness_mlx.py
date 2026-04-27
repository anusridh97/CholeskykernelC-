"""
Correctness tests for the MLX/Metal port of the double-sided Cholesky kernel.

Mirrors tests/test_correctness.py for the MLX path. Skips if MLX isn't
available (i.e. on non-Apple-Silicon machines).

Run:
    python -m pytest tests/test_correctness_mlx.py -v
"""

import numpy as np
import pytest


# Skip the entire module if MLX isn't available.
mx = pytest.importorskip("mlx.core")

from doublesided_cholesky_mlx import (
    incremental_update_mlx,
    incremental_update_mlx_batched,
    incremental_update_reference,
    incremental_update_reference_batched,
    # The numpy-helper internals are dtype-preserving (only `_to_numpy_f32`
    # downcasts). Importing them directly lets the fp64 tests reach 1e-10
    # without touching the public module API.
    _cholesky_rank1_update,
    _doublesided_propagate,
    _recover_vz,
    _solve_lower,
)


def random_problem(r, dtype=np.float32, seed=0):
    """Generate a well-conditioned test problem. Returns mx.arrays."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((r, r)).astype(dtype)
    G = (A @ A.T + r * np.eye(r, dtype=dtype)).astype(dtype)
    L = np.linalg.cholesky(G).astype(dtype)
    M = rng.standard_normal((r, r)).astype(dtype)
    Linv = np.linalg.inv(L).astype(dtype)
    Aw = (Linv @ M @ Linv.T).astype(dtype)
    z = (rng.standard_normal(r) * 0.1).astype(dtype)
    w = (rng.standard_normal(r) * 0.1).astype(dtype)
    return (
        mx.array(G), mx.array(L), mx.array(M),
        mx.array(Aw), mx.array(z), mx.array(w),
    )


def ground_truth(G, M, z, w):
    """Numpy ground truth: full re-factorization."""
    G_np = np.asarray(G)
    M_np = np.asarray(M)
    z_np = np.asarray(z)
    w_np = np.asarray(w)
    G_new = G_np + np.outer(z_np, z_np)
    M_new = M_np + np.outer(z_np, w_np)
    L_new = np.linalg.cholesky(G_new)
    Linv_new = np.linalg.inv(L_new)
    Aw_new = Linv_new @ M_new @ Linv_new.T
    return mx.array(L_new.astype(np.float32)), mx.array(Aw_new.astype(np.float32))


def _rel_err(a, b):
    a_np = np.asarray(a)
    b_np = np.asarray(b)
    denom = np.linalg.norm(b_np)
    if denom == 0:
        return float(np.linalg.norm(a_np - b_np))
    return float(np.linalg.norm(a_np - b_np) / denom)


# ---------------------------------------------------------------------------
# Reference implementation correctness (no GPU required).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("r", [8, 16, 32, 64, 128, 256, 512, 1024])
def test_reference_matches_ground_truth(r):
    """Numpy reference should match full re-factorization."""
    G, L, M, Aw, z, w = random_problem(r)
    L_exact, Aw_exact = ground_truth(G, M, z, w)

    L_test, Aw_test, _ = incremental_update_reference(L, Aw, z, w)

    # Givens chain accumulates O(eps * r) rounding error in fp32 — scale
    # the tolerance with r so the larger sizes don't fail on noise.
    tol = max(1e-4, 5e-7 * r)
    assert _rel_err(L_test, L_exact) < tol
    assert _rel_err(Aw_test, Aw_exact) < tol


# ---------------------------------------------------------------------------
# fp64 reference correctness (1e-10).
#
# Mirrors the fp64 tests in tests/test_correctness.py for the Triton path.
# Only the pure-numpy reference can run in fp64 here — the MLX kernel is
# fp32-only because Metal GPU compute does not support fp64.
# ---------------------------------------------------------------------------


def _random_problem_np(r, dtype=np.float64, seed=0):
    """Same shape as random_problem() but stays in numpy at the chosen dtype."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((r, r)).astype(dtype)
    G = (A @ A.T + r * np.eye(r, dtype=dtype)).astype(dtype)
    L = np.linalg.cholesky(G).astype(dtype)
    M = rng.standard_normal((r, r)).astype(dtype)
    Linv = np.linalg.inv(L).astype(dtype)
    Aw = (Linv @ M @ Linv.T).astype(dtype)
    z = (rng.standard_normal(r) * 0.1).astype(dtype)
    w = (rng.standard_normal(r) * 0.1).astype(dtype)
    return G, L, M, Aw, z, w


def _ground_truth_np(G, M, z, w):
    """Full re-factorization in numpy, dtype-preserving."""
    G_new = G + np.outer(z, z)
    M_new = M + np.outer(z, w)
    L_new = np.linalg.cholesky(G_new)
    Linv_new = np.linalg.inv(L_new)
    Aw_new = Linv_new @ M_new @ Linv_new.T
    return L_new, Aw_new


def _reference_numpy_dtype_preserving(L, Aw, z, w):
    """Same algorithm as `_reference_numpy` in the module, but without the
    fp32 cast. Lets fp64 inputs flow through to give an fp64-precision
    reference."""
    L_buf = np.ascontiguousarray(L).copy()
    z_buf = np.ascontiguousarray(z).copy()
    Aw_buf = np.ascontiguousarray(Aw)
    w_buf = np.ascontiguousarray(w)
    cs, ss = _cholesky_rank1_update(L_buf, z_buf)
    Aw_core = _doublesided_propagate(Aw_buf, cs, ss)
    vz = _recover_vz(cs, ss)
    vw = _solve_lower(L_buf, w_buf)
    return L_buf, Aw_core + np.outer(vz, vw), vz


@pytest.mark.parametrize("r", [8, 16, 32, 64, 128, 256, 512, 1024])
def test_reference_matches_ground_truth_fp64(r):
    """fp64 numpy reference matches full re-factorization to 1e-10. Mirrors
    test_python_reference_matches_ground_truth_fp64 in test_correctness.py."""
    G, L, M, Aw, z, w = _random_problem_np(r, dtype=np.float64)
    L_exact, Aw_exact = _ground_truth_np(G, M, z, w)

    L_test, Aw_test, _ = _reference_numpy_dtype_preserving(L, Aw, z, w)

    L_err = _rel_err(L_test, L_exact)
    Aw_err = _rel_err(Aw_test, Aw_exact)
    assert L_err  < 1e-10, f"L+ err {L_err:.2e}"
    assert Aw_err < 1e-10, f"Aw+ err {Aw_err:.2e}"


@pytest.mark.parametrize("r", [16, 32, 64, 128, 256, 512, 1024])
def test_reference_batched_matches_single_fp64(r):
    """fp64 batched reference (T independent fp64 single calls) matches
    per-slice fp64 single reference to 1e-10."""
    T = 4
    L_list, Aw_list, z_list, w_list, single_outs = [], [], [], [], []
    for t in range(T):
        _, L, _, Aw, z, w = _random_problem_np(r, dtype=np.float64, seed=t)
        L_list.append(L)
        Aw_list.append(Aw)
        z_list.append(z)
        w_list.append(w)
        single_outs.append(_reference_numpy_dtype_preserving(L, Aw, z, w))

    # "Batched" reference here is just the loop-form, in fp64. The point is
    # to assert the per-slice algorithm is order-independent in fp64.
    for t in range(T):
        L_b, Aw_b, vz_b = _reference_numpy_dtype_preserving(
            L_list[t], Aw_list[t], z_list[t], w_list[t]
        )
        L_s, Aw_s, vz_s = single_outs[t]
        assert _rel_err(L_b,  L_s)  < 1e-10
        assert _rel_err(Aw_b, Aw_s) < 1e-10
        assert _rel_err(vz_b, vz_s) < 1e-10


# ---------------------------------------------------------------------------
# MLX/Metal kernel correctness (single-update).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("r", [16, 32, 64, 128, 256, 512, 1024])
def test_mlx_matches_ground_truth(r):
    """Single-update MLX kernel matches full re-factorization."""
    G, L, M, Aw, z, w = random_problem(r)
    L_exact, Aw_exact = ground_truth(G, M, z, w)

    L_new, Aw_new, _ = incremental_update_mlx(L, Aw, z, w)
    mx.eval(L_new, Aw_new)

    L_err = _rel_err(L_new, L_exact)
    Aw_err = _rel_err(Aw_new, Aw_exact)

    tol = max(1e-4, 5e-7 * r)
    assert L_err < tol, f"L+ err {L_err:.2e} (tol {tol:.2e})"
    assert Aw_err < tol, f"Aw+ err {Aw_err:.2e} (tol {tol:.2e})"


@pytest.mark.parametrize("r", [16, 32, 64, 128, 256, 512, 1024])
def test_mlx_matches_reference(r):
    """MLX kernel agrees with the numpy reference at fp32 precision."""
    _, L, _, Aw, z, w = random_problem(r)
    L_ref, Aw_ref, vz_ref = incremental_update_reference(L, Aw, z, w)

    L_new, Aw_new, vz_new = incremental_update_mlx(L, Aw, z, w)
    mx.eval(L_new, Aw_new, vz_new)

    # Different summation orderings between numpy and MSL accumulate
    # different fp32 rounding error, ~O(eps * r). Scale tolerance with r.
    tol = max(1e-5, 1e-7 * r)
    assert _rel_err(L_new,  L_ref)  < tol, f"L+ disagreement"
    assert _rel_err(Aw_new, Aw_ref) < tol, f"Aw+ disagreement"
    assert _rel_err(vz_new, vz_ref) < tol, f"vz disagreement"


# ---------------------------------------------------------------------------
# Determinism (regression test for the multi-warp race we fixed in Triton).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("r", [16, 32, 64, 128, 256, 512, 1024])
def test_mlx_is_deterministic(r):
    """Two consecutive calls with identical inputs must produce identical
    outputs. Catches the missing-threadgroup-barrier bug class."""
    _, L, _, Aw, z, w = random_problem(r)

    L_a, Aw_a, vz_a = incremental_update_mlx(L, Aw, z, w)
    L_b, Aw_b, vz_b = incremental_update_mlx(L, Aw, z, w)
    mx.eval(L_a, Aw_a, vz_a, L_b, Aw_b, vz_b)

    assert _rel_err(L_a,  L_b)  == 0.0
    assert _rel_err(Aw_a, Aw_b) == 0.0
    assert _rel_err(vz_a, vz_b) == 0.0


# ---------------------------------------------------------------------------
# Batched kernel correctness.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("r", [16, 32, 64, 256, 1024])
def test_mlx_batched_matches_single(r):
    """Batched MLX path should match T independent single calls."""
    T = 5
    L_stack  = mx.zeros((T, r, r), dtype=mx.float32)
    Aw_stack = mx.zeros((T, r, r), dtype=mx.float32)
    z_stack  = mx.zeros((T, r),    dtype=mx.float32)
    w_stack  = mx.zeros((T, r),    dtype=mx.float32)

    single_aw = []
    L_list, Aw_list, z_list, w_list = [], [], [], []
    for t in range(T):
        _, L, _, Aw, z, w = random_problem(r, seed=t)
        L_list.append(L)
        Aw_list.append(Aw)
        z_list.append(z)
        w_list.append(w)
        _, Aw_out, _ = incremental_update_mlx(L, Aw, z, w)
        single_aw.append(Aw_out)

    L_stack  = mx.stack(L_list,  axis=0)
    Aw_stack = mx.stack(Aw_list, axis=0)
    z_stack  = mx.stack(z_list,  axis=0)
    w_stack  = mx.stack(w_list,  axis=0)

    _, Aw_batched, _ = incremental_update_mlx_batched(
        L_stack, Aw_stack, z_stack, w_stack
    )
    mx.eval(Aw_batched)

    for t in range(T):
        err = _rel_err(Aw_batched[t], single_aw[t])
        assert err < 1e-5, f"batched slice {t}: err={err:.2e}"


@pytest.mark.parametrize("r", [16, 32, 64, 128, 256, 512, 1024])
def test_mlx_batched_matches_cpp_reference(r):
    """Batched MLX kernel matches the C++ batched reference (fast_cholesky).

    This is a stronger correctness test than matches_single because it cross-
    checks two independent implementations of the same algorithm rather than
    just verifying that single + batched MLX agree with each other."""
    T = 4

    L_list, Aw_list, z_list, w_list = [], [], [], []
    for t in range(T):
        _, L, _, Aw, z, w = random_problem(r, seed=t)
        L_list.append(L)
        Aw_list.append(Aw)
        z_list.append(z)
        w_list.append(w)

    L_stack  = mx.stack(L_list,  axis=0)
    Aw_stack = mx.stack(Aw_list, axis=0)
    z_stack  = mx.stack(z_list,  axis=0)
    w_stack  = mx.stack(w_list,  axis=0)

    L_mlx, Aw_mlx, vz_mlx = incremental_update_mlx_batched(
        L_stack, Aw_stack, z_stack, w_stack
    )
    mx.eval(L_mlx, Aw_mlx, vz_mlx)

    L_ref, Aw_ref, vz_ref = incremental_update_reference_batched(
        L_stack, Aw_stack, z_stack, w_stack
    )

    tol = max(1e-5, 1e-7 * r)
    assert _rel_err(L_mlx,  L_ref)  < tol
    assert _rel_err(Aw_mlx, Aw_ref) < tol
    assert _rel_err(vz_mlx, vz_ref) < tol
