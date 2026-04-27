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


@pytest.mark.parametrize("r", [8, 16, 64, 128])
def test_reference_matches_ground_truth(r):
    """Numpy reference should match full re-factorization."""
    G, L, M, Aw, z, w = random_problem(r)
    L_exact, Aw_exact = ground_truth(G, M, z, w)

    L_test, Aw_test, _ = incremental_update_reference(L, Aw, z, w)

    assert _rel_err(L_test, L_exact) < 1e-4
    assert _rel_err(Aw_test, Aw_exact) < 1e-4


# ---------------------------------------------------------------------------
# MLX/Metal kernel correctness (single-update).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("r", [16, 64, 256])
def test_mlx_matches_ground_truth(r):
    """Single-update MLX kernel matches full re-factorization."""
    G, L, M, Aw, z, w = random_problem(r)
    L_exact, Aw_exact = ground_truth(G, M, z, w)

    L_new, Aw_new, _ = incremental_update_mlx(L, Aw, z, w)
    mx.eval(L_new, Aw_new)

    L_err = _rel_err(L_new, L_exact)
    Aw_err = _rel_err(Aw_new, Aw_exact)

    assert L_err < 1e-4, f"L+ err {L_err:.2e}"
    assert Aw_err < 1e-4, f"Aw+ err {Aw_err:.2e}"


@pytest.mark.parametrize("r", [16, 64, 256])
def test_mlx_matches_reference(r):
    """MLX kernel agrees with the numpy reference at fp32 precision."""
    _, L, _, Aw, z, w = random_problem(r)
    L_ref, Aw_ref, vz_ref = incremental_update_reference(L, Aw, z, w)

    L_new, Aw_new, vz_new = incremental_update_mlx(L, Aw, z, w)
    mx.eval(L_new, Aw_new, vz_new)

    assert _rel_err(L_new,  L_ref)  < 1e-5, f"L+ disagreement"
    assert _rel_err(Aw_new, Aw_ref) < 1e-5, f"Aw+ disagreement"
    assert _rel_err(vz_new, vz_ref) < 1e-5, f"vz disagreement"


# ---------------------------------------------------------------------------
# Determinism (regression test for the multi-warp race we fixed in Triton).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("r", [16, 64, 256])
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


def test_mlx_batched_matches_single():
    """Batched MLX path should match T independent single calls."""
    r = 32
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


@pytest.mark.parametrize("r", [16, 64, 128])
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

    assert _rel_err(L_mlx,  L_ref)  < 1e-5
    assert _rel_err(Aw_mlx, Aw_ref) < 1e-5
    assert _rel_err(vz_mlx, vz_ref) < 1e-5
