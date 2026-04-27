"""
Double-Sided Cholesky Propagation: MLX/Metal port.

This is a parallel of doublesided_cholesky.py for Apple Silicon. Same problem
(rank-1 update of L, A_w = L^{-1} M L^{-T} with G+ = G + zz^T, M+ = M + zw^T)
but the GPU kernel is a Metal Shading Language kernel dispatched through
mx.fast.metal_kernel instead of Triton.

Differences from the Triton version:

  1. MLX is functional. We do NOT mutate L in place — the kernel returns a
     fresh L_new tensor. Callers should expect (L_new, Aw_new, vz) instead
     of mutating L and getting (Aw_new, vz).

  2. The fused single + batched kernel is one MSL kernel, dispatched with
     grid=(BLOCK*T, 1, 1) / threadgroup=(BLOCK, 1, 1). T=1 is the single-
     update path; T>1 is the batched path.

  3. Same multi-warp memory race that bit the Triton kernel applies here —
     three threadgroup barriers between phases are mandatory for correctness.

  4. The tiled multi-block path from the Triton file is intentionally not
     ported. The fused kernel covers the common case; add the tiled path
     only if profiling on M-series shows the fused path is bandwidth-bound.

Requires Apple Silicon + macOS + mlx >= ~0.12. Importing this module on
non-Apple platforms is fine (mx is loaded lazily) but the kernel functions
will raise.
"""

from functools import lru_cache

import numpy as np

try:
    import mlx.core as mx
    _HAS_MLX = True
except ImportError:
    mx = None
    _HAS_MLX = False


# Optional: fast_cholesky is a torch C++ extension built from
# fast_update.cpp via setup_ext.py. When available it's the AVX-512 path
# and is by far the fastest reference (~10-30x faster than the manual
# numpy fallback). MLX itself does NOT need torch — the kernel works
# without it; we only import torch when fast_cholesky is also there.
try:
    import torch
    import fast_cholesky
    _HAS_FAST_CPP = True
except ImportError:
    torch = None
    fast_cholesky = None
    _HAS_FAST_CPP = False


# ---------------------------------------------------------------------------
# Reference / CPU baseline.
#
# Primary path: fast_cholesky.fast_incremental_update (built from
# fast_update.cpp). AVX-512, ~10-30x faster than the manual fallback below.
#
# Fallback path: a pure-numpy transcription of the same algorithm. We
# benchmarked both numpy and torch (via torch.linalg.solve_triangular +
# the pure-Python rotation loops) and numpy is 8-100x faster than torch
# for these shapes — the torch path's per-op overhead and .item() calls
# dominate at our problem sizes. So the fallback is numpy, not torch.
#
# This module is intentionally self-contained: it does NOT import from
# doublesided_cholesky. The same algorithmic helpers exist there in torch
# form, but reimplementing them here in numpy is faster AND keeps the
# MLX path independent of the Triton path.
# ---------------------------------------------------------------------------


def _to_numpy_f32(x):
    """mx.array / torch.Tensor / np.ndarray → contiguous float32 numpy."""
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32))


def _wrap_outputs(L_np, Aw_np, vz_np):
    """Convert numpy outputs to mx.array if MLX is available."""
    if _HAS_MLX:
        return mx.array(L_np), mx.array(Aw_np), mx.array(vz_np)
    return L_np, Aw_np, vz_np


# ---- Primary path: fast_cholesky (C++ AVX-512) ----------------------------


def _reference_fast_cpp(L, Aw, z, w):
    L_np  = _to_numpy_f32(L).copy()
    Aw_np = _to_numpy_f32(Aw)
    z_np  = _to_numpy_f32(z).copy()
    w_np  = _to_numpy_f32(w)
    L_t  = torch.from_numpy(L_np)
    Aw_t = torch.from_numpy(Aw_np)
    z_t  = torch.from_numpy(z_np)
    w_t  = torch.from_numpy(w_np)
    Aw_new_t, vz_t = fast_cholesky.fast_incremental_update(L_t, Aw_t, z_t, w_t)
    # L_t (and hence L_np) is modified in place to L+.
    return L_np, Aw_new_t.numpy(), vz_t.numpy()


def _reference_batched_fast_cpp(L, Aw, z, w):
    """Batched reference via single-update C++ entry, looped.

    Two reasons we don't call fast_incremental_update_batched directly:

      1. Correctness: the batched entry point clones each L[t] before
         calling update_impl, so L+ is written to a thread-local clone
         and discarded. We need L+ in the output.

      2. Speed: benchmarks on this CPU show fast_incremental_update_batched
         is actually 1.5-12x SLOWER than looping fast_incremental_update,
         because at::parallel_for's thread-pool overhead exceeds the gains
         at our T values. The loop is faster AND correct."""
    L_np  = _to_numpy_f32(L).copy()
    Aw_np = _to_numpy_f32(Aw)
    z_np  = _to_numpy_f32(z).copy()
    w_np  = _to_numpy_f32(w)
    T = L_np.shape[0]
    Aw_outs = np.empty_like(Aw_np)
    vz_outs = np.empty((T, L_np.shape[1]), dtype=L_np.dtype)
    for t in range(T):
        L_t  = torch.from_numpy(L_np[t])
        Aw_t = torch.from_numpy(Aw_np[t])
        z_t  = torch.from_numpy(z_np[t])
        w_t  = torch.from_numpy(w_np[t])
        Aw_new_t, vz_t = fast_cholesky.fast_incremental_update(L_t, Aw_t, z_t, w_t)
        Aw_outs[t] = Aw_new_t.numpy()
        vz_outs[t] = vz_t.numpy()
    return L_np, Aw_outs, vz_outs


# ---- Fallback path: pure numpy (self-contained) ---------------------------


def _givens_params(a: float, b: float):
    """Givens (c, s) such that c*a + s*b = rho, -s*a + c*b = 0."""
    rho = float(np.sqrt(a * a + b * b))
    if rho == 0.0:
        return 1.0, 0.0
    return a / rho, b / rho


def _cholesky_rank1_update(L: np.ndarray, z: np.ndarray):
    """In-place rank-1 Cholesky update via Givens rotations on [L | z].
    Returns (cs, ss) for use by Phases 2-3."""
    r = L.shape[0]
    cs = np.zeros(r, dtype=L.dtype)
    ss = np.zeros(r, dtype=L.dtype)
    for k in range(r):
        c, s = _givens_params(float(L[k, k]), float(z[k]))
        cs[k], ss[k] = c, s
        col_k = L[:, k].copy()
        col_z = z.copy()
        L[:, k] = c * col_k + s * col_z
        z[:] = -s * col_k + c * col_z
    return cs, ss


def _doublesided_propagate(Aw: np.ndarray, cs: np.ndarray, ss: np.ndarray):
    """Apply Q^T [Aw 0; 0 0] Q, return the top-left r x r block."""
    r = Aw.shape[0]
    Ah = np.zeros((r + 1, r + 1), dtype=Aw.dtype)
    Ah[:r, :r] = Aw
    for k in range(r):
        c, s = cs[k], ss[k]
        rk, rr = Ah[k].copy(), Ah[r].copy()
        Ah[k] = c * rk + s * rr
        Ah[r] = -s * rk + c * rr
    for k in range(r):
        c, s = cs[k], ss[k]
        ck, cr = Ah[:, k].copy(), Ah[:, r].copy()
        Ah[:, k] = c * ck + s * cr
        Ah[:, r] = -s * ck + c * cr
    return Ah[:r, :r].copy()


def _recover_vz(cs: np.ndarray, ss: np.ndarray):
    """Top r entries of Q^T e_{r+1} = (L+)^{-1} z."""
    r = cs.shape[0]
    vz = np.zeros(r, dtype=cs.dtype)
    e_last = 1.0
    for k in range(r):
        vz[k] = ss[k] * e_last
        e_last = cs[k] * e_last
    return vz


def _solve_lower(L: np.ndarray, b: np.ndarray):
    """Forward substitution L y = b for lower-triangular L."""
    r = L.shape[0]
    y = np.zeros(r, dtype=L.dtype)
    for i in range(r):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y


def _reference_numpy(L, Aw, z, w):
    L_np  = _to_numpy_f32(L).copy()
    Aw_np = _to_numpy_f32(Aw)
    z_np  = _to_numpy_f32(z).copy()
    w_np  = _to_numpy_f32(w)
    cs, ss = _cholesky_rank1_update(L_np, z_np)
    Aw_core = _doublesided_propagate(Aw_np, cs, ss)
    vz = _recover_vz(cs, ss)
    vw = _solve_lower(L_np, w_np)
    return L_np, Aw_core + np.outer(vz, vw), vz


def _reference_batched_numpy(L, Aw, z, w):
    L_np  = _to_numpy_f32(L)
    Aw_np = _to_numpy_f32(Aw)
    z_np  = _to_numpy_f32(z)
    w_np  = _to_numpy_f32(w)
    T = L_np.shape[0]
    L_outs = np.empty_like(L_np)
    Aw_outs = np.empty_like(Aw_np)
    vz_outs = np.empty((T, L_np.shape[1]), dtype=L_np.dtype)
    for t in range(T):
        L_outs[t], Aw_outs[t], vz_outs[t] = _reference_numpy(
            L_np[t], Aw_np[t], z_np[t], w_np[t]
        )
    return L_outs, Aw_outs, vz_outs


# ---- Dispatchers ----------------------------------------------------------


def incremental_update_reference(L, Aw, z, w):
    """Single-update reference. Prefers fast_cholesky.fast_incremental_update
    (C++/AVX-512) when the .so is built; falls back to a self-contained
    pure-numpy implementation otherwise. Returns mx.array if MLX is
    available, else numpy."""
    if _HAS_FAST_CPP:
        L_np, Aw_np, vz_np = _reference_fast_cpp(L, Aw, z, w)
    else:
        L_np, Aw_np, vz_np = _reference_numpy(L, Aw, z, w)
    return _wrap_outputs(L_np, Aw_np, vz_np)


def incremental_update_reference_batched(L, Aw, z, w):
    """Batched reference. Prefers a loop over fast_incremental_update
    (faster than fast_incremental_update_batched on benchmarks); falls
    back to a sequential numpy loop."""
    if _HAS_FAST_CPP:
        L_np, Aw_np, vz_np = _reference_batched_fast_cpp(L, Aw, z, w)
    else:
        L_np, Aw_np, vz_np = _reference_batched_numpy(L, Aw, z, w)
    return _wrap_outputs(L_np, Aw_np, vz_np)


# ---------------------------------------------------------------------------
# MSL kernel source. The header defines helpers + R/BLOCK constants. The
# source is the body that MLX wraps inside `kernel void ... { ... }`.
# ---------------------------------------------------------------------------


_GIVENS_HEADER = r"""
inline void givens_params_msl(thread float& c, thread float& s, float a, float b) {
    float rho = sqrt(a * a + b * b);
    bool is_zero = (rho == 0.0f);
    float safe_rho = rho + (is_zero ? 1.0f : 0.0f);
    c = is_zero ? 1.0f : (a / safe_rho);
    s = is_zero ? 0.0f : (b / safe_rho);
}
"""


# Body of `kernel void <name>(...)`. MLX auto-generates the function signature
# with the buffers from input_names/output_names plus standard built-ins:
#   thread_position_in_grid, thread_position_in_threadgroup,
#   threadgroup_position_in_grid, threads_per_threadgroup.
#
# Threadgroup memory budget (per-threadgroup):
#   cs, ss, z_work, vw, reduce_buf : 5 * 1024 * 4 bytes = 20 KB
# Within Apple Silicon's 32 KB per-threadgroup limit.
_KERNEL_SOURCE = r"""
    uint lane = thread_position_in_threadgroup.x;
    uint tg   = threadgroup_position_in_grid.x;
    uint off_mat = tg * R * R;
    uint off_vec = tg * R;

    bool active = lane < R;

    threadgroup float cs[1024];
    threadgroup float ss[1024];
    threadgroup float z_work[1024];
    threadgroup float vw[1024];
    threadgroup float reduce_buf[1024];

    // ---- Initialize working copies of L and z. MLX is functional, so the
    // kernel allocates a fresh L_new and we copy the input in. The copy is
    // O(r^2) extra traffic but happens once per kernel call.
    if (active) {
        for (uint j = 0; j < R; j++) {
            L_new[off_mat + lane * R + j] = L[off_mat + lane * R + j];
        }
        z_work[lane] = z[off_vec + lane];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    // ---- Phase 1: Rank-1 Cholesky update via Givens rotations on [L | z].
    for (uint k = 0; k < R; k++) {
        // Lkk and zk are scalars; broadcast via threadgroup memory / device read.
        float Lkk = L_new[off_mat + k * R + k];
        float zk  = z_work[k];

        float c, s;
        givens_params_msl(c, s, Lkk, zk);

        if (lane == 0) {
            cs[k] = c;
            ss[k] = s;
        }

        // Update column k for rows >= k. Rows < k stay zero (lower-triangular).
        if (lane >= k && lane < R) {
            float L_col = L_new[off_mat + lane * R + k];
            float z_val = z_work[lane];
            L_new[off_mat + lane * R + k] = c * L_col + s * z_val;
            z_work[lane] = -s * L_col + c * z_val;
        }
        // Cross-warp barrier: cs/ss/L_new/z_work writes from this iter must
        // be visible to next iter's reads.
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }

    // ---- Phase 2 left sweep: Q_k^T mixes row k with the virtual padded row.
    // pad_row stays in registers (one element per lane). Aw is read-only;
    // Aw_new gets the post-left-sweep matrix.
    float pad_row = 0.0f;
    for (uint k = 0; k < R; k++) {
        float c = cs[k];
        float s = ss[k];
        float rk = active ? Aw[off_mat + k * R + lane] : 0.0f;
        float rr = pad_row;
        if (active) {
            Aw_new[off_mat + k * R + lane] = c * rk + s * rr;
        }
        pad_row = -s * rk + c * rr;
    }
    // Left-sweep wrote rows; right-sweep reads columns. Same multi-warp race
    // we hit in the Triton port — the barrier is mandatory.
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    // ---- Phase 2 right sweep: Q_k mixes column k with the virtual padded column.
    // In-place on Aw_new. Each lane owns one column.
    float pad_col = 0.0f;
    for (uint k = 0; k < R; k++) {
        float c = cs[k];
        float s = ss[k];
        float ck = active ? Aw_new[off_mat + lane * R + k] : 0.0f;
        float cr = pad_col;
        if (active) {
            Aw_new[off_mat + lane * R + k] = c * ck + s * cr;
        }
        pad_col = -s * ck + c * cr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    // ---- Phase 3: recover v_z from the rotation chain.
    if (lane == 0) {
        float e_last = 1.0f;
        for (uint k = 0; k < R; k++) {
            vz[off_vec + k] = ss[k] * e_last;
            e_last = cs[k] * e_last;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    // ---- Phase 4: forward substitution L+ * v_w = w. vw lives in
    // threadgroup memory because the rank-1 correction needs a vector view.
    if (active) {
        vw[lane] = w[off_vec + lane];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0; k < R; k++) {
        float L_row = active ? L_new[off_mat + k * R + lane] : 0.0f;
        float vw_i  = active ? vw[lane] : 0.0f;
        float contrib = (lane < k) ? (L_row * vw_i) : 0.0f;

        // Threadgroup-wide tree reduction (no simdgroup intrinsics so this
        // works without simd_lane_id / simd_group_id which MLX does not auto-
        // expose to the source body).
        reduce_buf[lane] = contrib;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = BLOCK / 2; stride > 0; stride /= 2) {
            if (lane < stride) {
                reduce_buf[lane] += reduce_buf[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0) {
            float acc = reduce_buf[0];
            float wk  = vw[k];
            float Lkk = L_new[off_mat + k * R + k];
            vw[k] = (wk - acc) / Lkk;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Rank-1 correction: Aw_new += v_z v_w^T. Each lane owns one column.
    // No mem_device barrier needed before this — Phase 4 only touched
    // threadgroup memory; the right-sweep barrier above already published
    // Aw_new writes.
    if (active) {
        float vw_j = vw[lane];
        for (uint i_row = 0; i_row < R; i_row++) {
            float vzi = vz[off_vec + i_row];
            Aw_new[off_mat + i_row * R + lane] += vzi * vw_j;
        }
    }
"""


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


@lru_cache(maxsize=None)
def _make_kernel(r: int):
    """Compile (or cache) the Metal kernel specialized for matrix size r."""
    if not _HAS_MLX:
        raise RuntimeError(
            "MLX is not available. This module needs Apple Silicon + macOS + mlx."
        )

    BLOCK = _next_power_of_2(r)
    if BLOCK > 1024:
        raise ValueError(f"r={r} too large; BLOCK={BLOCK} exceeds Metal threadgroup max")

    header = f"#define R {r}\n#define BLOCK {BLOCK}\n" + _GIVENS_HEADER
    return mx.fast.metal_kernel(
        name=f"fused_doublesided_kernel_r{r}",
        input_names=["L", "Aw", "z", "w"],
        output_names=["L_new", "Aw_new", "vz"],
        header=header,
        source=_KERNEL_SOURCE,
        ensure_row_contiguous=True,
    )


def incremental_update_mlx(L, Aw, z, w):
    """Single-update fused MLX kernel.

    Args:
        L:  (r, r) lower-triangular Cholesky factor.
        Aw: (r, r) whitened operator.
        z:  (r,) update vector.
        w:  (r,) other-side update vector.

    Returns:
        L_new:  (r, r) updated Cholesky factor (functional, not in place).
        Aw_new: (r, r) updated whitened operator.
        vz:     (r,) the vector (L+)^{-1} z.

    Constraint: r <= 1024 so r fits in a single Metal threadgroup.
    """
    if not _HAS_MLX:
        raise RuntimeError("MLX is not available.")

    L_b  = mx.expand_dims(L,  axis=0)
    Aw_b = mx.expand_dims(Aw, axis=0)
    z_b  = mx.expand_dims(z,  axis=0)
    w_b  = mx.expand_dims(w,  axis=0)

    L_new_b, Aw_new_b, vz_b = incremental_update_mlx_batched(L_b, Aw_b, z_b, w_b)
    return L_new_b[0], Aw_new_b[0], vz_b[0]


def incremental_update_mlx_batched(L, Aw, z, w):
    """Batched fused MLX kernel. One threadgroup per update; all T updates
    run in parallel on the GPU.

    Args:
        L:  (T, r, r) batch of lower-triangular factors.
        Aw: (T, r, r) batch of whitened operators.
        z:  (T, r) batch of update vectors.
        w:  (T, r) batch of other-side update vectors.

    Returns:
        L_new:  (T, r, r) updated factors.
        Aw_new: (T, r, r) updated whitened operators.
        vz:     (T, r) recovered vectors.
    """
    if not _HAS_MLX:
        raise RuntimeError("MLX is not available.")

    T, r, _ = L.shape
    BLOCK = _next_power_of_2(r)
    kernel = _make_kernel(r)

    L_new, Aw_new, vz = kernel(
        inputs=[L, Aw, z, w],
        output_shapes=[L.shape, Aw.shape, (T, r)],
        output_dtypes=[L.dtype, Aw.dtype, L.dtype],
        grid=(BLOCK * T, 1, 1),
        threadgroup=(BLOCK, 1, 1),
    )
    return L_new, Aw_new, vz
