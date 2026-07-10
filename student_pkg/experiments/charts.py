"""Batched whitened-operator charts, bucketed by rank.

Both experiments maintain many independent low-rank charts

    G_t = L_t L_t^T,   A_w,t = L_t^{-1} M_t L_t^{-T}

and want to refresh all of them per outer step. A chart of rank r receives a
rank-1 update

    G_t += z_t z_t^T,   M_t += z_t w_t^T

and the kernel returns the new whitened operator A_w,t+ and the vector
v_z,t = (L_t+)^{-1} z_t, updating L_t in place. Charts of the *same* rank r are
stacked into one (T, r, r) batch and refreshed in a single launch; charts of
different rank are bucketed separately (the launch is dense over BLOCK =
next_pow2(r), so mixing ranks in one bucket would waste lanes).

This module hides two things from the experiments:

  * the CUDA/Triton batched kernel when a GPU is present, and
  * a pure-PyTorch reference fallback (per-slice ``incremental_update_reference``)
    when it is not, so the loops and their correctness checks run on CPU.

The public surface is :func:`refresh_bucket` (one batched rank-1 update over a
stack of equal-rank charts) and :func:`gpu_kernel_available`.
"""

from __future__ import annotations

import torch

from doublesided_cholesky import incremental_update_reference


def gpu_kernel_available() -> bool:
    """True when the batched Triton path can actually run (CUDA + Triton)."""
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return True


def _refresh_reference(L, Aw, z, w):
    """CPU fallback: loop the O(r^2) reference over the batch.

    Mirrors the batched kernel's contract: L is updated in place to L+, and
    (Aw_new, vz) are returned with shapes (T, r, r) and (T, r). v_z is the
    forward solve (L+)^{-1} z, recovered with a triangular solve since the
    reference does not return it.
    """
    T, r = L.shape[0], L.shape[1]
    Aw_out = torch.empty_like(Aw)
    vz = torch.empty(T, r, dtype=L.dtype, device=L.device)
    for t in range(T):
        # incremental_update_reference mutates L[t] in place to L+.
        Aw_out[t] = incremental_update_reference(L[t], Aw[t], z[t].clone(), w[t].clone())
        vz[t] = torch.linalg.solve_triangular(
            L[t], z[t].unsqueeze(1), upper=False
        ).squeeze(1)
    return Aw_out, vz


def refresh_bucket(L, Aw, z, w, force_reference: bool = False):
    """Refresh a stacked bucket of equal-rank charts.

    Args:
        L:  (T, r, r) lower-triangular factors, updated in place to L+.
        Aw: (T, r, r) current whitened operators.
        z:  (T, r) left rank-1 vectors (G += z z^T).
        w:  (T, r) right rank-1 vectors (M += z w^T).
        force_reference: use the CPU reference even if a GPU kernel is present
            (used by the cross-check tests).

    Returns:
        (Aw_new, vz): (T, r, r) refreshed operators and (T, r) solves
        v_z,t = (L_t+)^{-1} z_t.
    """
    if not force_reference and L.is_cuda and gpu_kernel_available():
        from doublesided_cholesky import incremental_update_triton_batched

        return incremental_update_triton_batched(L, Aw, z, w)
    return _refresh_reference(L, Aw, z, w)
