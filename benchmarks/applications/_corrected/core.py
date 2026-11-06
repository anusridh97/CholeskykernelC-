"""Correct double-sided Cholesky transport primitives.

The central state is

    G = L L^T,          A_w = L^{-1} M L^{-T}.

A signed metric update and an independently signed operator update are

    G+ = G + sigma_G z z^T,
    M+ = M + sigma_M z w^T,

where ``sigma_G`` is +1 (ordinary update) or -1 (admissible downdate), and
``sigma_M`` is -1, 0, or +1.  Positive metric updates use orthogonal Givens
rotations; downdates use leverage-guarded hyperbolic rotations.

The module exposes a correctness-first PyTorch reference and, when built, the
``fast_cholesky`` CPU extension.  Public functions are functional by default:
they return a new factor rather than relying on mutation of a possibly
Fortran-strided ``torch.linalg.cholesky`` result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch

try:  # Optional compiled backend.
    from . import _C as _fast_cholesky
except ImportError:  # pragma: no cover - exercised when extension is not built.
    _fast_cholesky = None

Backend = Literal["auto", "extension", "torch"]


@dataclass(frozen=True)
class UpdateResult:
    L: torch.Tensor
    Aw: torch.Tensor
    vz: torch.Tensor


def extension_available() -> bool:
    return _fast_cholesky is not None


def _validate_state(L: torch.Tensor, Aw: torch.Tensor) -> int:
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError(f"L must be square; got {tuple(L.shape)}")
    r = L.shape[0]
    if Aw.shape != (r, r):
        raise ValueError(f"Aw must have shape {(r, r)}; got {tuple(Aw.shape)}")
    if L.dtype != Aw.dtype or L.device != Aw.device:
        raise ValueError("L and Aw must have the same dtype and device")
    if L.dtype not in (torch.float32, torch.float64):
        raise TypeError("only float32 and float64 are supported")
    return r


def _validate_vector(x: torch.Tensor, r: int, L: torch.Tensor, name: str) -> None:
    if x.shape != (r,):
        raise ValueError(f"{name} must have shape {(r,)}; got {tuple(x.shape)}")
    if x.dtype != L.dtype or x.device != L.device:
        raise ValueError(f"{name} must match L's dtype and device")


def symmetrize(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.mT)


def whiten_vector(L: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return ``L^{-1} x`` without forming an inverse."""
    return torch.linalg.solve_triangular(L, x.unsqueeze(-1), upper=False).squeeze(-1)


def double_whiten(L: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Return ``L^{-1} M L^{-T}`` using two triangular solves."""
    left = torch.linalg.solve_triangular(L, M, upper=False)
    return torch.linalg.solve_triangular(L, left.mT, upper=False).mT


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(expected).clamp_min(torch.finfo(expected.dtype).tiny)
    return float(torch.linalg.vector_norm(actual - expected) / denom)


@torch.no_grad()
def incremental_update_reference(
    L: torch.Tensor,
    Aw: torch.Tensor,
    z: torch.Tensor,
    w: torch.Tensor | None = None,
    *,
    metric_sign: int = 1,
    operator_sign: int = 1,
    downdate_tol: float = 1e-8,
) -> UpdateResult:
    """Correct O(r^2) reference implementation.

    The input tensors are not mutated.  ``operator_sign=0`` performs only the
    metric/basis transport and ignores ``w``.
    """
    r = _validate_state(L, Aw)
    _validate_vector(z, r, L, "z")
    if metric_sign not in (-1, 1):
        raise ValueError("metric_sign must be +1 or -1")
    if operator_sign not in (-1, 0, 1):
        raise ValueError("operator_sign must be -1, 0, or +1")
    if w is None:
        w = torch.zeros_like(z)
    _validate_vector(w, r, L, "w")

    L_new = L.clone()
    z_work = z.clone()
    cs = torch.empty(r, dtype=L.dtype, device=L.device)
    ss = torch.empty_like(cs)

    if metric_sign < 0:
        leverage = torch.dot(whiten_vector(L_new, z), whiten_vector(L_new, z))
        if float(leverage) >= 1.0 - downdate_tol:
            raise ValueError(
                "inadmissible or poorly conditioned downdate: "
                f"z^T G^(-1) z={float(leverage):.8g}"
            )

    for k in range(r):
        a = L_new[k, k].clone()
        b = z_work[k].clone()

        if metric_sign > 0:
            rho = torch.hypot(a, b)
            c = a / rho
            s = b / rho
        else:
            radicand = (a - b) * (a + b)
            if float(radicand) <= 0.0:
                raise ValueError(f"downdate left the SPD cone at rotation {k}")
            rho = torch.sqrt(radicand)
            c = a / rho
            s = -b / rho

        cs[k] = c
        ss[k] = s

        col = L_new[k:, k].clone()
        tail = z_work[k:].clone()
        L_new[k:, k] = c * col + s * tail
        if metric_sign > 0:
            z_work[k:] = -s * col + c * tail
        else:
            z_work[k:] = s * col + c * tail

    Ah = torch.zeros((r + 1, r + 1), dtype=Aw.dtype, device=Aw.device)
    Ah[:r, :r] = Aw

    for k in range(r):
        c, s = cs[k], ss[k]
        row_k = Ah[k].clone()
        row_pad = Ah[r].clone()
        if metric_sign > 0:
            Ah[k] = c * row_k + s * row_pad
        else:
            Ah[k] = c * row_k - s * row_pad
        Ah[r] = -s * row_k + c * row_pad

    for k in range(r):
        c, s = cs[k], ss[k]
        col_k = Ah[:, k].clone()
        col_pad = Ah[:, r].clone()
        if metric_sign > 0:
            Ah[:, k] = c * col_k + s * col_pad
        else:
            Ah[:, k] = c * col_k - s * col_pad
        Ah[:, r] = -s * col_k + c * col_pad

    vz = torch.empty(r, dtype=L.dtype, device=L.device)
    e_last = L.new_tensor(1.0)
    for k in range(r):
        vz[k] = (ss[k] if metric_sign > 0 else -ss[k]) * e_last
        e_last = cs[k] * e_last

    Aw_new = Ah[:r, :r].clone()
    if operator_sign:
        vw = whiten_vector(L_new, w)
        Aw_new = Aw_new + operator_sign * torch.outer(vz, vw)

    return UpdateResult(L=L_new, Aw=Aw_new, vz=vz)


@torch.no_grad()
def incremental_update(
    L: torch.Tensor,
    Aw: torch.Tensor,
    z: torch.Tensor,
    w: torch.Tensor | None = None,
    *,
    metric_sign: int = 1,
    operator_sign: int = 1,
    downdate_tol: float = 1e-8,
    backend: Backend = "auto",
    inplace: bool = False,
) -> UpdateResult:
    """Run the signed double-sided update with the selected backend."""
    r = _validate_state(L, Aw)
    _validate_vector(z, r, L, "z")
    if w is None:
        w = torch.zeros_like(z)
    _validate_vector(w, r, L, "w")

    use_extension = backend == "extension" or (
        backend == "auto" and _fast_cholesky is not None and L.device.type == "cpu"
    )
    if backend == "extension" and (_fast_cholesky is None or L.device.type != "cpu"):
        raise RuntimeError("the fast_cholesky extension is unavailable for this tensor")

    if not use_extension:
        result = incremental_update_reference(
            L,
            Aw,
            z,
            w,
            metric_sign=metric_sign,
            operator_sign=operator_sign,
            downdate_tol=downdate_tol,
        )
        if inplace:
            L.copy_(result.L)
            return UpdateResult(L=L, Aw=result.Aw, vz=result.vz)
        return result

    if inplace:
        Aw_new, vz = _fast_cholesky.fast_incremental_update(
            L,
            Aw,
            z,
            w,
            metric_sign,
            operator_sign,
            downdate_tol,
        )
        return UpdateResult(L=L, Aw=Aw_new, vz=vz)

    L_new, Aw_new, vz = _fast_cholesky.fast_incremental_update_functional(
        L,
        Aw,
        z,
        w,
        metric_sign,
        operator_sign,
        downdate_tol,
    )
    return UpdateResult(L=L_new, Aw=Aw_new, vz=vz)


@torch.no_grad()
def add_operator_rank1(
    L: torch.Tensor,
    Aw: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    coefficient: float = 1.0,
) -> torch.Tensor:
    """Apply an arbitrary operator update ``M += coefficient * u v^T``.

    The metric is unchanged, so the correction is simply
    ``coefficient * (L^{-1}u)(L^{-1}v)^T``.
    """
    r = _validate_state(L, Aw)
    _validate_vector(u, r, L, "u")
    _validate_vector(v, r, L, "v")
    return Aw + coefficient * torch.outer(whiten_vector(L, u), whiten_vector(L, v))


@torch.no_grad()
def add_symmetric_operator_rank1(
    L: torch.Tensor,
    Aw: torch.Tensor,
    u: torch.Tensor,
    *,
    coefficient: float = 1.0,
) -> torch.Tensor:
    vu = whiten_vector(L, u)
    return symmetrize(Aw + coefficient * torch.outer(vu, vu))


@torch.no_grad()
def cyclic_shift_reference(
    L: torch.Tensor,
    Aw: torch.Tensor,
    mw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move coordinate 0 to the end and transport all whitened state.

    For the cyclic permutation ``P``, this computes the positive-diagonal
    Cholesky factor of ``P G P^T`` in O(r^2), together with the corresponding
    ``Q^T Aw Q`` and ``Q^T mw`` transports.
    """
    r = _validate_state(L, Aw)
    _validate_vector(mw, r, L, "mw")

    B = torch.roll(L.clone(), shifts=-1, dims=0)
    cs = torch.empty(max(r - 1, 0), dtype=L.dtype, device=L.device)
    ss = torch.empty_like(cs)

    for k in range(r - 1):
        a = B[k, k].clone()
        b = B[k, k + 1].clone()
        rho = torch.hypot(a, b)
        c, s = a / rho, b / rho
        cs[k], ss[k] = c, s
        col_k = B[:, k].clone()
        col_next = B[:, k + 1].clone()
        B[:, k] = c * col_k + s * col_next
        B[:, k + 1] = -s * col_k + c * col_next

    Aw_new = Aw.clone()
    mw_new = mw.clone()

    for k in range(r - 1):
        c, s = cs[k], ss[k]
        row_k = Aw_new[k].clone()
        row_next = Aw_new[k + 1].clone()
        Aw_new[k] = c * row_k + s * row_next
        Aw_new[k + 1] = -s * row_k + c * row_next

        x, y = mw_new[k].clone(), mw_new[k + 1].clone()
        mw_new[k] = c * x + s * y
        mw_new[k + 1] = -s * x + c * y

    for k in range(r - 1):
        c, s = cs[k], ss[k]
        col_k = Aw_new[:, k].clone()
        col_next = Aw_new[:, k + 1].clone()
        Aw_new[:, k] = c * col_k + s * col_next
        Aw_new[:, k + 1] = -s * col_k + c * col_next

    signs = torch.where(torch.diagonal(B) >= 0, 1.0, -1.0).to(B)
    B = B * signs.unsqueeze(0)
    Aw_new = signs.unsqueeze(1) * Aw_new * signs.unsqueeze(0)
    mw_new = signs * mw_new
    return torch.tril(B), Aw_new, mw_new


@torch.no_grad()
def cyclic_shift_state(
    L: torch.Tensor,
    Aw: torch.Tensor,
    mw: torch.Tensor,
    *,
    backend: Backend = "auto",
    inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    use_extension = backend == "extension" or (
        backend == "auto" and _fast_cholesky is not None and L.device.type == "cpu"
    )
    if backend == "extension" and (_fast_cholesky is None or L.device.type != "cpu"):
        raise RuntimeError("the fast_cholesky extension is unavailable for this tensor")

    if not use_extension:
        L_new, Aw_new, mw_new = cyclic_shift_reference(L, Aw, mw)
        if inplace:
            L.copy_(L_new)
            return L, Aw_new, mw_new
        return L_new, Aw_new, mw_new

    if inplace:
        Aw_new, mw_new = _fast_cholesky.fast_cyclic_shift(L, Aw, mw)
        return L, Aw_new, mw_new

    return _fast_cholesky.fast_cyclic_shift_functional(L, Aw, mw)


def arrowhead_signed_terms(
    delta: torch.Tensor,
    *,
    tolerance: float = 1e-12,
) -> list[tuple[int, torch.Tensor]]:
    """Decompose a symmetric last-row/last-column change into <=2 signed terms.

    ``delta`` must be zero away from its final row and final column.  The return
    value contains pairs ``(sign, z)`` satisfying
    ``delta = sum sign * z z^T``.  Positive terms are returned before negative
    terms so the intermediate metric stays farther inside the SPD cone.
    """
    if delta.ndim != 2 or delta.shape[0] != delta.shape[1]:
        raise ValueError("delta must be square")
    r = delta.shape[0]
    if r == 1:
        value = symmetrize(delta)[0, 0]
        if abs(float(value)) <= tolerance:
            return []
        return [(1 if value > 0 else -1, torch.sqrt(torch.abs(value)).reshape(1))]

    delta = symmetrize(delta)
    scale = max(float(torch.linalg.matrix_norm(delta)), 1.0)
    interior = delta[:-1, :-1]
    if float(torch.linalg.matrix_norm(interior)) > 100 * tolerance * scale:
        raise ValueError("delta is not an arrowhead last-coordinate update")

    e = torch.zeros(r, dtype=delta.dtype, device=delta.device)
    e[-1] = 1.0
    p = torch.zeros_like(e)
    p[:-1] = delta[:-1, -1]
    alpha = delta[-1, -1]
    beta = torch.linalg.vector_norm(p)

    terms: list[tuple[int, torch.Tensor]] = []
    if float(beta) <= tolerance * scale:
        if abs(float(alpha)) > tolerance * scale:
            terms.append((1 if alpha > 0 else -1, torch.sqrt(torch.abs(alpha)) * e))
        return terms

    u = p / beta
    small = torch.stack(
        [
            torch.stack([alpha, beta]),
            torch.stack([beta, alpha.new_zeros(())]),
        ]
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(small)
    for i in range(2):
        value = eigenvalues[i]
        if abs(float(value)) <= tolerance * scale:
            continue
        direction = eigenvectors[0, i] * e + eigenvectors[1, i] * u
        terms.append((1 if value > 0 else -1, torch.sqrt(torch.abs(value)) * direction))

    terms.sort(key=lambda item: 0 if item[0] > 0 else 1)
    return terms


@torch.no_grad()
def apply_arrowhead_metric_delta(
    L: torch.Tensor,
    Aw: torch.Tensor,
    delta: torch.Tensor,
    *,
    backend: Backend = "auto",
    downdate_tol: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a signed rank-at-most-two arrowhead metric change."""
    L_new, Aw_new = L, Aw
    zero = torch.zeros(L.shape[0], dtype=L.dtype, device=L.device)
    for sign, z in arrowhead_signed_terms(delta):
        result = incremental_update(
            L_new,
            Aw_new,
            z,
            zero,
            metric_sign=sign,
            operator_sign=0,
            downdate_tol=downdate_tol,
            backend=backend,
        )
        L_new, Aw_new = result.L, result.Aw
    return L_new, Aw_new


@torch.no_grad()
def apply_arrowhead_symmetric_operator_delta(
    L: torch.Tensor,
    Aw: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """Apply a rank-at-most-two symmetric operator change at fixed metric."""
    Aw_new = Aw
    for sign, u in arrowhead_signed_terms(delta):
        Aw_new = add_symmetric_operator_rank1(L, Aw_new, u, coefficient=float(sign))
    return symmetrize(Aw_new)


def reconstruct_unwhitened(L: torch.Tensor, Aw: torch.Tensor) -> torch.Tensor:
    return L @ Aw @ L.mT


def state_residuals(
    L: torch.Tensor,
    Aw: torch.Tensor,
    G: torch.Tensor,
    M: torch.Tensor,
) -> dict[str, float]:
    return {
        "gram_relative": relative_error(L @ L.mT, G),
        "operator_relative": relative_error(reconstruct_unwhitened(L, Aw), M),
    }


__all__ = [
    "Backend",
    "UpdateResult",
    "add_operator_rank1",
    "add_symmetric_operator_rank1",
    "apply_arrowhead_metric_delta",
    "apply_arrowhead_symmetric_operator_delta",
    "arrowhead_signed_terms",
    "cyclic_shift_reference",
    "cyclic_shift_state",
    "double_whiten",
    "extension_available",
    "incremental_update",
    "incremental_update_reference",
    "reconstruct_unwhitened",
    "relative_error",
    "state_residuals",
    "symmetrize",
    "whiten_vector",
]
