from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "rebuttal_experiments"
sys.path.insert(0, str(HERE))

candidate = pytest.importorskip("e1_positive_candidate")


def whiten(L: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    left = torch.linalg.solve_triangular(L, M, upper=False)
    return torch.linalg.solve_triangular(
        L, left.mT, upper=False
    ).mT.contiguous()


def relative_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        (
            torch.linalg.norm(value - reference)
            / torch.linalg.norm(reference).clamp_min(1e-30)
        ).item()
    )


@pytest.mark.parametrize("rank", (8, 17, 64))
def test_same_vector_update_matches_fresh(rank: int) -> None:
    generator = torch.Generator().manual_seed(9000 + rank)
    B = torch.randn(rank, min(rank, 16), generator=generator) * 0.05
    identity = torch.eye(rank)
    G = (identity + B @ B.mT).contiguous()
    M = (G - identity).contiguous()
    L = torch.linalg.cholesky(G).contiguous()
    Aw = whiten(L, M)
    z = (torch.randn(rank, generator=generator) * 0.01).contiguous()

    G_plus = G + torch.outer(z, z)
    M_plus = M + torch.outer(z, z)
    L_fresh = torch.linalg.cholesky(G_plus).contiguous()
    Aw_fresh = whiten(L_fresh, M_plus)
    vz_fresh = torch.linalg.solve_triangular(
        L_fresh, z[:, None], upper=False
    ).squeeze(1)

    Aw_out = torch.empty_like(Aw)
    vz_out = torch.empty(rank)
    workspace = torch.empty(4 * rank)
    candidate.positive_metric_operator_same_vector_out(
        L, Aw, z, Aw_out, vz_out, workspace
    )

    assert relative_error(L, L_fresh) < 5e-6
    assert relative_error(Aw_out, Aw_fresh) < 5e-5
    assert relative_error(vz_out, vz_fresh) < 5e-5


def test_ping_pong_recurrent_updates_match_fresh() -> None:
    rank = 64
    generator = torch.Generator().manual_seed(9100)
    identity = torch.eye(rank)
    G = identity.clone()
    M = torch.zeros_like(G)
    L = identity.clone()
    Aw_buffers = (torch.zeros_like(G), torch.empty_like(G))
    active = 0
    vz = torch.empty(rank)
    workspace = torch.empty(4 * rank)

    for _ in range(20):
        z = (torch.randn(rank, generator=generator) * 0.01).contiguous()
        candidate.positive_metric_operator_same_vector_out(
            L,
            Aw_buffers[active],
            z,
            Aw_buffers[1 - active],
            vz,
            workspace,
        )
        active = 1 - active
        outer = torch.outer(z, z)
        G.add_(outer)
        M.add_(outer)

    L_fresh = torch.linalg.cholesky(G).contiguous()
    Aw_fresh = whiten(L_fresh, M)
    assert relative_error(L, L_fresh) < 1e-5
    assert relative_error(Aw_buffers[active], Aw_fresh) < 1e-4


def test_rejects_aliasing_and_non_float32() -> None:
    rank = 8
    L = torch.eye(rank)
    Aw = torch.zeros(rank, rank)
    z = torch.ones(rank) * 0.01
    vz = torch.empty(rank)
    workspace = torch.empty(4 * rank)
    with pytest.raises(RuntimeError, match="must not alias"):
        candidate.positive_metric_operator_same_vector_out(
            L.clone(), Aw, z, Aw, vz, workspace
        )
    with pytest.raises(RuntimeError, match="float32"):
        candidate.positive_metric_operator_same_vector_out(
            L.double(),
            Aw.double(),
            z.double(),
            torch.empty_like(Aw, dtype=torch.float64),
            torch.empty(rank, dtype=torch.float64),
            torch.empty(4 * rank, dtype=torch.float64),
        )
