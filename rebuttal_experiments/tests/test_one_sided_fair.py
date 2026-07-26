from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

one_sided_fair = pytest.importorskip("one_sided_fair")


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    ((torch.float32, 2e-4), (torch.float64, 2e-12)),
)
@pytest.mark.parametrize(("rank", "width"), ((1, 1), (2, 4), (7, 3), (32, 8), (65, 17)))
@pytest.mark.parametrize("seed", (0, 1))
def test_fair_update_matches_fresh(
    dtype: torch.dtype,
    tolerance: float,
    rank: int,
    width: int,
    seed: int,
) -> None:
    generator = torch.Generator().manual_seed(30_000 + 31 * rank + seed)
    raw = torch.randn(rank, rank, dtype=dtype, generator=generator)
    G = raw @ raw.mT + rank * torch.eye(rank, dtype=dtype)
    L = torch.linalg.cholesky(G).contiguous()
    z = 0.1 * torch.randn(rank, dtype=dtype, generator=generator)
    B = torch.randn(rank, width, dtype=dtype, generator=generator)
    C = torch.randn(rank, width, dtype=dtype, generator=generator)
    Vb = torch.linalg.solve_triangular(L, B, upper=False).contiguous()
    Vc = torch.linalg.solve_triangular(L, C, upper=False).contiguous()

    L_work = L.clone()
    Vb_work = Vb.clone()
    Vc_work = Vc.clone()
    workspace = torch.empty(2 * rank + 2 * width, dtype=dtype)
    dense_candidate = torch.empty(rank, rank, dtype=dtype)
    result = one_sided_fair.one_sided_update_(
        L_work,
        Vb_work,
        Vc_work,
        z.clone(),
        workspace,
    )
    L_fresh = torch.linalg.cholesky(G + torch.outer(z, z)).contiguous()
    Vb_fresh = torch.linalg.solve_triangular(L_fresh, B, upper=False)
    Vc_fresh = torch.linalg.solve_triangular(L_fresh, C, upper=False)
    candidate_result = one_sided_fair.materialize_candidate_(
        Vb_work,
        Vc_work,
        dense_candidate,
    )
    assert result[0].data_ptr() == L_work.data_ptr()
    assert result[1].data_ptr() == Vb_work.data_ptr()
    assert result[2].data_ptr() == Vc_work.data_ptr()
    assert candidate_result.data_ptr() == dense_candidate.data_ptr()
    torch.testing.assert_close(L_work, L_fresh, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(Vb_work, Vb_fresh, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(Vc_work, Vc_fresh, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(
        Vb_work @ Vc_work.mT,
        Vb_fresh @ Vc_fresh.mT,
        rtol=tolerance,
        atol=tolerance,
    )
    torch.testing.assert_close(
        dense_candidate,
        Vb_fresh @ Vc_fresh.mT,
        rtol=tolerance,
        atol=tolerance,
    )


def test_rejects_short_workspace_and_noncontiguous_input() -> None:
    rank = 8
    width = 3
    L = torch.eye(rank, dtype=torch.float64)
    Vb = torch.ones(rank, width, dtype=torch.float64)
    Vc = torch.ones(rank, width, dtype=torch.float64)
    z = torch.ones(rank, dtype=torch.float64)
    with pytest.raises(RuntimeError, match=r"2\*r \+ 2\*p"):
        one_sided_fair.one_sided_update_(
            L.clone(),
            Vb.clone(),
            Vc.clone(),
            z.clone(),
            torch.empty(2 * rank + 2 * width - 1, dtype=torch.float64),
        )
    Vb_noncontiguous = torch.ones(width, rank, dtype=torch.float64).mT
    assert not Vb_noncontiguous.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        one_sided_fair.one_sided_update_(
            L.clone(),
            Vb_noncontiguous,
            Vc.clone(),
            z.clone(),
            torch.empty(2 * rank + 2 * width, dtype=torch.float64),
        )
