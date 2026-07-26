from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

e3_metric_only = pytest.importorskip("e3_metric_only")


def whiten(L: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    left = torch.linalg.solve_triangular(L, M, upper=False)
    return torch.linalg.solve_triangular(
        L, left.mT, upper=False
    ).mT.contiguous()


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        (torch.linalg.norm(actual - expected) / torch.linalg.norm(expected)).item()
    )


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    ((torch.float32, 2e-4), (torch.float64, 2e-12)),
)
@pytest.mark.parametrize("rank", (1, 2, 3, 7, 32, 65))
@pytest.mark.parametrize("seed", (0, 1))
def test_metric_only_and_generic_w0_match_fresh(
    dtype: torch.dtype,
    tolerance: float,
    rank: int,
    seed: int,
) -> None:
    generator = torch.Generator().manual_seed(10_000 + 17 * rank + seed)
    raw = torch.randn(rank, rank, dtype=dtype, generator=generator)
    G = raw @ raw.mT + rank * torch.eye(rank, dtype=dtype)
    L = torch.linalg.cholesky(G).contiguous()
    M = torch.randn(rank, rank, dtype=dtype, generator=generator)
    Aw = whiten(L, M)
    z = 0.1 * torch.randn(rank, dtype=dtype, generator=generator)
    zero = torch.zeros(rank, dtype=dtype)

    L_fresh = torch.linalg.cholesky(G + torch.outer(z, z)).contiguous()
    Aw_fresh = whiten(L_fresh, M)
    outputs = []
    for method in ("specialized", "generic"):
        L_work = L.clone()
        Aw_work = Aw.clone()
        workspace = torch.empty(5 * rank, dtype=dtype)
        if method == "specialized":
            output = e3_metric_only.metric_only_update_(
                L_work, Aw_work, z.clone(), workspace
            )
        else:
            output = e3_metric_only.generic_update_(
                L_work, Aw_work, z.clone(), zero, workspace
            )
        outputs.append(output)
        assert output[0].data_ptr() == L_work.data_ptr()
        assert output[1].data_ptr() == Aw_work.data_ptr()
        assert relative_error(output[0], L_fresh) <= tolerance
        assert relative_error(output[1], Aw_fresh) <= tolerance

    assert relative_error(outputs[0][0], outputs[1][0]) <= tolerance
    assert relative_error(outputs[0][1], outputs[1][1]) <= tolerance


def test_generic_nonzero_operator_update_matches_fresh() -> None:
    rank = 32
    generator = torch.Generator().manual_seed(42)
    raw = torch.randn(rank, rank, dtype=torch.float64, generator=generator)
    G = raw @ raw.mT + rank * torch.eye(rank, dtype=torch.float64)
    L = torch.linalg.cholesky(G).contiguous()
    M = torch.randn(rank, rank, dtype=torch.float64, generator=generator)
    Aw = whiten(L, M)
    z = 0.1 * torch.randn(rank, dtype=torch.float64, generator=generator)
    w = 0.1 * torch.randn(rank, dtype=torch.float64, generator=generator)

    output = e3_metric_only.generic_update_(
        L.clone(),
        Aw.clone(),
        z.clone(),
        w,
        torch.empty(5 * rank, dtype=torch.float64),
    )
    L_fresh = torch.linalg.cholesky(G + torch.outer(z, z))
    Aw_fresh = whiten(L_fresh, M + torch.outer(z, w))
    assert relative_error(output[0], L_fresh) <= 2e-12
    assert relative_error(output[1], Aw_fresh) <= 2e-12


def test_rejects_noncontiguous_dense_state() -> None:
    rank = 8
    L = torch.eye(rank, dtype=torch.float64)
    Aw = torch.eye(rank, dtype=torch.float64).mT
    assert not Aw.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        e3_metric_only.metric_only_update_(
            L,
            Aw,
            torch.ones(rank, dtype=torch.float64),
            torch.empty(5 * rank, dtype=torch.float64),
        )
