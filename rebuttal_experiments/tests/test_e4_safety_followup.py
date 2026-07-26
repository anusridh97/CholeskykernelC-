from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from rebuttal_experiments import e4_safety_followup as e4


def test_protocol_is_fixed_posthoc_followup():
    protocol, development = e4.validate_static_inputs()
    assert protocol["campaign_tag"] == "e4-safety-followup-v1"
    assert protocol["status"] == "frozen_before_confirmation"
    assert protocol["original_e4_status_must_remain"] == "criteria_failed"
    assert protocol["guard"]["probe_count"] == 8
    assert protocol["guard"]["safety_factor"] == 2.0
    assert protocol["guard"]["float32_trigger"] == 2.5e-4
    assert protocol["guard"]["float64_trigger"] == 5e-11
    assert development["future_confirmation"]["outcomes_inspected"] is False
    assert development["future_confirmation"]["artifacts_created"] is False


def test_development_and_confirmation_seeds_are_disjoint():
    protocol, development = e4.validate_static_inputs()
    used = {
        *development["selection"]["seeds"],
        *development["final_policy_development_rerun"]["check_seeds"],
        *development["retrospective_nonconfirmatory_check"]["source_seeds"],
    }
    confirmation = set(protocol["problem"]["confirmation_seeds"])
    assert confirmation == set(range(46001, 46021))
    assert used.isdisjoint(confirmation)


def test_original_failed_status_is_preserved():
    original = e4.read_json(e4.ORIGINAL_AUDIT)
    assert original["scientific_status"] == "criteria_failed"
    assert (
        original["scientific_criteria"]["checks"][
            "maximum_backward_error_float32"
        ]["passed"]
        is False
    )


def test_dispatch_signature_cannot_receive_exact_residual():
    signature = inspect.signature(e4.decide_guard)
    assert list(signature.parameters) == [
        "active_precision",
        "observed_probe_residual",
    ]


@pytest.mark.parametrize(
    ("precision", "below", "above", "expected"),
    [
        ("float32", 2.5e-4, 2.5000001e-4, "promote_fp64"),
        ("float64", 5e-11, 5.000001e-11, "refresh_fp64"),
    ],
)
def test_dispatch_boundary(precision, below, above, expected):
    assert e4.decide_guard(precision, below) == "none"
    assert e4.decide_guard(precision, above) == expected


def test_apply_guard_promotes_and_refreshes_from_authoritative_state():
    torch.set_num_threads(1)
    G = torch.diag(torch.linspace(1.0, 3.0, 8, dtype=torch.float64))
    generator = torch.Generator().manual_seed(5)
    M = torch.randn(8, 8, dtype=torch.float64, generator=generator)
    L32 = torch.linalg.cholesky(G.float()).contiguous()
    Aw32 = e4.whiten(L32, M.float())
    Aw32[0, 0] += 1.0
    L64, Aw64, active = e4.apply_guard(
        action="promote_fp64",
        L=L32,
        Aw=Aw32,
        G_authoritative=G,
        M_authoritative=M,
    )
    assert active == "float64"
    assert L64.dtype == torch.float64
    assert Aw64.dtype == torch.float64
    assert e4.exact_residual(L64, Aw64, M) < 1e-12
    Aw64[0, 0] += 1.0
    L_new, Aw_new, active = e4.apply_guard(
        action="refresh_fp64",
        L=L64,
        Aw=Aw64,
        G_authoritative=G,
        M_authoritative=M,
    )
    assert active == "float64"
    assert e4.exact_residual(L_new, Aw_new, M) < 1e-12


def test_probe_matrix_is_fixed_and_paired():
    left = e4.make_probes(32, 46001)
    right = e4.make_probes(32, 46001)
    other = e4.make_probes(32, 46002)
    assert torch.equal(left, right)
    assert not torch.equal(left, other)
    assert set(left.unique().tolist()) == {-1.0, 1.0}
    assert left.shape == (32, 8)


def test_write_once_json_refuses_overwrite(tmp_path: Path):
    path = tmp_path / "artifact.json"
    e4.write_once_json({"value": 1}, path)
    with pytest.raises(FileExistsError):
        e4.write_once_json({"value": 2}, path)
    assert json.loads(path.read_text()) == {"value": 1}


def test_expected_confirmation_grid_has_480_unique_trajectories():
    protocol, _ = e4.validate_static_inputs()
    identities = e4._expected_identities(protocol)
    assert len(identities) == 480


def test_nonfinite_probe_is_rejected():
    with pytest.raises(ValueError):
        e4.decide_guard("float32", float("nan"))
