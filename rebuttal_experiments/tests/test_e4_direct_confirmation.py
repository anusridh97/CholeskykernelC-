from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import pytest
import torch

import rebuttal_experiments.e4_direct_confirmation as e4
from rebuttal_experiments.audit_e4_direct_confirmation import (
    evaluate_scientific_criteria,
    expected_record_sequence,
    validate_diagnostics,
    validate_fingerprint_pairing,
    validate_journal_replay,
)


FROZEN_PROTOCOL_SHA256 = (
    "10f2ad4bc27bb2bcaf39001f2f76fea8ed474f4de3988c7aff857786563987ee"
)


def _protocol() -> dict:
    return json.loads(e4.PROTOCOL_PATH.read_text(encoding="utf-8"))


def _problem(
    dtype: torch.dtype,
    *,
    seed: int,
    rank: int = 24,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Generator,
]:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(rank, rank, dtype=torch.float64, generator=generator)
    G_original = raw @ raw.mT + rank * torch.eye(rank, dtype=torch.float64)
    raw_m = torch.randn(
        rank,
        rank,
        dtype=torch.float64,
        generator=generator,
    )
    M_original = raw_m / torch.linalg.matrix_norm(raw_m, ord=2)
    G_quantized = G_original.to(dtype).double()
    M_quantized = M_original.to(dtype).double()
    return G_original, M_original, G_quantized, M_quantized, generator


def _diagnostic_record(
    *,
    dtype: torch.dtype = torch.float64,
    step: int = 0,
    phase: str = "checkpoint",
) -> dict:
    G_original, M_original, G_quantized, M_quantized, _ = _problem(
        dtype,
        seed=91,
        rank=12,
    )
    L = torch.linalg.cholesky(G_quantized.to(dtype)).contiguous()
    Aw = e4.whiten(L, M_quantized.to(dtype))
    return e4.direct_diagnostics(
        step=step,
        phase=phase,
        dtype=dtype,
        G_quantized=G_quantized,
        M_quantized=M_quantized,
        G_original=G_original,
        M_original=M_original,
        L_incremental=L,
        Aw_incremental=Aw,
    )


def test_protocol_is_frozen_and_self_consistent():
    observed = hashlib.sha256(e4.PROTOCOL_PATH.read_bytes()).hexdigest()
    assert observed == e4.EXPECTED_PROTOCOL_SHA256 == FROZEN_PROTOCOL_SHA256
    e4.validate_protocol(_protocol())


def test_protocol_mutation_is_rejected_by_frozen_hash(tmp_path, monkeypatch):
    mutated = _protocol()
    mutated["positive_trajectories"]["seeds"][0] += 1
    mutation_path = tmp_path / "mutated_protocol.json"
    mutation_path.write_text(
        json.dumps(mutated, indent=2) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(e4, "PROTOCOL_PATH", mutation_path)
    with pytest.raises(ValueError, match="differs from its frozen hash"):
        e4.validate_protocol(mutated)


def test_write_once_json_refuses_overwrite(tmp_path):
    path = tmp_path / "result.json"
    e4.write_once_json({"value": 1}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        e4.write_once_json({"value": 2}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_direct_diagnostics_separates_identical_fresh_state(dtype):
    record = _diagnostic_record(dtype=dtype)
    validate_diagnostics(
        record,
        "test diagnostic",
        expected_step=0,
        expected_phase="checkpoint",
    )
    assert record["incremental_vs_fresh_same_dtype"]["L_relative_error"] == 0.0
    assert record["incremental_vs_fresh_same_dtype"]["Aw_relative_error"] == 0.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda record: record.update(phase="pre_reset"), "expected phase"),
        (
            lambda record: record["backward_errors"].pop("fresh_factor"),
            "field mismatch",
        ),
    ],
)
def test_diagnostic_schema_and_phase_tampering_is_rejected(mutation, message):
    record = _diagnostic_record()
    mutation(record)
    with pytest.raises(ValueError, match=message):
        validate_diagnostics(
            record,
            "tampered diagnostic",
            expected_step=0,
            expected_phase="checkpoint",
        )


def test_expected_record_sequence_is_phase_aware():
    protocol = _protocol()
    positive = protocol["positive_trajectories"]
    kernel = protocol["near_duplicate_kernels"]
    assert expected_record_sequence(positive, 0) == [
        (0, "checkpoint"),
        (1, "checkpoint"),
        (10, "checkpoint"),
        (50, "checkpoint"),
        (100, "checkpoint"),
        (200, "checkpoint"),
    ]
    assert expected_record_sequence(positive, 50) == [
        (0, "checkpoint"),
        (1, "checkpoint"),
        (10, "checkpoint"),
        (50, "pre_reset"),
        (50, "post_reset"),
        (100, "pre_reset"),
        (100, "post_reset"),
        (150, "pre_reset"),
        (200, "pre_reset"),
        (200, "post_reset"),
    ]
    assert expected_record_sequence(kernel, 50) == [
        (0, "checkpoint"),
        (1, "checkpoint"),
        (10, "checkpoint"),
        (50, "pre_reset"),
        (50, "post_reset"),
        (100, "pre_reset"),
        (100, "post_reset"),
    ]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("operator_sign", [-1, 0, 1])
def test_signed_extension_matches_fresh_at_tau_099(dtype, operator_sign):
    fast_cholesky = pytest.importorskip("fast_cholesky")
    (
        G_original,
        _,
        G_quantized,
        M_quantized,
        generator,
    ) = _problem(dtype, seed=1000 + operator_sign)
    rank = G_quantized.shape[0]
    L = torch.linalg.cholesky(G_quantized.to(dtype)).contiguous()
    Aw = e4.whiten(L, M_quantized.to(dtype))
    direction = torch.randn(
        rank,
        dtype=torch.float64,
        generator=generator,
    )
    direction /= torch.linalg.vector_norm(direction)
    z = (
        math.sqrt(0.99)
        * (torch.linalg.cholesky(G_original) @ direction)
    ).to(dtype).contiguous()
    if operator_sign == 0:
        w = torch.zeros_like(z)
    else:
        w = (
            torch.randn(
                rank,
                dtype=torch.float64,
                generator=generator,
            )
            / math.sqrt(rank)
        ).to(dtype).contiguous()
    Aw_before = Aw.clone()
    z_before = z.clone()
    w_before = w.clone()

    Aw_actual, _ = fast_cholesky.fast_incremental_update(
        L,
        Aw,
        z,
        w,
        -1,
        operator_sign,
        1e-8,
    )

    z64 = z.double()
    w64 = w.double()
    G_new = G_quantized - torch.outer(z64, z64)
    M_new = M_quantized + operator_sign * torch.outer(z64, w64)
    L_fresh = torch.linalg.cholesky(G_new.to(dtype))
    Aw_fresh = e4.whiten(L_fresh, M_new.to(dtype))
    L_error = (
        torch.linalg.vector_norm(L.double() - L_fresh.double())
        / torch.linalg.vector_norm(L_fresh.double())
    ).item()
    Aw_error = (
        torch.linalg.vector_norm(Aw_actual.double() - Aw_fresh.double())
        / torch.linalg.vector_norm(Aw_fresh.double())
    ).item()
    if dtype == torch.float32:
        assert L_error < 2e-5
        assert Aw_error < 5e-4
    else:
        assert L_error < 1e-11
        assert Aw_error < 1e-10
    assert torch.equal(Aw, Aw_before)
    assert torch.equal(z, z_before)
    assert torch.equal(w, w_before)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("leverage", [1.0, 1.01])
def test_inadmissible_downdate_rejection_is_mutation_safe(dtype, leverage):
    fast_cholesky = pytest.importorskip("fast_cholesky")
    _, _, G_quantized, M_quantized, generator = _problem(dtype, seed=2020)
    rank = G_quantized.shape[0]
    L = torch.linalg.cholesky(G_quantized.to(dtype)).contiguous()
    Aw = e4.whiten(L, M_quantized.to(dtype))
    direction = torch.randn(rank, dtype=dtype, generator=generator)
    direction /= torch.linalg.vector_norm(direction)
    z = (math.sqrt(leverage) * (L @ direction)).contiguous()
    w = torch.randn(
        rank,
        dtype=dtype,
        generator=generator,
    ).contiguous()
    before = tuple(tensor.clone() for tensor in (L, Aw, z, w))

    with pytest.raises(RuntimeError, match="inadmissible|must be below 1"):
        fast_cholesky.fast_incremental_update(
            L,
            Aw,
            z,
            w,
            -1,
            1,
            1e-8,
        )

    assert all(
        torch.equal(actual, original)
        for actual, original in zip((L, Aw, z, w), before)
    )


def test_fingerprint_pairing_detects_common_input_tampering():
    rows = [
        {
            "precision": precision,
            "target_condition": 100.0,
            "seed": 7,
            "input_fingerprints": {
                "metric": "a" * 64,
                "operator": "b" * 64,
            },
        }
        for precision in ("float32", "float64")
    ]
    summary = validate_fingerprint_pairing(
        rows,
        group_fields=("target_condition", "seed"),
        fingerprint_fields=("metric", "operator"),
        label="test common-input",
    )
    assert summary["group_count_with_fingerprints"] == 1
    assert summary["rows_with_fingerprints"] == 2

    tampered = deepcopy(rows)
    tampered[1]["input_fingerprints"]["metric"] = "c" * 64
    with pytest.raises(ValueError, match="pairing mismatch"):
        validate_fingerprint_pairing(
            tampered,
            group_fields=("target_condition", "seed"),
            fingerprint_fields=("metric", "operator"),
            label="test common-input",
        )


def test_journal_exact_replay_and_payload_tamper_detection(tmp_path):
    payload = {
        "positive_trajectories": [{"identity": "positive"}],
        "near_duplicate_kernel_trajectories": [{"identity": "kernel"}],
        "signed_downdates": [{"identity": "downdate"}],
    }
    entries = [
        {"kind": "positive", "row": payload["positive_trajectories"][0]},
        {
            "kind": "kernel",
            "row": payload["near_duplicate_kernel_trajectories"][0],
        },
        {"kind": "downdate", "row": payload["signed_downdates"][0]},
    ]
    journal = tmp_path / "rows.jsonl"
    journal.write_text(
        "".join(
            json.dumps(entry, sort_keys=True) + "\n" for entry in entries
        ),
        encoding="utf-8",
    )
    metadata = {
        "path": str(journal.resolve()),
        "sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
        "line_count": len(entries),
    }
    summary = validate_journal_replay(
        journal,
        metadata,
        payload,
        expected_path=journal,
    )
    assert summary["replay_exact"] is True

    tampered_payload = deepcopy(payload)
    tampered_payload["signed_downdates"][0]["identity"] = "changed"
    with pytest.raises(ValueError, match="replay mismatch"):
        validate_journal_replay(
            journal,
            metadata,
            tampered_payload,
            expected_path=journal,
        )


def test_journal_hash_and_final_newline_tampering_are_rejected(tmp_path):
    payload = {
        "positive_trajectories": [{"identity": "positive"}],
        "near_duplicate_kernel_trajectories": [],
        "signed_downdates": [],
    }
    journal = tmp_path / "rows.jsonl"
    journal.write_text(
        json.dumps(
            {
                "kind": "positive",
                "row": payload["positive_trajectories"][0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = {
        "path": str(journal.resolve()),
        "sha256": "0" * 64,
        "line_count": 1,
    }
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_journal_replay(
            journal,
            metadata,
            payload,
            expected_path=journal,
        )

    journal.write_bytes(journal.read_bytes().rstrip(b"\n"))
    metadata["sha256"] = hashlib.sha256(journal.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="final newline"):
        validate_journal_replay(
            journal,
            metadata,
            payload,
            expected_path=journal,
        )


def test_failed_scientific_criteria_do_not_fail_integrity_evaluation():
    scientific = evaluate_scientific_criteria(_protocol(), [], [], [])
    assert scientific["status"] == "criteria_failed"
    assert scientific["all_passed"] is False
    assert scientific["outcome_does_not_affect_integrity_status"] is True
    assert set(scientific["scope_declarations"]) == {
        "backward_error_scope",
        "guarded_error_scope",
    }


def test_scientific_scope_declaration_tampering_is_rejected():
    protocol = _protocol()
    protocol["predeclared_correctness_criteria"][
        "backward_error_scope"
    ] += " tampered"
    with pytest.raises(ValueError, match="scope declarations"):
        evaluate_scientific_criteria(protocol, [], [], [])
