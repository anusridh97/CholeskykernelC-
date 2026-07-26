from __future__ import annotations

import copy
import importlib
import json
import math
from pathlib import Path

import pytest


campaign = importlib.import_module(
    "rebuttal_experiments.e2_preregistered_campaign"
)


def test_protocol_and_seed_sets_are_frozen_and_disjoint() -> None:
    protocol, protocol_sha = campaign.load_and_verify_protocol()
    assert len(protocol_sha) == 64
    assert protocol["campaign"] == campaign.CAMPAIGN
    assert list(campaign.PILOT_SEEDS) == list(
        range(51001, 51006)
    )
    assert list(campaign.ACCURACY_SEEDS) == list(
        range(52001, 52021)
    )
    assert list(campaign.THROUGHPUT_SEEDS) == list(
        range(53001, 53021)
    )
    all_seeds = (
        campaign.PILOT_SEEDS
        + campaign.ACCURACY_SEEDS
        + campaign.THROUGHPUT_SEEDS
    )
    assert len(all_seeds) == len(set(all_seeds))
    assert not set(all_seeds) & {
        0,
        1,
        2,
        1000,
        1001,
        1002,
        6100,
        6101,
        6102,
        7777,
    }
    assert campaign.SPLIT_SEED == 381857132
    assert campaign.derived_split_seed() == (
        campaign.SPLIT_SEED,
        campaign.SPLIT_DERIVATION_SHA256,
    )
    assert protocol["dataset"]["historical_seed42_outcomes_viewed"] is True
    assert (
        protocol["dataset"][
            "v2_heldout_outcomes_viewed_before_preregistration"
        ]
        is False
    )


def test_confirmatory_estimands_are_separate_and_no_optional_stopping() -> None:
    protocol, _ = campaign.load_and_verify_protocol()
    accuracy = protocol["estimands"][
        "A_fixed_epoch_accuracy_equivalence"
    ]
    throughput = protocol["estimands"][
        "B_time_budget_throughput"
    ]
    assert accuracy["rank"] == throughput["rank"] == 1024
    assert accuracy["epochs"] == 10
    assert accuracy["test_evaluations_per_method_seed"] == 1
    assert throughput["test_evaluations_per_method_seed"] == 0
    assert protocol["analysis"]["no_optional_stopping"]
    assert (
        protocol["analysis"]["accuracy_equivalence"][
            "equivalence_margin_absolute_normalized_mse"
        ]
        == 0.01
    )


def test_runtime_and_avx512_contract_are_explicit() -> None:
    runtime, runtime_sha = campaign.load_and_verify_runtime_contract()
    assert len(runtime_sha) == 64
    assert runtime["slurm_account"] == "marlowe-m000151-pm06"
    assert runtime["slurm_partition"] == "batch"
    assert runtime["required_environment"]["PYTHONNOUSERSITE"] == "1"
    evidence = campaign.avx512_binary_evidence(
        campaign.ROOT / campaign.CANDIDATE_BINARY_RELATIVE
    )
    assert evidence["zmm_operand_count"] > 0
    assert evidence["symbols"]


def test_write_once_cannot_replace_an_artifact(tmp_path: Path) -> None:
    destination = tmp_path / "result.json"
    campaign.write_once({"value": 1}, destination)
    with pytest.raises(FileExistsError):
        campaign.write_once({"value": 2}, destination)
    assert json.loads(destination.read_text()) == {"value": 1}


def test_task_mapping_and_strict_file_set_are_exact() -> None:
    names = campaign.strict_expected_json_names()
    assert len(names) == 48
    assert "budget.json" in names
    assert "pilot_gate.json" in names
    assert "preflight.json" in names
    assert "summary.json" not in names
    assert {
        f"pilot_seed{seed}.json" for seed in campaign.PILOT_SEEDS
    }.issubset(names)
    assert {
        f"accuracy_seed{seed}.json"
        for seed in campaign.ACCURACY_SEEDS
    }.issubset(names)
    assert {
        f"throughput_seed{seed}.json"
        for seed in campaign.THROUGHPUT_SEEDS
    }.issubset(names)


def test_paired_t_intervals_use_all_twenty_pairs() -> None:
    values = [float(index) / 1000 for index in range(20)]
    summary_95 = campaign.paired_t_summary(
        values, confidence=0.95
    )
    summary_90 = campaign.paired_t_summary(
        values, confidence=0.90
    )
    assert summary_95["n_pairs"] == 20
    assert summary_95["student_t_df"] == 19
    assert summary_95["values"] == values
    assert summary_95["ci_lower"] < summary_95["mean"]
    assert summary_95["ci_upper"] > summary_95["mean"]
    assert (
        summary_95["ci_upper"] - summary_95["ci_lower"]
        > summary_90["ci_upper"] - summary_90["ci_lower"]
    )
    with pytest.raises(RuntimeError):
        campaign.paired_t_summary(values[:19], confidence=0.95)


def test_split_and_minibatch_pairing_are_deterministic() -> None:
    torch = pytest.importorskip("torch")
    campaign.torch = torch
    first = campaign.split_indices(101)
    second = campaign.split_indices(101)
    assert set(first) == {
        "development",
        "test",
        "pilot_train",
        "pilot_validation",
    }
    assert torch.equal(first["development"], second["development"])
    assert torch.equal(first["test"], second["test"])
    development = set(first["development"].tolist())
    test = set(first["test"].tolist())
    pilot_train = set(first["pilot_train"].tolist())
    pilot_validation = set(first["pilot_validation"].tolist())
    assert development.isdisjoint(test)
    assert pilot_train.isdisjoint(pilot_validation)
    assert pilot_train | pilot_validation == development
    assert development | test == set(range(101))

    schedule_a, hash_a = campaign.make_minibatch_schedule(
        len(pilot_train), 51001, 10
    )
    schedule_b, hash_b = campaign.make_minibatch_schedule(
        len(pilot_train), 51001, 10
    )
    schedule_c, hash_c = campaign.make_minibatch_schedule(
        len(pilot_train), 51002, 10
    )
    assert hash_a == hash_b
    assert hash_a != hash_c
    assert all(
        torch.equal(left, right)
        for left, right in zip(schedule_a, schedule_b, strict=True)
    )


def test_equivalence_decision_is_not_a_superiority_test() -> None:
    protocol, _ = campaign.load_and_verify_protocol()
    analysis = protocol["analysis"]["accuracy_equivalence"]
    assert analysis["superiority_is_not_the_estimand"] is True
    assert analysis["success_rule"].startswith(
        "the complete 20-pair 90% CI"
    )
    centered = campaign.paired_t_summary(
        [0.0] * 20, confidence=0.90
    )
    assert centered["ci_lower"] > -campaign.EQUIVALENCE_MARGIN
    assert centered["ci_upper"] < campaign.EQUIVALENCE_MARGIN
    wide = campaign.paired_t_summary(
        [-0.04, 0.04] * 10, confidence=0.90
    )
    assert (
        wide["ci_lower"] <= -campaign.EQUIVALENCE_MARGIN
        or wide["ci_upper"] >= campaign.EQUIVALENCE_MARGIN
    )
    assert math.isfinite(wide["sample_sd"])


def test_strict_aggregate_requires_and_uses_all_pairs(
    tmp_path: Path,
) -> None:
    locked_files = {
        str(Path(campaign.__file__).resolve().relative_to(campaign.ROOT)):
            campaign.sha256_file(Path(campaign.__file__).resolve()),
        str(campaign.PROTOCOL_PATH.relative_to(campaign.ROOT)):
            campaign.sha256_file(campaign.PROTOCOL_PATH),
        str(campaign.RUNTIME_CONTRACT_PATH.relative_to(campaign.ROOT)):
            campaign.sha256_file(campaign.RUNTIME_CONTRACT_PATH),
        str(campaign.DATA_RELATIVE):
            campaign.sha256_file(campaign.ROOT / campaign.DATA_RELATIVE),
        str(campaign.CANDIDATE_SOURCE_RELATIVE):
            campaign.sha256_file(
                campaign.ROOT / campaign.CANDIDATE_SOURCE_RELATIVE
            ),
        str(campaign.CANDIDATE_BINARY_RELATIVE):
            campaign.sha256_file(
                campaign.ROOT / campaign.CANDIDATE_BINARY_RELATIVE
            ),
    }
    context = {
        "git_commit": "a" * 40,
        "protocol_sha256": campaign.sha256_file(
            campaign.PROTOCOL_PATH
        ),
        "runtime_contract": campaign.read_json(
            campaign.RUNTIME_CONTRACT_PATH
        ),
        "runtime_contract_sha256": campaign.sha256_file(
            campaign.RUNTIME_CONTRACT_PATH
        ),
        "lock_sha256": "b" * 64,
        "lock": {"files": locked_files},
    }
    provenance = campaign.common_artifact_provenance(context)
    directory = tmp_path / campaign.CAMPAIGN
    directory.mkdir(parents=True)

    def pair_payload(
        *,
        stage: str,
        task_id: int,
        seed: int,
        budget_sha: str | None = None,
        gate_sha: str | None = None,
    ) -> dict:
        is_accuracy = stage == "accuracy-pair"
        is_throughput = stage == "throughput-pair"
        steps = 100 + task_id if is_throughput else 50
        is_pilot = stage == "pilot-pair"
        ours_mse = (
            0.5 + task_id * 1e-4
            if is_accuracy
            else (0.6 + task_id * 1e-4 if is_pilot else None)
        )
        refact_mse = (
            ours_mse + 2e-4 if ours_mse is not None else None
        )
        evaluation_kind = {
            "pilot-pair": "validation",
            "accuracy-pair": "test",
            "throughput-pair": "none",
        }[stage]

        def method_payload(
            method: str,
            evaluation: float | None,
        ) -> dict:
            method_steps = (
                steps
                if method == "ours"
                else (steps - 10 if is_throughput else steps)
            )
            return {
                "method": method,
                "rank": campaign.RANK,
                "initial_state_sha256": "c" * 64,
                "fixed_epoch_run": not is_throughput,
                "budget_seconds": 1.0 if is_throughput else None,
                "budget_crossed": is_throughput,
                "selected_steps": method_steps,
                "attempted_steps": (
                    method_steps + 1
                    if is_throughput
                    else method_steps
                ),
                "selected_training_seconds": (
                    0.9 if is_throughput else 1.0
                ),
                "attempted_training_seconds": (
                    1.1 if is_throughput else 1.0
                ),
                "selected_batch_mse": 0.7,
                "drop_last_incomplete_batch": True,
                "over_budget_attempt_excluded_from_reported_counters":
                    is_throughput,
                "selected_model_state_materialized":
                    not is_throughput,
                "refresh_count": 3,
                "core_seconds": 0.5,
                "refresh_seconds": 0.2,
                "reset_seconds": (
                    0.1 if method == "ours" else 0.0
                ),
                "max_reset_drift": (
                    1e-6 if method == "ours" else None
                ),
                "mean_unexplained_energy": 0.2,
                "evaluation_kind": evaluation_kind,
                "evaluation_calls": 0 if is_throughput else 1,
                "test_evaluations": 1 if is_accuracy else 0,
                "final_evaluation_mse": evaluation,
                "final_model_sha256": (
                    "e" * 64 if not is_throughput else None
                ),
            }

        claim_path = (
            directory
            / "claims"
            / f"{stage}_task{task_id}_seed{seed}.json"
        )
        campaign.write_once(
            {
                "stage": stage,
                "task_id": task_id,
                "seed": seed,
                "attempt_ordinal": 1,
                "slurm_restart_count": 0,
            },
            claim_path,
        )
        return {
            "schema_version": 1,
            "experiment": "e2_preregistered_paired_protein",
            "campaign": campaign.CAMPAIGN,
            "status": "complete",
            "stage": stage,
            "task_id": task_id,
            "seed": seed,
            "method_order": (
                ["ours", "refact"]
                if task_id % 2 == 0
                else ["refact", "ours"]
            ),
            "method_process_concurrency": 1,
            "methods_executed_sequentially": True,
            "shared_initial_state_sha256": "c" * 64,
            "minibatch_schedule_sha256": "d" * 64,
            "schedule_epochs": (
                campaign.THROUGHPUT_MAX_EPOCHS
                if is_throughput
                else campaign.EPOCHS
            ),
            "methods": {
                "ours": method_payload("ours", ours_mse),
                "refact": method_payload("refact", refact_mse),
            },
            "provenance": provenance,
            "runtime": {
                "cpu_model": "fixture-cpu",
                "platform": "fixture-platform",
                "python": "fixture-python",
                "python_executable": "/usr/bin/python3",
                "torch": "fixture-torch",
                "numpy": "fixture-numpy",
                "scipy": "fixture-scipy",
                "torch_parallel_info": "fixture-parallel-info",
                "runtime_contract_sha256": context[
                    "runtime_contract_sha256"
                ],
                "runtime_files": {
                    "python_executable_sha256": context[
                        "runtime_contract"
                    ]["python"]["executable_sha256"],
                    "packages": {},
                },
                "required_environment": {
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONHASHSEED": "0",
                    "OMP_NUM_THREADS": "10",
                    "MKL_NUM_THREADS": "10",
                    "OPENBLAS_NUM_THREADS": "10",
                    "OMP_DYNAMIC": "FALSE",
                    "MKL_DYNAMIC": "FALSE",
                    "OMP_PROC_BIND": "close",
                    "OMP_PLACES": "cores",
                    "E2_PROCESS_CONCURRENCY": "1",
                    "E2_EXCLUSIVE_NODE": "1",
                    "E2_ONE_ATTEMPT": "1",
                },
                "slurm": {
                    "SLURM_JOB_ID": f"job-{stage}-{task_id}",
                    "SLURM_ARRAY_JOB_ID": f"array-{stage}",
                    "SLURM_ARRAY_TASK_ID": str(task_id),
                    "SLURM_STEP_ID": "0",
                    "SLURM_CPUS_PER_TASK": "10",
                    "SLURM_NTASKS": "1",
                    "SLURM_JOB_NUM_NODES": "1",
                    "SLURM_LOCALID": "0",
                    "SLURM_PROCID": "0",
                    "SLURM_JOB_ACCOUNT":
                        campaign.EXPECTED_SLURM_ACCOUNT,
                    "SLURM_JOB_PARTITION":
                        campaign.EXPECTED_SLURM_PARTITION,
                    "SLURM_RESTART_COUNT": "0",
                },
                "runtime_values": {
                    "aten_intraop": 10,
                    "aten_interop": 1,
                    "openmp_max": 10,
                    "mkl_max": 10,
                },
                "affinity_before_torch_import": list(range(10)),
                "worker_affinity_union": list(range(10)),
                "cpu_topology": [
                    {
                        "logical_cpu": cpu,
                        "physical_package": 0,
                        "core_id": cpu,
                    }
                    for cpu in range(10)
                ],
                "cpu_flags": list(campaign.REQUIRED_AVX512_FLAGS),
                "candidate_avx512_evidence": {
                    "zmm_operand_count": 12,
                    "zmm_registers": ["zmm0"],
                    "symbols": ["fixture"],
                    "selected_disassembly_sha256": "9" * 64,
                },
            },
            "dataset": {
                "sha256": locked_files[str(campaign.DATA_RELATIVE)],
                "evaluation_kind": evaluation_kind,
                "test_tensor_constructed": is_accuracy,
                "split_seed": campaign.SPLIT_SEED,
                "split_seed_derivation": campaign.SPLIT_DERIVATION,
                "split_seed_derivation_sha256":
                    campaign.SPLIT_DERIVATION_SHA256,
                "steps_per_epoch": 5,
                "development_steps_per_epoch": 5,
                "train_examples": 1280,
                "index_sha256": {
                    "development": "1" * 64,
                    "test": "2" * 64,
                    "pilot_train": "3" * 64,
                    "pilot_validation": "4" * 64,
                },
                "normalizer_sha256": (
                    "5" * 64 if is_pilot else "6" * 64
                ),
            },
            "training": {
                "rank": campaign.RANK,
                "batch_size": campaign.BATCH_SIZE,
                "drop_last_incomplete_batch": True,
                "learning_rate": campaign.LEARNING_RATE,
                "damping": campaign.DAMPING,
                "momentum": campaign.MOMENTUM,
                "refresh_threshold": campaign.REFRESH_THRESHOLD,
                "ours_refactor_every": campaign.REFACTOR_EVERY,
            },
            "budget": (
                {"sha256": budget_sha} if is_throughput else None
            ),
            "manual_pilot_gate": (
                {
                    "path": str(
                        campaign.pilot_gate_path(tmp_path).resolve()
                    ),
                    "sha256": gate_sha,
                    "created_unix_ns": 1,
                }
                if not is_pilot
                else None
            ),
            "attempt_claim": {
                "path": str(claim_path.resolve()),
                "sha256": campaign.sha256_file(claim_path),
                "attempt_ordinal": 1,
            },
            "exact_node_preflight": {
                "status": "PASS",
                "cases": [
                    {
                        "rank": campaign.RANK,
                        "repeats": campaign.PREFLIGHT_RANK_REPEATS,
                        "production_reset": {
                            "after_update":
                                campaign.PREFLIGHT_RESET_AFTER
                        },
                    }
                ],
            },
            "scientific_guardrails": {
                "historical_seed42_outcomes_previously_viewed": True,
                "v2_test_outcomes_read_before_preregistration": False,
                "v2_split_selected_mechanically": True,
                "test_evaluations_total": 2 if is_accuracy else 0,
                "checkpoint_selection": False,
                "pair_complete_before_write": True,
            },
        }

    for task_id, seed in enumerate(campaign.PILOT_SEEDS):
        campaign.write_once(
            pair_payload(
                stage="pilot-pair",
                task_id=task_id,
                seed=seed,
            ),
            campaign.artifact_path(
                tmp_path, "pilot-pair", seed
            ),
        )
    pilot_sources = []
    pilot_rates = []
    for seed in campaign.PILOT_SEEDS:
        path = campaign.artifact_path(
            tmp_path, "pilot-pair", seed
        )
        payload = json.loads(path.read_text())
        ours = payload["methods"]["ours"]
        rate = (
            ours["selected_training_seconds"]
            / ours["selected_steps"]
        )
        pilot_rates.append(rate)
        pilot_sources.append(
            {
                "seed": seed,
                "path": str(path.resolve()),
                "sha256": campaign.sha256_file(path),
                "ours_seconds": ours["selected_training_seconds"],
                "ours_steps": ours["selected_steps"],
                "ours_seconds_per_step": rate,
            }
        )
    median_rate = 0.02
    budget = {
        "schema_version": 1,
        "experiment": "e2_preregistered_budget_calibration",
        "campaign": campaign.CAMPAIGN,
        "status": "complete",
        "calibration_rule": (
            "median pilot Ours application seconds per completed batch "
            "times batches in ten full-development epochs"
        ),
        "pilot_seeds": list(campaign.PILOT_SEEDS),
        "pilot_sources": pilot_sources,
        "pilot_ours_seconds_per_step": pilot_rates,
        "median_ours_seconds_per_step": median_rate,
        "target_development_steps": 50,
        "budget_seconds": 1.0,
        "test_values_used": False,
        "validation_values_used": False,
        "cpu_model": "fixture-cpu",
        "provenance": provenance,
    }
    budget_path = directory / "budget.json"
    campaign.write_once(budget, budget_path)
    budget_sha = campaign.sha256_file(budget_path)
    campaign.write_once(
        {"status": "PASS"},
        directory / "preflight.json",
    )
    gate_path = campaign.create_pilot_gate(
        tmp_path,
        context,
        campaign.MANUAL_GATE_ATTESTATION,
    )
    gate_sha = campaign.sha256_file(gate_path)

    for task_id, seed in enumerate(campaign.ACCURACY_SEEDS):
        campaign.write_once(
            pair_payload(
                stage="accuracy-pair",
                task_id=task_id,
                seed=seed,
                gate_sha=gate_sha,
            ),
            campaign.artifact_path(
                tmp_path, "accuracy-pair", seed
            ),
        )
    for task_id, seed in enumerate(campaign.THROUGHPUT_SEEDS):
        campaign.write_once(
            pair_payload(
                stage="throughput-pair",
                task_id=task_id,
                seed=seed,
                budget_sha=budget_sha,
                gate_sha=gate_sha,
            ),
            campaign.artifact_path(
                tmp_path, "throughput-pair", seed
            ),
        )

    accuracy_path = campaign.artifact_path(
        tmp_path, "accuracy-pair", campaign.ACCURACY_SEEDS[0]
    )
    valid_accuracy = json.loads(accuracy_path.read_text())
    campaign.validate_common_artifact(
        valid_accuracy,
        stage="accuracy-pair",
        task_id=0,
        seed=campaign.ACCURACY_SEEDS[0],
        context=context,
    )
    for mutation in (
        lambda item: item["runtime"]["slurm"].__setitem__(
            "SLURM_ARRAY_TASK_ID", "1"
        ),
        lambda item: item.__setitem__("schedule_epochs", 9),
        lambda item: item["methods"]["ours"].__setitem__(
            "test_evaluations", 0
        ),
        lambda item: item["scientific_guardrails"].__setitem__(
            "v2_test_outcomes_read_before_preregistration", True
        ),
    ):
        corrupted = copy.deepcopy(valid_accuracy)
        mutation(corrupted)
        with pytest.raises(RuntimeError):
            campaign.validate_common_artifact(
                corrupted,
                stage="accuracy-pair",
                task_id=0,
                seed=campaign.ACCURACY_SEEDS[0],
                context=context,
            )

    corrupted_budget = copy.deepcopy(budget)
    corrupted_budget["budget_seconds"] += 1e-3
    with pytest.raises(RuntimeError):
        campaign.audit_budget_against_pilots(
            tmp_path, corrupted_budget, context
        )

    summary_path = campaign.aggregate(
        tmp_path, context, gate_sha
    )
    summary = json.loads(summary_path.read_text())
    assert (
        summary["accuracy_equivalence"][
            "paired_95_percent_ci"
        ]["n_pairs"]
        == 20
    )
    assert (
        summary["time_budget_throughput"][
            "paired_batch_difference_95_percent_ci"
        ]["n_pairs"]
        == 20
    )
    assert summary["no_optional_stopping"] is True
    assert len(summary["accuracy_equivalence"]["pairs"]) == 20
    assert len(summary["time_budget_throughput"]["pairs"]) == 20
    with pytest.raises(FileExistsError):
        campaign.aggregate(tmp_path, context, gate_sha)
