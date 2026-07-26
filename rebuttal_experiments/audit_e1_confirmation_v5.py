#!/usr/bin/python3
"""Independent integrity audit for the E1 v5 confirmation artifacts.

This audit deliberately works from raw pair artifacts and exactly-once claims.
It checks the source/runtime lock, pairing and test-access contract, Slurm
launch map, live thread/affinity observations, raw timing reconciliation, and
the report-level per-invocation core aggregate.  It does not choose or filter
seeds and does not modify any scientific artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any


CAMPAIGN = "e1-confirm-samevector-out-v5"
SEEDS = tuple(range(62001, 62011))
RANKS = (256, 512, 1024)
THREADS = (1, 10)
METHODS = ("ours", "refact")
RUN_COMMIT = "cfd9fb419444b9420268368a5ecec24e0cc1cfac"
LOCKED_GIT_COMMIT = "2788ff1222d84563f102b079f55cb3924686c6fa"
PREFLIGHT_RUN_COMMIT = "27de255712f80d9fb6938c424832465481c02ca2"
PILOT_RUN_COMMIT = "7ee97661286fe0e77cd74849701b6998161294eb"
SOURCE_LOCK_SHA256 = (
    "59373aad6adebc1d1902b00811bb13bc99ad1afbc2460da89cd9cf9d24db8f81"
)
PREFLIGHT_RECEIPT_SHA256 = (
    "b1648ffd726ec5793fa9ce9b42e8bd6bf3a6d82838034f90035b6638c658de37"
)
PILOT_SUMMARY_SHA256 = (
    "d8b6706be425193157d35e2304457898b01d4f812cea32f7c99feda07a1d9f4b"
)
PILOT_GATE_SHA256 = (
    "78baa19f3736783d0620090e39890ad57479bb86a71db69676a529f208a23615"
)
CANDIDATE_SOURCE_SHA256 = (
    "a0d7bba11f178b40fb5ab5c2674640ee8b2f7c5a63836eb931ac9749db0dc8b0"
)
CANDIDATE_BINARY_SHA256 = (
    "7e884903b910a5c0b87c5c93f15e3279f29399cddb7e183185d323c4415a9b1d"
)
PROTEIN_DATA_SHA256 = (
    "5ce538fa05f6834de3acc727e220e777a02fb1ebc3a7bfebcb5aaad9769fed13"
)
REPORT_AGGREGATOR_SHA256 = (
    "d7c6f9a4d180eec8d39dd922ac81fbbc14743983cdff4aadd5ff973e664f91dd"
)
T_CRITICAL_DF9_95 = 2.2621571627409915
TIMING_COMPONENTS = (
    "batch_fetch_seconds",
    "forward_backward_seconds",
    "projection_decision_seconds",
    "refresh_seconds",
    "nonrefresh_core_seconds",
    "reset_seconds",
    "direction_seconds",
    "parameter_update_seconds",
)
EXPECTED_TEST_ACCESS = {
    "per_method_evaluations": 1,
    "pair_total_evaluations": 2,
    "selection_using_test_values": False,
    "terminal_after_both_methods_trained": True,
}
EXPECTED_PROBE_TOLERANCES = {
    "L_relative_error": 5e-6,
    "Aw_relative_error": 5e-5,
    "vz_relative_error": 5e-5,
}
EXPECTED_RAW_KEYS = {
    "step",
    "epoch",
    "batch_offset",
    "batch_mse",
    "unexplained_energy",
    "refreshed",
    "step_total_seconds",
    "unaccounted_seconds",
    *TIMING_COMPONENTS,
}
EXPECTED_OFFSETS = list(range(0, 36584 - 256 + 1, 256))
EXPECTED_EPOCH_TRACE = [
    epoch for epoch in range(1, 11) for _ in EXPECTED_OFFSETS
]
EXPECTED_BATCH_OFFSET_TRACE = EXPECTED_OFFSETS * 10


class AuditError(RuntimeError):
    """Raised when a retained artifact violates the frozen protocol."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def is_finite_nonnegative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant {token}")
            ),
        )
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read {path}: {error}") from error
    require(isinstance(payload, dict), f"{path} is not a JSON object")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left: float, right: float, *, atol: float = 1e-10) -> bool:
    return math.isclose(left, right, rel_tol=1e-11, abs_tol=atol)


def estimate(values: list[float]) -> dict[str, Any]:
    require(len(values) == 10, f"expected ten paired values, got {len(values)}")
    require(all(math.isfinite(value) for value in values), "non-finite estimate")
    mean = statistics.mean(values)
    sample_sd = statistics.stdev(values)
    half_width = T_CRITICAL_DF9_95 * sample_sd / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "sample_sd": sample_sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "values": values,
    }


def validate_native_stat(
    recorded: dict[str, Any], values: list[float], context: str
) -> None:
    require(recorded["values"] == values, f"{context}: retained values changed")
    require(
        close(recorded["mean"], statistics.mean(values)),
        f"{context}: mean changed",
    )
    expected_sd = statistics.stdev(values) if len(values) > 1 else None
    if expected_sd is None:
        require(recorded["sample_sd"] is None, f"{context}: SD changed")
    else:
        require(
            close(recorded["sample_sd"], expected_sd),
            f"{context}: sample SD changed",
        )


def same_json_structure_and_values(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is bool and type(right) is bool and left == right
    if isinstance(left, float) and isinstance(right, float):
        return (
            math.isfinite(left)
            and math.isfinite(right)
            and math.isclose(left, right, rel_tol=1e-15, abs_tol=1e-15)
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            same_json_structure_and_values(left[key], right[key])
            for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            same_json_structure_and_values(a, b)
            for a, b in zip(left, right, strict=True)
        )
    return left == right


def expected_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(SEEDS):
        for rank_index, rank in enumerate(RANKS):
            for thread_index, threads in enumerate(THREADS):
                ours_first = (seed_index + rank_index + thread_index) % 2 == 0
                specs.append(
                    {
                        "phase": "confirmation",
                        "task_index": len(specs),
                        "seed": seed,
                        "seed_index": seed_index,
                        "rank": rank,
                        "rank_index": rank_index,
                        "threads": threads,
                        "thread_index": thread_index,
                        "method_order": (
                            ["ours", "refact"]
                            if ours_first
                            else ["refact", "ours"]
                        ),
                    }
                )
    return specs


def result_name(spec: dict[str, Any]) -> str:
    return (
        f"confirmation_pair_r{spec['rank']}_t{spec['threads']}"
        f"_seed{spec['seed']}.json"
    )


def validate_spec(
    actual: dict[str, Any],
    expected: dict[str, Any],
    context: str,
) -> None:
    require(set(actual) == set(expected), f"{context}: spec fields changed")
    for key in (
        "task_index",
        "seed",
        "seed_index",
        "rank",
        "rank_index",
        "threads",
        "thread_index",
    ):
        require(
            type(actual[key]) is int and actual[key] == expected[key],
            f"{context}: invalid integer spec field {key}",
        )
    require(
        type(actual["phase"]) is str
        and actual["phase"] == expected["phase"]
        and type(actual["method_order"]) is list
        and len(actual["method_order"]) == 2
        and all(type(value) is str for value in actual["method_order"])
        and actual["method_order"] == expected["method_order"],
        f"{context}: invalid categorical spec fields",
    )


def claim_name(spec: dict[str, Any]) -> str:
    return (
        f"confirmation_pair_r{spec['rank']}_t{spec['threads']}"
        f"_seed{spec['seed']}.claim.json"
    )


def core_ms_per_call(method: dict[str, Any]) -> float:
    calls = int(method["steps"]) - int(method["refresh_count"])
    require(calls > 0, "non-refresh invocation denominator is not positive")
    seconds = float(
        method["timing"]["component_totals"]["nonrefresh_core_seconds"]
    )
    require(math.isfinite(seconds) and seconds >= 0, "invalid core seconds")
    return 1000.0 * seconds / calls


def validate_probe(probe: dict[str, Any], rank: int, context: str) -> float:
    require(
        probe["passed"] is True
        and probe["rank"] == rank
        and probe["dtype"] == "torch.float32"
        and probe["contract"] == "positive metric/operator same-vector update"
        and probe["caller_owned_outputs_and_workspace"] is True,
        f"{context}: kernel-probe contract failed",
    )
    require(
        probe["predeclared_tolerances"] == EXPECTED_PROBE_TOLERANCES,
        f"{context}: kernel-probe tolerances changed",
    )
    require(
        set(probe["errors"]) == set(EXPECTED_PROBE_TOLERANCES),
        f"{context}: kernel-probe error fields changed",
    )
    maximum_fraction = 0.0
    for key, tolerance in EXPECTED_PROBE_TOLERANCES.items():
        error = probe["errors"][key]
        require(
            isinstance(error, (int, float))
            and not isinstance(error, bool)
            and math.isfinite(error)
            and 0 <= error <= tolerance,
            f"{context}: kernel-probe error {key}",
        )
        maximum_fraction = max(maximum_fraction, error / tolerance)
    return maximum_fraction


def validate_thread_report(
    report: dict[str, Any], threads: int, context: str
) -> None:
    initial_cpus = report["affinity_before_numeric_import"]
    require(
        len(initial_cpus) == 10 and len(set(initial_cpus)) == 10,
        f"{context}: initial ten-CPU mask",
    )
    topology = report["affinity_before_topology"]
    require(
        [row["logical_cpu"] for row in topology] == initial_cpus,
        f"{context}: initial topology mismatch",
    )
    initial_physical = {
        (row["physical_package"], row["core_id"]) for row in topology
    }
    require(
        len(initial_physical) == 10
        and len({row["physical_package"] for row in topology}) == 1
        and report["allocation_unique_physical_cores"] == 10
        and report["allocation_physical_package"]
        == topology[0]["physical_package"],
        f"{context}: allocation is not ten physical cores on one socket",
    )
    selected = report["selected_cpus"]
    selected_topology = report["selected_cpu_topology"]
    require(
        report["requested_threads"] == threads
        and len(selected) == threads
        and report["affinity_after"] == selected
        and [row["logical_cpu"] for row in selected_topology] == selected
        and set(selected).issubset(set(initial_cpus)),
        f"{context}: selected affinity mismatch",
    )
    selected_physical = {
        (row["physical_package"], row["core_id"]) for row in selected_topology
    }
    require(
        len(selected_physical) == threads
        and len({row["physical_package"] for row in selected_topology}) == 1
        and report["unique_physical_cores"] == threads
        and report["selected_physical_package"]
        == selected_topology[0]["physical_package"],
        f"{context}: selected cores are not physical/one-socket",
    )
    task_affinities = list(report["tasks_after_workload"].values())
    require(task_affinities, f"{context}: missing live worker affinity evidence")
    require(
        all(set(mask).issubset(set(selected)) for mask in task_affinities),
        f"{context}: worker escaped selected affinity",
    )
    worker_union = sorted(
        {cpu for mask in task_affinities for cpu in mask}
    )
    require(
        worker_union == selected
        and report["worker_affinity_union"] == selected,
        f"{context}: worker affinity union mismatch",
    )
    require(
        report["runtime_values"]
        == {
            "aten_interop": 1,
            "aten_intraop": threads,
            "mkl_max": threads,
            "openmp_max": threads,
        },
        f"{context}: live runtime thread counts",
    )
    environment = report["environment"]
    expected_environment = {
        "E1_EXCLUSIVE_NODE": "1",
        "E1_PROCESS_CONCURRENCY": "1",
        "MKL_DYNAMIC": "FALSE",
        "MKL_NUM_THREADS": str(threads),
        "OMP_DYNAMIC": "FALSE",
        "OMP_NUM_THREADS": str(threads),
        "OMP_PLACES": "cores",
        "OMP_PROC_BIND": "close",
        "OPENBLAS_NUM_THREADS": str(threads),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
    }
    require(
        environment == expected_environment,
        f"{context}: thread environment changed",
    )


def load_array_map(path: Path) -> dict[int, str]:
    payload = read_json(path)
    require(
        payload.get("schema_version") == 1
        and payload.get("campaign_tag") == CAMPAIGN
        and payload.get("run_git_commit") == RUN_COMMIT
        and payload.get("scientific_configuration_changed") is False,
        "wrong launch-map identity",
    )
    mapping: dict[int, str] = {}
    for launch in payload.get("launches", []):
        require(isinstance(launch, dict), "malformed launch-map row")
        array_job_id = launch["array_job_id"]
        require(
            isinstance(array_job_id, str)
            and array_job_id.isdigit()
            and array_job_id,
            "invalid launch-map array job ID",
        )
        task_indices = launch["task_indices"]
        require(isinstance(task_indices, list) and task_indices, "empty launch")
        for index in task_indices:
            require(
                type(index) is int and 0 <= index < 60,
                "invalid launch-map task index",
            )
            require(index not in mapping, f"task {index} launched more than once")
            mapping[index] = array_job_id
    require(sorted(mapping) == list(range(60)), "launch map does not cover 0..59")
    return mapping


def load_slurm_accounting(path: Path) -> dict[int, dict[str, str]]:
    records: dict[int, dict[str, str]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        fields = line.split("|")
        require(
            len(fields) == 10 and fields[-1] == "",
            f"malformed Slurm accounting line {line_number}",
        )
        (
            logical_id,
            raw_id,
            job_name,
            state,
            submit,
            start,
            end,
            restarts,
            exit_code,
        ) = fields[:-1]
        require(job_name == "e1-v5-confirm", "unexpected accounting job name")
        require("_" in logical_id and "." not in logical_id, "unexpected job ID")
        array_job_id, task_text = logical_id.split("_", 1)
        require(
            array_job_id.isdigit() and task_text.isdigit() and raw_id.isdigit(),
            "invalid accounting identifier",
        )
        task_index = int(task_text)
        require(task_index not in records, "duplicate accounting task")
        require(
            state == "COMPLETED"
            and restarts == "0"
            and exit_code == "0:0",
            f"task {task_index} did not complete successfully",
        )
        for timestamp in (submit, start, end):
            datetime.fromisoformat(timestamp)
        records[task_index] = {
            "array_job_id": array_job_id,
            "task_job_id": raw_id,
            "state": state,
            "submit": submit,
            "start": start,
            "end": end,
            "restarts": restarts,
            "exit_code": exit_code,
        }
    require(sorted(records) == list(range(60)), "accounting does not cover 0..59")
    require(
        len({record["task_job_id"] for record in records.values()}) == 60,
        "accounting task job IDs are not unique",
    )
    return records


def audit(
    directory: Path,
    array_map_path: Path,
    slurm_accounting_path: Path,
    report_summary_path: Path,
) -> dict[str, Any]:
    directory = directory.resolve()
    lock_path = directory / "source_lock.json"
    receipt_path = directory / "preflight_receipt.json"
    pilot_summary_path = directory / "pilot_summary.json"
    gate_path = directory / "pilot_approved.json"
    summary_path = directory / "confirmation_summary.json"
    lock = read_json(lock_path)
    receipt = read_json(receipt_path)
    pilot_summary = read_json(pilot_summary_path)
    gate = read_json(gate_path)
    summary = read_json(summary_path)
    report_summary = read_json(report_summary_path)
    lock_hash = sha256(lock_path)
    pilot_summary_hash = sha256(pilot_summary_path)
    expected_array_jobs = load_array_map(array_map_path)
    accounting = load_slurm_accounting(slurm_accounting_path)
    require(
        {
            index: record["array_job_id"]
            for index, record in accounting.items()
        }
        == expected_array_jobs,
        "launch ledger disagrees with authoritative Slurm accounting",
    )

    require(lock_hash == SOURCE_LOCK_SHA256, "canonical source-lock hash changed")
    require(
        sha256(receipt_path) == PREFLIGHT_RECEIPT_SHA256,
        "canonical preflight receipt hash changed",
    )
    require(
        pilot_summary_hash == PILOT_SUMMARY_SHA256,
        "canonical pilot summary hash changed",
    )
    require(
        sha256(gate_path) == PILOT_GATE_SHA256,
        "canonical pilot gate hash changed",
    )
    require(lock["schema_version"] == 2, "wrong source-lock schema")
    require(
        lock["experiment"] == "e1_paired_confirmation_source_lock",
        "wrong source-lock experiment",
    )
    require(lock["campaign_tag"] == CAMPAIGN, "wrong source-lock campaign")
    require(lock["git_commit"] == LOCKED_GIT_COMMIT, "wrong locked Git commit")
    require(
        lock["files"]["incremental_source"]["sha256"]
        == CANDIDATE_SOURCE_SHA256,
        "wrong optimized candidate source",
    )
    require(
        lock["files"]["incremental_binary"]["sha256"]
        == CANDIDATE_BINARY_SHA256,
        "wrong executed candidate binary",
    )
    require(
        lock["files"]["protein_data"]["sha256"] == PROTEIN_DATA_SHA256,
        "wrong Protein input",
    )
    require(
        lock["fixed_config"]["confirmation_seeds"] == list(SEEDS)
        and lock["fixed_config"]["ranks"] == list(RANKS)
        and lock["fixed_config"]["thread_counts"] == list(THREADS)
        and lock["fixed_config"]["refresh_threshold"] == 0.1
        and lock["fixed_config"]["epochs"] == 10
        and lock["fixed_config"]["batch_size"] == 256,
        "canonical fixed configuration changed",
    )
    aggregator_path = Path(__file__).with_name(
        "aggregate_e1_confirmation_report.py"
    )
    require(
        sha256(aggregator_path) == REPORT_AGGREGATOR_SHA256,
        "predeclared report aggregator changed",
    )
    require(receipt["source_lock_sha256"] == lock_hash, "preflight lock mismatch")
    require(pilot_summary["source_lock_sha256"] == lock_hash, "pilot lock mismatch")
    require(gate["source_lock_sha256"] == lock_hash, "gate lock mismatch")
    require(
        gate["pilot_summary_sha256"] == pilot_summary_hash,
        "gate pilot-summary mismatch",
    )
    require(gate["decision"] == "approve_untouched_confirmation", "bad gate")
    require(gate["confirmation_seeds"] == list(SEEDS), "gate seeds changed")
    require(
        gate["confirmation_configuration_changed_after_pilot"] is False,
        "configuration changed after pilot",
    )
    require(
        gate["schema_version"] == 1
        and gate["experiment"] == "e1_pilot_review_gate"
        and gate["campaign_tag"] == CAMPAIGN
        and gate["approval_git_commit"] == PILOT_RUN_COMMIT
        and gate["reviewer"] == "Codex_independent_protocol_audit"
        and isinstance(gate["review_note"], str)
        and bool(gate["review_note"].strip()),
        "pilot gate provenance",
    )
    gate_time = datetime.fromisoformat(gate["created_utc"])
    require(gate_time.tzinfo is not None, "pilot gate timestamp is not aware UTC")
    require(summary["source_lock_sha256"] == lock_hash, "summary lock mismatch")

    require(receipt["schema_version"] == 1, "preflight receipt schema")
    require(
        receipt["experiment"] == "e1_confirmation_preflight",
        "preflight receipt experiment",
    )
    require(
        receipt["locked_git_commit"] == LOCKED_GIT_COMMIT
        and receipt["run_git_commit"] == PREFLIGHT_RUN_COMMIT,
        "preflight commit provenance",
    )
    require(
        receipt["confirmation_outcomes_present_before_launch"] is False
        and receipt["pilot_outcomes_present_before_launch"] is False,
        "preflight found prior campaign outcomes",
    )
    require(receipt["runtime"] == lock["runtime"], "preflight runtime changed")
    require(
        receipt["data"] == lock["fixed_config"]["expected_data_partition"],
        "preflight data partition changed",
    )
    preflight_slurm = receipt["slurm"]
    for key, value in {
        "account": "marlowe-m000151-pm06",
        "partition": "batch",
        "cpus_per_task": "10",
        "ntasks": "1",
        "job_num_nodes": "1",
        "restart_count": "0",
    }.items():
        require(preflight_slurm[key] == value, f"preflight Slurm {key}")
    require(
        preflight_slurm["exclusive_allocation_verified"] is True,
        "preflight was not exclusive",
    )
    preflight_thread = receipt["thread_verification"]
    validate_thread_report(preflight_thread, 10, "preflight")
    preflight_ranks: list[int] = []
    for probe in receipt["kernel_probes"]:
        preflight_ranks.append(probe["rank"])
        validate_probe(probe, probe["rank"], "preflight")
    require(preflight_ranks == list(RANKS), "preflight probe grid changed")

    require(
        pilot_summary["schema_version"] == 1
        and pilot_summary["experiment"] == "e1_paired_pilot_summary"
        and pilot_summary["campaign_tag"] == CAMPAIGN
        and pilot_summary["artifact_count"] == 18
        and pilot_summary["fixed_config"] == lock["fixed_config"],
        "pilot summary identity/configuration failed",
    )
    require(
        pilot_summary["audit"]
        == {
            "all_affinities_exact": True,
            "all_expected_artifacts_present": True,
            "all_paired_input_hashes_equal": True,
            "all_runtime_thread_counts_verified": True,
            "all_write_once_artifacts_source_locked": True,
            "confirmation_test_evaluations_exactly_once_per_method": True,
            "pilot_test_values_absent": True,
        },
        "pilot summary audit flags failed",
    )
    pilot_specs: list[dict[str, Any]] = []
    for seed_index, seed in enumerate((63001, 63002, 63003)):
        for rank_index, rank in enumerate(RANKS):
            for thread_index, threads in enumerate(THREADS):
                ours_first = (
                    seed_index + rank_index + thread_index
                ) % 2 == 0
                pilot_specs.append(
                    {
                        "phase": "pilot",
                        "task_index": len(pilot_specs),
                        "seed": seed,
                        "seed_index": seed_index,
                        "rank": rank,
                        "rank_index": rank_index,
                        "threads": threads,
                        "thread_index": thread_index,
                        "method_order": (
                            ["ours", "refact"]
                            if ours_first
                            else ["refact", "ours"]
                        ),
                    }
                )
    pilot_names = {
        (
            f"pilot_pair_r{spec['rank']}_t{spec['threads']}"
            f"_seed{spec['seed']}.json"
        )
        for spec in pilot_specs
    }
    pilot_paths = sorted(directory.glob("pilot_pair_*.json"))
    require(
        {path.name for path in pilot_paths} == pilot_names,
        "pilot artifact coverage changed",
    )
    pilot_times: list[datetime] = []
    for spec in pilot_specs:
        pilot_path = directory / (
            f"pilot_pair_r{spec['rank']}_t{spec['threads']}"
            f"_seed{spec['seed']}.json"
        )
        pilot = read_json(pilot_path)
        index = spec["task_index"]
        pilot_time = datetime.fromisoformat(pilot["created_utc"])
        require(pilot_time.tzinfo is not None, f"pilot {index}: timestamp")
        pilot_times.append(pilot_time)
        validate_spec(pilot["spec"], spec, f"pilot {index}")
        require(pilot["source_lock"]["sha256"] == lock_hash, f"pilot {index}: lock")
        require(
            pilot["source_lock"]["run_git_commit"] == PILOT_RUN_COMMIT,
            f"pilot {index}: run commit",
        )
        require(
            pilot["source_lock"]["files"] == lock["files"]
            and pilot["source_lock"]["runtime"] == lock["runtime"],
            f"pilot {index}: source/runtime",
        )
        require(
            pilot["test_access"]
            == {
                "per_method_evaluations": 0,
                "pair_total_evaluations": 0,
                "selection_using_test_values": False,
                "terminal_after_both_methods_trained": True,
            },
            f"pilot {index}: test policy",
        )
        require(
            type(pilot["test_access"]["per_method_evaluations"]) is int
            and type(pilot["test_access"]["pair_total_evaluations"]) is int,
            f"pilot {index}: test counters are not strict integers",
        )
        require(
            pilot["data"]["heldout_tensors_materialized"] is False
            and pilot["timing_outside_measured_training"][
                "test_materialization_seconds"
            ]
            == 0,
            f"pilot {index}: held-out materialization",
        )
        pilot_probe = pilot["implementation"]["kernel_equivalence_probe"]
        validate_probe(pilot_probe, spec["rank"], f"pilot {index}")
        pilot_thread = pilot["environment"]["thread_verification"]
        validate_thread_report(pilot_thread, spec["threads"], f"pilot {index}")
        require(
            set(pilot["methods"]) == set(METHODS),
            f"pilot {index}: method set",
        )
        pilot_inputs = pilot["paired_inputs"]
        require(
            pilot_inputs["identical_data_object"] is True
            and pilot_inputs["identical_minibatch_permutations"] is True
            and pilot_inputs["identical_model_and_curvature_clones"] is True,
            f"pilot {index}: paired-input flags",
        )
        pilot_initial_hashes: set[str] = set()
        pilot_permutation_hashes: set[str] = set()
        for method_name in METHODS:
            pilot_method = pilot["methods"][method_name]
            require(
                type(pilot_method["test_evaluations"]) is int
                and pilot_method["test_evaluations"] == 0
                and pilot_method["final_test_mse"] is None
                and pilot_method["timing"]["test_evaluation_seconds"] == 0,
                f"pilot {index}: held-out value present",
            )
            require(
                type(pilot_method["steps"]) is int
                and pilot_method["steps"] == 1420
                and type(pilot_method["epochs"]) is int
                and pilot_method["epochs"] == 10,
                f"pilot {index}: incomplete training",
            )
            pilot_initial_hashes.add(pilot_method["initial_state_sha256"])
            pilot_permutation_hashes.add(
                pilot_method["permutations_sha256"]
            )
            pilot_raw = pilot_method["raw_step_samples"]
            require(
                set(pilot_raw) == EXPECTED_RAW_KEYS
                and all(
                    isinstance(values, list) and len(values) == 1420
                    for values in pilot_raw.values()
                )
                and pilot_raw["step"] == list(range(1, 1421))
                and pilot_raw["epoch"] == EXPECTED_EPOCH_TRACE
                and pilot_raw["batch_offset"] == EXPECTED_BATCH_OFFSET_TRACE,
                f"pilot {index}: raw grid/schema",
            )
            require(
                all(
                    type(value) is int
                    for key in ("step", "epoch", "batch_offset")
                    for value in pilot_raw[key]
                ),
                f"pilot {index}: raw integer grid types",
            )
            require(
                all(
                    is_finite_nonnegative_number(value)
                    for key in ("batch_mse", "unexplained_energy")
                    for value in pilot_raw[key]
                ),
                f"pilot {index}: raw scientific values",
            )
            require(
                all(type(value) is bool for value in pilot_raw["refreshed"]),
                f"pilot {index}: refresh trace type",
            )
            for timing_key in (
                *TIMING_COMPONENTS,
                "step_total_seconds",
                "unaccounted_seconds",
            ):
                require(
                    len(pilot_raw[timing_key]) == 1420
                    and all(
                        is_finite_nonnegative_number(value)
                        for value in pilot_raw[timing_key]
                    ),
                    f"pilot {index}: raw timing {timing_key}",
                )
            for sample_index in range(1420):
                component_total = math.fsum(
                    pilot_raw[component][sample_index]
                    for component in TIMING_COMPONENTS
                )
                require(
                    close(
                        pilot_raw["step_total_seconds"][sample_index],
                        component_total
                        + pilot_raw["unaccounted_seconds"][sample_index],
                        atol=1e-12,
                    ),
                    f"pilot {index}: per-step timing reconciliation",
                )
                if pilot_raw["refreshed"][sample_index]:
                    require(
                        pilot_raw["refresh_seconds"][sample_index] > 0
                        and pilot_raw["nonrefresh_core_seconds"][
                            sample_index
                        ]
                        == 0,
                        f"pilot {index}: refresh/core timing",
                    )
                else:
                    require(
                        pilot_raw["refresh_seconds"][sample_index] == 0
                        and pilot_raw["nonrefresh_core_seconds"][
                            sample_index
                        ]
                        > 0,
                        f"pilot {index}: nonrefresh/core timing",
                    )
            pilot_reset_steps = [
                step
                for step, seconds in zip(
                    pilot_raw["step"],
                    pilot_raw["reset_seconds"],
                    strict=True,
                )
                if seconds > 0
            ]
            require(
                pilot_reset_steps
                == (
                    list(range(50, 1401, 50))
                    if method_name == "ours"
                    else []
                ),
                f"pilot {index}: reset schedule",
            )
        require(
            len(pilot_initial_hashes) == 1
            and len(pilot_permutation_hashes) == 1
            and pilot_inputs["initial_state_sha256"]
            == next(iter(pilot_initial_hashes))
            and pilot_inputs["permutations_sha256"]
            == next(iter(pilot_permutation_hashes)),
            f"pilot {index}: paired-input hashes",
        )
        for key in (
            "same_initial_state_hash",
            "same_permutation_hash",
            "same_step_count",
            "same_declared_precision",
        ):
            require(
                pilot["paired_diagnostics"][key] is True,
                f"pilot {index}: paired diagnostic {key}",
            )

    lock_time = datetime.fromisoformat(lock["created_utc"])
    receipt_time = datetime.fromisoformat(receipt["created_utc"])
    pilot_summary_time = datetime.fromisoformat(pilot_summary["created_utc"])
    require(
        all(
            timestamp.tzinfo is not None
            for timestamp in (lock_time, receipt_time, pilot_summary_time)
        )
        and lock_time < receipt_time < min(pilot_times)
        and max(pilot_times) <= pilot_summary_time < gate_time,
        "lock/preflight/pilot/summary/gate chronology failed",
    )

    specs = expected_specs()
    expected_results = {result_name(spec) for spec in specs}
    expected_claims = {claim_name(spec) for spec in specs}
    result_paths = sorted(directory.glob("confirmation_pair_*.json"))
    claim_directory = directory / "confirmation_attempt_claims"
    claim_paths = sorted(claim_directory.glob("*.json"))
    require(
        {path.name for path in result_paths} == expected_results,
        "missing or extra confirmation pair artifact",
    )
    require(
        {path.name for path in claim_paths} == expected_claims,
        "missing or extra exactly-once claim",
    )
    require(
        all(path.stat().st_size > 0 for path in result_paths + claim_paths),
        "empty retained artifact",
    )

    by_task: dict[int, dict[str, Any]] = {}
    task_job_ids: set[str] = set()
    claim_hashes: set[str] = set()
    nodes: set[str] = set()
    test_evaluations = 0
    max_kernel_tolerance_ratio = 0.0
    max_reset_drift = 0.0
    max_timing_reconciliation_error = 0.0

    for spec in specs:
        path = directory / result_name(spec)
        artifact = read_json(path)
        index = spec["task_index"]
        require(artifact["schema_version"] == 1, f"task {index}: schema")
        require(
            artifact["experiment"] == "e1_paired_confirmation",
            f"task {index}: experiment",
        )
        require(artifact["campaign_tag"] == CAMPAIGN, f"task {index}: campaign")
        validate_spec(artifact["spec"], spec, f"task {index}")
        require(
            artifact["fixed_config"] == lock["fixed_config"],
            f"task {index}: fixed config changed",
        )
        source = artifact["source_lock"]
        require(source["sha256"] == lock_hash, f"task {index}: lock hash")
        require(
            source["locked_git_commit"] == lock["git_commit"],
            f"task {index}: locked commit",
        )
        require(
            source["run_git_commit"] == RUN_COMMIT,
            f"task {index}: run commit",
        )
        require(source["files"] == lock["files"], f"task {index}: file lock")
        require(source["runtime"] == lock["runtime"], f"task {index}: runtime")
        require(artifact["pilot_gate"] == gate, f"task {index}: gate payload")

        data = artifact["data"]
        for key, value in lock["fixed_config"]["expected_data_partition"].items():
            require(data[key] == value, f"task {index}: data partition {key}")
        require(
            data["sha256"] == lock["files"]["protein_data"]["sha256"],
            f"task {index}: Protein hash",
        )
        require(data["split_seed"] == 42, f"task {index}: split seed")
        require(
            data["normalization"] == "fit on fixed training partition only",
            f"task {index}: normalization",
        )
        require(
            data["heldout_tensors_materialized"] is True,
            f"task {index}: held-out tensors not terminally materialized",
        )
        require(
            artifact["test_access"] == EXPECTED_TEST_ACCESS,
            f"task {index}: test policy",
        )
        require(
            type(artifact["test_access"]["per_method_evaluations"]) is int
            and type(artifact["test_access"]["pair_total_evaluations"]) is int,
            f"task {index}: test counters are not strict integers",
        )
        require(
            artifact["environment"]["deterministic_algorithms"] is True
            and artifact["environment"]["runtime_lock"] == lock["runtime"],
            f"task {index}: deterministic/runtime evidence",
        )
        require(
            artifact["timing_outside_measured_training"][
                "test_materialization_seconds"
            ]
            > 0,
            f"task {index}: terminal test materialization",
        )

        paired = artifact["paired_diagnostics"]
        for key in (
            "same_initial_state_hash",
            "same_permutation_hash",
            "same_step_count",
            "same_declared_precision",
        ):
            require(paired[key] is True, f"task {index}: paired invariant {key}")

        implementation = artifact["implementation"]
        require(
            implementation["incremental_dispatch"]
            == "explicit positive same-vector contract",
            f"task {index}: dispatch",
        )
        require(
            "caller-owned" in implementation["incremental_kernel"],
            f"task {index}: allocation contract",
        )
        probe = implementation["kernel_equivalence_probe"]
        max_kernel_tolerance_ratio = max(
            max_kernel_tolerance_ratio,
            validate_probe(probe, spec["rank"], f"task {index}"),
        )

        thread = artifact["environment"]["thread_verification"]
        threads = spec["threads"]
        validate_thread_report(thread, threads, f"task {index}")

        slurm = artifact["environment"]["slurm"]
        require(
            slurm["array_job_id"] == expected_array_jobs[index],
            f"task {index}: unexpected array job",
        )
        require(slurm["array_task_id"] == str(index), f"task {index}: array ID")
        require(
            slurm["job_id"] == accounting[index]["task_job_id"],
            f"task {index}: task job ID disagrees with Slurm accounting",
        )
        for key, value in {
            "account": "marlowe-m000151-pm06",
            "partition": "batch",
            "cpus_per_task": "10",
            "ntasks": "1",
            "job_num_nodes": "1",
            "local_id": "0",
            "process_id": "0",
            "restart_count": "0",
        }.items():
            require(slurm[key] == value, f"task {index}: Slurm {key}")
        require(
            slurm["exclusive_allocation_verified"] is True,
            f"task {index}: non-exclusive allocation",
        )
        task_job_ids.add(slurm["job_id"])
        nodes.add(slurm["node_list"])

        claim_path = claim_directory / claim_name(spec)
        claim = read_json(claim_path)
        embedded_claim = artifact["confirmation_exactly_once_claim"]
        require(
            embedded_claim["payload"] == claim,
            f"task {index}: embedded claim payload",
        )
        require(
            embedded_claim["sha256"] == sha256(claim_path),
            f"task {index}: claim hash",
        )
        require(
            Path(embedded_claim["path"]).name == claim_path.name,
            f"task {index}: embedded claim path",
        )
        require(
            claim["schema_version"] == 1
            and claim["experiment"] == "e1_confirmation_exactly_once_claim"
            and claim["campaign_tag"] == CAMPAIGN
            and isinstance(claim["policy"], str)
            and "may never be retried" in claim["policy"],
            f"task {index}: claim schema/policy",
        )
        validate_spec(claim["spec"], spec, f"task {index} claim")
        require(claim["source_lock_sha256"] == lock_hash, f"task {index}: claim lock")
        require(claim["run_git_commit"] == RUN_COMMIT, f"task {index}: claim commit")
        require(claim["slurm"] == slurm, f"task {index}: claim Slurm payload")
        claim_time = datetime.fromisoformat(claim["created_utc"])
        artifact_time = datetime.fromisoformat(artifact["created_utc"])
        require(
            claim_time.tzinfo is not None
            and artifact_time.tzinfo is not None
            and gate_time < claim_time < artifact_time,
            f"task {index}: gate/claim/artifact chronology",
        )
        pair_total_seconds = artifact["timing_outside_measured_training"][
            "pair_total_seconds"
        ]
        require(
            (artifact_time - claim_time).total_seconds()
            >= pair_total_seconds - 0.1,
            f"task {index}: claim did not precede measured pair",
        )
        claim_hashes.add(embedded_claim["sha256"])

        steps = 1420
        require(
            set(artifact["methods"]) == set(METHODS),
            f"task {index}: method set changed",
        )
        paired_inputs = artifact["paired_inputs"]
        require(
            paired_inputs["identical_data_object"] is True
            and paired_inputs["identical_minibatch_permutations"] is True
            and paired_inputs["identical_model_and_curvature_clones"] is True,
            f"task {index}: paired clone evidence",
        )
        initial_hashes: set[str] = set()
        permutation_hashes: set[str] = set()
        for method_name in METHODS:
            method = artifact["methods"][method_name]
            require(method["method"] == method_name, f"task {index}: method")
            require(
                type(method["steps"]) is int and method["steps"] == steps,
                f"task {index}: step count",
            )
            require(
                type(method["epochs"]) is int and method["epochs"] == 10,
                f"task {index}: epoch count",
            )
            require(method["dtype"] == "torch.float32", f"task {index}: dtype")
            require(
                type(method["test_evaluations"]) is int
                and method["test_evaluations"] == 1,
                f"task {index}: method test count",
            )
            require(
                method["timing"]["test_evaluation_seconds"] > 0,
                f"task {index}: method terminal test timing",
            )
            require(
                math.isfinite(method["final_train_mse"])
                and method["final_train_mse"] >= 0
                and math.isfinite(method["final_test_mse"])
                and method["final_test_mse"] >= 0,
                f"task {index}: non-finite MSE",
            )
            test_evaluations += method["test_evaluations"]
            initial_hashes.add(method["initial_state_sha256"])
            permutation_hashes.add(method["permutations_sha256"])

            raw = method["raw_step_samples"]
            require(
                set(raw) == EXPECTED_RAW_KEYS,
                f"task {index}: {method_name} raw schema",
            )
            for key, values in raw.items():
                require(
                    isinstance(values, list) and len(values) == steps,
                    f"task {index}: {method_name} raw {key}",
                )
            require(raw["step"] == list(range(1, steps + 1)), f"task {index}: steps")
            require(
                all(
                    type(value) is int
                    for key in ("step", "epoch", "batch_offset")
                    for value in raw[key]
                ),
                f"task {index}: raw integer grid types",
            )
            require(len(EXPECTED_OFFSETS) == 142, "internal offset grid")
            require(
                raw["epoch"] == EXPECTED_EPOCH_TRACE
                and raw["batch_offset"] == EXPECTED_BATCH_OFFSET_TRACE,
                f"task {index}: epoch/batch-offset grid",
            )
            require(
                all(type(value) is bool for value in raw["refreshed"]),
                f"task {index}: refresh trace type",
            )
            for key in ("batch_mse", "unexplained_energy"):
                require(
                    all(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(value)
                        and value >= 0
                        for value in raw[key]
                    ),
                    f"task {index}: invalid raw {key}",
                )
            for refreshed, refresh_seconds, core_seconds in zip(
                raw["refreshed"],
                raw["refresh_seconds"],
                raw["nonrefresh_core_seconds"],
                strict=True,
            ):
                if refreshed:
                    require(
                        refresh_seconds > 0 and core_seconds == 0,
                        f"task {index}: refresh/core timing semantics",
                    )
                else:
                    require(
                        refresh_seconds == 0 and core_seconds > 0,
                        f"task {index}: nonrefresh/core timing semantics",
                    )
            require(
                method["refresh_count"] == sum(bool(x) for x in raw["refreshed"]),
                f"task {index}: refresh count",
            )
            require(
                close(method["refresh_fraction"], method["refresh_count"] / steps),
                f"task {index}: refresh fraction",
            )
            require(
                close(
                    method["timing"]["sum_step_seconds"],
                    math.fsum(raw["step_total_seconds"]),
                ),
                f"task {index}: total step timing",
            )
            require(
                close(
                    method["timing"]["application_step_seconds"],
                    method["timing"]["sum_step_seconds"],
                ),
                f"task {index}: application timing is not raw step total",
            )
            require(
                all(
                    is_finite_nonnegative_number(value)
                    for key in ("step_total_seconds", "unaccounted_seconds")
                    for value in raw[key]
                ),
                f"task {index}: invalid raw total/unaccounted timing",
            )
            for sample_index in range(steps):
                per_step_components = math.fsum(
                    raw[component][sample_index]
                    for component in TIMING_COMPONENTS
                )
                require(
                    close(
                        raw["step_total_seconds"][sample_index],
                        per_step_components
                        + raw["unaccounted_seconds"][sample_index],
                        atol=1e-12,
                    ),
                    f"task {index}: per-step timing reconciliation",
                )
            component_sum = 0.0
            for component in TIMING_COMPONENTS:
                require(
                    all(
                        is_finite_nonnegative_number(value)
                        for value in raw[component]
                    ),
                    f"task {index}: invalid raw timing {component}",
                )
                observed = math.fsum(raw[component])
                recorded = method["timing"]["component_totals"][component]
                require(
                    close(observed, recorded),
                    f"task {index}: component {component}",
                )
                component_sum += observed
            unaccounted = math.fsum(raw["unaccounted_seconds"])
            require(
                close(method["timing"]["sum_unaccounted_seconds"], unaccounted),
                f"task {index}: unaccounted timing",
            )
            reconciliation = abs(
                method["timing"]["sum_step_seconds"]
                - component_sum
                - unaccounted
            )
            require(reconciliation < 1e-8, f"task {index}: timing reconciliation")
            max_timing_reconciliation_error = max(
                max_timing_reconciliation_error, reconciliation
            )
            for value in method["reset_drift_values"]:
                require(
                    math.isfinite(value) and value >= 0,
                    f"task {index}: reset drift",
                )
                max_reset_drift = max(max_reset_drift, value)
            reset_steps = [
                step
                for step, seconds in zip(
                    raw["step"], raw["reset_seconds"], strict=True
                )
                if seconds > 0
            ]
            if method_name == "ours":
                require(
                    reset_steps == list(range(50, 1401, 50))
                    and len(method["reset_drift_values"]) == 28,
                    f"task {index}: Ours reset schedule",
                )
                require(
                    close(
                        method["max_reset_drift"],
                        max(method["reset_drift_values"]),
                    )
                    and close(
                        method["mean_reset_drift"],
                        statistics.mean(method["reset_drift_values"]),
                    ),
                    f"task {index}: Ours reset-drift aggregate",
                )
            else:
                require(
                    reset_steps == []
                    and method["reset_drift_values"] == []
                    and method["max_reset_drift"] is None
                    and method["mean_reset_drift"] is None,
                    f"task {index}: Refact reset evidence",
                )
            require(
                set(method["final_state"])
                == {"model_sha256", "U_sha256", "G_sha256", "L_sha256", "Aw_sha256"}
                and all(
                    isinstance(value, str) and len(value) == 64
                    for value in method["final_state"].values()
                ),
                f"task {index}: final-state hashes",
            )
            core_ms_per_call(method)
        require(len(initial_hashes) == 1, f"task {index}: initial-state pairing")
        require(
            len(permutation_hashes) == 1,
            f"task {index}: permutation pairing",
        )
        require(
            paired_inputs["initial_state_sha256"] == next(iter(initial_hashes))
            and paired_inputs["permutations_sha256"]
            == next(iter(permutation_hashes)),
            f"task {index}: paired-input hashes",
        )
        require(
            close(
                paired["final_test_mse_difference_ours_minus_refact"],
                artifact["methods"]["ours"]["final_test_mse"]
                - artifact["methods"]["refact"]["final_test_mse"],
            ),
            f"task {index}: paired MSE contrast",
        )
        ours_raw = artifact["methods"]["ours"]["raw_step_samples"]
        refact_raw = artifact["methods"]["refact"]["raw_step_samples"]
        mismatch_count = sum(
            ours != refact
            for ours, refact in zip(
                ours_raw["refreshed"],
                refact_raw["refreshed"],
                strict=True,
            )
        )
        max_unexplained_difference = max(
            abs(ours - refact)
            for ours, refact in zip(
                ours_raw["unexplained_energy"],
                refact_raw["unexplained_energy"],
                strict=True,
            )
        )
        require(
            paired["refresh_decision_mismatches"] == mismatch_count
            and close(
                paired["refresh_decision_mismatch_fraction"],
                mismatch_count / steps,
            )
            and close(
                paired["max_absolute_unexplained_energy_difference"],
                max_unexplained_difference,
            )
            and close(
                paired["final_train_mse_difference_ours_minus_refact"],
                artifact["methods"]["ours"]["final_train_mse"]
                - artifact["methods"]["refact"]["final_train_mse"],
            ),
            f"task {index}: paired divergence traces",
        )
        for key in (
            "final_model_relative_error_ours_vs_refact",
            "final_train_prediction_relative_error_ours_vs_refact",
            "final_test_prediction_relative_error_ours_vs_refact",
        ):
            value = paired[key]
            require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value >= 0,
                f"task {index}: paired divergence {key}",
            )
        by_task[index] = artifact

    require(sorted(by_task) == list(range(60)), "task coverage")
    require(len(task_job_ids) == 60, "task job IDs are not unique")
    require(len(claim_hashes) == 60, "claim hashes are not unique")
    require(test_evaluations == 120, "test-evaluation count is not 120")

    require(summary["schema_version"] == 1, "locked summary schema")
    require(
        summary["experiment"] == "e1_paired_confirmation_summary",
        "locked summary experiment",
    )
    require(summary["artifact_count"] == 60, "locked summary artifact count")
    require(summary["fixed_config"] == lock["fixed_config"], "summary config")
    require(len(summary["groups"]) == 6, "locked summary group count")
    require(
        summary["audit"]
        == {
            "all_affinities_exact": True,
            "all_expected_artifacts_present": True,
            "all_paired_input_hashes_equal": True,
            "all_runtime_thread_counts_verified": True,
            "all_write_once_artifacts_source_locked": True,
            "confirmation_test_evaluations_exactly_once_per_method": True,
            "pilot_test_values_absent": True,
        },
        "locked summary audit flags",
    )
    require(
        [(group["rank"], group["threads"]) for group in summary["groups"]]
        == [(rank, threads) for rank in RANKS for threads in THREADS],
        "locked summary group identities/order changed",
    )

    require(
        report_summary["experiment"] == "e1_v5_report_aggregate",
        "report aggregate experiment",
    )
    require(report_summary["artifact_count"] == 60, "report aggregate count")
    require(
        report_summary["source_lock_sha256"] == lock_hash,
        "report aggregate lock",
    )
    require(
        report_summary["locked_confirmation_summary_sha256"]
        == sha256(summary_path),
        "report aggregate summary hash",
    )
    from aggregate_e1_confirmation_report import aggregate as report_aggregate

    require(
        same_json_structure_and_values(
            report_summary, report_aggregate(directory)
        ),
        "report aggregate is not the exact predeclared raw reconstruction",
    )

    groups: list[dict[str, Any]] = []
    for rank in RANKS:
        for threads in THREADS:
            artifacts = sorted(
                (
                    artifact
                    for artifact in by_task.values()
                    if artifact["spec"]["rank"] == rank
                    and artifact["spec"]["threads"] == threads
                ),
                key=lambda artifact: artifact["spec"]["seed"],
            )
            require(
                [artifact["spec"]["seed"] for artifact in artifacts]
                == list(SEEDS),
                f"incomplete r={rank}, t={threads} cell",
            )
            orders = {
                "ours_first": sum(
                    artifact["spec"]["method_order"][0] == "ours"
                    for artifact in artifacts
                ),
                "refact_first": sum(
                    artifact["spec"]["method_order"][0] == "refact"
                    for artifact in artifacts
                ),
            }
            require(
                orders == {"ours_first": 5, "refact_first": 5},
                f"order imbalance r={rank}, t={threads}",
            )
            app_ratios: list[float] = []
            core_ratios: list[float] = []
            mse_differences: list[float] = []
            per_method: dict[str, dict[str, list[float]]] = {
                method: {
                    "application_seconds": [],
                    "core_ms_per_nonrefresh_call": [],
                    "test_mse": [],
                    "refresh_fraction": [],
                    "refresh_count": [],
                }
                for method in METHODS
            }
            for artifact in artifacts:
                ours = artifact["methods"]["ours"]
                refact = artifact["methods"]["refact"]
                ours_core = core_ms_per_call(ours)
                refact_core = core_ms_per_call(refact)
                app_ratios.append(
                    refact["timing"]["application_step_seconds"]
                    / ours["timing"]["application_step_seconds"]
                )
                core_ratios.append(refact_core / ours_core)
                mse_differences.append(
                    ours["final_test_mse"] - refact["final_test_mse"]
                )
                for method_name, method, core in (
                    ("ours", ours, ours_core),
                    ("refact", refact, refact_core),
                ):
                    per_method[method_name]["application_seconds"].append(
                        method["timing"]["application_step_seconds"]
                    )
                    per_method[method_name][
                        "core_ms_per_nonrefresh_call"
                    ].append(core)
                    per_method[method_name]["test_mse"].append(
                        method["final_test_mse"]
                    )
                    per_method[method_name]["refresh_fraction"].append(
                        method["refresh_fraction"]
                    )
                    per_method[method_name]["refresh_count"].append(
                        float(method["refresh_count"])
                    )

            native_group = next(
                group
                for group in summary["groups"]
                if group["rank"] == rank and group["threads"] == threads
            )
            require(
                native_group["seeds"] == list(SEEDS)
                and native_group["method_order_counts"] == orders,
                f"native summary group identity r={rank}, t={threads}",
            )
            for method_name in METHODS:
                native_method = native_group["methods"][method_name]
                method_rows = [
                    artifact["methods"][method_name]
                    for artifact in artifacts
                ]
                native_values = {
                    "training_wall_seconds": [
                        row["timing"]["training_wall_seconds"]
                        for row in method_rows
                    ],
                    "sum_step_seconds": [
                        row["timing"]["sum_step_seconds"] for row in method_rows
                    ],
                    "nonrefresh_core_seconds": [
                        row["timing"]["component_totals"][
                            "nonrefresh_core_seconds"
                        ]
                        for row in method_rows
                    ],
                    "refresh_seconds": [
                        row["timing"]["component_totals"]["refresh_seconds"]
                        for row in method_rows
                    ],
                    "reset_seconds": [
                        row["timing"]["component_totals"]["reset_seconds"]
                        for row in method_rows
                    ],
                    "refresh_fraction": [
                        row["refresh_fraction"] for row in method_rows
                    ],
                    "final_train_mse": [
                        row["final_train_mse"] for row in method_rows
                    ],
                    "final_test_mse": [
                        row["final_test_mse"] for row in method_rows
                    ],
                }
                require(
                    set(native_method) == set(native_values),
                    f"native summary method fields r={rank}, t={threads}",
                )
                for key, values in native_values.items():
                    validate_native_stat(
                        native_method[key],
                        values,
                        (
                            f"native summary r={rank}, t={threads}, "
                            f"{method_name}, {key}"
                        ),
                    )
            native_paired_values = {
                "paired_application_speedup_refact_over_ours": app_ratios,
                "paired_instrumented_wall_speedup_refact_over_ours": [
                    artifact["methods"]["refact"]["timing"][
                        "training_wall_seconds"
                    ]
                    / artifact["methods"]["ours"]["timing"][
                        "training_wall_seconds"
                    ]
                    for artifact in artifacts
                ],
                "paired_nonrefresh_core_speedup_refact_over_ours": [
                    artifact["methods"]["refact"]["timing"][
                        "component_totals"
                    ]["nonrefresh_core_seconds"]
                    / artifact["methods"]["ours"]["timing"][
                        "component_totals"
                    ]["nonrefresh_core_seconds"]
                    for artifact in artifacts
                ],
                "paired_test_mse_difference_ours_minus_refact": (
                    mse_differences
                ),
                "refresh_decision_mismatch_count": [
                    float(
                        artifact["paired_diagnostics"][
                            "refresh_decision_mismatches"
                        ]
                    )
                    for artifact in artifacts
                ],
            }
            for key, values in native_paired_values.items():
                validate_native_stat(
                    native_group[key],
                    values,
                    f"native summary r={rank}, t={threads}, {key}",
                )
            expected_individual = [
                {
                    "seed": artifact["spec"]["seed"],
                    "method_order": artifact["spec"]["method_order"],
                    "ours_training_wall_seconds": artifact["methods"]["ours"][
                        "timing"
                    ]["training_wall_seconds"],
                    "refact_training_wall_seconds": artifact["methods"][
                        "refact"
                    ]["timing"]["training_wall_seconds"],
                    "application_speedup_refact_over_ours": (
                        artifact["methods"]["refact"]["timing"][
                            "application_step_seconds"
                        ]
                        / artifact["methods"]["ours"]["timing"][
                            "application_step_seconds"
                        ]
                    ),
                    "ours_test_mse": artifact["methods"]["ours"][
                        "final_test_mse"
                    ],
                    "refact_test_mse": artifact["methods"]["refact"][
                        "final_test_mse"
                    ],
                    "refresh_decision_mismatches": artifact[
                        "paired_diagnostics"
                    ]["refresh_decision_mismatches"],
                }
                for artifact in artifacts
            ]
            require(
                native_group["individual_pairs"] == expected_individual,
                f"native summary individual pairs r={rank}, t={threads}",
            )

            computed = {
                "rank": rank,
                "threads": threads,
                "method_order_counts": orders,
                "paired_application_speedup_refact_over_ours": estimate(
                    app_ratios
                ),
                "paired_core_speedup_refact_over_ours_per_nonrefresh_call": (
                    estimate(core_ratios)
                ),
                "paired_test_mse_difference_ours_minus_refact": estimate(
                    mse_differences
                ),
                "methods": {
                    method: {
                        key: estimate(values)
                        for key, values in metrics.items()
                    }
                    for method, metrics in per_method.items()
                },
            }
            recorded = next(
                group
                for group in report_summary["groups"]
                if group["rank"] == rank and group["threads"] == threads
            )
            comparisons = (
                (
                    computed["paired_application_speedup_refact_over_ours"],
                    recorded["paired_application_speedup_refact_over_ours"],
                ),
                (
                    computed[
                        "paired_core_speedup_refact_over_ours_per_nonrefresh_call"
                    ],
                    recorded[
                        "paired_core_speedup_refact_over_ours_per_nonrefresh_call"
                    ],
                ),
                (
                    computed["paired_test_mse_difference_ours_minus_refact"],
                    recorded["paired_test_mse_difference_ours_minus_refact"],
                ),
            )
            for left, right in comparisons:
                for key in (
                    "mean",
                    "sample_sd",
                    "ci95_low",
                    "ci95_high",
                ):
                    require(
                        close(left[key], right[key]),
                        f"report aggregate mismatch r={rank}, t={threads}, {key}",
                    )
            for method_name in METHODS:
                report_prefix = "ours" if method_name == "ours" else "refact"
                for local_key, report_key in (
                    (
                        "core_ms_per_nonrefresh_call",
                        f"{report_prefix}_core_ms_per_nonrefresh_call",
                    ),
                    ("test_mse", f"{report_prefix}_final_test_mse"),
                    ("refresh_fraction", f"{report_prefix}_refresh_fraction"),
                ):
                    left = computed["methods"][method_name][local_key]
                    right = recorded[report_key]
                    for key in ("mean", "sample_sd", "ci95_low", "ci95_high"):
                        require(
                            close(left[key], right[key]),
                            (
                                "report method aggregate mismatch "
                                f"r={rank}, t={threads}, {method_name}, {local_key}"
                            ),
                        )
            groups.append(computed)

    return {
        "audit_passed": True,
        "campaign_tag": CAMPAIGN,
        "protocol": {
            "dataset": "UCI Protein",
            "train_examples": 36584,
            "test_examples": 9146,
            "split_seed": 42,
            "architecture": [9, 512, 512, 128, 1],
            "ranks": list(RANKS),
            "threads": list(THREADS),
            "confirmation_seeds": list(SEEDS),
            "epochs": 10,
            "batch_size": 256,
            "steps_per_method": 1420,
            "tau": 0.1,
            "reset_every_steps_ours": 50,
            "dtype": "CPU FP32",
            "test_policy": (
                "exactly one terminal test evaluation per method after both "
                "paired trainings; no test-based selection"
            ),
            "confidence_intervals": (
                "two-sided 95% Student-t intervals across ten paired seeds"
            ),
        },
        "provenance": {
            "source_lock_sha256": lock_hash,
            "preflight_receipt_sha256": sha256(receipt_path),
            "pilot_summary_sha256": pilot_summary_hash,
            "pilot_gate_sha256": sha256(gate_path),
            "confirmation_summary_sha256": sha256(summary_path),
            "report_summary_sha256": sha256(report_summary_path),
            "independent_audit_source_sha256": sha256(Path(__file__)),
            "locked_git_commit": lock["git_commit"],
            "run_git_commit": RUN_COMMIT,
            "array_launch_map_sha256": sha256(array_map_path),
            "slurm_accounting_sha256": sha256(slurm_accounting_path),
            "nodes": sorted(nodes),
            "result_artifacts": len(result_paths),
            "unique_task_job_ids": len(task_job_ids),
            "exactly_once_claims": len(claim_paths),
            "total_test_evaluations": test_evaluations,
        },
        "integrity": {
            "all_frozen_validator_checks_passed": True,
            "all_independent_checks_passed": True,
            "max_kernel_error_fraction_of_tolerance": (
                max_kernel_tolerance_ratio
            ),
            "max_reset_relative_drift": max_reset_drift,
            "max_timing_reconciliation_error_seconds": (
                max_timing_reconciliation_error
            ),
            "authoritative_slurm_accounting_has_one_successful_allocation_per_task": (
                True
            ),
            "all_live_thread_and_affinity_checks_passed": True,
            "core_ratios_use_per_nonrefresh_invocation_normalization": True,
        },
        "groups": groups,
    }


def write_once_json(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise AuditError(f"refusing to overwrite {output}") from error
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_directory", type=Path)
    parser.add_argument("--array-map", required=True, type=Path)
    parser.add_argument("--slurm-accounting", required=True, type=Path)
    parser.add_argument("--report-summary", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = audit(
        args.campaign_directory,
        args.array_map,
        args.slurm_accounting,
        args.report_summary,
    )
    if args.output is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        write_once_json(payload, args.output)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
