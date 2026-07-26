"""Outcome-blind report aggregation for the E1 v5 confirmation.

This analysis was fixed after an independent protocol audit and before any
E1-v5 confirmation task was submitted.  The locked experiment summary stores
the *total* non-refresh core time.  Because two long FP32 trajectories can
make different refresh decisions, totals can contain different numbers of
core invocations.  The report-level core comparison therefore uses

    total non-refresh core seconds / (steps - refresh_count)

for each method in each paired seed, then forms Refact/Ours ratios within
pairs.  Application timing already includes refreshes, numerical resets, and
the rest of every training step, so its predeclared paired ratio is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


CAMPAIGN = "e1-confirm-samevector-out-v5"
RANKS = (256, 512, 1024)
THREADS = (1, 10)
SEEDS = tuple(range(62001, 62011))
METHODS = ("ours", "refact")
T_CRITICAL_DF9_95 = 2.2621571627409915


class AuditError(RuntimeError):
    """Raised when a confirmation artifact is missing or inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise AuditError(f"{path} is not a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _estimate(values: list[float]) -> dict[str, Any]:
    if len(values) != len(SEEDS):
        raise AuditError(
            f"expected {len(SEEDS)} paired values, found {len(values)}"
        )
    if not all(math.isfinite(value) for value in values):
        raise AuditError("non-finite value in report aggregate")
    mean = statistics.mean(values)
    sample_sd = statistics.stdev(values)
    half_width = (
        T_CRITICAL_DF9_95 * sample_sd / math.sqrt(len(values))
    )
    return {
        "n": len(values),
        "mean": mean,
        "sample_sd": sample_sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "values": values,
    }


def core_ms_per_call(method: dict[str, Any]) -> float:
    """Return core milliseconds per actual non-refresh invocation."""
    try:
        steps = int(method["steps"])
        refresh_count = int(method["refresh_count"])
        total_seconds = float(
            method["timing"]["component_totals"][
                "nonrefresh_core_seconds"
            ]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AuditError("malformed method timing record") from error
    calls = steps - refresh_count
    if calls <= 0:
        raise AuditError(
            f"non-refresh call count must be positive, got {calls}"
        )
    if not math.isfinite(total_seconds) or total_seconds < 0:
        raise AuditError(f"invalid core total {total_seconds!r}")
    return 1000.0 * total_seconds / calls


def _expected_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(SEEDS):
        for rank_index, rank in enumerate(RANKS):
            for thread_index, threads in enumerate(THREADS):
                ours_first = (
                    seed_index + rank_index + thread_index
                ) % 2 == 0
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


def _result_name(spec: dict[str, Any]) -> str:
    return (
        f"confirmation_pair_r{spec['rank']}_t{spec['threads']}"
        f"_seed{spec['seed']}.json"
    )


def _validate_artifact(
    artifact: dict[str, Any],
    spec: dict[str, Any],
    lock: dict[str, Any],
    lock_sha256: str,
) -> None:
    if artifact.get("schema_version") != 1:
        raise AuditError("unsupported E1 artifact schema")
    if artifact.get("experiment") != "e1_paired_confirmation":
        raise AuditError("wrong E1 experiment name")
    if artifact.get("campaign_tag") != CAMPAIGN:
        raise AuditError("wrong E1 campaign tag")
    if artifact.get("spec") != spec:
        raise AuditError(f"task {spec['task_index']} specification changed")
    if artifact.get("fixed_config") != lock.get("fixed_config"):
        raise AuditError(f"task {spec['task_index']} fixed config changed")
    source = artifact.get("source_lock", {})
    if source.get("sha256") != lock_sha256:
        raise AuditError(f"task {spec['task_index']} source lock changed")
    if source.get("files") != lock.get("files"):
        raise AuditError(f"task {spec['task_index']} file manifest changed")
    if source.get("runtime") != lock.get("runtime"):
        raise AuditError(f"task {spec['task_index']} runtime changed")

    access = artifact.get("test_access")
    if access != {
        "per_method_evaluations": 1,
        "pair_total_evaluations": 2,
        "selection_using_test_values": False,
        "terminal_after_both_methods_trained": True,
    }:
        raise AuditError(f"task {spec['task_index']} test policy failed")

    paired = artifact.get("paired_diagnostics", {})
    for name in (
        "same_initial_state_hash",
        "same_permutation_hash",
        "same_step_count",
        "same_declared_precision",
    ):
        if paired.get(name) is not True:
            raise AuditError(
                f"task {spec['task_index']} failed paired invariant {name}"
            )

    slurm = artifact.get("environment", {}).get("slurm", {})
    if (
        slurm.get("account") != "marlowe-m000151-pm06"
        or slurm.get("partition") != "batch"
        or slurm.get("cpus_per_task") != "10"
        or slurm.get("ntasks") != "1"
        or slurm.get("job_num_nodes") != "1"
        or slurm.get("restart_count") != "0"
        or slurm.get("exclusive_allocation_verified") is not True
    ):
        raise AuditError(f"task {spec['task_index']} Slurm policy failed")

    threads = spec["threads"]
    runtime_threads = artifact.get("environment", {}).get(
        "thread_verification", {}
    )
    if runtime_threads.get("runtime_values") != {
        "aten_interop": 1,
        "aten_intraop": threads,
        "mkl_max": threads,
        "openmp_max": threads,
    }:
        raise AuditError(
            f"task {spec['task_index']} runtime threads changed"
        )
    if runtime_threads.get("worker_affinity_union") != (
        runtime_threads.get("selected_cpus")
    ):
        raise AuditError(
            f"task {spec['task_index']} worker affinity escaped"
        )

    methods = artifact.get("methods", {})
    if set(methods) != set(METHODS):
        raise AuditError(f"task {spec['task_index']} methods changed")
    for name in METHODS:
        method = methods[name]
        if method.get("test_evaluations") != 1:
            raise AuditError(
                f"task {spec['task_index']} {name} test count changed"
            )
        if not math.isfinite(float(method["final_test_mse"])):
            raise AuditError(
                f"task {spec['task_index']} {name} MSE is non-finite"
            )
        core_ms_per_call(method)


def _aggregate_cell(
    artifacts: list[dict[str, Any]], rank: int, threads: int
) -> dict[str, Any]:
    cell = sorted(
        (
            artifact
            for artifact in artifacts
            if artifact["spec"]["rank"] == rank
            and artifact["spec"]["threads"] == threads
        ),
        key=lambda artifact: artifact["spec"]["seed"],
    )
    if [artifact["spec"]["seed"] for artifact in cell] != list(SEEDS):
        raise AuditError(f"incomplete r={rank}, threads={threads} cell")

    app_ratios: list[float] = []
    core_ratios: list[float] = []
    per_method_core: dict[str, list[float]] = {
        method: [] for method in METHODS
    }
    per_method_mse: dict[str, list[float]] = {
        method: [] for method in METHODS
    }
    per_method_refresh: dict[str, list[float]] = {
        method: [] for method in METHODS
    }
    mse_differences: list[float] = []
    individual_pairs = []
    for artifact in cell:
        ours = artifact["methods"]["ours"]
        refact = artifact["methods"]["refact"]
        ours_core = core_ms_per_call(ours)
        refact_core = core_ms_per_call(refact)
        app_ratio = (
            float(refact["timing"]["application_step_seconds"])
            / float(ours["timing"]["application_step_seconds"])
        )
        core_ratio = refact_core / ours_core
        ours_mse = float(ours["final_test_mse"])
        refact_mse = float(refact["final_test_mse"])
        difference = ours_mse - refact_mse
        recorded_difference = float(
            artifact["paired_diagnostics"][
                "final_test_mse_difference_ours_minus_refact"
            ]
        )
        if not math.isclose(
            difference,
            recorded_difference,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise AuditError(
                f"seed {artifact['spec']['seed']} MSE contrast changed"
            )

        app_ratios.append(app_ratio)
        core_ratios.append(core_ratio)
        mse_differences.append(difference)
        for name, method, core in (
            ("ours", ours, ours_core),
            ("refact", refact, refact_core),
        ):
            per_method_core[name].append(core)
            per_method_mse[name].append(float(method["final_test_mse"]))
            per_method_refresh[name].append(
                float(method["refresh_fraction"])
            )
        individual_pairs.append(
            {
                "seed": artifact["spec"]["seed"],
                "method_order": artifact["spec"]["method_order"],
                "application_speedup_refact_over_ours": app_ratio,
                "ours_core_ms_per_nonrefresh_call": ours_core,
                "refact_core_ms_per_nonrefresh_call": refact_core,
                "core_speedup_refact_over_ours": core_ratio,
                "ours_refresh_count": ours["refresh_count"],
                "refact_refresh_count": refact["refresh_count"],
                "ours_test_mse": ours_mse,
                "refact_test_mse": refact_mse,
                "test_mse_difference_ours_minus_refact": difference,
            }
        )

    return {
        "rank": rank,
        "threads": threads,
        "seeds": list(SEEDS),
        "method_order_counts": {
            "ours_first": sum(
                artifact["spec"]["method_order"][0] == "ours"
                for artifact in cell
            ),
            "refact_first": sum(
                artifact["spec"]["method_order"][0] == "refact"
                for artifact in cell
            ),
        },
        "paired_application_speedup_refact_over_ours": _estimate(
            app_ratios
        ),
        "paired_core_speedup_refact_over_ours_per_nonrefresh_call": (
            _estimate(core_ratios)
        ),
        "ours_core_ms_per_nonrefresh_call": _estimate(
            per_method_core["ours"]
        ),
        "refact_core_ms_per_nonrefresh_call": _estimate(
            per_method_core["refact"]
        ),
        "ours_refresh_fraction": _estimate(per_method_refresh["ours"]),
        "refact_refresh_fraction": _estimate(
            per_method_refresh["refact"]
        ),
        "ours_final_test_mse": _estimate(per_method_mse["ours"]),
        "refact_final_test_mse": _estimate(per_method_mse["refact"]),
        "paired_test_mse_difference_ours_minus_refact": _estimate(
            mse_differences
        ),
        "individual_pairs": individual_pairs,
    }


def aggregate(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    lock_path = directory / "source_lock.json"
    summary_path = directory / "confirmation_summary.json"
    lock = _read_json(lock_path)
    locked_summary = _read_json(summary_path)
    lock_sha256 = _sha256(lock_path)
    if lock.get("campaign_tag") != CAMPAIGN:
        raise AuditError("source lock is not the E1 v5 campaign")
    if locked_summary.get("campaign_tag") != CAMPAIGN:
        raise AuditError("locked summary is not the E1 v5 campaign")
    if locked_summary.get("source_lock_sha256") != lock_sha256:
        raise AuditError("locked summary source hash changed")
    if locked_summary.get("artifact_count") != 60:
        raise AuditError("locked summary is incomplete")

    specs = _expected_specs()
    expected_names = {_result_name(spec) for spec in specs}
    paths = sorted(directory.glob("confirmation_pair_*.json"))
    if {path.name for path in paths} != expected_names:
        raise AuditError("missing or extra E1 confirmation result")
    claim_paths = sorted(
        (directory / "confirmation_attempt_claims").glob("*.json")
    )
    if len(claim_paths) != 60:
        raise AuditError("expected exactly 60 exactly-once claims")

    artifacts: list[dict[str, Any]] = []
    for spec in specs:
        artifact = _read_json(directory / _result_name(spec))
        _validate_artifact(artifact, spec, lock, lock_sha256)
        artifacts.append(artifact)

    return {
        "schema_version": 1,
        "experiment": "e1_v5_report_aggregate",
        "campaign_tag": CAMPAIGN,
        "source_lock_sha256": lock_sha256,
        "locked_confirmation_summary_sha256": _sha256(summary_path),
        "artifact_count": len(artifacts),
        "analysis_policy": {
            "fixed_before_confirmation_launch": True,
            "application_ratio": (
                "paired Refact/Ours ratio of complete per-step application "
                "time; refresh and reset work included"
            ),
            "core_normalization": (
                "each method's total non-refresh core seconds divided by "
                "its own steps minus refresh_count; paired Refact/Ours "
                "ratios are then summarized across seeds"
            ),
            "confidence_intervals": (
                "two-sided Student-t 95% intervals across ten paired seeds "
                "with df=9"
            ),
            "test_policy": (
                "one terminal test evaluation per method per pair after "
                "both trainings; no test-based selection"
            ),
        },
        "audit": {
            "all_60_results_present": True,
            "all_60_exactly_once_claims_present": True,
            "all_locked_sources_and_runtimes_match": True,
            "all_runtime_threads_and_affinities_match": True,
            "all_paired_inputs_match": True,
            "all_test_evaluations_exactly_once": True,
            "core_denominators_use_actual_nonrefresh_invocations": True,
        },
        "groups": [
            _aggregate_cell(artifacts, rank, threads)
            for rank in RANKS
            for threads in THREADS
        ],
    }


def _write_once_json(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise AuditError(f"refusing to replace existing {output}") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = aggregate(args.campaign_directory)
    if args.output is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _write_once_json(payload, args.output)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
