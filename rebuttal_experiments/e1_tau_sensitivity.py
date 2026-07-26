"""Source-locked E1 refresh-threshold sensitivity confirmation.

This campaign is deliberately separate from ``e1_confirmation.py``.  It
holds every training choice fixed, runs the optimized positive-update method
at six predeclared refresh thresholds, and treats all thresholds for one
seed/rank as a paired block.  The held-out partition is materialized only
after all six trainings finish, then evaluated exactly once per threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Import this module before importing any numerical package.  It captures
# Slurm's task affinity before PyTorch/OpenMP can narrow the calling thread.
from rebuttal_experiments import e1_confirmation as base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "rebuttal_results" / "e1_tau_sensitivity"
DEFAULT_CAMPAIGN = "e1-tau-sensitivity-v1"
CONFIRMATION_SEEDS = tuple(range(3000, 3010))
RANKS = (256, 1024)
TAUS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
THREADS = 10
REFERENCE_TAU = 0.1

FIXED_CONFIG: dict[str, Any] = {
    "dataset": "UCI Protein protein.mat",
    "split_seed": 42,
    "train_fraction": 0.8,
    "dtype": "float32",
    "method": "ours optimized exact positive update",
    "architecture": [9, 512, 512, 128, 1],
    "epochs": 10,
    "batch_size": 256,
    "learning_rate": 0.1,
    "damping": 1.0,
    "momentum": 0.9,
    "refactor_every": 50,
    "ranks": list(RANKS),
    "refresh_thresholds": list(TAUS),
    "reference_threshold": REFERENCE_TAU,
    "threads": THREADS,
    "confirmation_seeds": list(CONFIRMATION_SEEDS),
    "task_block": (
        "one exclusive task per seed/rank; all six thresholds clone one "
        "byte-identical initial state and use identical permutations"
    ),
    "threshold_order": (
        "six forward cyclic orders followed by four reverse cyclic orders; "
        "each threshold occupies each position once or twice"
    ),
    "test_policy": (
        "all six fixed trainings and pre-test invariants finish before the "
        "held-out tensors are materialized; exactly one terminal test "
        "evaluation per threshold; no selection, early stopping, or retry"
    ),
    "primary_cost_axis": "refresh_count / training_steps",
    "timing_interpretation": (
        "descriptive application-level timing under equal controlled "
        "10-thread execution; not isolated kernel timing"
    ),
    "outcome_adaptive_changes": False,
    "expected_data_partition": base.EXPECTED_DATA_PARTITION,
}

LOCKED_FILES = {
    "driver": "rebuttal_experiments/e1_tau_sensitivity.py",
    "protocol": "rebuttal_experiments/E1_TAU_SENSITIVITY_PROTOCOL.md",
    "prior_outcome_audit": (
        "rebuttal_experiments/E1_TAU_PRIOR_OUTCOME_AUDIT.md"
    ),
    "freeze_sbatch": "rebuttal_experiments/slurm/e1_tau_freeze.sbatch",
    "preflight_sbatch": (
        "rebuttal_experiments/slurm/e1_tau_preflight.sbatch"
    ),
    "confirmation_sbatch": (
        "rebuttal_experiments/slurm/e1_tau_confirmation.sbatch"
    ),
    "paired_e1_infrastructure": "rebuttal_experiments/e1_confirmation.py",
    "ng_reference_driver": "rebuttal_experiments/ng_sweep.py",
    "common_helpers": "rebuttal_experiments/common.py",
    "corrected_data_loader": (
        "Cholesky-main-2/corrected_incremental_cholesky/data_utils.py"
    ),
    "incremental_source": (
        "Cholesky-main-2/corrected_incremental_cholesky/fast_update.cpp"
    ),
    "extension_build": (
        "Cholesky-main-2/corrected_incremental_cholesky/setup_ext.py"
    ),
    "incremental_binary": (
        "Cholesky-main-2/corrected_incremental_cholesky/fast_cholesky.so"
    ),
    "protein_data": "Cholesky-main-2/data/protein.mat",
}
TRACKED_LOCK_KEYS = tuple(
    key for key in LOCKED_FILES if key != "incremental_binary"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tau_key(tau: float) -> str:
    if tau not in TAUS:
        raise ValueError(f"unregistered threshold {tau}")
    return f"{tau:g}"


def _threshold_order(seed_index: int) -> list[float]:
    """Return a predeclared near-balanced order for one paired block."""
    if not 0 <= seed_index < len(CONFIRMATION_SEEDS):
        raise ValueError(f"invalid seed index {seed_index}")
    if seed_index < len(TAUS):
        return [
            TAUS[(seed_index + position) % len(TAUS)]
            for position in range(len(TAUS))
        ]
    start = seed_index - len(TAUS)
    return [
        TAUS[(start - position) % len(TAUS)]
        for position in range(len(TAUS))
    ]


def _task_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(CONFIRMATION_SEEDS):
        for rank_index, rank in enumerate(RANKS):
            specs.append(
                {
                    "phase": "confirmation",
                    "task_index": len(specs),
                    "seed": seed,
                    "seed_index": seed_index,
                    "rank": rank,
                    "rank_index": rank_index,
                    "threads": THREADS,
                    "threshold_order": _threshold_order(seed_index),
                }
            )
    return specs


def _task_spec(task_index: int) -> dict[str, Any]:
    specs = _task_specs()
    if not 0 <= task_index < len(specs):
        raise ValueError(
            f"task index must be in [0, {len(specs) - 1}], got {task_index}"
        )
    return specs[task_index]


def _campaign_dir(args: argparse.Namespace) -> Path:
    return Path(args.results_dir).resolve() / args.campaign_tag


def _lock_path(args: argparse.Namespace) -> Path:
    if args.lock_file:
        return Path(args.lock_file).resolve()
    return _campaign_dir(args) / "source_lock.json"


def _artifact_path(directory: Path, spec: dict[str, Any]) -> Path:
    return directory / (
        f"confirmation_block_r{spec['rank']}_t{THREADS}"
        f"_seed{spec['seed']}.json"
    )


def _claim_path(directory: Path, spec: dict[str, Any]) -> Path:
    return (
        directory
        / "confirmation_attempt_claims"
        / f"{_artifact_path(directory, spec).stem}.claim.json"
    )


def freeze_lock(args: argparse.Namespace) -> dict[str, Any]:
    if args.campaign_tag != DEFAULT_CAMPAIGN:
        raise ValueError(
            f"reviewed campaign tag is frozen as {DEFAULT_CAMPAIGN!r}"
        )
    slurm = base._validate_preflight_slurm()
    thread_report = base.configure_and_verify_threads(THREADS)
    base.torch.use_deterministic_algorithms(True)
    tracked = [LOCKED_FILES[key] for key in TRACKED_LOCK_KEYS]
    base._scoped_git_clean(tracked)
    head = base._git("rev-parse", "HEAD").stdout.strip()
    files: dict[str, Any] = {}
    for key, relative in LOCKED_FILES.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing locked file: {path}")
        files[key] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": base._sha256(path),
            "git_tracked": key in TRACKED_LOCK_KEYS,
        }
    base._finalize_thread_affinity(thread_report)
    payload = {
        "schema_version": 1,
        "experiment": "e1_tau_sensitivity_source_lock",
        "campaign_tag": args.campaign_tag,
        "created_utc": _utc_now(),
        "git_commit": head,
        "fixed_config": FIXED_CONFIG,
        "task_count": len(_task_specs()),
        "runtime": base._runtime_manifest(),
        "freeze_environment": {
            "slurm": slurm,
            "thread_verification": thread_report,
        },
        "files": files,
    }
    output = _lock_path(args)
    try:
        output.relative_to(ROOT)
    except ValueError as error:
        raise RuntimeError("source lock must be inside the repository") from error
    base._write_once_json(payload, output)
    print(f"wrote immutable source lock {output}", flush=True)
    return payload


def _verify_lock(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str, str]:
    path = _lock_path(args)
    lock = base._read_json(path)
    if lock.get("schema_version") != 1:
        raise RuntimeError("unsupported or missing source-lock schema")
    if lock.get("campaign_tag") != DEFAULT_CAMPAIGN:
        raise RuntimeError("campaign tag differs from reviewed campaign")
    if lock.get("fixed_config") != FIXED_CONFIG:
        raise RuntimeError("driver configuration differs from source lock")
    if lock.get("task_count") != len(_task_specs()):
        raise RuntimeError("task count differs from source lock")
    locked_commit = lock.get("git_commit")
    if not isinstance(locked_commit, str):
        raise RuntimeError("source lock omitted git commit")
    current_head = base._git("rev-parse", "HEAD").stdout.strip()
    if (
        base._git(
            "merge-base",
            "--is-ancestor",
            locked_commit,
            current_head,
            check=False,
        ).returncode
        != 0
    ):
        raise RuntimeError("locked commit is not an ancestor of run HEAD")
    relative_lock = str(path.relative_to(ROOT))
    base._scoped_git_clean(
        [*[LOCKED_FILES[key] for key in TRACKED_LOCK_KEYS], relative_lock]
    )
    records = lock.get("files")
    if not isinstance(records, dict):
        raise RuntimeError("source lock omitted file manifest")
    for key, relative in LOCKED_FILES.items():
        record = records.get(key)
        check_path = ROOT / relative
        if not isinstance(record, dict) or record.get("path") != relative:
            raise RuntimeError(f"source-lock path mismatch for {key}")
        if check_path.stat().st_size != record.get("bytes"):
            raise RuntimeError(f"locked byte size changed for {relative}")
        if base._sha256(check_path) != record.get("sha256"):
            raise RuntimeError(f"locked SHA-256 changed for {relative}")
    if lock.get("runtime") != base._runtime_manifest():
        raise RuntimeError("runtime differs from source lock")
    return lock, base._sha256(path), current_head


def _require_preflight(
    args: argparse.Namespace,
    lock_sha256: str,
) -> dict[str, Any]:
    path = _campaign_dir(args) / "preflight_receipt.json"
    if not path.is_file():
        raise RuntimeError("confirmation requires a completed preflight")
    receipt = base._read_json(path)
    if receipt.get("schema_version") != 1:
        raise RuntimeError("preflight receipt schema mismatch")
    if (
        receipt.get("experiment") != "e1_tau_sensitivity_preflight"
        or receipt.get("campaign_tag") != DEFAULT_CAMPAIGN
        or receipt.get("source_lock_sha256") != lock_sha256
        or receipt.get("passed") is not True
        or receipt.get("prior_outcomes_or_claims_present") is not False
    ):
        raise RuntimeError("preflight receipt does not approve this lock")
    lock = base._read_json(_lock_path(args))
    if (
        receipt.get("locked_git_commit") != lock["git_commit"]
        or receipt.get("runtime") != lock["runtime"]
        or receipt.get("data") != base.EXPECTED_DATA_PARTITION
    ):
        raise RuntimeError("preflight receipt provenance mismatch")
    run_commit = receipt.get("run_git_commit")
    if not isinstance(run_commit, str) or (
        base._git(
            "merge-base",
            "--is-ancestor",
            run_commit,
            "HEAD",
            check=False,
        ).returncode
        != 0
    ):
        raise RuntimeError("preflight run commit is not an ancestor of HEAD")
    probes = receipt.get("kernel_probes")
    if (
        not isinstance(probes, list)
        or [probe.get("rank") for probe in probes] != list(RANKS)
        or not all(probe.get("passed") is True for probe in probes)
    ):
        raise RuntimeError("preflight kernel probes are incomplete")
    runtime = receipt["thread_verification"]["runtime_values"]
    thread_report = receipt["thread_verification"]
    if not (
        runtime["aten_intraop"]
        == runtime["openmp_max"]
        == runtime["mkl_max"]
        == THREADS
        and runtime["aten_interop"] == 1
        and len(thread_report["affinity_before_numeric_import"]) == THREADS
        and thread_report["allocation_unique_physical_cores"] == THREADS
        and thread_report["worker_affinity_union"]
        == thread_report["selected_cpus"]
    ):
        raise RuntimeError("preflight thread/affinity audit mismatch")
    slurm = receipt["slurm"]
    if not (
        slurm["account"] == base.EXPECTED_SLURM_ACCOUNT
        and slurm["partition"] == base.EXPECTED_SLURM_PARTITION
        and slurm["cpus_per_task"] == str(THREADS)
        and slurm["ntasks"] == "1"
        and slurm["job_num_nodes"] == "1"
        and slurm["restart_count"] == "0"
        and slurm["exclusive_allocation_verified"]
    ):
        raise RuntimeError("preflight Slurm audit mismatch")
    base._scoped_git_clean([str(path.relative_to(ROOT))])
    return {
        "path": str(path),
        "sha256": base._sha256(path),
        "payload": receipt,
    }


def _claim_confirmation(
    directory: Path,
    spec: dict[str, Any],
    lock_sha256: str,
    preflight: dict[str, Any],
    run_git_commit: str,
    slurm: dict[str, Any],
) -> dict[str, Any]:
    claim = {
        "schema_version": 1,
        "experiment": "e1_tau_confirmation_exactly_once_claim",
        "campaign_tag": DEFAULT_CAMPAIGN,
        "created_utc": _utc_now(),
        "spec": spec,
        "source_lock_sha256": lock_sha256,
        "preflight_sha256": preflight["sha256"],
        "run_git_commit": run_git_commit,
        "slurm": slurm,
        "policy": (
            "This seed/rank block may never be retried. If it fails before "
            "publishing, retire the campaign rather than risk a repeated "
            "held-out evaluation."
        ),
    }
    path = _claim_path(directory, spec)
    try:
        base._write_once_json(claim, path)
    except FileExistsError as error:
        raise RuntimeError(
            f"confirmation block already claimed and cannot retry: {path}"
        ) from error
    return {"path": str(path), "sha256": base._sha256(path), "payload": claim}


def _run_threshold(
    tau: float,
    rank: int,
    X_train: Any,
    y_train: Any,
    initial: dict[str, Any],
    permutations: list[Any],
    input_hashes: dict[str, str],
    fast_cholesky: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    original = base.FIXED_CONFIG["refresh_threshold"]
    if original != REFERENCE_TAU:
        raise RuntimeError("shared E1 configuration was unexpectedly mutated")
    print(f"starting fixed threshold tau={tau:g} rank={rank}", flush=True)
    try:
        base.FIXED_CONFIG["refresh_threshold"] = tau
        result, private = base._run_method(
            "ours",
            "confirmation",
            rank,
            X_train,
            y_train,
            initial,
            permutations,
            input_hashes,
            fast_cholesky,
        )
    finally:
        base.FIXED_CONFIG["refresh_threshold"] = original
    result["refresh_threshold"] = tau
    return result, private


def _pre_test_invariants(
    results: dict[str, dict[str, Any]],
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    expected_keys = {_tau_key(tau) for tau in TAUS}
    if set(results) != expected_keys:
        raise RuntimeError("threshold block is incomplete before test access")
    steps = {result["steps"] for result in results.values()}
    dtypes = {result["dtype"] for result in results.values()}
    initial_hashes = {
        result["initial_state_sha256"] for result in results.values()
    }
    permutation_hashes = {
        result["permutations_sha256"] for result in results.values()
    }
    test_counts = {result["test_evaluations"] for result in results.values()}
    checks = {
        "all_six_thresholds_finished": len(results) == len(TAUS),
        "same_initial_state_hash": initial_hashes
        == {input_hashes["initial_state_sha256"]},
        "same_permutation_hash": permutation_hashes
        == {input_hashes["permutations_sha256"]},
        "same_step_count": len(steps) == 1,
        "same_declared_precision": dtypes == {"torch.float32"},
        "zero_test_evaluations_before_terminal_pass": test_counts == {0},
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"pre-test paired invariants failed: {failed}")
    return checks


def _evaluate_test_once(
    order: list[float],
    results: dict[str, dict[str, Any]],
    private: dict[str, dict[str, Any]],
    X_test: Any,
    y_test: Any,
) -> None:
    for tau in order:
        key = _tau_key(tau)
        if results[key]["test_evaluations"] != 0:
            raise RuntimeError(f"tau={tau:g} accessed test before terminal pass")
        start = time.perf_counter()
        with base.torch.no_grad():
            prediction = private[key]["model"](X_test).squeeze(-1)
            mse = float(
                base.torch.nn.functional.mse_loss(prediction, y_test).item()
            )
        if not math.isfinite(mse):
            raise RuntimeError(f"tau={tau:g} produced non-finite test MSE")
        results[key]["final_test_mse"] = mse
        results[key]["test_evaluations"] = 1
        results[key]["timing"]["test_evaluation_seconds"] = (
            time.perf_counter() - start
        )
        private[key]["test_prediction"] = prediction


def _cross_threshold_diagnostics(
    results: dict[str, dict[str, Any]],
    private: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    reference_key = _tau_key(REFERENCE_TAU)
    reference_result = results[reference_key]
    reference_private = private[reference_key]
    output: dict[str, Any] = {}
    for tau in TAUS:
        key = _tau_key(tau)
        result = results[key]
        hidden = private[key]
        refresh_mismatches = sum(
            left != right
            for left, right in zip(
                hidden["refresh_trace"],
                reference_private["refresh_trace"],
                strict=True,
            )
        )
        output[key] = {
            "tau": tau,
            "reference_tau": REFERENCE_TAU,
            "refresh_trace_mismatch_count": refresh_mismatches,
            "refresh_trace_mismatch_fraction": (
                refresh_mismatches / result["steps"]
            ),
            "final_model_relative_error_vs_reference": base._relative_error(
                hidden["model_vector"], reference_private["model_vector"]
            ),
            "final_train_prediction_relative_error_vs_reference": (
                base._relative_error(
                    hidden["train_prediction"],
                    reference_private["train_prediction"],
                )
            ),
            "final_test_prediction_relative_error_vs_reference": (
                base._relative_error(
                    hidden["test_prediction"],
                    reference_private["test_prediction"],
                )
            ),
            "final_train_mse_difference_vs_reference": (
                result["final_train_mse"]
                - reference_result["final_train_mse"]
            ),
            "final_test_mse_difference_vs_reference": (
                result["final_test_mse"]
                - reference_result["final_test_mse"]
            ),
            "refresh_fraction_difference_vs_reference": (
                result["refresh_fraction"]
                - reference_result["refresh_fraction"]
            ),
        }
    return output


def run_confirmation(args: argparse.Namespace) -> dict[str, Any]:
    if args.task_index is None:
        raise ValueError("--task-index is required")
    spec = _task_spec(args.task_index)
    lock, lock_sha256, run_git_commit = _verify_lock(args)
    slurm = base._validate_slurm_task(spec)
    thread_report = base.configure_and_verify_threads(THREADS)
    base.torch.use_deterministic_algorithms(True)
    preflight = _require_preflight(args, lock_sha256)
    directory = _campaign_dir(args)
    claim = _claim_confirmation(
        directory,
        spec,
        lock_sha256,
        preflight,
        run_git_commit,
        slurm,
    )

    import fast_cholesky

    extension = Path(fast_cholesky.__file__).resolve()
    expected_extension = (ROOT / LOCKED_FILES["incremental_binary"]).resolve()
    if extension != expected_extension:
        raise RuntimeError(f"loaded {extension}, expected {expected_extension}")

    block_start = time.perf_counter()
    data_start = time.perf_counter()
    X_train, y_train, data_metadata, preprocessing = (
        base._load_training_partition()
    )
    data_load_seconds = time.perf_counter() - data_start
    setup_start = time.perf_counter()
    initial, permutations, input_hashes = base._prepare_initial_state(
        X_train, spec["seed"], spec["rank"]
    )
    setup_seconds = time.perf_counter() - setup_start
    kernel_probe = base._kernel_equivalence_probe(
        spec["rank"], fast_cholesky
    )
    warmup_seconds = base._warm_up_model(X_train, y_train, initial)

    results: dict[str, dict[str, Any]] = {}
    private: dict[str, dict[str, Any]] = {}
    for tau in spec["threshold_order"]:
        key = _tau_key(tau)
        results[key], private[key] = _run_threshold(
            tau,
            spec["rank"],
            X_train,
            y_train,
            initial,
            permutations,
            input_hashes,
            fast_cholesky,
        )
    pre_test_checks = _pre_test_invariants(results, input_hashes)

    test_start = time.perf_counter()
    X_test, y_test = base._materialize_confirmation_test(
        preprocessing, data_metadata
    )
    test_materialization_seconds = time.perf_counter() - test_start
    _evaluate_test_once(
        spec["threshold_order"], results, private, X_test, y_test
    )
    diagnostics = _cross_threshold_diagnostics(results, private)
    base._finalize_thread_affinity(thread_report)

    payload = {
        "schema_version": 1,
        "experiment": "e1_tau_sensitivity_confirmation",
        "campaign_tag": args.campaign_tag,
        "created_utc": _utc_now(),
        "spec": spec,
        "fixed_config": FIXED_CONFIG,
        "data": {
            "path": str(base.DATA_PATH.resolve()),
            "sha256": lock["files"]["protein_data"]["sha256"],
            **data_metadata,
            "normalization": "fit on fixed training partition only",
            "split_seed": FIXED_CONFIG["split_seed"],
            "heldout_tensors_materialized_after_all_trainings": True,
        },
        "paired_inputs": {
            **input_hashes,
            "identical_data_object": True,
            "identical_model_and_curvature_clones": True,
            "identical_minibatch_permutations": True,
        },
        "source_lock": {
            "path": str(_lock_path(args)),
            "sha256": lock_sha256,
            "locked_git_commit": lock["git_commit"],
            "run_git_commit": run_git_commit,
            "files": lock["files"],
            "runtime": lock["runtime"],
        },
        "implementation": {
            "incremental_extension": str(extension),
            "incremental_dispatch": "metric_sign=1, operator_sign=1",
            "incremental_kernel": (
                "row-major virtual-padding contiguous SIMD positive path"
            ),
            "kernel_equivalence_probe": kernel_probe,
        },
        "threshold_results": results,
        "cross_threshold_diagnostics": diagnostics,
        "pre_test_invariants": pre_test_checks,
        "test_access": {
            "per_threshold_evaluations": 1,
            "block_total_evaluations": len(TAUS),
            "selection_using_test_values": False,
            "test_used_for_early_stopping": False,
            "terminal_after_all_six_trainings": True,
        },
        "timing_outside_measured_training": {
            "data_load_seconds": data_load_seconds,
            "paired_initialization_seconds": setup_seconds,
            "model_warmup_seconds": warmup_seconds,
            "test_materialization_seconds": test_materialization_seconds,
            "block_total_seconds": time.perf_counter() - block_start,
        },
        "preflight": preflight,
        "confirmation_exactly_once_claim": claim,
        "environment": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": base.torch.__version__,
            "torch_config": base.torch.__config__.show(),
            "runtime_lock": lock["runtime"],
            "deterministic_algorithms": (
                base.torch.are_deterministic_algorithms_enabled()
            ),
            "thread_verification": thread_report,
            "slurm": slurm,
        },
    }
    output = _artifact_path(directory, spec)
    base._write_once_json(payload, output)
    print(f"wrote immutable threshold block {output}", flush=True)
    return payload


def _validate_artifact(
    payload: dict[str, Any],
    spec: dict[str, Any],
    directory: Path,
    lock: dict[str, Any],
    lock_sha256: str,
    preflight_sha256: str,
) -> None:
    if payload.get("schema_version") != 1:
        raise RuntimeError("artifact schema mismatch")
    if payload.get("campaign_tag") != DEFAULT_CAMPAIGN:
        raise RuntimeError("artifact campaign mismatch")
    if payload.get("spec") != spec:
        raise RuntimeError("artifact task mapping mismatch")
    if payload.get("fixed_config") != FIXED_CONFIG:
        raise RuntimeError("artifact fixed configuration mismatch")
    source = payload["source_lock"]
    if (
        source["path"] != str(directory / "source_lock.json")
        or source["sha256"] != lock_sha256
        or source["locked_git_commit"] != lock["git_commit"]
        or source["files"] != lock["files"]
        or source["runtime"] != lock["runtime"]
    ):
        raise RuntimeError("artifact source-lock hash mismatch")
    if (
        base._git(
            "merge-base",
            "--is-ancestor",
            source["run_git_commit"],
            "HEAD",
            check=False,
        ).returncode
        != 0
    ):
        raise RuntimeError("artifact run commit is not an ancestor of HEAD")
    data = payload["data"]
    for key, expected in base.EXPECTED_DATA_PARTITION.items():
        if data.get(key) != expected:
            raise RuntimeError(f"artifact data manifest mismatch for {key}")
    if (
        data.get("sha256") != lock["files"]["protein_data"]["sha256"]
        or data.get("heldout_tensors_materialized_after_all_trainings")
        is not True
    ):
        raise RuntimeError("artifact data provenance/test timing mismatch")
    claim = payload["confirmation_exactly_once_claim"]
    expected_claim_path = _claim_path(directory, spec)
    if (
        Path(claim["path"]) != expected_claim_path
        or claim["payload"]["spec"] != spec
        or claim["payload"]["source_lock_sha256"] != lock_sha256
        or claim["payload"]["preflight_sha256"] != preflight_sha256
    ):
        raise RuntimeError("artifact exactly-once claim mismatch")
    if base._sha256(Path(claim["path"])) != claim["sha256"]:
        raise RuntimeError("artifact claim bytes changed")
    if base._read_json(Path(claim["path"])) != claim["payload"]:
        raise RuntimeError("embedded claim differs from write-once claim")
    if payload["preflight"]["sha256"] != preflight_sha256:
        raise RuntimeError("artifact preflight mismatch")
    if (
        payload["preflight"]["path"]
        != str(directory / "preflight_receipt.json")
        or base._read_json(Path(payload["preflight"]["path"]))
        != payload["preflight"]["payload"]
    ):
        raise RuntimeError("embedded preflight differs from committed receipt")
    expected_keys = {_tau_key(tau) for tau in TAUS}
    results = payload["threshold_results"]
    if set(results) != expected_keys:
        raise RuntimeError("artifact threshold grid is incomplete")
    expected_steps = (
        FIXED_CONFIG["epochs"]
        * (
            base.EXPECTED_DATA_PARTITION["train_examples"]
            // FIXED_CONFIG["batch_size"]
        )
    )
    for tau in TAUS:
        key = _tau_key(tau)
        result = results[key]
        if result["refresh_threshold"] != tau:
            raise RuntimeError(f"artifact mislabeled threshold {tau}")
        if result["steps"] != expected_steps:
            raise RuntimeError("artifact training step count mismatch")
        if result["test_evaluations"] != 1:
            raise RuntimeError("artifact did not test exactly once")
        if not math.isfinite(result["final_test_mse"]):
            raise RuntimeError("artifact test MSE is non-finite")
        if result["refresh_count"] != sum(result["raw_step_samples"]["refreshed"]):
            raise RuntimeError("artifact refresh count does not match raw trace")
        if result["refresh_fraction"] != result["refresh_count"] / expected_steps:
            raise RuntimeError("artifact refresh fraction is inconsistent")
        for name, values in result["raw_step_samples"].items():
            if len(values) != expected_steps:
                raise RuntimeError(
                    f"artifact tau={tau:g} raw field {name} has wrong length"
                )
        if (
            result["initial_state_sha256"]
            != payload["paired_inputs"]["initial_state_sha256"]
            or result["permutations_sha256"]
            != payload["paired_inputs"]["permutations_sha256"]
        ):
            raise RuntimeError("artifact paired input hashes differ")
    if not all(payload["pre_test_invariants"].values()):
        raise RuntimeError("artifact pre-test invariant is false")
    if payload["test_access"] != {
        "per_threshold_evaluations": 1,
        "block_total_evaluations": len(TAUS),
        "selection_using_test_values": False,
        "test_used_for_early_stopping": False,
        "terminal_after_all_six_trainings": True,
    }:
        raise RuntimeError("artifact test-access policy mismatch")
    runtime = payload["environment"]["thread_verification"]["runtime_values"]
    if not (
        runtime["aten_intraop"]
        == runtime["openmp_max"]
        == runtime["mkl_max"]
        == THREADS
        and runtime["aten_interop"] == 1
    ):
        raise RuntimeError("artifact runtime thread counts mismatch")
    thread_report = payload["environment"]["thread_verification"]
    if (
        len(thread_report["affinity_before_numeric_import"]) != THREADS
        or thread_report["allocation_unique_physical_cores"] != THREADS
        or thread_report["worker_affinity_union"]
        != thread_report["selected_cpus"]
        or len(thread_report["selected_cpus"]) != THREADS
    ):
        raise RuntimeError("artifact CPU affinity audit failed")
    slurm = payload["environment"]["slurm"]
    if (
        slurm["account"] != base.EXPECTED_SLURM_ACCOUNT
        or slurm["partition"] != base.EXPECTED_SLURM_PARTITION
        or slurm["ntasks"] != "1"
        or slurm["job_num_nodes"] != "1"
        or not slurm["exclusive_allocation_verified"]
        or slurm["restart_count"] != "0"
    ):
        raise RuntimeError("artifact Slurm audit failed")
    if not payload["implementation"]["kernel_equivalence_probe"]["passed"]:
        raise RuntimeError("artifact kernel probe failed")
    if (
        payload["implementation"]["incremental_extension"]
        != str((ROOT / LOCKED_FILES["incremental_binary"]).resolve())
        or payload["environment"]["runtime_lock"] != lock["runtime"]
    ):
        raise RuntimeError("artifact implementation/runtime provenance mismatch")


def _mean_sd(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
        "values": values,
    }


def _paired_95_ci(values: list[float]) -> dict[str, Any]:
    if len(values) != len(CONFIRMATION_SEEDS):
        raise RuntimeError("paired CI requires all ten frozen seeds")
    center = statistics.mean(values)
    sample_sd = statistics.stdev(values)
    half_width = 2.2621571628540993 * sample_sd / math.sqrt(len(values))
    return {
        "mean": center,
        "sample_sd": sample_sd,
        "student_t_df": 9,
        "confidence_level": 0.95,
        "ci_low": center - half_width,
        "ci_high": center + half_width,
        "values": values,
    }


def aggregate_confirmation(args: argparse.Namespace) -> dict[str, Any]:
    lock, lock_sha256, _ = _verify_lock(args)
    preflight = _require_preflight(args, lock_sha256)
    directory = _campaign_dir(args)
    payloads = []
    expected_artifacts = {_artifact_path(directory, spec) for spec in _task_specs()}
    actual_artifacts = set(directory.glob("confirmation_block_*.json"))
    if actual_artifacts != expected_artifacts:
        raise RuntimeError("missing or extra confirmation artifacts")
    expected_claims = {_claim_path(directory, spec) for spec in _task_specs()}
    actual_claims = set(
        (directory / "confirmation_attempt_claims").glob("*.claim.json")
    )
    if actual_claims != expected_claims:
        raise RuntimeError("missing or extra confirmation claims")
    for spec in _task_specs():
        payload = base._read_json(_artifact_path(directory, spec))
        _validate_artifact(
            payload,
            spec,
            directory,
            lock,
            lock_sha256,
            preflight["sha256"],
        )
        payloads.append(payload)

    groups = []
    reference_key = _tau_key(REFERENCE_TAU)
    for rank in RANKS:
        rank_payloads = sorted(
            (item for item in payloads if item["spec"]["rank"] == rank),
            key=lambda item: item["spec"]["seed"],
        )
        if len(rank_payloads) != len(CONFIRMATION_SEEDS):
            raise RuntimeError("rank group omitted a frozen seed")
        threshold_rows = []
        for tau in TAUS:
            key = _tau_key(tau)
            records = [item["threshold_results"][key] for item in rank_payloads]
            reference_records = [
                item["threshold_results"][reference_key]
                for item in rank_payloads
            ]
            threshold_rows.append(
                {
                    "rank": rank,
                    "tau": tau,
                    "seeds": list(CONFIRMATION_SEEDS),
                    "refresh_fraction": _mean_sd(
                        [record["refresh_fraction"] for record in records]
                    ),
                    "refresh_count": _mean_sd(
                        [float(record["refresh_count"]) for record in records]
                    ),
                    "final_train_mse": _mean_sd(
                        [record["final_train_mse"] for record in records]
                    ),
                    "final_test_mse": _mean_sd(
                        [record["final_test_mse"] for record in records]
                    ),
                    "application_step_seconds": _mean_sd(
                        [
                            record["timing"]["application_step_seconds"]
                            for record in records
                        ]
                    ),
                    "training_wall_seconds": _mean_sd(
                        [
                            record["timing"]["training_wall_seconds"]
                            for record in records
                        ]
                    ),
                    "refresh_seconds": _mean_sd(
                        [
                            record["timing"]["component_totals"][
                                "refresh_seconds"
                            ]
                            for record in records
                        ]
                    ),
                    "nonrefresh_core_seconds": _mean_sd(
                        [
                            record["timing"]["component_totals"][
                                "nonrefresh_core_seconds"
                            ]
                            for record in records
                        ]
                    ),
                    "reset_seconds": _mean_sd(
                        [
                            record["timing"]["component_totals"][
                                "reset_seconds"
                            ]
                            for record in records
                        ]
                    ),
                    "paired_test_mse_difference_vs_tau_0.1": _paired_95_ci(
                        [
                            record["final_test_mse"]
                            - reference["final_test_mse"]
                            for record, reference in zip(
                                records, reference_records, strict=True
                            )
                        ]
                    ),
                    "paired_refresh_fraction_difference_vs_tau_0.1": (
                        _paired_95_ci(
                            [
                                record["refresh_fraction"]
                                - reference["refresh_fraction"]
                                for record, reference in zip(
                                    records,
                                    reference_records,
                                    strict=True,
                                )
                            ]
                        )
                    ),
                    "individual_runs": [
                        {
                            "seed": item["spec"]["seed"],
                            "order_position": item["spec"][
                                "threshold_order"
                            ].index(tau),
                            "refresh_count": item["threshold_results"][key][
                                "refresh_count"
                            ],
                            "refresh_fraction": item["threshold_results"][key][
                                "refresh_fraction"
                            ],
                            "final_train_mse": item["threshold_results"][key][
                                "final_train_mse"
                            ],
                            "final_test_mse": item["threshold_results"][key][
                                "final_test_mse"
                            ],
                            "application_step_seconds": item[
                                "threshold_results"
                            ][key]["timing"]["application_step_seconds"],
                            "training_wall_seconds": item[
                                "threshold_results"
                            ][key]["timing"]["training_wall_seconds"],
                        }
                        for item in rank_payloads
                    ],
                }
            )
        groups.append({"rank": rank, "thresholds": threshold_rows})

    payload = {
        "schema_version": 1,
        "experiment": "e1_tau_sensitivity_confirmation_summary",
        "campaign_tag": args.campaign_tag,
        "created_utc": _utc_now(),
        "source_lock_sha256": lock_sha256,
        "preflight_sha256": preflight["sha256"],
        "fixed_config": FIXED_CONFIG,
        "artifact_count": len(payloads),
        "groups": groups,
        "interpretation": {
            "primary_cost_axis": "refresh_fraction",
            "wall_time": (
                "descriptive application-level 10-thread timing, not a "
                "kernel comparison"
            ),
            "test_policy": (
                "all predeclared cells are reported; test values selected "
                "neither thresholds nor checkpoints"
            ),
        },
        "audit": {
            "all_20_expected_blocks_present": True,
            "all_20_exactly_once_claims_present": True,
            "all_120_fixed_runs_present": True,
            "all_test_evaluations_exactly_once": True,
            "all_test_evaluations_terminal_after_block_training": True,
            "all_paired_input_hashes_equal_within_block": True,
            "all_runtime_thread_counts_verified": True,
            "all_affinities_exact": True,
            "all_source_and_runtime_hashes_verified": True,
            "no_extra_artifacts_or_claims": True,
        },
    }
    output = directory / "confirmation_summary.json"
    base._write_once_json(payload, output)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"wrote immutable summary {output}", flush=True)
    return payload


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    self_test()
    lock, lock_sha256, run_git_commit = _verify_lock(args)
    slurm = base._validate_preflight_slurm()
    thread_report = base.configure_and_verify_threads(THREADS)
    base.torch.use_deterministic_algorithms(True)
    directory = _campaign_dir(args)
    existing = [
        *directory.glob("confirmation_block_*.json"),
        *directory.glob("confirmation_attempt_claims/*.json"),
        *directory.glob("confirmation_summary.json"),
    ]
    if existing:
        raise RuntimeError(f"preflight found prior outcomes/claims: {existing}")

    import fast_cholesky

    extension = Path(fast_cholesky.__file__).resolve()
    expected_extension = (ROOT / LOCKED_FILES["incremental_binary"]).resolve()
    if extension != expected_extension:
        raise RuntimeError(f"loaded {extension}, expected {expected_extension}")
    X_train, y_train, data_metadata, _ = base._load_training_partition()
    if not base.torch.isfinite(X_train).all():
        raise RuntimeError("training features contain non-finite values")
    if not base.torch.isfinite(y_train).all():
        raise RuntimeError("training labels contain non-finite values")
    probes = [
        base._kernel_equivalence_probe(rank, fast_cholesky)
        for rank in RANKS
    ]
    base._finalize_thread_affinity(thread_report)
    payload = {
        "schema_version": 1,
        "experiment": "e1_tau_sensitivity_preflight",
        "campaign_tag": args.campaign_tag,
        "created_utc": _utc_now(),
        "source_lock_sha256": lock_sha256,
        "locked_git_commit": lock["git_commit"],
        "run_git_commit": run_git_commit,
        "runtime": lock["runtime"],
        "slurm": slurm,
        "thread_verification": thread_report,
        "data": data_metadata,
        "kernel_probes": probes,
        "prior_outcomes_or_claims_present": False,
        "passed": True,
    }
    output = directory / "preflight_receipt.json"
    base._write_once_json(payload, output)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"wrote immutable preflight receipt {output}", flush=True)
    return payload


def self_test() -> None:
    specs = _task_specs()
    if len(specs) != 20:
        raise AssertionError("expected 20 seed/rank blocks")
    if [spec["task_index"] for spec in specs] != list(range(20)):
        raise AssertionError("task indices are not contiguous")
    for rank in RANKS:
        group = [spec for spec in specs if spec["rank"] == rank]
        if [spec["seed"] for spec in group] != list(CONFIRMATION_SEEDS):
            raise AssertionError("rank group seed mapping changed")
        for tau in TAUS:
            position_counts = [
                sum(spec["threshold_order"][position] == tau for spec in group)
                for position in range(len(TAUS))
            ]
            if any(count not in (1, 2) for count in position_counts):
                raise AssertionError("threshold order position is imbalanced")
    if any(
        set(spec["threshold_order"]) != set(TAUS) for spec in specs
    ):
        raise AssertionError("a task does not contain the exact threshold grid")
    old_outcome_roots = (
        ROOT / "rebuttal_results" / "ng",
        ROOT / "rebuttal_results" / "e1_confirmation",
    )
    for seed in CONFIRMATION_SEEDS:
        for outcome_root in old_outcome_roots:
            if outcome_root.exists() and list(outcome_root.rglob(f"*seed{seed}*.json")):
                raise AssertionError(
                    f"confirmation seed {seed} has a prior E1 outcome"
                )
    for relative in (
        LOCKED_FILES["freeze_sbatch"],
        LOCKED_FILES["preflight_sbatch"],
        LOCKED_FILES["confirmation_sbatch"],
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        required = (
            "#SBATCH --account=marlowe-m000151-pm06",
            "#SBATCH --partition=batch",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=10",
            "#SBATCH --exclusive",
            "#SBATCH --hint=nomultithread",
            "#SBATCH --no-requeue",
            "PYTHONNOUSERSITE=1",
            "OMP_NUM_THREADS=10",
            "MKL_NUM_THREADS=10",
            "OPENBLAS_NUM_THREADS=10",
            "E1_PROCESS_CONCURRENCY=1",
            "E1_EXCLUSIVE_NODE=1",
            "SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf",
            "/cm/shared/apps/slurm/current/bin:${PATH}",
            "exec /cm/shared/apps/slurm/current/bin/srun",
            "--kill-on-bad-exit=1",
        )
        for line in required:
            if line not in text:
                raise AssertionError(f"{relative} omitted {line!r}")
        if "#SBATCH --requeue" in text or "--distribution" in text:
            raise AssertionError(f"{relative} enables an unreviewed launch mode")
    confirmation_text = (
        ROOT / LOCKED_FILES["confirmation_sbatch"]
    ).read_text(encoding="utf-8")
    if "#SBATCH --array=0-19%10" not in confirmation_text:
        raise AssertionError("confirmation array mapping changed")
    print("E1 tau-sensitivity self-test passed", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        required=True,
        choices=(
            "plan",
            "self-test",
            "freeze-lock",
            "preflight",
            "run-confirmation",
            "aggregate-confirmation",
        ),
    )
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--campaign-tag", default=DEFAULT_CAMPAIGN)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--lock-file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "plan":
        print(
            json.dumps(
                {"fixed_config": FIXED_CONFIG, "tasks": _task_specs()},
                indent=2,
                sort_keys=True,
            )
        )
    elif args.action == "self-test":
        self_test()
    elif args.action == "freeze-lock":
        freeze_lock(args)
    elif args.action == "preflight":
        preflight(args)
    elif args.action == "run-confirmation":
        run_confirmation(args)
    elif args.action == "aggregate-confirmation":
        aggregate_confirmation(args)
    else:
        raise AssertionError(f"unhandled action {args.action}")


if __name__ == "__main__":
    main()
