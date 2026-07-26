"""Run the frozen untouched-seed E3 drag fairness confirmation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typing import Any, Callable

import torch

from .common import ROOT, configure_threads
from .e3_metric_only_pilot import (
    DENSE_PATHS,
    REQUIRED_THREAD_ENV,
    THIN_PATHS,
    aggregate_rows,
    atomic_json_dump,
    first_cpu_model,
    relative_path,
    scalar_stats,
    sha256,
    time_cases,
    validate_outputs,
    whiten,
)


PROTOCOL_PATH = Path(__file__).with_name(
    "e3_drag_fairness_confirmation_protocol.json"
)
CAMPAIGN_TAG = "e3-drag-fair-confirm-v1"
SEEDS = tuple(range(76001, 76011))
RANKS_AND_WIDTHS = {
    256: (1, 4, 8, 16, 32, 64, 128, 256, 512),
    1024: (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048),
}
WARMUPS = 5
REPEATS = 30
TOLERANCE = 1e-11
EXPECTED_ACCOUNT = "marlowe-m000151-pm06"
EXPECTED_PARTITION = "batch"
DEVELOPMENT_RESULT_SHA256 = (
    "247ac9399769633450d5dbd3b0d80b29b4a3806f8a4001060b22179e5fd94292"
)
DEVELOPMENT_AUDIT_SHA256 = (
    "89277f779f050efb3bb289277016fe21d010c20b7b5ccf314562d516d8f399c5"
)


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def validate_protocol(protocol: dict[str, Any]) -> None:
    expected = {
        "campaign_tag": CAMPAIGN_TAG,
        "status": "frozen_after_development_before_confirmation",
        "dtype": "float64",
        "threads": 1,
        "seeds": list(SEEDS),
        "warmups_per_path": WARMUPS,
        "timed_repeats_per_path": REPEATS,
        "ranks_and_widths": {
            str(rank): list(widths)
            for rank, widths in RANKS_AND_WIDTHS.items()
        },
        "relative_error_tolerance": TOLERANCE,
        "retain_all_rows": True,
    }
    actual = {key: protocol.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"E3 fairness confirmation protocol mismatch: "
            f"expected {expected}, got {actual}"
        )
    names = list(protocol.get("paths", {}))
    if names != list(DENSE_PATHS + THIN_PATHS):
        raise ValueError(f"unexpected confirmation path order: {names}")
    for name in DENSE_PATHS:
        if protocol["paths"][name].get("output_contract") != "dense_same_output":
            raise ValueError(f"{name} lacks dense same-output contract")
    for name in THIN_PATHS:
        if protocol["paths"][name].get("output_contract") != "thin_factor_only":
            raise ValueError(f"{name} lacks thin factor-only contract")
    resource = protocol.get("resource_contract", {})
    if (
        resource.get("account") != EXPECTED_ACCOUNT
        or resource.get("partition") != EXPECTED_PARTITION
        or resource.get("qos") != "medium"
        or resource.get("physical_cpus_per_process") != 1
        or not resource.get("exclusive_node")
    ):
        raise ValueError(f"unexpected resource contract: {resource}")
    selection = protocol.get("development_selection", {})
    expected_selection = {
        "pilot_id": "e3-drag-fairness-dev-v1",
        "source_commit": "0050862f8980bf442694be50f1fef1a02829c2a5",
        "result_sha256": DEVELOPMENT_RESULT_SHA256,
        "audit_sha256": DEVELOPMENT_AUDIT_SHA256,
        "audit_status": "passed",
        "selected_update": "one_sided_fair.one_sided_update_",
        "selected_materialization": (
            "single-thread torch.mm(out=...) for every tested width"
        ),
        "custom_materializer_selected": False,
        "confirmation_seeds_evaluated_during_development": False,
    }
    for key, expected_value in expected_selection.items():
        if selection.get(key) != expected_value:
            raise ValueError(
                f"development selection mismatch for {key}: "
                f"expected {expected_value}, got {selection.get(key)}"
            )
    if protocol.get("confirmation_outcome_blinding") is None:
        raise ValueError("confirmation outcome-blinding rule is absent")
    if protocol.get("old_e3_preservation", {}).get("mutation_authorized"):
        raise ValueError("old frozen E3 mutation must remain unauthorized")


def validate_development_gate(protocol: dict[str, Any]) -> None:
    selection = protocol["development_selection"]
    result_path = ROOT / selection["result_path"]
    audit_path = ROOT / selection["audit_path"]
    if sha256(result_path) != DEVELOPMENT_RESULT_SHA256:
        raise ValueError("development result hash changed")
    if sha256(audit_path) != DEVELOPMENT_AUDIT_SHA256:
        raise ValueError("development audit hash changed")
    result = _read(result_path)
    audit = _read(audit_path)
    if (
        result.get("status") != "complete"
        or result.get("report_use") != "development_only_not_reportable"
        or result.get("integrity", {}).get("confirmation_seeds_evaluated")
    ):
        raise ValueError("development result integrity gate failed")
    if (
        audit.get("status") != "passed"
        or audit.get("reserved_confirmation_seeds_evaluated")
        or audit.get("selection", {}).get("update", {}).get("selected")
        != "fair_drag_factor"
        or audit.get("selection", {})
        .get("selected_materializer", {})
        .get("name")
        != "all_mkl"
    ):
        raise ValueError("development audit selection gate failed")
    development_seeds = set(result["protocol"]["seeds"])
    if development_seeds & set(SEEDS):
        raise ValueError("development and confirmation seeds overlap")


def validate_runtime() -> dict[str, Any]:
    required_environment = {
        **{name: "1" for name in REQUIRED_THREAD_ENV},
        "PYTHONNOUSERSITE": "1",
    }
    invalid = {
        name: os.environ.get(name)
        for name, expected in required_environment.items()
        if os.environ.get(name) != expected
    }
    if invalid:
        raise RuntimeError(f"non-hermetic thread environment: {invalid}")
    if os.environ.get("SLURM_JOB_ACCOUNT") != EXPECTED_ACCOUNT:
        raise RuntimeError("confirmation is not on the frozen Slurm account")
    if os.environ.get("SLURM_JOB_PARTITION") != EXPECTED_PARTITION:
        raise RuntimeError("confirmation is not on the batch partition")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 1:
        raise RuntimeError(
            f"confirmation requires exactly one physical CPU: {affinity}"
        )
    configure_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("PyTorch did not retain the one-thread contract")
    declared_commit = os.environ.get("SOURCE_COMMIT")
    actual_commit = _git_output("rev-parse", "HEAD")
    if not declared_commit or declared_commit != actual_commit:
        raise RuntimeError(
            f"source-lock mismatch: declared={declared_commit}, actual={actual_commit}"
        )
    if subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode:
        raise RuntimeError("tracked worktree differs from source commit")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        raise RuntimeError("index differs from source commit")
    return {
        "thread_environment": {
            name: os.environ.get(name) for name in REQUIRED_THREAD_ENV
        },
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "process_affinity": affinity,
        "source_commit": actual_commit,
    }


def load_extensions():
    path = str(PROTOCOL_PATH.parent)
    if path not in sys.path:
        sys.path.insert(0, path)
    import e3_metric_only
    import one_sided_fair

    return e3_metric_only, one_sided_fair


def make_cases(
    *,
    L: torch.Tensor,
    Aw: torch.Tensor,
    z: torch.Tensor,
    zero: torch.Tensor,
    G: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    Vb: torch.Tensor,
    Vc: torch.Tensor,
    e3_metric_only,
    one_sided_fair,
) -> dict[str, tuple[Callable[[], tuple[Any, ...]], Callable[..., Any]]]:
    rank, width = Vb.shape

    def prepare_specialized():
        return (
            L.clone(),
            Aw.clone(),
            z.clone(),
            torch.empty(5 * rank, dtype=L.dtype),
        )

    def execute_specialized(L_work, Aw_work, z_work, workspace):
        return e3_metric_only.metric_only_update_(
            L_work, Aw_work, z_work, workspace
        )

    def prepare_ablation():
        return (
            L.clone(),
            Aw.clone(),
            z.clone(),
            zero,
            torch.empty(5 * rank, dtype=L.dtype),
        )

    def execute_ablation(L_work, Aw_work, z_work, w_work, workspace):
        return e3_metric_only.generic_update_(
            L_work, Aw_work, z_work, w_work, workspace
        )

    def prepare_drag_factor():
        return (
            L.clone(),
            Vb.clone(),
            Vc.clone(),
            z.clone(),
            torch.empty(2 * rank + 2 * width, dtype=L.dtype),
        )

    def execute_drag_factor(L_work, left, right, z_work, workspace):
        return one_sided_fair.one_sided_update_(
            L_work, left, right, z_work, workspace
        )

    def prepare_drag_materialize():
        return (*prepare_drag_factor(), torch.empty_like(Aw))

    def execute_drag_materialize(
        L_work, left, right, z_work, workspace, output
    ):
        one_sided_fair.one_sided_update_(
            L_work, left, right, z_work, workspace
        )
        torch.mm(left, right.mT, out=output)
        return L_work, output

    def prepare_fresh():
        return ()

    def execute_fresh_factor():
        L_new = torch.linalg.cholesky(G + torch.outer(z, z)).contiguous()
        left = torch.linalg.solve_triangular(L_new, B, upper=False)
        right = torch.linalg.solve_triangular(L_new, C, upper=False)
        return L_new, left, right

    def execute_fresh_materialize():
        L_new, left, right = execute_fresh_factor()
        return L_new, left @ right.mT

    return {
        "metric_only_specialized": (
            prepare_specialized,
            execute_specialized,
        ),
        "generic_w0_ablation": (prepare_ablation, execute_ablation),
        "drag_materialize": (
            prepare_drag_materialize,
            execute_drag_materialize,
        ),
        "fresh_materialize": (prepare_fresh, execute_fresh_materialize),
        "drag_factor_only": (prepare_drag_factor, execute_drag_factor),
        "fresh_factor_only": (prepare_fresh, execute_fresh_factor),
    }


def run_cell(
    *,
    rank: int,
    width: int,
    width_index: int,
    seed: int,
    e3_metric_only,
    one_sided_fair,
) -> dict[str, Any]:
    dtype = torch.float64
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(rank, rank, dtype=dtype, generator=generator)
    G = raw @ raw.mT + rank * torch.eye(rank, dtype=dtype)
    L = torch.linalg.cholesky(G).contiguous()
    z = 0.1 * torch.randn(rank, dtype=dtype, generator=generator)
    B = C = None
    for current_width in RANKS_AND_WIDTHS[rank]:
        B_candidate = torch.randn(
            rank, current_width, dtype=dtype, generator=generator
        ) / math.sqrt(current_width)
        C_candidate = torch.randn(
            rank, current_width, dtype=dtype, generator=generator
        ) / math.sqrt(current_width)
        if current_width == width:
            B, C = B_candidate, C_candidate
            break
    if B is None or C is None:
        raise ValueError(f"width {width} is not frozen for rank {rank}")
    M = B @ C.mT
    Vb = torch.linalg.solve_triangular(L, B, upper=False).contiguous()
    Vc = torch.linalg.solve_triangular(L, C, upper=False).contiguous()
    Aw = (Vb @ Vc.mT).contiguous()
    zero = torch.zeros(rank, dtype=dtype)
    L_fresh = torch.linalg.cholesky(G + torch.outer(z, z)).contiguous()
    left_fresh = torch.linalg.solve_triangular(L_fresh, B, upper=False)
    right_fresh = torch.linalg.solve_triangular(L_fresh, C, upper=False)
    Aw_fresh = whiten(L_fresh, M)
    cases = make_cases(
        L=L,
        Aw=Aw,
        z=z,
        zero=zero,
        G=G,
        B=B,
        C=C,
        Vb=Vb,
        Vc=Vc,
        e3_metric_only=e3_metric_only,
        one_sided_fair=one_sided_fair,
    )
    accuracy = validate_outputs(
        cases,
        L_fresh=L_fresh,
        Aw_fresh=Aw_fresh,
        left_fresh=left_fresh,
        right_fresh=right_fresh,
        tolerance=TOLERANCE,
    )
    names = list(DENSE_PATHS + THIN_PATHS)
    timings, schedule = time_cases(
        cases,
        names=names,
        seed=seed,
        width_index=width_index,
        warmups=WARMUPS,
        repeats=REPEATS,
    )
    speedups = {}
    denominator = timings["metric_only_specialized"]["values"]
    for competitor in (
        "generic_w0_ablation",
        "drag_materialize",
        "fresh_materialize",
    ):
        numerator = timings[competitor]["values"]
        speedups[f"metric_only_vs_{competitor}"] = scalar_stats(
            [left / right for left, right in zip(numerator, denominator)]
        )
    return {
        "rank": rank,
        "factor_width": width,
        "seed": seed,
        "dtype": "float64",
        "threads": 1,
        "warmups_per_path": WARMUPS,
        "timed_repeats_per_path": REPEATS,
        "accuracy": accuracy,
        "timings_seconds": timings,
        "dense_same_output_speedups": speedups,
        "timing_schedule": schedule,
        "output_contracts": {
            name: (
                "dense_same_output" if name in DENSE_PATHS else "thin_factor_only"
            )
            for name in names
        },
    }


def _provenance(protocol, e3_metric_only, one_sided_fair) -> dict[str, Any]:
    metric_binary = Path(e3_metric_only.__file__).resolve()
    fair_binary = Path(one_sided_fair.__file__).resolve()
    paths = [
        PROTOCOL_PATH,
        Path(__file__).resolve(),
        Path(__file__).with_name("e3_metric_only_pilot.py"),
        Path(__file__).with_name("e3_metric_only_ext.cpp"),
        Path(__file__).with_name("setup_e3_metric_only.py"),
        Path(__file__).with_name("one_sided_fair_ext.cpp"),
        Path(__file__).with_name("setup_one_sided_fair.py"),
        metric_binary,
        fair_binary,
        ROOT / protocol["development_selection"]["result_path"],
        ROOT / protocol["development_selection"]["audit_path"],
    ]
    return {
        "files": {
            relative_path(path): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
        },
        "metric_only_binary": str(metric_binary),
        "fair_drag_binary": str(fair_binary),
        "build_flags": {
            "metric_only": [
                "-O3",
                "-march=native",
                "-ffast-math",
                "-fno-math-errno",
                "-DNDEBUG",
            ],
            "fair_drag": [
                "-O3",
                "-march=native",
                "-ffast-math",
                "-fno-math-errno",
                "-DNDEBUG",
            ],
        },
    }


def run() -> dict[str, Any]:
    protocol = _read(PROTOCOL_PATH)
    validate_protocol(protocol)
    validate_development_gate(protocol)
    output = ROOT / protocol["result_path"]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite confirmation: {output}")
    runtime = validate_runtime()
    e3_metric_only, one_sided_fair = load_extensions()
    started = datetime.now(timezone.utc)
    load_before = os.getloadavg()
    rows = []
    for rank, widths in RANKS_AND_WIDTHS.items():
        for width_index, width in enumerate(widths):
            for seed in SEEDS:
                row = run_cell(
                    rank=rank,
                    width=width,
                    width_index=width_index,
                    seed=seed,
                    e3_metric_only=e3_metric_only,
                    one_sided_fair=one_sided_fair,
                )
                rows.append(row)
                print(
                    f"completed r={rank} p={width} seed={seed}",
                    flush=True,
                )
    ended = datetime.now(timezone.utc)
    expected_cells = sum(len(widths) for widths in RANKS_AND_WIDTHS.values()) * len(
        SEEDS
    )
    if len(rows) != expected_cells:
        raise AssertionError(
            f"expected {expected_cells} cells, observed {len(rows)}"
        )
    names = list(DENSE_PATHS + THIN_PATHS)
    payload = {
        "schema_version": 1,
        "experiment": "e3_drag_fairness_confirmation",
        "campaign_tag": CAMPAIGN_TAG,
        "status": "complete",
        "protocol": protocol,
        "protocol_path": relative_path(PROTOCOL_PATH),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "rows": rows,
        "aggregate": aggregate_rows(
            rows,
            names,
            expected_seed_count=len(SEEDS),
        ),
        "integrity": {
            "declared_cell_count": expected_cells,
            "observed_cell_count": len(rows),
            "all_rows_retained": True,
            "samples_per_path_per_cell": REPEATS,
            "development_only_seeds_excluded": True,
            "confirmation_tuned_from_outcomes": False,
            "partial_outcomes_inspected_before_completion": False,
        },
        "contract_boundary": {
            "dense_same_output_paths": list(DENSE_PATHS),
            "thin_factor_only_paths": list(THIN_PATHS),
            "steady_state_representation_boundary": protocol[
                "steady_state_representation_boundary"
            ],
            "claim_scope": (
                "Performance claims compare only dense same-output paths. "
                "Thin factor-only paths are reported separately because they "
                "return and retain a different representation."
            ),
        },
        "environment": {
            **runtime,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "cpu_model": first_cpu_model(),
            "torch": torch.__version__,
            "torch_config": torch.__config__.show(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_account": os.environ.get("SLURM_JOB_ACCOUNT"),
            "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_job_qos": os.environ.get("SLURM_JOB_QOS"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "started_utc": started.isoformat(),
            "ended_utc": ended.isoformat(),
            "elapsed_wall_seconds": (ended - started).total_seconds(),
            "load_average_before": list(load_before),
            "load_average_after": list(os.getloadavg()),
        },
        "provenance": _provenance(
            protocol, e3_metric_only, one_sided_fair
        ),
    }
    atomic_json_dump(payload, output)
    print(output, flush=True)
    return payload


if __name__ == "__main__":
    run()
