"""Run the frozen exclusive-node, multi-seed E3 confirmation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
from typing import Any

import torch

from .common import ROOT, configure_threads
from .e3_metric_only_pilot import (
    DENSE_PATHS,
    REQUIRED_THREAD_ENV,
    THIN_PATHS,
    aggregate_rows,
    atomic_json_dump,
    first_cpu_model,
    load_extensions,
    relative_path,
    run_cell,
    sha256,
)


PROTOCOL_PATH = Path(__file__).with_name(
    "e3_metric_only_confirmation_protocol.json"
)
CAMPAIGN_TAG = "e3-metric-only-confirm-v1"
SEEDS = tuple(range(74001, 74011))
RANKS_AND_WIDTHS = {
    256: (1, 4, 8, 16, 32, 64, 128, 256, 512),
    1024: (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048),
}
WARMUPS = 5
REPEATS = 30
TOLERANCE = 1e-11
EXPECTED_ACCOUNT = "marlowe-m000151-pm06"
EXPECTED_PARTITION = "batch"


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
        "status": "frozen_before_confirmation",
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
            f"E3 confirmation protocol mismatch: expected {expected}, got {actual}"
        )
    names = list(protocol.get("paths", {}))
    if names != list(DENSE_PATHS + THIN_PATHS):
        raise ValueError(f"unexpected path order: {names}")
    for name in DENSE_PATHS:
        if protocol["paths"][name].get("output_contract") != "dense_same_output":
            raise ValueError(f"{name} lacks the dense same-output contract")
    for name in THIN_PATHS:
        if protocol["paths"][name].get("output_contract") != "thin_factor_only":
            raise ValueError(f"{name} lacks the thin factor-only contract")
    resource = protocol.get("resource_contract", {})
    if (
        resource.get("account") != EXPECTED_ACCOUNT
        or resource.get("partition") != EXPECTED_PARTITION
        or resource.get("physical_cpus_per_process") != 1
        or not resource.get("exclusive_node")
    ):
        raise ValueError(f"unexpected resource contract: {resource}")


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
        raise RuntimeError("E3 confirmation is not on the frozen Slurm account")
    if os.environ.get("SLURM_JOB_PARTITION") != EXPECTED_PARTITION:
        raise RuntimeError("E3 confirmation is not on the batch partition")
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("CPU affinity is not observable")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 1:
        raise RuntimeError(
            f"E3 confirmation requires exactly one physical CPU: {affinity}"
        )
    configure_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError(
            "PyTorch did not retain the required single-thread runtime"
        )
    declared_commit = os.environ.get("SOURCE_COMMIT")
    actual_commit = _git_output("rev-parse", "HEAD")
    if not declared_commit or declared_commit != actual_commit:
        raise RuntimeError(
            f"source-lock mismatch: declared={declared_commit!r}, "
            f"actual={actual_commit!r}"
        )
    if subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0:
        raise RuntimeError("tracked worktree differs from the source-locked commit")
    if (
        subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode
        != 0
    ):
        raise RuntimeError("index differs from the source-locked commit")
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


def _provenance(
    *,
    protocol: dict[str, Any],
    e3_metric_only,
    one_sided_cholesky,
) -> dict[str, Any]:
    metric_binary = Path(e3_metric_only.__file__).resolve()
    one_sided_binary = Path(one_sided_cholesky.__file__).resolve()
    paths = [
        PROTOCOL_PATH,
        Path(__file__).resolve(),
        Path(__file__).with_name("e3_metric_only_pilot.py"),
        Path(__file__).with_name("e3_metric_only_ext.cpp"),
        Path(__file__).with_name("setup_e3_metric_only.py"),
        Path(__file__).with_name("one_sided_ext.cpp"),
        Path(__file__).with_name("setup_one_sided.py"),
        metric_binary,
        one_sided_binary,
    ]
    paths.extend(ROOT / item for item in protocol["reference_source_jsons"])
    paths.append(ROOT / protocol["pilot_context"]["path"])
    return {
        "files": {
            relative_path(path): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
        },
        "metric_only_binary": str(metric_binary),
        "one_sided_binary": str(one_sided_binary),
        "build_flags": {
            "metric_only": [
                "-O3",
                "-march=native",
                "-ffast-math",
                "-fno-math-errno",
                "-DNDEBUG",
            ],
            "one_sided": ["-O3", "-march=native", "-ffast-math"],
        },
    }


def run() -> dict[str, Any]:
    with PROTOCOL_PATH.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    validate_protocol(protocol)
    output = ROOT / protocol["result_path"]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    runtime = validate_runtime()
    e3_metric_only, one_sided_cholesky = load_extensions()
    names = list(protocol["paths"])
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
                    warmups=WARMUPS,
                    repeats=REPEATS,
                    tolerance=TOLERANCE,
                    names=names,
                    e3_metric_only=e3_metric_only,
                    one_sided_cholesky=one_sided_cholesky,
                )
                rows.append(row)
                print(
                    f"r={rank} p={width} seed={seed} "
                    f"metric={1e3 * row['timings_seconds']['metric_only_specialized']['median']:.3f}ms "
                    f"drag+mat={1e3 * row['timings_seconds']['drag_materialize']['median']:.3f}ms "
                    f"fresh+mat={1e3 * row['timings_seconds']['fresh_materialize']['median']:.3f}ms",
                    flush=True,
                )
    ended = datetime.now(timezone.utc)
    expected_cells = sum(len(widths) for widths in RANKS_AND_WIDTHS.values()) * len(
        SEEDS
    )
    if len(rows) != expected_cells:
        raise AssertionError(f"expected {expected_cells} cells, observed {len(rows)}")
    payload = {
        "schema_version": 1,
        "experiment": "e3_metric_only_confirmation",
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
            "pilot_excluded_from_statistics": True,
            "confirmation_tuned_from_outcomes": False,
        },
        "contract_boundary": {
            "dense_same_output_paths": list(DENSE_PATHS),
            "thin_factor_only_paths": list(THIN_PATHS),
            "claim_scope": (
                "Performance claims compare only dense same-output paths. "
                "Factor-only paths are reported separately because they return "
                "a different representation."
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
            protocol=protocol,
            e3_metric_only=e3_metric_only,
            one_sided_cholesky=one_sided_cholesky,
        ),
    }
    atomic_json_dump(payload, output)
    print(output, flush=True)
    return payload


if __name__ == "__main__":
    run()
