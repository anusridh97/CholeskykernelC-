"""Read-only E4 reanalysis separating fresh and incremental FP error.

This module never invokes the incremental-update implementation and never
modifies an existing experiment artifact.  It deterministically reconstructs
the exact ``G`` and ``M`` states from the recorded seeds/configuration, then
computes a fresh factorization/whitening diagnostic at every saved checkpoint.

The raw E4 files save scalar diagnostics rather than the computed ``L`` and
``Aw`` tensors.  Consequently, at non-reset checkpoints the observable
quantity is the signed scalar difference

    raw forward error - fresh forward error,

not the norm of the difference between incremental and fresh tensors.  The
output records that distinction explicitly and supplies triangle-inequality
bounds for the unavailable direct tensor comparison.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

import torch

from .common import ROOT, configure_threads, tensor_relative_error
from .kernel_stability import (
    make_near_duplicate_kernel,
    rbf,
    whiten as kernel_whiten,
)
from .stability import make_spd, whiten as controlled_whiten
from .task_matrix import synthetic_tasks


RAW_DIR = ROOT / "rebuttal_results" / "stability"
DEFAULT_JSON = ROOT / "rebuttal_results" / "e4_reanalysis.json"
DEFAULT_MARKDOWN = ROOT / "rebuttal_results" / "e4_reanalysis.md"
DEFAULT_SIGNED = (
    ROOT
    / "Cholesky-main-2"
    / "corrected_incremental_cholesky"
    / "results"
    / "correctness.json"
)
SUMMARY_PATH = ROOT / "rebuttal_results" / "summary.json"

METRICS = (
    "L_relative_error",
    "Aw_relative_error",
    "factor_backward_error",
    "operator_backward_error",
)
FORWARD_METRICS = ("L_relative_error", "Aw_relative_error")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def finite_stats(values: Iterable[float]) -> dict[str, Any]:
    materialized = [float(value) for value in values]
    if not materialized:
        return {"mean": None, "sample_sd": None, "n": 0, "values": []}
    if not all(math.isfinite(value) for value in materialized):
        raise ValueError("non-finite value encountered during E4 aggregation")
    return {
        "mean": statistics.mean(materialized),
        "sample_sd": (
            statistics.stdev(materialized) if len(materialized) > 1 else 0.0
        ),
        "n": len(materialized),
        "values": materialized,
    }


def state_kind(step: int, reset_interval: int) -> str:
    if step == 0:
        return "fresh_initial"
    if reset_interval > 0 and step % reset_interval == 0:
        return "fresh_reset"
    return "incremental"


def fresh_diagnostics(
    G_exact: torch.Tensor,
    M_exact: torch.Tensor,
    dtype: torch.dtype,
    whiten,
) -> dict[str, float]:
    L_exact = torch.linalg.cholesky(G_exact)
    Aw_exact = whiten(L_exact, M_exact)
    L_fresh = torch.linalg.cholesky(G_exact.to(dtype)).contiguous()
    Aw_fresh = whiten(L_fresh, M_exact.to(dtype))
    L64 = L_fresh.double()
    Aw64 = Aw_fresh.double()
    return {
        "L_relative_error": tensor_relative_error(L64, L_exact),
        "Aw_relative_error": tensor_relative_error(Aw64, Aw_exact),
        "factor_backward_error": tensor_relative_error(
            L64 @ L64.mT, G_exact
        ),
        "operator_backward_error": tensor_relative_error(
            L64 @ Aw64 @ L64.mT, M_exact
        ),
    }


def metric_comparison(
    raw: dict[str, Any],
    fresh: dict[str, float],
    kind: str,
) -> dict[str, Any]:
    delta = {metric: float(raw[metric] - fresh[metric]) for metric in METRICS}
    ratio = {
        metric: (
            float(raw[metric] / fresh[metric])
            if fresh[metric] != 0.0
            else (1.0 if raw[metric] == 0.0 else None)
        )
        for metric in METRICS
    }
    bounds = {
        metric: {
            "lower": abs(delta[metric]),
            "upper": float(raw[metric] + fresh[metric]),
            "normalization": "the corresponding FP64 exact tensor norm",
        }
        for metric in FORWARD_METRICS
    }
    if kind.startswith("fresh_"):
        direct = {
            "available": True,
            "basis": "known_by_protocol",
            "L_relative_to_exact_norm": 0.0,
            "Aw_relative_to_exact_norm": 0.0,
        }
    else:
        direct = {
            "available": False,
            "basis": "raw artifact stores scalar diagnostics, not state tensors",
            "L_relative_to_exact_norm": None,
            "Aw_relative_to_exact_norm": None,
        }
    return {
        "raw_minus_fresh_scalar_error": delta,
        "raw_to_fresh_scalar_error_ratio": ratio,
        "incremental_vs_fresh_direct_tensor_error": direct,
        "direct_tensor_error_bounds": bounds,
    }


def expected_path(raw_dir: Path, task: dict[str, Any]) -> Path:
    kind = task["kind"]
    if kind == "stability":
        return raw_dir / (
            f"stability_r{task['rank']}_cond{task['condition']:g}_"
            f"{task['precision']}_reset{task['reset_interval']}_"
            f"seed{task['seed']}.json"
        )
    if kind == "kernel_stability":
        return raw_dir / (
            f"kernel_r{task['rank']}_{task['precision']}_{task['operator']}.json"
        )
    if kind == "downdate":
        return raw_dir / f"downdate_r{task['rank']}_seed{task['seed']}.json"
    raise ValueError(f"not an E4 task: {task}")


def canonical_artifacts(
    raw_dir: Path,
) -> tuple[dict[str, list[tuple[Path, dict[str, Any]]]], list[dict[str, Any]]]:
    task_groups: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    expected: set[Path] = set()
    for task in synthetic_tasks():
        if task["kind"] not in {"stability", "kernel_stability", "downdate"}:
            continue
        path = expected_path(raw_dir, task)
        expected.add(path.resolve())
        if not path.is_file():
            raise FileNotFoundError(f"missing planned E4 artifact: {path}")
        task_groups[task["kind"]].append((path, read_json(path)))

    extras = []
    for path in sorted(raw_dir.glob("*.json")):
        if path.resolve() in expected:
            continue
        payload = read_json(path)
        extras.append(
            {
                "path": relative_path(path),
                "sha256": sha256(path),
                "experiment": payload.get("experiment"),
                "rank": payload.get("rank"),
                "reason_excluded": (
                    "not present in the canonical synthetic task matrix "
                    "(smoke/development artifact)"
                ),
                "payload": payload,
            }
        )
    return task_groups, extras


def reconstruct_controlled_fresh(
    payload: dict[str, Any],
) -> tuple[dict[int, dict[str, float]], float]:
    rank = int(payload["rank"])
    condition = float(payload["target_condition"])
    seed = int(payload["seed"])
    steps = int(payload["steps"])
    update_scale = float(payload["update_scale"])
    checkpoints = {int(record["step"]) for record in payload["records"]}
    dtype = torch.float32 if payload["precision"] == "float32" else torch.float64

    G_exact = make_spd(rank, condition, seed)
    lambda_floor = float(torch.linalg.eigvalsh(G_exact)[0].item())
    M_exact = G_exact - lambda_floor * torch.eye(rank, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed + 991)
    output: dict[int, dict[str, float]] = {}
    for step in range(steps + 1):
        if step in checkpoints:
            output[step] = fresh_diagnostics(
                G_exact, M_exact, dtype, controlled_whiten
            )
        if step == steps:
            break
        scale = update_scale * (torch.trace(G_exact) / rank).sqrt()
        z64 = (
            torch.randn(rank, dtype=torch.float64, generator=generator)
            * scale
            / rank**0.5
        )
        G_exact = G_exact + torch.outer(z64, z64)
        M_exact = M_exact + torch.outer(z64, z64)
    if set(output) != checkpoints:
        raise AssertionError("failed to reconstruct every controlled checkpoint")
    return output, lambda_floor


def analyze_controlled(
    artifacts: list[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    if len(artifacts) != 72:
        raise AssertionError(
            f"expected 72 controlled artifacts, found {len(artifacts)}"
        )
    identities: set[tuple[Any, ...]] = set()
    detailed = []
    raw_artifacts = []
    record_lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
    reset_validation = {metric: 0.0 for metric in METRICS}
    cache: dict[tuple[Any, ...], tuple[dict[int, dict[str, float]], float]] = {}

    for path, payload in sorted(artifacts, key=lambda item: str(item[0])):
        if payload.get("experiment") != "positive_update_stability":
            raise ValueError(f"unexpected experiment in {path}")
        identity = (
            int(payload["rank"]),
            float(payload["target_condition"]),
            str(payload["precision"]),
            int(payload["reset_interval"]),
            int(payload["seed"]),
        )
        if identity in identities:
            raise ValueError(f"duplicate controlled identity: {identity}")
        identities.add(identity)
        if payload["status"] != "ok" or payload["failure"] is not None:
            raise ValueError(f"failed controlled artifact: {path}")
        steps = [int(record["step"]) for record in payload["records"]]
        if steps != [0, 1, 10, 50, 100, 200]:
            raise ValueError(f"unexpected checkpoints in {path}: {steps}")

        cache_key = (
            payload["rank"],
            payload["target_condition"],
            payload["precision"],
            payload["seed"],
            payload["steps"],
            payload["update_scale"],
        )
        if cache_key not in cache:
            cache[cache_key] = reconstruct_controlled_fresh(payload)
        fresh_by_step, reconstructed_floor = cache[cache_key]
        if reconstructed_floor != float(payload["lambda_floor"]):
            raise AssertionError(f"lambda-floor reconstruction mismatch in {path}")

        raw_artifacts.append(
            {
                "path": relative_path(path),
                "sha256": sha256(path),
                "payload": payload,
            }
        )
        for raw in payload["records"]:
            step = int(raw["step"])
            kind = state_kind(step, int(payload["reset_interval"]))
            fresh = fresh_by_step[step]
            comparison = metric_comparison(raw, fresh, kind)
            row = {
                "rank": int(payload["rank"]),
                "target_condition": float(payload["target_condition"]),
                "precision": str(payload["precision"]),
                "reset_interval": int(payload["reset_interval"]),
                "seed": int(payload["seed"]),
                "step": step,
                "state_kind": kind,
                "source": relative_path(path),
                "raw": raw,
                "fresh_same_precision": fresh,
                **comparison,
            }
            detailed.append(row)
            key = (
                row["rank"],
                row["target_condition"],
                row["precision"],
                row["reset_interval"],
                row["seed"],
                row["step"],
            )
            record_lookup[key] = row
            if kind.startswith("fresh_"):
                for metric in METRICS:
                    reset_validation[metric] = max(
                        reset_validation[metric],
                        abs(float(raw[metric]) - fresh[metric]),
                    )

    paired = []
    for rank, condition, precision, reset, seed in sorted(identities):
        if reset != 0:
            continue
        for step in (0, 1, 10, 50, 100, 200):
            rows = {
                interval: record_lookup[
                    (rank, condition, precision, interval, seed, step)
                ]
                for interval in (0, 50, 200)
            }
            raw_no_reset = rows[0]["raw"]
            paired.append(
                {
                    "rank": rank,
                    "target_condition": condition,
                    "precision": precision,
                    "seed": seed,
                    "step": step,
                    "fresh_same_precision": rows[0]["fresh_same_precision"],
                    "raw_by_reset_interval": {
                        str(interval): rows[interval]["raw"]
                        for interval in (0, 50, 200)
                    },
                    "state_kind_by_reset_interval": {
                        str(interval): rows[interval]["state_kind"]
                        for interval in (0, 50, 200)
                    },
                    "no_reset_minus_fresh": {
                        metric: float(
                            raw_no_reset[metric]
                            - rows[0]["fresh_same_precision"][metric]
                        )
                        for metric in METRICS
                    },
                    "no_reset_minus_reset50": {
                        metric: float(
                            raw_no_reset[metric] - rows[50]["raw"][metric]
                        )
                        for metric in METRICS
                    },
                    "no_reset_minus_reset200": {
                        metric: float(
                            raw_no_reset[metric] - rows[200]["raw"][metric]
                        )
                        for metric in METRICS
                    },
                }
            )

    aggregate_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in detailed:
        aggregate_groups[
            (
                row["rank"],
                row["target_condition"],
                row["precision"],
                row["reset_interval"],
                row["step"],
            )
        ].append(row)
    aggregate = []
    for key, rows in sorted(aggregate_groups.items()):
        rank, condition, precision, reset_interval, step = key
        aggregate.append(
            {
                "rank": rank,
                "target_condition": condition,
                "precision": precision,
                "reset_interval": reset_interval,
                "step": step,
                "n": len(rows),
                "metrics": {
                    metric: {
                        "raw": finite_stats(row["raw"][metric] for row in rows),
                        "fresh": finite_stats(
                            row["fresh_same_precision"][metric] for row in rows
                        ),
                        "raw_minus_fresh": finite_stats(
                            row["raw_minus_fresh_scalar_error"][metric]
                            for row in rows
                        ),
                    }
                    for metric in METRICS
                },
            }
        )

    paired_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        paired_groups[
            (
                row["rank"],
                row["target_condition"],
                row["precision"],
                row["step"],
            )
        ].append(row)
    paired_aggregate = []
    for key, rows in sorted(paired_groups.items()):
        rank, condition, precision, step = key
        paired_aggregate.append(
            {
                "rank": rank,
                "target_condition": condition,
                "precision": precision,
                "step": step,
                "n": len(rows),
                "metrics": {
                    metric: {
                        "fresh": finite_stats(
                            row["fresh_same_precision"][metric] for row in rows
                        ),
                        "no_reset": finite_stats(
                            row["raw_by_reset_interval"]["0"][metric]
                            for row in rows
                        ),
                        "reset50": finite_stats(
                            row["raw_by_reset_interval"]["50"][metric]
                            for row in rows
                        ),
                        "reset200": finite_stats(
                            row["raw_by_reset_interval"]["200"][metric]
                            for row in rows
                        ),
                        "no_reset_minus_fresh": finite_stats(
                            row["no_reset_minus_fresh"][metric] for row in rows
                        ),
                        "no_reset_minus_reset50": finite_stats(
                            row["no_reset_minus_reset50"][metric] for row in rows
                        ),
                        "no_reset_minus_reset200": finite_stats(
                            row["no_reset_minus_reset200"][metric] for row in rows
                        ),
                    }
                    for metric in METRICS
                },
            }
        )

    return {
        "artifact_count": len(raw_artifacts),
        "record_count": len(detailed),
        "fresh_checkpoint_validation_max_abs_difference": reset_validation,
        "raw_artifacts": raw_artifacts,
        "checkpoint_records": detailed,
        "paired_no_reset_comparisons": paired,
        "aggregate_by_checkpoint": aggregate,
        "paired_aggregate_by_checkpoint": paired_aggregate,
    }


def reconstruct_kernel_fresh(
    payload: dict[str, Any],
    raw_row: dict[str, Any],
) -> dict[int, dict[str, float]]:
    rank = int(payload["rank"])
    delta = float(raw_row["delta"])
    seed = int(raw_row["seed"])
    jitter = float(payload["jitter"])
    steps = int(payload["steps"])
    update_scale = float(payload["update_scale"])
    operator = str(payload["operator"])
    dtype = torch.float32 if payload["precision"] == "float32" else torch.float64
    checkpoints = {int(record["step"]) for record in raw_row["records"]}

    G_exact, points = make_near_duplicate_kernel(rank, delta, jitter, seed)
    generator = torch.Generator().manual_seed(seed + 7001)
    if operator == "cumulative":
        M_exact = G_exact - jitter * torch.eye(rank, dtype=torch.float64)
    elif operator == "random":
        raw_operator = torch.randn(
            rank, rank, dtype=torch.float64, generator=generator
        )
        M_exact = raw_operator / torch.linalg.matrix_norm(raw_operator, ord=2)
    else:
        raise ValueError(f"unknown kernel operator: {operator}")

    output: dict[int, dict[str, float]] = {}
    for step in range(steps + 1):
        if step in checkpoints:
            output[step] = fresh_diagnostics(
                G_exact, M_exact, dtype, kernel_whiten
            )
        if step == steps:
            break
        x_new = torch.rand(1, 3, dtype=torch.float64, generator=generator)
        z64 = update_scale * rbf(points, x_new, lengthscale=0.4).squeeze(1)
        G_exact.add_(torch.outer(z64, z64))
        M_exact.add_(torch.outer(z64, z64))
    if set(output) != checkpoints:
        raise AssertionError("failed to reconstruct every kernel checkpoint")
    return output


def analyze_kernels(
    artifacts: list[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    if len(artifacts) != 4:
        raise AssertionError(f"expected 4 kernel artifacts, found {len(artifacts)}")
    detailed = []
    raw_artifacts = []
    reset_validation = {metric: 0.0 for metric in METRICS}
    identities: set[tuple[Any, ...]] = set()

    for path, payload in sorted(artifacts, key=lambda item: str(item[0])):
        if payload.get("experiment") != "near_duplicate_kernel_stability":
            raise ValueError(f"unexpected experiment in {path}")
        file_identity = (
            int(payload["rank"]),
            str(payload["precision"]),
            str(payload["operator"]),
        )
        if file_identity in identities:
            raise ValueError(f"duplicate kernel identity: {file_identity}")
        identities.add(file_identity)
        if len(payload["rows"]) != 18:
            raise ValueError(f"expected 18 kernel rows in {path}")
        raw_artifacts.append(
            {
                "path": relative_path(path),
                "sha256": sha256(path),
                "payload": payload,
            }
        )
        row_identities: set[tuple[float, int]] = set()
        for raw_row in payload["rows"]:
            row_identity = (float(raw_row["delta"]), int(raw_row["seed"]))
            if row_identity in row_identities:
                raise ValueError(f"duplicate kernel row in {path}: {row_identity}")
            row_identities.add(row_identity)
            if raw_row["status"] != "ok" or raw_row["failure"] is not None:
                raise ValueError(f"failed kernel row in {path}: {row_identity}")
            steps = [int(record["step"]) for record in raw_row["records"]]
            if steps != [0, 1, 10, 50, 100]:
                raise ValueError(
                    f"unexpected kernel checkpoints in {path}: {steps}"
                )
            fresh_by_step = reconstruct_kernel_fresh(payload, raw_row)
            for raw in raw_row["records"]:
                step = int(raw["step"])
                kind = state_kind(step, int(payload["reset_interval"]))
                fresh = fresh_by_step[step]
                comparison = metric_comparison(raw, fresh, kind)
                detailed.append(
                    {
                        "rank": int(payload["rank"]),
                        "precision": str(payload["precision"]),
                        "operator": str(payload["operator"]),
                        "delta": float(raw_row["delta"]),
                        "seed": int(raw_row["seed"]),
                        "step": step,
                        "reset_interval": int(payload["reset_interval"]),
                        "state_kind": kind,
                        "source": relative_path(path),
                        "raw": raw,
                        "fresh_same_precision": fresh,
                        **comparison,
                    }
                )
                if kind.startswith("fresh_"):
                    for metric in METRICS:
                        reset_validation[metric] = max(
                            reset_validation[metric],
                            abs(float(raw[metric]) - fresh[metric]),
                        )

    aggregate_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in detailed:
        aggregate_groups[
            (
                row["rank"],
                row["precision"],
                row["operator"],
                row["delta"],
                row["step"],
            )
        ].append(row)
    aggregate = []
    for key, rows in sorted(aggregate_groups.items()):
        rank, precision, operator, delta, step = key
        aggregate.append(
            {
                "rank": rank,
                "precision": precision,
                "operator": operator,
                "delta": delta,
                "step": step,
                "n": len(rows),
                "metrics": {
                    metric: {
                        "raw": finite_stats(row["raw"][metric] for row in rows),
                        "fresh": finite_stats(
                            row["fresh_same_precision"][metric] for row in rows
                        ),
                        "raw_minus_fresh": finite_stats(
                            row["raw_minus_fresh_scalar_error"][metric]
                            for row in rows
                        ),
                    }
                    for metric in METRICS
                },
            }
        )
    return {
        "artifact_count": len(raw_artifacts),
        "record_count": len(detailed),
        "fresh_checkpoint_validation_max_abs_difference": reset_validation,
        "raw_artifacts": raw_artifacts,
        "checkpoint_records": detailed,
        "aggregate_by_checkpoint": aggregate,
        "unavailable_comparisons": [
            {
                "comparison": "kernel no-reset minus reset",
                "reason": (
                    "the canonical kernel campaign saved only reset_interval=50; "
                    "there is no paired no-reset trajectory"
                ),
            }
        ],
    }


def analyze_downdates(
    artifacts: list[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    if len(artifacts) != 6:
        raise AssertionError(f"expected 6 downdate artifacts, found {len(artifacts)}")
    rows = []
    identities: set[tuple[int, int]] = set()
    for path, payload in sorted(artifacts, key=lambda item: str(item[0])):
        if payload.get("experiment") != "downdate_amplification":
            raise ValueError(f"unexpected experiment in {path}")
        identity = (int(payload["rank"]), int(payload["seed"]))
        if identity in identities:
            raise ValueError(f"duplicate downdate identity: {identity}")
        identities.add(identity)
        if len(payload["rows"]) != 5:
            raise ValueError(f"unexpected downdate row count in {path}")
        rows.append(
            {
                "path": relative_path(path),
                "sha256": sha256(path),
                "payload": payload,
            }
        )
    return {
        "artifact_count": len(rows),
        "raw_artifacts": rows,
        "fresh_comparison": {
            "available": False,
            "reason": (
                "E4 downdate artifacts measure analytic transport amplification; "
                "they do not execute or save a signed incremental state"
            ),
        },
    }


def analyze_signed_correctness(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "available": False,
            "reason": "no signed-update correctness artifact was supplied",
        }
    payload = read_json(path)
    if payload.get("experiment") != "correctness_signed":
        raise ValueError(f"unexpected signed-correctness artifact: {path}")
    required = {
        (rank, sign)
        for rank in (16, 64, 128, 256)
        for sign in (-1, 1)
    }
    observed = {
        (int(row["r"]), int(row["metric_sign"])) for row in payload["rows"]
    }
    if observed != required:
        raise ValueError(
            f"signed-correctness cells differ from expected: {observed ^ required}"
        )
    return {
        "available": True,
        "canonical_e4_artifact": False,
        "path": relative_path(path),
        "sha256": sha256(path),
        "payload": payload,
        "interpretation": (
            "Each stored L/Aw/Gram/vz error is already measured directly against "
            "a fresh factorization for one signed update, then aggregated over "
            "five seeds."
        ),
        "limitations": [
            "FP64 only",
            "one update only; no checkpoint trajectory",
            "per-seed values are not stored",
            (
                "operator_sign is mixed across seeds and absent from the result "
                "rows, so it cannot be disaggregated from this artifact"
            ),
            (
                "the artifact is supplemental and is not one of the canonical "
                "E4 task-matrix outputs"
            ),
        ],
    }


def audit_existing_summary(
    controlled: dict[str, Any],
    kernels: dict[str, Any],
    downdates: dict[str, Any],
) -> dict[str, Any]:
    if not SUMMARY_PATH.is_file():
        return {
            "available": False,
            "reason": "rebuttal_results/summary.json does not exist",
        }
    summary = read_json(SUMMARY_PATH)
    stability = summary.get("stability", {})
    positive = stability.get("positive_updates", [])
    kernel_rows = stability.get("near_duplicate_kernels", [])
    down_rows = stability.get("downdates", [])

    canonical_positive = {
        (
            row["target_condition"],
            row["precision"],
            row["reset_interval"],
            row["seed"],
        ): row
        for row in controlled["checkpoint_records"]
        if row["step"] == 200
    }
    summary_positive = {
        (
            row["target_condition"],
            row["precision"],
            row["reset_interval"],
            row["seed"],
        ): row
        for row in positive
    }
    missing_positive = sorted(
        str(key) for key in canonical_positive.keys() - summary_positive.keys()
    )
    extra_positive = sorted(
        str(key) for key in summary_positive.keys() - canonical_positive.keys()
    )
    mismatched_positive = []
    for key in canonical_positive.keys() & summary_positive.keys():
        if summary_positive[key].get("final") != canonical_positive[key]["raw"]:
            mismatched_positive.append(str(key))

    canonical_kernel_ids = {
        (
            artifact["payload"]["rank"],
            artifact["payload"]["precision"],
            artifact["payload"]["operator"],
        )
        for artifact in kernels["raw_artifacts"]
    }
    summary_kernel_ids = {
        (row["rank"], row["precision"], row["operator"]) for row in kernel_rows
    }
    canonical_down_ids = {
        (artifact["payload"]["rank"], artifact["payload"]["seed"])
        for artifact in downdates["raw_artifacts"]
    }
    summary_down_ids = {(row["rank"], row["seed"]) for row in down_rows}

    return {
        "available": True,
        "path": relative_path(SUMMARY_PATH),
        "sha256": sha256(SUMMARY_PATH),
        "stored_counts": {
            "positive_updates": len(positive),
            "near_duplicate_kernels": len(kernel_rows),
            "downdates": len(down_rows),
        },
        "canonical_counts": {
            "positive_updates": controlled["artifact_count"],
            "near_duplicate_kernels": kernels["artifact_count"],
            "downdates": downdates["artifact_count"],
        },
        "positive_missing_cells": missing_positive,
        "positive_extra_cells": extra_positive,
        "positive_final_mismatches": sorted(mismatched_positive),
        "kernel_missing_cells": sorted(
            str(key) for key in canonical_kernel_ids - summary_kernel_ids
        ),
        "kernel_extra_cells": sorted(
            str(key) for key in summary_kernel_ids - canonical_kernel_ids
        ),
        "downdate_missing_cells": sorted(
            str(key) for key in canonical_down_ids - summary_down_ids
        ),
        "downdate_extra_cells": sorted(
            str(key) for key in summary_down_ids - canonical_down_ids
        ),
        "aggregator_limitation": (
            "aggregate_stability retains only the final controlled-spectrum "
            "record, so checkpoint-level fresh/incremental separation must use "
            "the raw JSONs. Kernel and downdate payloads are retained in full."
        ),
    }


def source_hashes() -> dict[str, str]:
    sources = (
        ROOT / "rebuttal_experiments" / "reanalyze_e4.py",
        ROOT / "rebuttal_experiments" / "stability.py",
        ROOT / "rebuttal_experiments" / "kernel_stability.py",
        ROOT / "rebuttal_experiments" / "downdate_amplification.py",
        ROOT / "rebuttal_experiments" / "aggregate.py",
        ROOT / "rebuttal_experiments" / "task_matrix.py",
        ROOT
        / "Cholesky-main-2"
        / "corrected_incremental_cholesky"
        / "run_experiments.py",
    )
    return {relative_path(path): sha256(path) for path in sources if path.is_file()}


def build_analysis(raw_dir: Path, signed_path: Path | None) -> dict[str, Any]:
    configure_threads(1)
    groups, extras = canonical_artifacts(raw_dir)
    controlled = analyze_controlled(groups["stability"])
    kernels = analyze_kernels(groups["kernel_stability"])
    downdates = analyze_downdates(groups["downdate"])
    signed = analyze_signed_correctness(signed_path)
    summary_audit = audit_existing_summary(controlled, kernels, downdates)
    return {
        "schema_version": 1,
        "analysis": "E4 fresh-factorization versus incremental scalar errors",
        "determinism": {
            "threads": 1,
            "torch_version": torch.__version__,
            "timing_performed": False,
            "incremental_kernel_invoked": False,
            "existing_artifacts_modified": False,
            "source_sha256": source_hashes(),
        },
        "methodology": {
            "fresh_baseline": (
                "Replay only the deterministic FP64 G/M state construction and "
                "compute a fresh factorization plus two triangular solves in "
                "the recorded precision at every raw checkpoint."
            ),
            "incremental_excess_definition": (
                "signed scalar diagnostic gap: raw_relative_error - "
                "fresh_relative_error"
            ),
            "non_additivity_warning": (
                "Normwise forward errors do not add. The signed scalar gap is "
                "not ||incremental_state - fresh_state||. Because E4 did not "
                "save tensors, that direct norm is unavailable except where "
                "the protocol itself freshly resets state."
            ),
        },
        "canonical_inventory": {
            "controlled_artifacts": controlled["artifact_count"],
            "kernel_artifacts": kernels["artifact_count"],
            "downdate_artifacts": downdates["artifact_count"],
            "excluded_noncanonical_artifacts": extras,
        },
        "controlled_positive_updates": controlled,
        "near_duplicate_kernels": kernels,
        "downdate_amplification": downdates,
        "supplemental_signed_update_vs_fresh": signed,
        "existing_summary_audit": summary_audit,
        "unavailable_comparisons": [
            {
                "comparison": (
                    "direct incremental-state versus fresh-state tensor norm "
                    "at non-reset checkpoints"
                ),
                "reason": "raw E4 JSONs contain diagnostics but not L/Aw tensors",
            },
            {
                "comparison": "signed downdate trajectory versus fresh",
                "reason": (
                    "canonical E4 downdate files contain amplification analysis, "
                    "not executions of the signed update implementation"
                ),
            },
            {
                "comparison": "near-duplicate kernel no-reset versus reset",
                "reason": "only reset_interval=50 was run for the kernel family",
            },
        ],
    }


def sci(value: float | None) -> str:
    return "—" if value is None else f"{value:.3e}"


def cond_label(value: float) -> str:
    exponent = round(math.log10(value))
    return f"1e{exponent}"


def render_markdown(analysis: dict[str, Any]) -> str:
    controlled = analysis["controlled_positive_updates"]
    kernels = analysis["near_duplicate_kernels"]
    controlled_reset_difference = max(
        controlled["fresh_checkpoint_validation_max_abs_difference"].values()
    )
    kernel_reset_difference = max(
        kernels["fresh_checkpoint_validation_max_abs_difference"].values()
    )
    lines = [
        "# E4 fresh-versus-incremental reanalysis",
        "",
        (
            "This is a deterministic, read-only reanalysis of the canonical E4 "
            "JSONs. No experiment or timing kernel was rerun. Fresh diagnostics "
            "were reconstructed from the saved seeds and configuration."
        ),
        "",
        (
            "For non-reset checkpoints, `raw − fresh` is a signed difference "
            "between two scalar relative errors. It is not a direct norm between "
            "the incremental and fresh tensors because E4 did not save those "
            "tensors."
        ),
        "",
        "## Audit",
        "",
        (
            f"- Canonical artifacts: {controlled['artifact_count']} controlled, "
            f"{kernels['artifact_count']} kernel, "
            f"{analysis['downdate_amplification']['artifact_count']} downdate."
        ),
        (
            "- Maximum absolute reconstruction mismatch at protocol-fresh "
            "controlled checkpoints: "
            f"{controlled_reset_difference:.3e}."
        ),
        (
            "- Maximum absolute reconstruction mismatch at protocol-fresh "
            "kernel checkpoints: "
            f"{kernel_reset_difference:.3e}."
        ),
    ]
    extras = analysis["canonical_inventory"]["excluded_noncanonical_artifacts"]
    lines.append(
        f"- Excluded {len(extras)} noncanonical smoke/development artifacts: "
        + (", ".join(item["path"] for item in extras) if extras else "none")
        + "."
    )
    summary = analysis["existing_summary_audit"]
    if summary["available"]:
        lines.extend(
            [
                (
                    "- Existing summary counts are "
                    f"{summary['stored_counts']}; canonical counts are "
                    f"{summary['canonical_counts']}."
                ),
                (
                    "- Existing-summary canonical final-cell mismatches: "
                    f"{len(summary['positive_final_mismatches'])}."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Controlled-spectrum FP32 checkpoint means",
            "",
            (
                "All values are means over three seeds. `reset50` and `reset200` "
                "show the raw trajectories; before their first reset they are "
                "identical to the no-reset trajectory by construction."
            ),
            "",
            (
                "| target κ | step | fresh | no reset | no reset − fresh | "
                "reset50 | reset200 |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    paired = {
        (
            row["target_condition"],
            row["precision"],
            row["step"],
        ): row
        for row in controlled["paired_aggregate_by_checkpoint"]
    }
    for condition in (1e2, 1e4, 1e6, 1e8):
        for step in (0, 1, 10, 50, 100, 200):
            metric = paired[(condition, "float32", step)]["metrics"][
                "Aw_relative_error"
            ]
            lines.append(
                f"| {cond_label(condition)} | {step} | "
                f"{sci(metric['fresh']['mean'])} | "
                f"{sci(metric['no_reset']['mean'])} | "
                f"{sci(metric['no_reset_minus_fresh']['mean'])} | "
                f"{sci(metric['reset50']['mean'])} | "
                f"{sci(metric['reset200']['mean'])} |"
            )

    lines.extend(
        [
            "",
            "## Controlled-spectrum FP64 no-reset summary",
            "",
            "| target κ | largest mean | step | final mean | fresh mean |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in (1e2, 1e4, 1e6, 1e8):
        candidates = [
            paired[(condition, "float64", step)]
            for step in (0, 1, 10, 50, 100, 200)
        ]
        largest = max(
            candidates,
            key=lambda row: row["metrics"]["Aw_relative_error"]["no_reset"][
                "mean"
            ],
        )
        final = paired[(condition, "float64", 200)]["metrics"][
            "Aw_relative_error"
        ]
        lines.append(
            f"| {cond_label(condition)} | "
            f"{sci(largest['metrics']['Aw_relative_error']['no_reset']['mean'])} | "
            f"{largest['step']} | {sci(final['no_reset']['mean'])} | "
            f"{sci(final['fresh']['mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Near-duplicate kernel summary",
            "",
            (
                "The largest signed gap is selected over the two non-reset "
                "checkpoints (steps 1 and 10). Step 100 follows a fresh reset."
            ),
            "",
            (
                "| precision | operator | δ | initial fresh | "
                "largest raw − fresh | step | final fresh/reset |"
            ),
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    kernel_aggregate = {
        (
            row["precision"],
            row["operator"],
            row["delta"],
            row["step"],
        ): row
        for row in kernels["aggregate_by_checkpoint"]
    }
    for precision in ("float32", "float64"):
        for operator in ("cumulative", "random"):
            for delta in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
                candidates = [
                    kernel_aggregate[(precision, operator, delta, step)]
                    for step in (1, 10)
                ]
                largest = max(
                    candidates,
                    key=lambda row: abs(
                        row["metrics"]["Aw_relative_error"]["raw_minus_fresh"][
                            "mean"
                        ]
                    ),
                )
                initial = kernel_aggregate[
                    (precision, operator, delta, 0)
                ]["metrics"]["Aw_relative_error"]["fresh"]["mean"]
                final = kernel_aggregate[
                    (precision, operator, delta, 100)
                ]["metrics"]["Aw_relative_error"]["raw"]["mean"]
                largest_gap = largest["metrics"]["Aw_relative_error"][
                    "raw_minus_fresh"
                ]["mean"]
                lines.append(
                    f"| {precision} | {operator} | {delta:.0e} | "
                    f"{sci(initial)} | "
                    f"{sci(largest_gap)} | "
                    f"{largest['step']} | {sci(final)} |"
                )

    signed = analysis["supplemental_signed_update_vs_fresh"]
    lines.extend(["", "## Supplemental signed single-update comparison", ""])
    if signed["available"]:
        lines.extend(
            [
                (
                    "This artifact is supplemental, FP64-only, and not part of "
                    "the canonical E4 task matrix. Its stored errors are direct "
                    "comparisons to fresh factorization, averaged over five seeds; "
                    "operator signs are mixed and cannot be disaggregated."
                ),
                "",
                "| r | metric sign | L mean | L max | Aw mean | Aw max |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(
            signed["payload"]["rows"],
            key=lambda item: (item["r"], item["metric_sign"]),
        ):
            lines.append(
                f"| {row['r']} | {row['metric_sign']:+d} | "
                f"{sci(row['L_mean'])} | {sci(row['L_max'])} | "
                f"{sci(row['Aw_mean'])} | {sci(row['Aw_max'])} |"
            )
    else:
        lines.append(f"Unavailable: {signed['reason']}.")

    lines.extend(
        [
            "",
            "## Comparisons the raw campaign cannot support",
            "",
        ]
    )
    for item in analysis["unavailable_comparisons"]:
        lines.append(f"- {item['comparison']}: {item['reason']}.")
    lines.append("")
    return "\n".join(lines)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def run_self_test(raw_dir: Path, signed_path: Path | None) -> None:
    sample = finite_stats([1.0, 3.0])
    assert sample["mean"] == 2.0
    assert sample["sample_sd"] == math.sqrt(2.0)
    assert state_kind(0, 50) == "fresh_initial"
    assert state_kind(10, 50) == "incremental"
    assert state_kind(50, 50) == "fresh_reset"
    comparison = metric_comparison(
        {metric: 3.0 for metric in METRICS},
        {metric: 2.0 for metric in METRICS},
        "incremental",
    )
    assert comparison["raw_minus_fresh_scalar_error"]["Aw_relative_error"] == 1.0
    assert comparison["direct_tensor_error_bounds"]["Aw_relative_error"] == {
        "lower": 1.0,
        "upper": 5.0,
        "normalization": "the corresponding FP64 exact tensor norm",
    }
    analysis = build_analysis(raw_dir, signed_path)
    assert analysis["controlled_positive_updates"]["artifact_count"] == 72
    assert analysis["controlled_positive_updates"]["record_count"] == 432
    assert analysis["near_duplicate_kernels"]["artifact_count"] == 4
    assert analysis["near_duplicate_kernels"]["record_count"] == 360
    assert analysis["downdate_amplification"]["artifact_count"] == 6
    assert (
        max(
            analysis["controlled_positive_updates"][
                "fresh_checkpoint_validation_max_abs_difference"
            ].values()
        )
        == 0.0
    )
    assert (
        max(
            analysis["near_duplicate_kernels"][
                "fresh_checkpoint_validation_max_abs_difference"
            ].values()
        )
        == 0.0
    )
    json.dumps(analysis, allow_nan=False, sort_keys=True)
    markdown = render_markdown(analysis)
    assert "raw − fresh" in markdown
    print(
        "self-test passed: 72 controlled artifacts / 432 checkpoints, "
        "4 kernel artifacts / 360 checkpoints, 6 downdate artifacts; "
        "fresh reset reconstruction is bit-exact",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--signed-correctness", type=Path, default=DEFAULT_SIGNED)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signed_path = args.signed_correctness
    if args.self_test:
        run_self_test(args.raw_dir, signed_path)
        return
    analysis = build_analysis(args.raw_dir, signed_path)
    encoded = json.dumps(
        analysis, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    atomic_write_text(args.output_json, encoded)
    atomic_write_text(args.output_markdown, render_markdown(analysis))
    print(f"wrote {args.output_json}", flush=True)
    print(f"wrote {args.output_markdown}", flush=True)


if __name__ == "__main__":
    main()
