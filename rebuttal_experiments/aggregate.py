from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from .common import RESULTS, atomic_json_dump
from .task_matrix import gpu_tasks, ng_tasks, synthetic_tasks
from .worker import expected_output


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def mean_std(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def completion() -> dict:
    groups = {
        "ng": len(ng_tasks()),
        "synthetic": len(synthetic_tasks()),
        "gpu": len(gpu_tasks()),
    }
    output = {}
    for group, count in groups.items():
        missing = [
            index for index in range(count) if not expected_output(group, index).is_file()
        ]
        output[group] = {
            "expected": count,
            "complete": count - len(missing),
            "missing_indices": missing,
        }
    return output


def planned_synthetic_payloads(kind: str):
    for index, task in enumerate(synthetic_tasks()):
        if task["kind"] != kind:
            continue
        path = expected_output("synthetic", index)
        if path.is_file():
            yield read_json(path)


def aggregate_ng() -> dict:
    grouped = defaultdict(list)
    payloads = []
    for index in range(len(ng_tasks())):
        path = expected_output("ng", index)
        if not path.is_file():
            continue
        item = read_json(path)
        payloads.append(item)
        grouped[(item["method"], item["rank"], item["tau"])].append(item)

    rows = []
    for (method, rank, tau), items in sorted(grouped.items()):
        rows.append(
            {
                "method": method,
                "rank": rank,
                "tau": tau,
                "test_mse": mean_std([item["final_test_mse"] for item in items]),
                "wall_seconds": mean_std([item["wall_seconds"] for item in items]),
                "refresh_frequency": mean_std(
                    [item["refresh_count"] / item["steps"] for item in items]
                ),
                "refresh_time_share": mean_std(
                    [item["refresh_time_share"] for item in items]
                ),
                "max_reset_drift": max(
                    (
                        item["max_reset_drift"]
                        for item in items
                        if item["max_reset_drift"] is not None
                    ),
                    default=None,
                ),
            }
        )

    budget = None
    ours_1024 = [
        item
        for item in payloads
        if item["method"] == "ours"
        and item["rank"] == 1024
        and item["tau"] == 0.1
    ]
    if ours_1024:
        budget = statistics.mean(item["wall_seconds"] for item in ours_1024)
    budget_rows = []
    if budget is not None:
        by_method_rank = defaultdict(list)
        for item in payloads:
            if item["tau"] != 0.1:
                continue
            eligible = [
                record["test_mse"]
                for record in item["history"]
                if record["wall_seconds"] <= budget
            ]
            if eligible:
                by_method_rank[(item["method"], item["rank"])].append(min(eligible))
        for (method, rank), values in sorted(by_method_rank.items()):
            budget_rows.append(
                {
                    "method": method,
                    "rank": rank,
                    "budget_seconds": budget,
                    "best_test_mse": mean_std(values),
                }
            )
    return {"rows": rows, "measured_budget_seconds": budget, "budget_rows": budget_rows}


def aggregate_representation() -> list[dict]:
    output = []
    for item in planned_synthetic_payloads("representation"):
        rows = item["rows"]

        def first_crossing(field: str):
            return next(
                (
                    row["factor_width"]
                    for row in rows
                    if row["two_sided_seconds"] <= row[field]
                ),
                None,
            )

        output.append(
            {
                "rank": item["rank"],
                "observed_drag_crossover_p": first_crossing("drag_seconds"),
                "observed_drag_materialize_crossover_p": first_crossing(
                    "drag_materialize_seconds"
                ),
                "observed_fresh_crossover_p": first_crossing("fresh_solve_seconds"),
                "observed_fresh_materialize_crossover_p": first_crossing(
                    "fresh_materialize_seconds"
                ),
                "observed_memory_crossover_p": next(
                    (
                        row["factor_width"]
                        for row in rows
                        if row["factor_state_bytes"] >= row["dense_state_bytes"]
                    ),
                    None,
                ),
                "analytic_memory_crossover_p": item["analytic_memory_crossover_p"],
                "analytic_drag_time_crossover_p": item[
                    "analytic_drag_time_crossover_p"
                ],
                "max_relative_error": max(
                    max(row["drag_relative_error"], row["fresh_relative_error"])
                    for row in rows
                ),
            }
        )
    return output


def aggregate_rank_k() -> list[dict]:
    output = []
    for item in planned_synthetic_payloads("rank_k"):
        output.append(
            {
                "rank": item["rank"],
                "predicted_flop_break_even_k": item["predicted_flop_break_even_k"],
                "rows": item["rows"],
            }
        )
    return output


def aggregate_stability() -> dict:
    positive = []
    downdates = []
    kernels = []
    for kind in ("stability", "downdate", "kernel_stability"):
        for item in planned_synthetic_payloads(kind):
            if item["experiment"] == "positive_update_stability":
                final = item["records"][-1] if item["records"] else None
                positive.append(
                    {
                        "target_condition": item["target_condition"],
                        "precision": item["precision"],
                        "reset_interval": item["reset_interval"],
                        "seed": item["seed"],
                        "status": item["status"],
                        "final": final,
                    }
                )
            elif item["experiment"] == "downdate_amplification":
                downdates.append(
                    {
                        "rank": item["rank"],
                        "seed": item["seed"],
                        "rows": item["rows"],
                    }
                )
            elif item["experiment"] == "near_duplicate_kernel_stability":
                kernels.append(item)
    return {
        "positive_updates": positive,
        "near_duplicate_kernels": kernels,
        "downdates": downdates,
    }


def aggregate_cuda() -> list[dict]:
    paths = (expected_output("gpu", index) for index in range(len(gpu_tasks())))
    return [read_json(path) for path in paths if path.is_file()]


def render_markdown(summary: dict) -> str:
    lines = ["# Rebuttal experiment summary", ""]
    lines.append("## Completion")
    lines.append("")
    lines.append("| Group | Complete | Expected |")
    lines.append("|---|---:|---:|")
    for group, state in summary["completion"].items():
        lines.append(f"| {group} | {state['complete']} | {state['expected']} |")

    lines.extend(["", "## E1: refresh-threshold sweep", ""])
    lines.append(
        "| Method | r | tau | Refresh frequency | Test MSE | Wall time (s) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in summary["ng"]["rows"]:
        lines.append(
            f"| {row['method']} | {row['rank']} | {row['tau']:g} | "
            f"{row['refresh_frequency']['mean']:.3f} ± "
            f"{row['refresh_frequency']['std']:.3f} | "
            f"{row['test_mse']['mean']:.4f} ± {row['test_mse']['std']:.4f} | "
            f"{row['wall_seconds']['mean']:.1f} ± "
            f"{row['wall_seconds']['std']:.1f} |"
        )

    lines.extend(["", "## E3: representation crossovers", ""])
    lines.append(
        "| r | Drag p | Drag+materialize p | Fresh p | Fresh+materialize p | "
        "Memory p |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in summary["representation"]:
        lines.append(
            f"| {row['rank']} | {row['observed_drag_crossover_p']} | "
            f"{row['observed_drag_materialize_crossover_p']} | "
            f"{row['observed_fresh_crossover_p']} | "
            f"{row['observed_fresh_materialize_crossover_p']} | "
            f"{row['observed_memory_crossover_p']} |"
        )

    lines.extend(["", "## E6: rank-k", ""])
    lines.append("| r | k | Repeated rank-1 / refact speedup |")
    lines.append("|---:|---:|---:|")
    for item in summary["rank_k"]:
        for row in item["rows"]:
            lines.append(f"| {item['rank']} | {row['k']} | {row['speedup']:.3f}x |")

    lines.extend(["", "## E5: CUDA", ""])
    lines.append("| Impl | r | Batch | Speedup | Aw relative error |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in summary["cuda"]:
        lines.append(
            f"| {row['impl']} | {row['rank']} | {row['batch']} | "
            f"{row['event_speedup']:.3f}x | {row['Aw_relative_error']:.3e} |"
        )
    lines.append("")
    lines.append(
        "Natural-gradient results use the reconstructed corrected §3.5 backend; "
        "the copied Cholesky-main artifact does not reproduce the submitted "
        "Setup-B result or contain the original Protein .mat file."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    summary = {
        "completion": completion(),
        "ng": aggregate_ng(),
        "representation": aggregate_representation(),
        "rank_k": aggregate_rank_k(),
        "stability": aggregate_stability(),
        "cuda": aggregate_cuda(),
    }
    atomic_json_dump(summary, RESULTS / "summary.json")
    (RESULTS / "SUMMARY.md").write_text(render_markdown(summary), encoding="utf-8")
    print(RESULTS / "summary.json")
    print(RESULTS / "SUMMARY.md")


if __name__ == "__main__":
    main()
