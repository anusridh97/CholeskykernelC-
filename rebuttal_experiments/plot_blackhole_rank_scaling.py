"""Plot the audited locked black-hole rank-scaling aggregate.

The plot treats the ten predeclared physical transitions as the statistical
units.  Repeat-level timings are intentionally not promoted to independent
observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_TAG = "blackhole-s4-v1-locked-rerun2"
DEFAULT_CAMPAIGN_DIR = (
    ROOT / "rebuttal_results" / "blackhole" / CAMPAIGN_TAG
)
EXPECTED_RANKS = [2_048, 4_096, 8_192, 16_384, 32_768, 49_152, 65_536]


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("campaign_tag") != CAMPAIGN_TAG:
        raise ValueError(f"unexpected campaign tag in {path}")
    audit = payload.get("audit", {})
    if not audit.get("passed") or not audit.get(
        "all_headline_correctness_valid"
    ):
        raise ValueError("refusing to plot an unaudited or invalid campaign")
    rows = payload.get("ladder", [])
    if [row.get("rank") for row in rows] != EXPECTED_RANKS:
        raise ValueError("aggregate rank ladder does not match the frozen design")
    for row in rows:
        if row.get("n_real_transition_cases") != 10:
            raise ValueError("each rank must contain ten real transitions")
        if not row.get("headline_valid"):
            raise ValueError(f"rank {row.get('rank')} failed the correctness gate")
    return payload


def _interval_errors(
    rows: list[dict[str, Any]], field: str, *, scale: float = 1.0
) -> tuple[list[float], list[list[float]]]:
    means = [row[field]["mean"] * scale for row in rows]
    lower = [
        (row[field]["mean"] - row[field]["ci95_low"]) * scale
        for row in rows
    ]
    upper = [
        (row[field]["ci95_high"] - row[field]["mean"]) * scale
        for row in rows
    ]
    return means, [lower, upper]


def plot(aggregate: Path, output: Path) -> None:
    payload = _load(aggregate)
    rows = payload["ladder"]
    ranks = [row["rank"] for row in rows]

    inverse_ms, inverse_err = _interval_errors(
        rows, "inverse_event_seconds", scale=1_000.0
    )
    fresh_ms, fresh_err = _interval_errors(
        rows, "fresh_event_seconds", scale=1_000.0
    )
    speedup, speedup_err = _interval_errors(
        rows, "paired_event_speedup_fresh_over_inverse"
    )

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, (latency_axis, speedup_axis) = plt.subplots(
        1, 2, figsize=(7.0, 2.75), constrained_layout=True
    )
    colors = {"inverse": "#2878B5", "fresh": "#D95F02"}

    latency_axis.errorbar(
        ranks,
        inverse_ms,
        yerr=inverse_err,
        marker="o",
        markersize=4,
        linewidth=1.5,
        capsize=2,
        color=colors["inverse"],
        label="Inverse-factor update",
    )
    latency_axis.errorbar(
        ranks,
        fresh_ms,
        yerr=fresh_err,
        marker="s",
        markersize=3.8,
        linewidth=1.5,
        capsize=2,
        color=colors["fresh"],
        label="Fresh refactorization",
    )
    latency_axis.set_xscale("log", base=2)
    latency_axis.set_yscale("log")
    latency_axis.set_xlabel("rank $r$")
    latency_axis.set_ylabel("CUDA-event latency (ms)")
    latency_axis.set_title("(a) Real-transition latency")
    latency_axis.grid(True, which="both", color="#D9D9D9", linewidth=0.5)
    latency_axis.legend(frameon=False, loc="upper left")

    speedup_axis.errorbar(
        ranks,
        speedup,
        yerr=speedup_err,
        marker="o",
        markersize=4,
        linewidth=1.6,
        capsize=2,
        color="#3B8F55",
    )
    speedup_axis.axhline(1.0, color="#555555", linewidth=0.9, linestyle="--")
    speedup_axis.set_xscale("log", base=2)
    speedup_axis.set_yscale("log")
    speedup_axis.set_xlabel("rank $r$")
    speedup_axis.set_ylabel("paired speedup ($\\times$)")
    speedup_axis.set_title("(b) Fresh / inverse-factor")
    speedup_axis.grid(True, which="both", color="#D9D9D9", linewidth=0.5)

    tick_labels = ["2k", "4k", "8k", "16k", "33k", "49k", "66k"]
    for axis in (latency_axis, speedup_axis):
        axis.set_xticks(ranks, labels=tick_labels)
        axis.get_xaxis().set_minor_formatter(plt.NullFormatter())
        axis.tick_params(axis="x", labelrotation=25)
        for tick_label in axis.get_xticklabels():
            tick_label.set_horizontalalignment("right")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": "Locked black-hole real-data rank scaling",
            "Author": "Anonymous Authors",
            "Subject": CAMPAIGN_TAG,
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=DEFAULT_CAMPAIGN_DIR / "aggregate.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CAMPAIGN_DIR / "rank_scaling.pdf",
    )
    args = parser.parse_args()
    plot(args.aggregate.resolve(), args.output.resolve())
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
