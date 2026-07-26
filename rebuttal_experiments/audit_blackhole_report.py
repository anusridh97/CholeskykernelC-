"""Reconcile the black-hole report section with its locked raw artifacts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_TAG = "blackhole-s4-v1-locked-rerun2"
CAMPAIGN_DIR = ROOT / "rebuttal_results" / "blackhole" / CAMPAIGN_TAG
REPORT = ROOT / "rebuttal_results" / "rebuttal_experiment_report.tex"
PLOT = CAMPAIGN_DIR / "rank_scaling.pdf"
RANKS = [2_048, 4_096, 8_192, 16_384, 32_768, 49_152, 65_536]
STATE_GIB = ["0.0625", "0.25", "1", "4", "16", "36", "64"]


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _require_once(haystack: str, needle: str, label: str) -> None:
    count = haystack.count(needle)
    if count != 1:
        raise AssertionError(f"{label}: expected once, observed {count}: {needle}")


def _scientific(value: float) -> str:
    if value <= 0.0 or not math.isfinite(value):
        raise AssertionError(f"invalid scientific value {value}")
    exponent = math.floor(math.log10(value))
    mantissa = value / (10.0**exponent)
    return f"{mantissa:.2f}\\times10^{{{exponent}}}"


def _fixed_tex(value: float, digits: int) -> str:
    return f"{value:,.{digits}f}".replace(",", "{,}")


def _walk_relative_errors(value: Any) -> Iterable[float]:
    if isinstance(value, dict):
        for key, child in value.items():
            if (
                "relative_error" in key
                and isinstance(child, (int, float))
                and not isinstance(child, bool)
            ):
                yield float(child)
            yield from _walk_relative_errors(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_relative_errors(child)


def main() -> None:
    aggregate = _load(CAMPAIGN_DIR / "aggregate.json")
    if aggregate["campaign_tag"] != CAMPAIGN_TAG:
        raise AssertionError("campaign tag mismatch")
    audit = aggregate["audit"]
    for key, expected in (
        ("passed", True),
        ("all_headline_correctness_valid", True),
        ("headline_claims_withheld", False),
    ):
        if audit.get(key) is not expected:
            raise AssertionError(
                f"aggregate audit did not authorize headline claims: {key}"
            )
    rows = aggregate["ladder"]
    if [row["rank"] for row in rows] != RANKS:
        raise AssertionError("rank ladder mismatch")
    if any(row["n_real_transition_cases"] != 10 for row in rows):
        raise AssertionError("every rank must contain ten real transitions")
    if any(not row["headline_valid"] for row in rows):
        raise AssertionError("a rank failed the correctness gate")
    if any(
        min(row["paired_event_speedup_fresh_over_inverse"]["values"]) <= 1.0
        for row in rows
    ):
        raise AssertionError("not every paired transition favors inverse-factor")

    raw_by_rank = {
        rank: _load(CAMPAIGN_DIR / f"ladder_r{rank}.json") for rank in RANKS
    }
    report = _compact(REPORT.read_text(encoding="utf-8"))

    inverse_decimals = [3, 3, 3, 3, 3, 3, 3]
    fresh_decimals = [4, 4, 4, 3, 3, 3, 3]
    fresh_sd_decimals = [4, 4, 4, 3, 3, 3, 3]
    for index, (row, state_gib) in enumerate(zip(rows, STATE_GIB)):
        inverse = row["inverse_event_seconds"]
        fresh = row["fresh_event_seconds"]
        speedup = row["paired_event_speedup_fresh_over_inverse"]
        inv_digits = inverse_decimals[index]
        fresh_digits = fresh_decimals[index]
        fresh_sd_digits = fresh_sd_decimals[index]
        expected = (
            f"{row['rank']:,} & "
            f"\\({_fixed_tex(inverse['mean'] * 1_000, inv_digits)}"
            f"\\pm{_fixed_tex(inverse['sample_sd'] * 1_000, inv_digits)}\\) & "
            f"\\({_fixed_tex(fresh['mean'] * 1_000, fresh_digits)}"
            f"\\pm{_fixed_tex(fresh['sample_sd'] * 1_000, fresh_sd_digits)}\\) & "
            f"\\({speedup['mean']:.4f}\\pm{speedup['sample_sd']:.4f}\\) & "
            f"\\([{speedup['ci95_low']:.4f},{speedup['ci95_high']:.4f}]\\) & "
            f"{state_gib} \\\\"
        )
        _require_once(report, expected, f"timing table rank {row['rank']}")

        cross_factor = max(
            item["factor"] for item in row["cross_method_probe_max_relative_errors"]
        )
        cross_operator = max(
            item["operator"]
            for item in row["cross_method_probe_max_relative_errors"]
        )
        raw = raw_by_rank[row["rank"]]
        dense_factor = [
            case["methods"]["inverse"]["validation"]["full_dense"][
                "factor_relative_error"
            ]
            for case in raw["cases"]
            if case["methods"]["inverse"]["validation"]["full_dense"] is not None
        ]
        dense_operator = [
            case["methods"]["inverse"]["validation"]["full_dense"][
                "operator_relative_error"
            ]
            for case in raw["cases"]
            if case["methods"]["inverse"]["validation"]["full_dense"] is not None
        ]
        if row["rank"] <= 32_768:
            dense = (
                f"\\({_scientific(max(dense_factor))}/"
                f"{_scientific(max(dense_operator))}\\)"
            )
            scope = "dense+probe+finite"
        else:
            if dense_factor or dense_operator:
                raise AssertionError("large-rank full-dense check unexpectedly exists")
            dense = "--"
            scope = "probe+finite"
        expected_validation = (
            f"{row['rank']:,} & "
            f"{min(row['condition_numbers']):.2f}--"
            f"{max(row['condition_numbers']):.2f} & "
            f"\\({_scientific(cross_factor)}\\) & "
            f"\\({_scientific(cross_operator)}\\) & "
            f"{dense} & {scope} \\\\"
        )
        _require_once(
            report,
            expected_validation,
            f"validation table rank {row['rank']}",
        )

    all_relative_errors = [
        value
        for raw in raw_by_rank.values()
        for value in _walk_relative_errors(raw)
    ]
    worst = max(all_relative_errors)
    if f"{worst:.2e}" != "1.84e-04":
        raise AssertionError(f"unexpected worst diagnostic {worst}")

    first_speedup = rows[0]["paired_event_speedup_fresh_over_inverse"]
    last_speedup = rows[-1]["paired_event_speedup_fresh_over_inverse"]
    last_inverse = rows[-1]["inverse_event_seconds"]
    last_fresh = rows[-1]["fresh_event_seconds"]
    required_claims = [
        f"\\(1.7119\\pm0.0085\\times\\)",
        f"\\(71.7546\\pm0.2247\\times\\)",
        f"\\([{last_speedup['ci95_low']:.4f},{last_speedup['ci95_high']:.4f}]\\)",
        f"\\({last_inverse['mean'] * 1_000:.3f}"
        f"\\pm{last_inverse['sample_sd'] * 1_000:.3f}\\)\\,ms",
        f"\\({last_fresh['mean']:.6f}"
        f"\\pm{last_fresh['sample_sd']:.6f}\\)\\,s",
        "\\(1.84\\times10^{-4}\\)",
    ]
    if round(first_speedup["mean"], 4) != 1.7119:
        raise AssertionError("first-rank speedup changed")
    for claim in required_claims:
        if claim not in report:
            raise AssertionError(f"missing aggregate-grounded claim: {claim}")

    for exact_identifier in (
        aggregate["audit"]["locked_git_commit"],
        aggregate["audit"]["campaign_lock_git_commit"],
        aggregate["audit"]["campaign_lock_sha256"],
        aggregate["audit"]["feature_manifest_sha256"],
        aggregate["audit"]["feature_trace_sha256"],
        aggregate["audit"]["slurm_array_job_id"],
    ):
        if exact_identifier not in report:
            raise AssertionError(f"missing provenance identifier {exact_identifier}")

    labels = set(re.findall(r"\\label\{([^}]+)\}", report))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", report))
    for label in (
        "sec:e5-blackhole",
        "tab:e5-blackhole-scale",
        "fig:e5-blackhole-scale",
        "tab:e5-blackhole-valid",
    ):
        if label not in labels:
            raise AssertionError(f"missing label {label}")
    for reference in (
        "sec:e5-blackhole",
        "tab:e5-blackhole-scale",
        "fig:e5-blackhole-scale",
        "tab:e5-blackhole-valid",
    ):
        if reference not in refs:
            raise AssertionError(f"black-hole artifact is not referenced: {reference}")

    if not PLOT.is_file() or PLOT.stat().st_size < 1_000:
        raise AssertionError("rank-scaling plot is missing")
    log_path = ROOT / "rebuttal_results" / "rebuttal_experiment_report.log"
    if log_path.is_file():
        log = log_path.read_text(encoding="utf-8", errors="replace")
        for forbidden in ("Overfull \\hbox", "Undefined control sequence"):
            if forbidden in log:
                raise AssertionError(f"LaTeX log contains {forbidden}")

    print(
        "black-hole report audit passed: "
        "7 ranks, 70 paired transitions, 14 displayed rows"
    )


if __name__ == "__main__":
    main()
