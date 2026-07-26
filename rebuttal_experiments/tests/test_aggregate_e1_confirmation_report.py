from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "rebuttal_experiments"
sys.path.insert(0, str(HERE))

from aggregate_e1_confirmation_report import (  # noqa: E402
    AuditError,
    _estimate,
    core_ms_per_call,
)


def _method(
    *,
    steps: int,
    refresh_count: int,
    core_seconds: float,
) -> dict:
    return {
        "steps": steps,
        "refresh_count": refresh_count,
        "timing": {
            "component_totals": {
                "nonrefresh_core_seconds": core_seconds,
            }
        },
    }


class ReportAggregateTest(unittest.TestCase):
    def test_core_time_uses_actual_nonrefresh_invocation_count(self) -> None:
        ours = _method(steps=100, refresh_count=20, core_seconds=2.0)
        refact = _method(steps=100, refresh_count=40, core_seconds=3.0)

        ours_ms = core_ms_per_call(ours)
        refact_ms = core_ms_per_call(refact)

        self.assertAlmostEqual(ours_ms, 25.0)
        self.assertAlmostEqual(refact_ms, 50.0)
        self.assertAlmostEqual(refact_ms / ours_ms, 2.0)
        self.assertAlmostEqual(3.0 / 2.0, 1.5)

    def test_core_time_rejects_zero_nonrefresh_calls(self) -> None:
        method = _method(steps=100, refresh_count=100, core_seconds=0.0)
        with self.assertRaisesRegex(AuditError, "must be positive"):
            core_ms_per_call(method)

    def test_t_interval_uses_ten_paired_seed_values(self) -> None:
        values = [float(value) for value in range(1, 11)]
        result = _estimate(values)
        expected_half_width = (
            2.2621571627409915
            * result["sample_sd"]
            / math.sqrt(len(values))
        )
        self.assertEqual(result["n"], 10)
        self.assertAlmostEqual(result["mean"], 5.5)
        self.assertAlmostEqual(
            result["ci95_low"], 5.5 - expected_half_width
        )
        self.assertAlmostEqual(
            result["ci95_high"], 5.5 + expected_half_width
        )

    def test_t_interval_rejects_unpaired_or_nonfinite_input(self) -> None:
        with self.assertRaisesRegex(AuditError, "expected 10 paired"):
            _estimate([1.0] * 9)
        with self.assertRaisesRegex(AuditError, "non-finite"):
            _estimate([1.0] * 9 + [math.nan])


if __name__ == "__main__":
    unittest.main()
