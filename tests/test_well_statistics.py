"""Exercise rank-ladder reporting without a CUDA device."""

import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from benchmarks.the_well import rank_ladder


class TestWellStatistics(unittest.TestCase):
    def test_probe_error_summary(self):
        """Report per-probe errors and their median from known CPU tensors."""
        expected = torch.tensor([[3.0, 0.0], [4.0, 5.0]], dtype=torch.float64)
        actual = expected * torch.tensor([1.1, 1.2], dtype=torch.float64)
        result = rank_ladder._column_errors(actual, expected)
        for actual_error, expected_error in zip(result["per_probe_relative_errors"], [0.1, 0.2]):
            self.assertAlmostEqual(actual_error, expected_error)
        self.assertAlmostEqual(result["median_relative_error"], 0.15)
        self.assertAlmostEqual(result["max_relative_error"], 0.2)
        self.assertAlmostEqual(result["frobenius_relative_error"], math.sqrt(0.025))

    def test_timing_summary(self):
        """Convert event milliseconds to seconds and summarize unsorted samples."""
        event = Mock()
        event.elapsed_time.side_effect = [30.0, 10.0, 20.0]
        state = SimpleNamespace(prepare=Mock(return_value=0.5))
        with (
            patch.object(torch.cuda, "Event", return_value=event),
            patch.object(torch.cuda, "synchronize"),
            patch.object(torch.cuda, "reset_peak_memory_stats"),
            patch.object(torch.cuda, "max_memory_allocated", return_value=1024),
            patch.object(torch.cuda, "max_memory_reserved", return_value=2048),
            patch.object(rank_ladder, "_update_inverse") as update,
        ):
            result = rank_ladder._time_method(
                method="inverse", state=state, z=None, w=None, workspace=None,
                update_function=None, tile=8, repeats=3,
            )
        self.assertEqual(result["event_seconds_samples"], [0.03, 0.01, 0.02])
        self.assertEqual(result["event_seconds_median"], 0.02)
        self.assertEqual(result["updates_per_second_from_event_median"], 50.0)
        self.assertEqual(result["state_restoration_seconds_samples"], [0.5] * 3)
        self.assertEqual(result["state_restoration_seconds_median"], 0.5)
        self.assertEqual(update.call_count, 3)
        self.assertEqual(state.prepare.call_count, 3)
