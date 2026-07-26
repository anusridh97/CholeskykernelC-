"""Regression and adversarial mutation tests for the E1 v5 audit."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rebuttal_experiments"))

from audit_e1_confirmation_v5 import AuditError, audit  # noqa: E402


SOURCE = (
    ROOT
    / "rebuttal_results"
    / "e1_confirmation"
    / "e1-confirm-samevector-out-v5"
)


class E1V5AuditMutationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(dir=SOURCE.parent)
        cls.campaign = Path(cls.temporary.name) / SOURCE.name
        shutil.copytree(SOURCE, cls.campaign, copy_function=os.link)
        cls.ledger = cls.campaign / "confirmation_launch_ledger.json"
        cls.accounting = cls.campaign / "confirmation_slurm_accounting.psv"
        cls.report = cls.campaign / "confirmation_report_summary.json"
        # Regression for the frozen driver's one-based epoch trace.
        audit(cls.campaign, cls.ledger, cls.accounting, cls.report)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def mutate_json_and_require_failure(
        self, relative: str, mutation
    ) -> None:
        source = SOURCE / relative
        target = self.campaign / relative
        payload = json.loads(source.read_text(encoding="utf-8"))
        mutation(payload)
        target.unlink()
        target.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            with self.assertRaises(AuditError):
                audit(
                    self.campaign,
                    self.ledger,
                    self.accounting,
                    self.report,
                )
        finally:
            target.unlink()
            os.link(source, target)

    def test_zero_based_epoch_is_rejected(self) -> None:
        self.mutate_json_and_require_failure(
            "confirmation_pair_r256_t1_seed62001.json",
            lambda payload: payload["methods"]["ours"]["raw_step_samples"][
                "epoch"
            ].__setitem__(0, 0),
        )

    def test_unreconciled_application_time_is_rejected(self) -> None:
        def mutation(payload):
            payload["methods"]["ours"]["timing"][
                "application_step_seconds"
            ] += 1.0

        self.mutate_json_and_require_failure(
            "confirmation_pair_r256_t1_seed62001.json", mutation
        )

    def test_empty_kernel_error_map_is_rejected(self) -> None:
        def mutation(payload):
            probe = payload["implementation"]["kernel_equivalence_probe"]
            probe["errors"] = {}
            probe["predeclared_tolerances"] = {}

        self.mutate_json_and_require_failure(
            "confirmation_pair_r256_t1_seed62001.json", mutation
        )

    def test_refresh_core_semantic_mismatch_is_rejected(self) -> None:
        def mutation(payload):
            raw = payload["methods"]["ours"]["raw_step_samples"]
            index = raw["refreshed"].index(False)
            raw["nonrefresh_core_seconds"][index] = 0.0

        self.mutate_json_and_require_failure(
            "confirmation_pair_r256_t1_seed62001.json", mutation
        )

    def test_nonzero_scheduler_restart_is_rejected(self) -> None:
        source = SOURCE / "confirmation_slurm_accounting.psv"
        target = self.accounting
        rows = source.read_text(encoding="utf-8").splitlines()
        fields = rows[0].split("|")
        fields[-3] = "1"
        rows[0] = "|".join(fields)
        target.unlink()
        target.write_text("\n".join(rows) + "\n", encoding="utf-8")
        try:
            with self.assertRaises(AuditError):
                audit(
                    self.campaign,
                    self.ledger,
                    self.accounting,
                    self.report,
                )
        finally:
            target.unlink()
            os.link(source, target)

    def test_boolean_raw_timing_is_rejected(self) -> None:
        self.mutate_json_and_require_failure(
            "confirmation_pair_r256_t1_seed62001.json",
            lambda payload: payload["methods"]["refact"]["raw_step_samples"][
                "reset_seconds"
            ].__setitem__(0, False),
        )

    def test_boolean_test_count_is_rejected(self) -> None:
        self.mutate_json_and_require_failure(
            "confirmation_pair_r256_t1_seed62001.json",
            lambda payload: payload["methods"]["ours"].__setitem__(
                "test_evaluations", True
            ),
        )

    def test_aggregate_preserving_per_step_corruption_is_rejected(self) -> None:
        def mutation(payload):
            values = payload["methods"]["ours"]["raw_step_samples"][
                "step_total_seconds"
            ]
            values[0] += 0.001
            values[1] -= 0.001

        self.mutate_json_and_require_failure(
            "confirmation_pair_r256_t1_seed62001.json", mutation
        )

    def test_negative_unaccounted_timing_is_rejected(self) -> None:
        def mutation(payload):
            values = payload["methods"]["ours"]["raw_step_samples"][
                "unaccounted_seconds"
            ]
            delta = values[0] + 0.001
            values[0] -= delta
            values[1] += delta

        self.mutate_json_and_require_failure(
            "confirmation_pair_r256_t1_seed62001.json", mutation
        )

    def test_false_pilot_pairing_flag_is_rejected(self) -> None:
        self.mutate_json_and_require_failure(
            "pilot_pair_r256_t1_seed63001.json",
            lambda payload: payload["paired_inputs"].__setitem__(
                "identical_model_and_curvature_clones", False
            ),
        )

    def test_corrupted_pilot_epoch_grid_is_rejected(self) -> None:
        self.mutate_json_and_require_failure(
            "pilot_pair_r256_t1_seed63001.json",
            lambda payload: payload["methods"]["ours"]["raw_step_samples"][
                "epoch"
            ].__setitem__(0, 0),
        )

    def test_boolean_task_index_is_rejected(self) -> None:
        self.mutate_json_and_require_failure(
            "confirmation_pair_r256_t1_seed62001.json",
            lambda payload: payload["spec"].__setitem__("task_index", False),
        )


if __name__ == "__main__":
    unittest.main()
