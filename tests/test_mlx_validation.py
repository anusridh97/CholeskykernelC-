"""Small CPU-only checks of the evidence gate; never imports MLX or its kernel."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from benchmarks.mlx import validate_comparison as driver


class FakeMetal:
    """Array plumbing only, used to test orchestration without a GPU."""
    array = staticmethod(np.array)
    eval = staticmethod(lambda *args: None)
    synchronize = staticmethod(lambda: None)


class ExactCPUUpdate:
    @staticmethod
    def incremental_update_mlx_batched(L, Aw, z, w):
        L, Aw, z, w = [v.astype(np.float64) for v in (L, Aw, z, w)]
        G = L @ L.swapaxes(-1, -2)
        M = L @ Aw @ L.swapaxes(-1, -2)
        updated = driver.make_reference(np, dict(G=G, M=M, z=z, w=w))
        return updated[2:]


class EvidenceGateTest(unittest.TestCase):
    def setUp(self):
        self.p = driver.problem(np, 7, 4, 6)
        self.ref = driver.make_reference(np, self.p)

    def test_consistent_input_and_exact_output(self):
        L = self.p['L'].astype(np.float64)
        Aw = self.p['Aw'].astype(np.float64)
        self.assertTrue(driver.errors(np, L @ Aw @ L.swapaxes(-1, -2), self.p['M'])['passed'])
        result = driver.validate(np, {'mlx_batched_update': self.ref[2:]}, self.ref)
        self.assertTrue(result['passed'])

    def test_single_corrupted_batch_is_rejected(self):
        output = [v.copy() for v in self.ref[2:]]
        output[1][3] *= 1.01
        result = driver.validate(np, {'mlx_batched_update': output}, self.ref)
        self.assertFalse(result['passed'])
        errors = result['methods']['mlx_batched_update']['checks']['operator_vs_fp64']
        self.assertEqual(len(errors['per_item_relative_frobenius']), 4)
        self.assertGreater(errors['per_item_relative_frobenius'][3], driver.TOL)

    def test_nonfinite_shape_and_auxiliary_failures(self):
        self.assertFalse(driver.errors(np, np.array([[np.nan]]), np.ones((1, 1)))['passed'])
        self.assertFalse(driver.errors(np, np.ones((2, 1)), np.ones((1, 1)))['passed'])
        output = [v.copy() for v in self.ref[2:]]
        output[2][0] *= 1.1
        self.assertFalse(driver.validate(np, {'mlx_batched_update': output}, self.ref)['passed'])

    def test_all_baselines_and_record_flow_on_tiny_cpu_problem(self):
        args = SimpleNamespace(seed=0, time=False)
        result = driver.one_cell(np, torch, FakeMetal, ExactCPUUpdate, args, 7, 4)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(len(result['correctness']['methods']), 3)
        self.assertEqual(len(result['input_hashes']), 6)
        self.assertNotIn('timing', result)

    def test_old_independent_Aw_input_is_rejected_before_execution(self):
        args = SimpleNamespace(seed=0, time=False)
        self.p['Aw'] = np.ones_like(self.p['Aw'])
        with patch.object(driver, 'problem', return_value=self.p):
            result = driver.one_cell(np, torch, None, None, args, 7, 4)
        self.assertEqual(result['status'], 'failed_initial_consistency')


if __name__ == '__main__':
    unittest.main()
