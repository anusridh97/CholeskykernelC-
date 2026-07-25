from __future__ import annotations

import itertools
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rebuttal_experiments.blackhole_features import (
    CHANNELS,
    EXPECTED_ROOT_KEYS,
    SCALAR_FIELDS,
    SPATIAL_DIMS,
    VECTOR_FIELDS,
    RunningMoments,
    _inspect_hdf5,
    _select_low_frequencies,
    _write_channel_coefficients,
)


class RunningMomentsTest(unittest.TestCase):
    def test_streamed_merges_match_direct_population_statistics(self) -> None:
        batches = (
            np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            np.asarray([-5.0, 9.0], dtype=np.float64),
            np.linspace(-2.0, 4.0, 17, dtype=np.float64),
        )
        moments = RunningMoments()
        for batch in batches:
            moments.update(batch)
        expected = np.concatenate(batches).astype(np.float64)
        self.assertEqual(moments.count, expected.size)
        self.assertAlmostEqual(moments.mean, float(expected.mean()), places=14)
        self.assertAlmostEqual(moments.std, float(expected.std(ddof=0)), places=14)


class FrequencyOrderTest(unittest.TestCase):
    def test_integer_order_matches_exhaustive_normalized_frequency_rule(self) -> None:
        shape = (4, 3, 2)
        count = 17
        actual = _select_low_frequencies(shape, count)
        expected = sorted(
            itertools.product(*(range(size) for size in shape)),
            key=lambda index: (
                sum(
                    (index[axis] / shape[axis]) ** 2
                    for axis in range(3)
                ),
                index,
            ),
        )[:count]
        np.testing.assert_array_equal(
            actual, np.asarray(expected, dtype=np.int32)
        )

    def test_frequency_major_channel_interleave_is_a_nested_prefix(self) -> None:
        shape = (4, 3, 2)
        indices = _select_low_frequencies(shape, 2)
        row = np.empty(19, dtype=np.float32)
        for channel_index in range(len(CHANNELS)):
            coefficients = (
                np.arange(math.prod(shape), dtype=np.float64).reshape(shape)
                + 1000.0 * channel_index
            )
            _write_channel_coefficients(
                row,
                channel_index,
                coefficients,
                indices,
            )
        for position, value in enumerate(row):
            frequency = position // len(CHANNELS)
            channel = position % len(CHANNELS)
            coordinate = tuple(indices[frequency])
            expected = (
                np.arange(math.prod(shape), dtype=np.float64).reshape(shape)[
                    coordinate
                ]
                + 1000.0 * channel
            )
            self.assertEqual(float(value), float(expected))


class Hdf5SchemaTest(unittest.TestCase):
    def test_tiny_well_file_passes_and_dtype_change_fails_closed(self) -> None:
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py is unavailable")

        spatial_shape = (4, 3, 2)
        n_snapshots = 5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.hdf5"
            with h5py.File(path, "w") as handle:
                handle.attrs["dataset_name"] = "post_neutron_star_merger"
                handle.attrs["grid_type"] = "spherical"
                handle.attrs["n_spatial_dims"] = 3
                handle.attrs["n_trajectories"] = 1
                for name in EXPECTED_ROOT_KEYS:
                    handle.create_group(name)

                dimensions = handle["dimensions"]
                dimensions.attrs["spatial_dims"] = np.asarray(
                    SPATIAL_DIMS, dtype="S"
                )
                dimensions.create_dataset("time", shape=(n_snapshots,))
                for name, size in zip(SPATIAL_DIMS, spatial_shape):
                    dimensions.create_dataset(name, shape=(size,))

                t0 = handle["t0_fields"]
                t0.attrs["field_names"] = np.asarray(
                    SCALAR_FIELDS, dtype="S"
                )
                scalar_shape = (1, n_snapshots, *spatial_shape)
                for name in SCALAR_FIELDS:
                    dataset = t0.create_dataset(
                        name, shape=scalar_shape, dtype=np.float32
                    )
                    dataset.attrs["sample_varying"] = True
                    dataset.attrs["time_varying"] = True
                    dataset.attrs["dim_varying"] = np.ones(3, dtype=bool)

                t1 = handle["t1_fields"]
                t1.attrs["field_names"] = np.asarray(
                    VECTOR_FIELDS, dtype="S"
                )
                for name in VECTOR_FIELDS:
                    dataset = t1.create_dataset(
                        name,
                        shape=(*scalar_shape, 3),
                        dtype=np.float32,
                    )
                    dataset.attrs["sample_varying"] = True
                    dataset.attrs["time_varying"] = True
                    dataset.attrs["dim_varying"] = np.ones(3, dtype=bool)

                handle["t2_fields"].attrs["field_names"] = np.asarray(
                    (), dtype="S1"
                )
                handle["additional_information"].create_dataset(
                    "gcon", shape=(*spatial_shape, 10), dtype=np.float32
                )

            with h5py.File(path, "r") as handle:
                metadata = _inspect_hdf5(
                    handle,
                    expected_n_trajectories=1,
                    expected_n_snapshots=n_snapshots,
                    expected_spatial_shape=spatial_shape,
                )
                self.assertEqual(
                    metadata["metric_dataset_paths_excluded"], ["additional_information/gcon"]
                )
                self.assertEqual(metadata["spatial_shape"], list(spatial_shape))

            with h5py.File(path, "r+") as handle:
                del handle["t0_fields"]["density"]
                dataset = handle["t0_fields"].create_dataset(
                    "density", shape=scalar_shape, dtype=np.float64
                )
                dataset.attrs["sample_varying"] = True
                dataset.attrs["time_varying"] = True
                dataset.attrs["dim_varying"] = np.ones(3, dtype=bool)
            with h5py.File(path, "r") as handle:
                with self.assertRaisesRegex(ValueError, "expected float32"):
                    _inspect_hdf5(
                        handle,
                        expected_n_trajectories=1,
                        expected_n_snapshots=n_snapshots,
                        expected_spatial_shape=spatial_shape,
                    )


if __name__ == "__main__":
    unittest.main()
