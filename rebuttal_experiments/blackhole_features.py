"""Prepare the frozen real-data feature trace for the black-hole rank sweep.

The input is one pinned trajectory from The Well's
``post_neutron_star_merger`` dataset.  This program deliberately does not
download it: the raw file must already be present and must match both the
frozen byte count and SHA-256 digest before HDF5 is opened.

The output is an unscaled, C-contiguous ``float32`` NumPy array with one row
per snapshot.  For each of the twelve dynamic channels, we

1. estimate a scalar mean and population standard deviation in FP64 from
   snapshots 0--63 (inclusive);
2. normalize every snapshot in FP64;
3. apply a separable, orthonormal three-dimensional DCT-II; and
4. retain coefficients in increasing normalized spatial frequency.

Coefficients are interleaved frequency-major, with the frozen channel order
inside each frequency.  Consequently every tested rank is a prefix of the
same rank-65536 representation.  The trace itself remains unscaled.  The
manifest records, for each tested rank r,

    s_r^2 = (1 / 64) * sum_{t=0}^{63} ||phi_r(x_t)||_2^2,

and consumers divide the rank-r prefix by s_r.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "blackhole_features_v1"
CAMPAIGN_TAG = "blackhole-s4-v1"

DATASET_REPOSITORY = "polymathic-ai/post_neutron_star_merger"
DATASET_REVISION = "1fd582f3e29749ac4d0c14bc816bddb2394258cc"
DATASET_RELATIVE_PATH = (
    "data/train/post_neutron_star_merger_scenario_4.hdf5"
)
DATASET_EXPECTED_BYTES = 14_109_638_656
DATASET_EXPECTED_SHA256 = (
    "21e587c90c006c4b7d37e2b04899bd7e928967ae506e9ceb5afd05dfca6909fd"
)
# Hugging Face exposes the content digest in the Xet-backed file metadata.
DATASET_HF_X_LINKED_ETAG = DATASET_EXPECTED_SHA256

DEFAULT_RAW_PATH = (
    ROOT
    / "data"
    / "the_well"
    / "post_neutron_star_merger"
    / DATASET_RELATIVE_PATH
)
DEFAULT_TRACE_PATH = (
    ROOT
    / "data"
    / "the_well"
    / "processed"
    / "post_neutron_star_merger_scenario_4_dct65536.npy"
)
DEFAULT_MANIFEST_PATH = (
    ROOT
    / "rebuttal_results"
    / "blackhole"
    / CAMPAIGN_TAG
    / "feature_manifest.json"
)

N_TRAJECTORIES = 1
N_SNAPSHOTS = 181
SPATIAL_DIMS = ("log_r", "theta", "phi")
SPATIAL_SHAPE = (192, 128, 66)
MAX_RANK = 65_536
RANKS = (2_048, 4_096, 8_192, 16_384, 32_768, 49_152, 65_536)

CALIBRATION_SNAPSHOTS = tuple(range(64))
INITIAL_TRANSITIONS = tuple(range(64))
WARMUP_TRANSITION = 64
EVALUATION_TRANSITIONS = (80, 91, 102, 113, 124, 135, 146, 157, 168, 179)
STABILITY_TRANSITIONS = tuple(range(64, 164))
RIDGE = 1.0

SCALAR_FIELDS = (
    "density",
    "internal_energy",
    "pressure",
    "temperature",
    "electron_fraction",
    "entropy",
)
VECTOR_FIELDS = ("magnetic_field", "velocity")
CHANNEL_ORDER = (
    "density",
    "internal_energy",
    "pressure",
    "temperature",
    "electron_fraction",
    "entropy",
    "magnetic_field/log_r",
    "magnetic_field/theta",
    "magnetic_field/phi",
    "velocity/log_r",
    "velocity/theta",
    "velocity/phi",
)
EXPECTED_ROOT_KEYS = (
    "additional_information",
    "boundary_conditions",
    "dimensions",
    "scalars",
    "t0_fields",
    "t1_fields",
    "t2_fields",
)


@dataclass(frozen=True)
class Channel:
    """One scalar component of a dynamic HDF5 field."""

    name: str
    group: str
    field: str
    component: int | None
    component_name: str | None


@dataclass
class RunningMoments:
    """Mergeable FP64 population moments for one channel."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            raise ValueError("cannot update moments with an empty array")
        if not bool(np.isfinite(values).all()):
            raise ValueError("dynamic field contains a non-finite value")

        batch_count = int(values.size)
        batch_mean = float(np.mean(values, dtype=np.float64))
        centered = values - batch_mean
        batch_m2 = float(np.sum(centered * centered, dtype=np.float64))

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        self.m2 += (
            batch_m2
            + delta * delta * self.count * batch_count / total
        )
        self.count = total

    @property
    def std(self) -> float:
        if self.count <= 0:
            raise ValueError("moments have no observations")
        variance = self.m2 / self.count
        if not math.isfinite(variance) or variance <= 0.0:
            raise ValueError(
                f"channel has invalid population variance {variance!r}"
            )
        return math.sqrt(variance)


def _channels() -> tuple[Channel, ...]:
    scalar_channels = tuple(
        Channel(
            name=field,
            group="t0_fields",
            field=field,
            component=None,
            component_name=None,
        )
        for field in SCALAR_FIELDS
    )
    vector_channels = tuple(
        Channel(
            name=f"{field}/{component_name}",
            group="t1_fields",
            field=field,
            component=component,
            component_name=component_name,
        )
        for field in VECTOR_FIELDS
        for component, component_name in enumerate(SPATIAL_DIMS)
    )
    result = scalar_channels + vector_channels
    if len(result) != 12:
        raise AssertionError(f"expected twelve dynamic channels, got {len(result)}")
    return result


CHANNELS = _channels()
if tuple(channel.name for channel in CHANNELS) != CHANNEL_ORDER:
    raise AssertionError("dynamic channel order differs from the frozen protocol")


def _sha256_file(path: Path, *, report_progress: bool = False) -> str:
    digest = hashlib.sha256()
    byte_count = 0
    next_report = 1 << 30
    started = time.monotonic()
    with path.open("rb") as handle:
        while True:
            block = handle.read(16 << 20)
            if not block:
                break
            digest.update(block)
            byte_count += len(block)
            if report_progress and byte_count >= next_report:
                elapsed = max(time.monotonic() - started, 1e-9)
                gib = byte_count / (1 << 30)
                rate = byte_count / elapsed / (1 << 20)
                print(
                    f"hashed {gib:.1f} GiB ({rate:.1f} MiB/s)",
                    flush=True,
                )
                next_report += 1 << 30
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _text_sequence(value: Any) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim == 0:
        return (_decode_text(array.item()),)
    return tuple(_decode_text(item) for item in array.tolist())


def _json_value(value: Any) -> Any:
    """Convert HDF5/NumPy metadata to strict JSON-compatible values."""

    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return {"encoding": "hex", "value": value.hex()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite metadata value {value!r}")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _attrs_dict(attrs: Any) -> dict[str, Any]:
    return {
        str(key): _json_value(attrs[key])
        for key in sorted(attrs.keys())
    }


def _require_bool_attr(dataset: Any, name: str, expected: bool) -> None:
    if name not in dataset.attrs:
        raise ValueError(f"{dataset.name} is missing required attribute {name!r}")
    actual = bool(np.asarray(dataset.attrs[name]).item())
    if actual != expected:
        raise ValueError(
            f"{dataset.name} attribute {name!r}: expected {expected}, got {actual}"
        )


def _require_dim_varying(dataset: Any) -> None:
    if "dim_varying" not in dataset.attrs:
        raise ValueError(
            f"{dataset.name} is missing required attribute 'dim_varying'"
        )
    actual = tuple(
        bool(item) for item in np.asarray(dataset.attrs["dim_varying"]).tolist()
    )
    expected = (True, True, True)
    if actual != expected:
        raise ValueError(
            f"{dataset.name} dim_varying: expected {expected}, got {actual}"
        )


def _dataset_metadata(dataset: Any) -> dict[str, Any]:
    return {
        "shape": [int(size) for size in dataset.shape],
        "dtype": str(np.dtype(dataset.dtype)),
        "chunks": (
            [int(size) for size in dataset.chunks]
            if dataset.chunks is not None
            else None
        ),
        "compression": dataset.compression,
        "compression_opts": _json_value(dataset.compression_opts),
        "shuffle": bool(dataset.shuffle),
        "fletcher32": bool(dataset.fletcher32),
        "scaleoffset": _json_value(dataset.scaleoffset),
        "attrs": _attrs_dict(dataset.attrs),
    }


def _inspect_hdf5(
    handle: Any,
    *,
    expected_n_trajectories: int = N_TRAJECTORIES,
    expected_n_snapshots: int = N_SNAPSHOTS,
    expected_spatial_shape: Sequence[int] = SPATIAL_SHAPE,
) -> dict[str, Any]:
    """Validate the pinned The Well schema and return its storage metadata."""

    import h5py

    expected_spatial_shape = tuple(
        int(size) for size in expected_spatial_shape
    )
    if (
        expected_n_trajectories <= 0
        or expected_n_snapshots <= 0
        or len(expected_spatial_shape) != 3
        or any(size <= 0 for size in expected_spatial_shape)
    ):
        raise ValueError("invalid expected HDF5 dimensions")

    root_keys = tuple(sorted(handle.keys()))
    if root_keys != tuple(sorted(EXPECTED_ROOT_KEYS)):
        raise ValueError(
            "unexpected HDF5 root keys: "
            f"expected {EXPECTED_ROOT_KEYS}, got {root_keys}"
        )

    required_root_attrs = (
        "dataset_name",
        "grid_type",
        "n_spatial_dims",
        "n_trajectories",
    )
    missing_root_attrs = [
        name for name in required_root_attrs if name not in handle.attrs
    ]
    if missing_root_attrs:
        raise ValueError(
            f"HDF5 root is missing attributes {missing_root_attrs}"
        )

    dataset_name = _decode_text(np.asarray(handle.attrs["dataset_name"]).item())
    if dataset_name != "post_neutron_star_merger":
        raise ValueError(
            "unexpected dataset_name: "
            f"expected 'post_neutron_star_merger', got {dataset_name!r}"
        )
    n_spatial_dims = int(np.asarray(handle.attrs["n_spatial_dims"]).item())
    n_trajectories = int(np.asarray(handle.attrs["n_trajectories"]).item())
    if n_spatial_dims != 3:
        raise ValueError(f"expected 3 spatial dimensions, got {n_spatial_dims}")
    if n_trajectories != expected_n_trajectories:
        raise ValueError(
            f"expected {expected_n_trajectories} trajectory, got {n_trajectories}"
        )

    dimensions = handle["dimensions"]
    if "spatial_dims" not in dimensions.attrs:
        raise ValueError("dimensions group is missing spatial_dims")
    spatial_dims = _text_sequence(dimensions.attrs["spatial_dims"])
    if spatial_dims != SPATIAL_DIMS:
        raise ValueError(
            f"expected spatial_dims {SPATIAL_DIMS}, got {spatial_dims}"
        )
    required_dimensions = {"time", *SPATIAL_DIMS}
    missing_dimensions = sorted(required_dimensions.difference(dimensions.keys()))
    if missing_dimensions:
        raise ValueError(
            f"dimensions group is missing datasets {missing_dimensions}"
        )
    expected_dimension_shapes = {
        "time": (expected_n_snapshots,),
        **{
            name: (length,)
            for name, length in zip(SPATIAL_DIMS, expected_spatial_shape)
        },
    }
    for name, expected_shape in expected_dimension_shapes.items():
        actual_shape = tuple(int(size) for size in dimensions[name].shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"dimensions/{name} shape: expected {expected_shape}, "
                f"got {actual_shape}"
            )

    t0_names = _text_sequence(handle["t0_fields"].attrs.get("field_names", ()))
    t1_names = _text_sequence(handle["t1_fields"].attrs.get("field_names", ()))
    t2_names = _text_sequence(handle["t2_fields"].attrs.get("field_names", ()))
    if set(t0_names) != set(SCALAR_FIELDS) or len(t0_names) != len(SCALAR_FIELDS):
        raise ValueError(
            f"unexpected t0_fields: expected {SCALAR_FIELDS}, got {t0_names}"
        )
    if set(t1_names) != set(VECTOR_FIELDS) or len(t1_names) != len(VECTOR_FIELDS):
        raise ValueError(
            f"unexpected t1_fields: expected {VECTOR_FIELDS}, got {t1_names}"
        )
    if t2_names:
        raise ValueError(f"expected no dynamic t2_fields, got {t2_names}")
    if set(handle["t0_fields"].keys()) != set(SCALAR_FIELDS):
        raise ValueError("t0_fields datasets do not match field_names")
    if set(handle["t1_fields"].keys()) != set(VECTOR_FIELDS):
        raise ValueError("t1_fields datasets do not match field_names")
    if set(handle["t2_fields"].keys()):
        raise ValueError("expected t2_fields group to contain no datasets")

    scalar_shape = (
        expected_n_trajectories,
        expected_n_snapshots,
        *expected_spatial_shape,
    )
    vector_shape = (*scalar_shape, len(SPATIAL_DIMS))
    for field in SCALAR_FIELDS:
        dataset = handle["t0_fields"][field]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"t0_fields/{field} is not a dataset")
        if tuple(dataset.shape) != scalar_shape:
            raise ValueError(
                f"{dataset.name} shape: expected {scalar_shape}, got {dataset.shape}"
            )
        if np.dtype(dataset.dtype) != np.dtype(np.float32):
            raise ValueError(
                f"{dataset.name} dtype: expected float32, got {dataset.dtype}"
            )
        _require_bool_attr(dataset, "sample_varying", True)
        _require_bool_attr(dataset, "time_varying", True)
        _require_dim_varying(dataset)

    for field in VECTOR_FIELDS:
        dataset = handle["t1_fields"][field]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"t1_fields/{field} is not a dataset")
        if tuple(dataset.shape) != vector_shape:
            raise ValueError(
                f"{dataset.name} shape: expected {vector_shape}, got {dataset.shape}"
            )
        if np.dtype(dataset.dtype) != np.dtype(np.float32):
            raise ValueError(
                f"{dataset.name} dtype: expected float32, got {dataset.dtype}"
            )
        _require_bool_attr(dataset, "sample_varying", True)
        _require_bool_attr(dataset, "time_varying", True)
        _require_dim_varying(dataset)

    # ``gcon`` is a time-independent spacetime metric.  It may be useful for
    # other tasks, but it is intentionally not one of our twelve dynamic
    # channels.  Fail if a metric-like dataset appears under a dynamic group.
    metric_paths: list[str] = []
    all_datasets: dict[str, Any] = {}
    all_groups: dict[str, Any] = {}

    def collect(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            all_datasets[name] = _dataset_metadata(obj)
            leaf = name.rsplit("/", 1)[-1].lower().replace("_", "")
            if "gcon" in leaf:
                metric_paths.append(name)
        elif isinstance(obj, h5py.Group):
            all_groups[name] = {
                "attrs": _attrs_dict(obj.attrs),
                "keys": sorted(str(key) for key in obj.keys()),
            }

    handle.visititems(collect)
    misplaced_metric = [
        path
        for path in metric_paths
        if path.startswith(("t0_fields/", "t1_fields/", "t2_fields/"))
    ]
    if misplaced_metric:
        raise ValueError(
            "gcon must be excluded from dynamic channels; found "
            f"{misplaced_metric}"
        )

    return {
        "root_attributes": _attrs_dict(handle.attrs),
        "root_keys": list(root_keys),
        "spatial_dims": list(spatial_dims),
        "spatial_shape": list(expected_spatial_shape),
        "n_snapshots": expected_n_snapshots,
        "n_trajectories": expected_n_trajectories,
        "dynamic_scalar_fields_in_file": list(t0_names),
        "dynamic_vector_fields_in_file": list(t1_names),
        "dynamic_tensor_fields_in_file": list(t2_names),
        "metric_dataset_paths_excluded": metric_paths,
        "hdf5_groups": all_groups,
        "hdf5_datasets": all_datasets,
    }


def _iter_snapshot_field_arrays(
    handle: Any, snapshot: int
) -> Iterator[tuple[Channel, np.ndarray]]:
    """Read each stored field once and yield its scalar channel components."""

    for field in SCALAR_FIELDS:
        channel = next(
            item
            for item in CHANNELS
            if item.group == "t0_fields" and item.field == field
        )
        values = np.asarray(
            handle["t0_fields"][field][0, snapshot, ...],
            dtype=np.float64,
            order="C",
        )
        yield channel, values

    for field in VECTOR_FIELDS:
        values = np.asarray(
            handle["t1_fields"][field][0, snapshot, ...],
            dtype=np.float64,
            order="C",
        )
        for component, component_name in enumerate(SPATIAL_DIMS):
            channel = next(
                item
                for item in CHANNELS
                if item.group == "t1_fields"
                and item.field == field
                and item.component == component
            )
            if channel.component_name != component_name:
                raise AssertionError("vector channel order changed")
            yield channel, values[..., component]


def _normalization_stats(handle: Any) -> dict[str, RunningMoments]:
    stats = {channel.name: RunningMoments() for channel in CHANNELS}
    for position, snapshot in enumerate(CALIBRATION_SNAPSHOTS, start=1):
        started = time.monotonic()
        for channel, values in _iter_snapshot_field_arrays(handle, snapshot):
            stats[channel.name].update(values)
        print(
            "normalization "
            f"{position}/{len(CALIBRATION_SNAPSHOTS)} "
            f"(snapshot {snapshot}, {time.monotonic() - started:.2f}s)",
            flush=True,
        )

    expected_count = len(CALIBRATION_SNAPSHOTS) * math.prod(SPATIAL_SHAPE)
    for channel in CHANNELS:
        moments = stats[channel.name]
        if moments.count != expected_count:
            raise ValueError(
                f"{channel.name} normalization count: expected "
                f"{expected_count}, got {moments.count}"
            )
        # Evaluate now so zero/invalid variance aborts before trace allocation.
        _ = moments.std
    return stats


def _select_low_frequencies(
    spatial_shape: Sequence[int], count: int
) -> np.ndarray:
    """Return low-frequency DCT indices with exact normalized-radius ties.

    Frequencies are ordered by

        sum_i (k_i / n_i)^2,

    then lexicographically by ``(k_0, k_1, k_2)``.  A common-denominator
    integer score makes equality and tie-breaking independent of FP rounding.
    """

    shape = tuple(int(size) for size in spatial_shape)
    if len(shape) != 3 or any(size <= 0 for size in shape):
        raise ValueError(f"expected a positive 3D shape, got {shape}")
    total = math.prod(shape)
    if count <= 0 or count > total:
        raise ValueError(
            f"frequency count must lie in [1, {total}], got {count}"
        )

    common = math.lcm(*shape)
    maximum_term = sum(
        ((size - 1) * (common // size)) ** 2 for size in shape
    )
    if maximum_term > np.iinfo(np.int64).max:
        raise OverflowError("normalized-frequency integer score exceeds int64")

    scores = (
        (
            np.arange(shape[0], dtype=np.int64)
            * np.int64(common // shape[0])
        )[:, None, None]
        ** 2
        + (
            np.arange(shape[1], dtype=np.int64)
            * np.int64(common // shape[1])
        )[None, :, None]
        ** 2
        + (
            np.arange(shape[2], dtype=np.int64)
            * np.int64(common // shape[2])
        )[None, None, :]
        ** 2
    )
    flat_scores = scores.reshape(-1)
    cutoff = int(np.partition(flat_scores, count - 1)[count - 1])
    before = np.flatnonzero(flat_scores < cutoff)
    at_cutoff = np.flatnonzero(flat_scores == cutoff)
    needed = count - int(before.size)
    if needed < 0 or needed > int(at_cutoff.size):
        raise AssertionError("invalid normalized-frequency cutoff")
    selected_flat = np.concatenate((before, at_cutoff[:needed]))
    order = np.lexsort((selected_flat, flat_scores[selected_flat]))
    selected_flat = selected_flat[order]
    coordinates = np.column_stack(
        np.unravel_index(selected_flat, shape)
    ).astype(np.int32, copy=False)
    if coordinates.shape != (count, 3):
        raise AssertionError(
            f"expected {(count, 3)} frequency indices, got {coordinates.shape}"
        )
    return np.ascontiguousarray(coordinates)


def _selected_frequency_hash(indices: np.ndarray) -> str:
    canonical = np.ascontiguousarray(indices, dtype="<i4")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _write_channel_coefficients(
    row: np.ndarray,
    channel_index: int,
    coefficients: np.ndarray,
    frequency_indices: np.ndarray,
) -> None:
    target = row[channel_index:: len(CHANNELS)]
    selected = coefficients[
        frequency_indices[: target.size, 0],
        frequency_indices[: target.size, 1],
        frequency_indices[: target.size, 2],
    ]
    if not bool(np.isfinite(selected).all()):
        raise ValueError("DCT produced a non-finite selected coefficient")
    target[:] = np.asarray(selected, dtype=np.float32)


def _write_trace(
    handle: Any,
    trace_path: Path,
    stats: dict[str, RunningMoments],
    frequency_indices: np.ndarray,
    *,
    workers: int,
) -> None:
    try:
        from scipy.fft import dctn
    except ImportError as error:
        raise RuntimeError(
            "SciPy is required; run with /usr/bin/python3 and "
            "PYTHONPATH=<repo>/rebuttal_deps:<repo>"
        ) from error

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = trace_path.with_name(
        f".{trace_path.name}.{os.getpid()}.tmp.npy"
    )
    if temporary.exists():
        temporary.unlink()

    trace = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(N_SNAPSHOTS, MAX_RANK),
        fortran_order=False,
    )
    try:
        for snapshot in range(N_SNAPSHOTS):
            started = time.monotonic()
            row = trace[snapshot]
            observed_channels: list[str] = []
            for channel, values in _iter_snapshot_field_arrays(handle, snapshot):
                moments = stats[channel.name]
                normalized = (values - moments.mean) / moments.std
                coefficients = dctn(
                    normalized,
                    type=2,
                    axes=(0, 1, 2),
                    norm="ortho",
                    overwrite_x=True,
                    workers=workers,
                )
                channel_index = CHANNELS.index(channel)
                _write_channel_coefficients(
                    row,
                    channel_index,
                    coefficients,
                    frequency_indices,
                )
                observed_channels.append(channel.name)
            if tuple(observed_channels) != tuple(
                channel.name for channel in CHANNELS
            ):
                raise AssertionError(
                    f"dynamic channel order changed: {observed_channels}"
                )
            if not bool(np.isfinite(row).all()):
                raise ValueError(
                    f"trace row for snapshot {snapshot} contains non-finite values"
                )
            print(
                f"trace {snapshot + 1}/{N_SNAPSHOTS} "
                f"({time.monotonic() - started:.2f}s)",
                flush=True,
            )
        trace.flush()
    except BaseException:
        del trace
        temporary.unlink(missing_ok=True)
        raise
    del trace

    # Ensure bytes reach the filesystem before the atomic rename.
    with temporary.open("rb") as handle_for_sync:
        os.fsync(handle_for_sync.fileno())
    os.replace(temporary, trace_path)


def _rank_scales(trace_path: Path) -> dict[str, float]:
    trace = np.load(trace_path, mmap_mode="r", allow_pickle=False)
    if trace.shape != (N_SNAPSHOTS, MAX_RANK):
        raise ValueError(f"unexpected trace shape {trace.shape}")
    if trace.dtype != np.dtype(np.float32):
        raise ValueError(f"unexpected trace dtype {trace.dtype}")
    calibration = np.asarray(
        trace[np.asarray(CALIBRATION_SNAPSHOTS), :],
        dtype=np.float64,
    )
    squared = calibration * calibration
    cumulative = np.cumsum(squared, axis=1, dtype=np.float64)
    scales: dict[str, float] = {}
    for rank in RANKS:
        mean_energy = float(
            np.mean(cumulative[:, rank - 1], dtype=np.float64)
        )
        scale = math.sqrt(mean_energy)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"rank {rank} has invalid calibration scale {scale!r}"
            )
        scales[str(rank)] = scale
    return scales


def _normalization_manifest(
    stats: dict[str, RunningMoments]
) -> dict[str, Any]:
    entries = []
    for channel in CHANNELS:
        moments = stats[channel.name]
        entries.append(
            {
                "name": channel.name,
                "source": f"{channel.group}/{channel.field}",
                "component": channel.component,
                "component_name": channel.component_name,
                "count": moments.count,
                "mean": moments.mean,
                "std": moments.std,
                "m2": moments.m2,
            }
        )
    return {
        "accumulation_dtype": "float64",
        "definition": (
            "per-channel scalar z-score over every spatial cell in "
            "snapshots 0..63; population standard deviation (ddof=0)"
        ),
        "snapshots": list(CALIBRATION_SNAPSHOTS),
        "snapshot_range_inclusive": [0, 63],
        "channels": entries,
    }


def _manifest(
    *,
    raw_path: Path,
    raw_sha256: str,
    trace_path: Path,
    trace_sha256: str,
    hdf5_metadata: dict[str, Any],
    stats: dict[str, RunningMoments],
    frequency_indices: np.ndarray,
    scales: dict[str, float],
    workers: int,
) -> dict[str, Any]:
    source_path = Path(__file__).resolve()
    frequencies_per_trace = math.ceil(MAX_RANK / len(CHANNELS))
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_tag": CAMPAIGN_TAG,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": {
            "repository": DATASET_REPOSITORY,
            "revision": DATASET_REVISION,
            "relative_path": DATASET_RELATIVE_PATH,
            "local_path": _repo_relative(raw_path),
            "expected_bytes": DATASET_EXPECTED_BYTES,
            "actual_bytes": raw_path.stat().st_size,
            "expected_sha256": DATASET_EXPECTED_SHA256,
            "sha256": raw_sha256,
            "hf_x_linked_etag": DATASET_HF_X_LINKED_ETAG,
            **hdf5_metadata,
        },
        "trace": {
            "path": _repo_relative(trace_path),
            "sha256": trace_sha256,
            "shape": [N_SNAPSHOTS, MAX_RANK],
            "dtype": "float32",
            "format": "npy",
            "format_version": _npy_version(trace_path),
            "fortran_order": False,
            "contiguous_order": "C",
            "chunks": None,
            "compression": None,
            "scaled": False,
            "max_rank": MAX_RANK,
        },
        "source": {
            "path": _repo_relative(source_path),
            "sha256": _sha256_file(source_path),
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": _scipy_version(),
            "h5py": _h5py_version(),
            "dct_workers": workers,
        },
        "normalization": _normalization_manifest(stats),
        "frequency_order": {
            "transform": "separable 3D DCT-II",
            "normalization": "ortho",
            "spatial_axes": [0, 1, 2],
            "spatial_dims": list(SPATIAL_DIMS),
            "spatial_shape": list(SPATIAL_SHAPE),
            "score": (
                "(k_log_r/192)^2 + (k_theta/128)^2 + (k_phi/66)^2"
            ),
            "tie_break": ["k_log_r", "k_theta", "k_phi"],
            "tie_break_order": "ascending lexicographic",
            "selected_frequency_count": frequencies_per_trace,
            "selected_frequency_indices_dtype": "little-endian int32",
            "selected_frequency_indices_sha256": _selected_frequency_hash(
                frequency_indices
            ),
            "first_indices": frequency_indices[:16].tolist(),
            "last_indices": frequency_indices[-16:].tolist(),
            "channel_interleave": "frequency-major, then channel order",
            "channel_order": [channel.name for channel in CHANNELS],
            "channel_count": len(CHANNELS),
            "prefix_rule": (
                "rank r uses trace[:, :r], so all tested ranks are nested"
            ),
        },
        "rank_scaling": {
            "definition": (
                "s_r^2 = (1/64) * sum_{t=0}^{63} "
                "||trace[t, :r]||_2^2"
            ),
            "computed_from": "persisted unscaled float32 trace",
            "accumulation_dtype": "float64",
            "consumer_operation": "trace[t, :r] / s_r",
            "calibration_snapshots": list(CALIBRATION_SNAPSHOTS),
            "ranks": list(RANKS),
            "scales": scales,
        },
        "transitions": {
            "semantics": (
                "transition i is the ordered snapshot pair (i, i+1)"
            ),
            "initial": list(INITIAL_TRANSITIONS),
            "warmup": WARMUP_TRANSITION,
            "evaluation": list(EVALUATION_TRANSITIONS),
            "stability": list(STABILITY_TRANSITIONS),
        },
        "protocol": {
            "ridge": RIDGE,
            "scenario": 4,
            "scenario_name": "torus_MBH_10",
            "description": (
                "disk from a black-hole--neutron-star merger with a "
                "10-solar-mass black hole"
            ),
            "dynamic_channels": 12,
            "gcon_included": False,
            "raw_trace_scaled": False,
            "tested_ranks": list(RANKS),
        },
    }


def _npy_version(path: Path) -> list[int]:
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
    return [int(version[0]), int(version[1])]


def _scipy_version() -> str:
    import scipy

    return scipy.__version__


def _h5py_version() -> str:
    import h5py

    return h5py.__version__


def _atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _check_output_space(trace_path: Path) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    required = (
        N_SNAPSHOTS * MAX_RANK * np.dtype(np.float32).itemsize
        + (16 << 20)
    )
    free = shutil.disk_usage(trace_path.parent).free
    if free < required:
        raise OSError(
            f"{trace_path.parent} has {free} free bytes; "
            f"at least {required} are required"
        )


def prepare(
    *,
    raw_path: Path,
    trace_path: Path,
    manifest_path: Path,
    workers: int,
    force: bool,
) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError(
            "h5py is required; run with /usr/bin/python3 and "
            "PYTHONPATH=<repo>/rebuttal_deps:<repo>"
        ) from error

    raw_path = raw_path.resolve()
    trace_path = trace_path.resolve()
    manifest_path = manifest_path.resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw dataset is missing: {raw_path}")
    actual_bytes = raw_path.stat().st_size
    if actual_bytes != DATASET_EXPECTED_BYTES:
        raise ValueError(
            f"raw byte count mismatch: expected {DATASET_EXPECTED_BYTES}, "
            f"got {actual_bytes}"
        )
    if not force:
        existing = [
            str(path)
            for path in (trace_path, manifest_path)
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing output(s) without --force: "
                + ", ".join(existing)
            )
    if raw_path == trace_path or raw_path == manifest_path:
        raise ValueError("output path must differ from the raw input path")
    if trace_path == manifest_path:
        raise ValueError("trace and manifest paths must differ")
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    _check_output_space(trace_path)

    print(
        f"validating pinned raw file ({actual_bytes} bytes): {raw_path}",
        flush=True,
    )
    raw_sha256 = _sha256_file(raw_path, report_progress=True)
    if raw_sha256 != DATASET_EXPECTED_SHA256:
        raise ValueError(
            "raw SHA-256 mismatch: "
            f"expected {DATASET_EXPECTED_SHA256}, got {raw_sha256}"
        )

    with h5py.File(raw_path, "r") as handle:
        hdf5_metadata = _inspect_hdf5(handle)
        print("HDF5 schema inspection passed", flush=True)
        stats = _normalization_stats(handle)
        print("FP64 normalization pass complete", flush=True)
        frequency_count = math.ceil(MAX_RANK / len(CHANNELS))
        frequency_indices = _select_low_frequencies(
            SPATIAL_SHAPE, frequency_count
        )
        _write_trace(
            handle,
            trace_path,
            stats,
            frequency_indices,
            workers=workers,
        )

    trace = np.load(trace_path, mmap_mode="r", allow_pickle=False)
    if trace.shape != (N_SNAPSHOTS, MAX_RANK):
        raise ValueError(f"trace shape changed after write: {trace.shape}")
    if trace.dtype != np.dtype(np.float32):
        raise ValueError(f"trace dtype changed after write: {trace.dtype}")
    if not trace.flags.c_contiguous or trace.flags.f_contiguous:
        raise ValueError("trace is not exclusively C-contiguous")
    del trace

    scales = _rank_scales(trace_path)
    print("hashing completed feature trace", flush=True)
    trace_sha256 = _sha256_file(trace_path, report_progress=False)
    payload = _manifest(
        raw_path=raw_path,
        raw_sha256=raw_sha256,
        trace_path=trace_path,
        trace_sha256=trace_sha256,
        hdf5_metadata=hdf5_metadata,
        stats=stats,
        frequency_indices=frequency_indices,
        scales=scales,
        workers=workers,
    )
    _atomic_json_dump(payload, manifest_path)
    print(f"wrote trace: {trace_path}", flush=True)
    print(f"wrote manifest: {manifest_path}", flush=True)
    print(f"trace SHA-256: {trace_sha256}", flush=True)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
        help="workers passed to each scipy.fft.dctn call",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="atomically replace an existing trace and manifest",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    prepare(
        raw_path=args.raw,
        trace_path=args.trace,
        manifest_path=args.manifest,
        workers=args.workers,
        force=args.force,
    )


if __name__ == "__main__":
    main()
