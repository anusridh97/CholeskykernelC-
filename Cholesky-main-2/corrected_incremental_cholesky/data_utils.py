"""Dataset loading and leakage-free train/test preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat


def split_and_standardize(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0,1)")
    generator = torch.Generator(device=X.device)
    generator.manual_seed(seed)
    permutation = torch.randperm(X.shape[0], generator=generator, device=X.device)
    split = int(train_fraction * X.shape[0])
    train_index, test_index = permutation[:split], permutation[split:]
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    x_min = X_train.amin(dim=0)
    x_range = (X_train.amax(dim=0) - x_min).clamp_min(1e-8)
    X_train = (X_train - x_min) / x_range
    X_test = (X_test - x_min) / x_range

    y_mean = y_train.mean()
    y_std = y_train.std(unbiased=False).clamp_min(1e-8)
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
    return X_train, y_train, X_test, y_test


def load_uci_mat(
    name: str,
    *,
    data_dir: str | Path = "data",
    train_fraction: float = 0.8,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    path = Path(data_dir) / f"{name}.mat"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}. Place the UCI .mat file there before running this experiment."
        )
    raw = loadmat(path)
    if "data" not in raw:
        raise KeyError(f"{path} does not contain a 'data' matrix")
    data = torch.as_tensor(raw["data"], dtype=dtype, device=device)
    X, y = data[:, :-1], data[:, -1]
    return split_and_standardize(
        X, y, train_fraction=train_fraction, seed=seed
    )


def load_gb1(
    *,
    data_dir: str | Path = "data",
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    data_dir = Path(data_dir)
    x_path = data_dir / "gb1_X.npy"
    y_path = data_dir / "gb1_y.npy"
    if not x_path.is_file() or not y_path.is_file():
        raise FileNotFoundError(
            f"expected {x_path} and {y_path}; the corrected package does not silently "
            "substitute synthetic data for the paper's GB1 result"
        )
    X = torch.as_tensor(np.load(x_path), dtype=dtype, device=device)
    y = torch.as_tensor(np.load(y_path), dtype=dtype, device=device).reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError("GB1 X and y have inconsistent first dimensions")
    return X, y


def make_synthetic_regression(
    *,
    n: int = 2048,
    d: int = 8,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    X = torch.randn(n, d, generator=generator, dtype=dtype, device=device)
    weights = torch.randn(d, generator=generator, dtype=dtype, device=device)
    y = (
        torch.sin(X @ weights)
        + 0.25 * torch.cos(2.0 * X[:, 0])
        + 0.05 * torch.randn(n, generator=generator, dtype=dtype, device=device)
    )
    return split_and_standardize(X, y, seed=seed)


def make_synthetic_landscape(
    *,
    n: int = 20_000,
    d: int = 8,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    X = torch.randn(n, d, generator=generator, dtype=dtype, device=device)
    y = (
        2.5 * torch.exp(-0.5 * X[:, :3].square().sum(dim=1))
        + 3.0 * torch.exp(-0.7 * (X[:, 3:6] - 0.6).square().sum(dim=1))
        + 0.4 * X[:, 0] * X[:, 1]
        - 0.3 * X[:, 2] * X[:, 3]
        + 0.05 * torch.randn(n, generator=generator, dtype=dtype, device=device)
    )
    return X, y


__all__ = [
    "load_gb1",
    "load_uci_mat",
    "make_synthetic_landscape",
    "make_synthetic_regression",
    "split_and_standardize",
]
