from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
STUDENT_PKG = ROOT / "student_pkg"
RESULTS = ROOT / "rebuttal_results"


def add_student_pkg() -> None:
    path = str(STUDENT_PKG)
    if path not in sys.path:
        sys.path.insert(0, path)


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_threads(num_threads: int) -> None:
    num_threads = max(1, int(num_threads))
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def load_regression_dataset(name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and normalize the exact UCI datasets used by the copied code."""
    if name == "elevators":
        from scipy.io import loadmat

        data = np.asarray(loadmat(ROOT / "data" / "elevators.mat")["data"])
        x_np = data[:, :-1]
        y_np = data[:, -1]
    elif name == "protein":
        import pandas as pd

        frame = pd.read_csv(ROOT / "data" / "uci_protein" / "CASP.csv")
        y_np = frame["RMSD"].to_numpy()
        x_np = frame.drop(columns=["RMSD"]).to_numpy()
    else:
        raise ValueError(f"unknown dataset: {name}")

    X = torch.as_tensor(x_np, dtype=torch.float32)
    y = torch.as_tensor(y_np, dtype=torch.float32)
    x_min = X.amin(dim=0)
    x_max = X.amax(dim=0)
    X = (X - x_min) / (x_max - x_min + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return X.contiguous(), y.contiguous()


def split_80_20(
    X: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_train = int(0.8 * X.shape[0])
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def result_path(group: str, stem: str) -> Path:
    directory = RESULTS / group
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{stem}.json"


def atomic_json_dump(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def tensor_relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.linalg.norm(expected).clamp_min(torch.finfo(expected.dtype).tiny)
    return float((torch.linalg.norm(actual - expected) / denominator).item())
