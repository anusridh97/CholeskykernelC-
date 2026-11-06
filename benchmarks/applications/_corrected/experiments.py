"""Preserved corrected application routines; configuration supplied by run.py."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from .core import Backend
from .adaptive_svgp import AdaptiveInducingGaussian
from .data_utils import load_gb1, load_uci_mat, make_synthetic_landscape, make_synthetic_regression
from .refresh import transported_ng_experiment

RESULTS_DIR = None

def _write_json(name, payload):
    if RESULTS_DIR is None:
        raise RuntimeError("Set an explicit results directory before running")
    path = RESULTS_DIR / name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    return path

def _candidate_pool(
    n_total: int,
    count: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return torch.randperm(n_total, generator=generator, device=device)[:count]


def run_protein_bo(
    *,
    landscape: str = "synthetic",
    data_dir: str = "data",
    rank: int = 64,
    initial: int = 96,
    rounds: int = 30,
    batch_size: int = 96,
    candidate_count: int = 5000,
    seeds: int = 5,
    backend: Backend = "auto",
    lengthscale: float = 5.0,
    noise_std: float = 0.1,
    refactor_every: int = 50,
    dtype: torch.dtype = torch.float32,
) -> dict[str, object]:
    if landscape.lower() == "gb1":
        X_all, y_all = load_gb1(data_dir=data_dir, dtype=dtype)
    else:
        X_all, y_all = make_synthetic_landscape(dtype=dtype, seed=17)
    n_total = X_all.shape[0]
    if rank > initial:
        raise ValueError("rank must not exceed the initial observation count")

    aggregate: dict[str, dict[str, list[object]]] = {
        method: {"times": [], "best_curves": [], "fallbacks": []}
        for method in ("ours", "refact")
    }

    for seed in range(seeds):
        generator = torch.Generator(device=X_all.device)
        generator.manual_seed(seed)
        initial_perm = torch.randperm(n_total, generator=generator, device=X_all.device)
        initial_indices = initial_perm[:initial]
        candidate_pools = [
            _candidate_pool(
                n_total,
                min(candidate_count, n_total),
                generator=generator,
                device=X_all.device,
            )
            for _ in range(rounds)
        ]

        for method in ("ours", "refact"):
            observed = torch.zeros(n_total, dtype=torch.bool, device=X_all.device)
            observed[initial_indices] = True
            Z = X_all[initial_indices[:rank]].clone()
            model = AdaptiveInducingGaussian(
                Z,
                method=method,
                backend=backend,
                lengthscale=lengthscale,
                noise_std=noise_std,
                refactor_every=refactor_every,
            )
            for index in initial_indices.tolist():
                model.condition(X_all[index], y_all[index])

            best_curve = [float(y_all[initial_indices].max())]
            start = time.perf_counter()
            for round_index in range(rounds):
                pool = candidate_pools[round_index]
                available = pool[~observed[pool]]
                if available.numel() < batch_size:
                    available = torch.arange(n_total, device=X_all.device)[~observed]
                X_candidates = X_all[available]
                mean, variance = model.predict(X_candidates)
                acquisition = mean + 2.0 * variance.sqrt()
                selected_local = torch.topk(
                    acquisition, k=min(batch_size, acquisition.numel())
                ).indices
                selected = available[selected_local]

                for index in selected.tolist():
                    model.replace_and_condition(X_all[index], y_all[index])
                observed[selected] = True
                best_curve.append(float(y_all[observed].max()))

            elapsed = time.perf_counter() - start
            aggregate[method]["times"].append(elapsed)
            aggregate[method]["best_curves"].append(best_curve)
            aggregate[method]["fallbacks"].append(
                model.diagnostics.fallback_refactors
            )
            print(
                f"seed={seed:2d} {method:>7}: {elapsed:.3f}s, "
                f"best={best_curve[-1]:.4f}, fallbacks="
                f"{model.diagnostics.fallback_refactors}"
            )

    summary: dict[str, object] = {}
    for method in ("ours", "refact"):
        times = np.asarray(aggregate[method]["times"], dtype=float)
        curves = np.asarray(aggregate[method]["best_curves"], dtype=float)
        summary[method] = {
            "mean_time": float(times.mean()),
            "std_time": float(times.std()),
            "mean_best": float(curves[:, -1].mean()),
            "std_best": float(curves[:, -1].std()),
            "mean_curve": curves.mean(axis=0).tolist(),
            "std_curve": curves.std(axis=0).tolist(),
            "fallbacks": aggregate[method]["fallbacks"],
            "raw_times": times.tolist(),
            "raw_best_curves": curves.tolist(),
        }
    summary["speedup"] = summary["refact"]["mean_time"] / summary["ours"]["mean_time"]

    payload = {
        "experiment": "adaptive_inducing_protein_bo",
        "landscape": landscape,
        "rank": rank,
        "initial": initial,
        "rounds": rounds,
        "batch_size": batch_size,
        "candidate_count": candidate_count,
        "seeds": seeds,
        "summary": summary,
    }
    _write_json(f"protein_{landscape}_r{rank}.json", payload)
    return payload


def run_ng(
    *,
    dataset: str = "synthetic",
    data_dir: str = "data",
    rank: int = 32,
    epochs: int = 5,
    hidden_dims: list[int] | None = None,
    chart_mode: str = "sandwich",
    backend: Backend = "auto",
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
) -> dict[str, object]:
    if dataset == "synthetic":
        X_train, y_train, X_test, y_test = make_synthetic_regression(
            n=4096, d=9, seed=seed, dtype=dtype
        )
    else:
        X_train, y_train, X_test, y_test = load_uci_mat(
            dataset, data_dir=data_dir, seed=seed, dtype=dtype
        )
    results = transported_ng_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        hidden_dims=hidden_dims or [128],
        chart_rank=rank,
        epochs=epochs,
        seed=seed,
        backend=backend,
        chart_mode=chart_mode,
    )
    payload = {
        "experiment": "transported_chart_preconditioning",
        "dataset": dataset,
        "rank": rank,
        "epochs": epochs,
        "chart_mode": chart_mode,
        "results": results,
    }
    _write_json(f"ng_{dataset}_{chart_mode}_r{rank}.json", payload)
    return payload
