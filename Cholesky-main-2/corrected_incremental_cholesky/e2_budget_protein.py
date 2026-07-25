"""Leakage-free, budget-matched E2 rerun on UCI Protein.

The budget is calibrated using excluded seeds and frozen before evaluation.
Each evaluation run restores the last *completed batch* whose measured
training time did not exceed that budget, then evaluates the test set once.
This avoids both the old epoch-boundary artifact and test-set checkpoint
selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import statistics
import time
from pathlib import Path

import torch

from data_utils import load_uci_mat
from rebuttal_experiments.common import (
    atomic_json_dump,
    configure_threads,
    set_reproducible_seed,
)
from rebuttal_experiments.ng_sweep import (
    apply_first_layer_step,
    build_model,
    flatten_first_layer_gradient,
    whiten,
)


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
DEFAULT_DATA = PACKAGE_ROOT / "data" / "protein.mat"
DEFAULT_RESULTS = HERE / "results" / "e2_budget_protein"
PILOT_SEEDS = (1000, 1001, 1002)
EVALUATION_SEEDS = (0, 1, 2)
METHOD_RANKS = (("ours", 1024), ("refact", 256), ("refact", 512), ("refact", 1024))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }


def _restore(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state)


def _output_path(
    results_dir: Path,
    campaign_tag: str,
    phase: str,
    method: str,
    rank: int,
    seed: int,
) -> Path:
    return (
        results_dir
        / campaign_tag
        / f"{phase}_{method}_r{rank}_seed{seed}.json"
    )


def run_training(args: argparse.Namespace) -> dict:
    configure_threads(args.threads)
    set_reproducible_seed(args.seed)

    data_path = Path(args.data_path).resolve()
    # Keep the submitted held-out split fixed across training seeds.  The
    # loader fits normalization on the 80% training partition only.
    X_train, y_train, X_test, y_test = load_uci_mat(
        "protein",
        data_dir=data_path.parent,
        train_fraction=0.8,
        seed=args.split_seed,
        dtype=torch.float32,
        device="cpu",
    )

    model = build_model(X_train.shape[1])
    d_param = model[0].weight.numel() + model[0].bias.numel()
    if args.rank > d_param:
        raise ValueError(f"rank {args.rank} exceeds parameter dimension {d_param}")

    identity = torch.eye(args.rank)
    U, _ = torch.linalg.qr(torch.randn(d_param, args.rank), mode="reduced")
    G = (args.damping * identity).contiguous()
    L = (args.damping**0.5 * identity).contiguous()
    Aw = torch.zeros(args.rank, args.rank).contiguous()
    first_velocity = torch.zeros(d_param)
    other_velocities = {
        layer_index: {
            "weight": torch.zeros_like(model[layer_index].weight),
            "bias": torch.zeros_like(model[layer_index].bias),
        }
        for layer_index in (2, 4, 6)
    }

    if args.method == "ours":
        import fast_cholesky

        extension_path = str(Path(fast_cholesky.__file__).resolve())
        if HERE not in Path(extension_path).parents:
            raise RuntimeError(
                "E2 must use the extension rebuilt from Cholesky-main-2; "
                f"loaded {extension_path}"
            )
    else:
        fast_cholesky = None
        extension_path = None

    budget = args.budget_seconds
    selected_state = _snapshot(model)
    selected_training_seconds = 0.0
    selected_step = 0
    selected_epoch = 0
    selected_refreshes = 0
    selected_core_seconds = 0.0
    selected_refresh_seconds = 0.0
    selected_reset_seconds = 0.0
    selected_max_reset_drift: float | None = None

    refresh_count = 0
    core_seconds = 0.0
    refresh_seconds = 0.0
    reset_seconds = 0.0
    reset_drift: list[float] = []
    unexplained_values: list[float] = []
    training_seconds = 0.0
    attempted_steps = 0
    selected_batch_loss = float("nan")
    budget_crossed = False
    progress: list[dict[str, float | int]] = []

    steps_per_epoch = X_train.shape[0] // args.batch_size
    for epoch in range(args.max_epochs):
        generator = torch.Generator().manual_seed(args.seed * 100003 + epoch)
        permutation = torch.randperm(X_train.shape[0], generator=generator)

        for offset in range(
            0,
            X_train.shape[0] - args.batch_size + 1,
            args.batch_size,
        ):
            step_start = time.perf_counter()
            index = permutation[offset : offset + args.batch_size]
            xb, yb = X_train[index], y_train[index]
            prediction = model(xb).squeeze(-1)
            loss = torch.nn.functional.mse_loss(prediction, yb)
            model.zero_grad(set_to_none=True)
            loss.backward()
            g = flatten_first_layer_gradient(model)

            a = U.mT @ g
            residual = g - U @ a
            g_energy = float(torch.dot(g, g).item())
            residual_energy = float(torch.dot(residual, residual).item())
            unexplained = residual_energy / max(g_energy, 1e-30)
            unexplained_values.append(unexplained)
            refreshed = (
                unexplained > args.refresh_threshold
                and residual_energy > 1e-24
            )

            if refreshed:
                refresh_start = time.perf_counter()
                rho = residual_energy**0.5
                U_bar = torch.cat((U, (residual / rho).unsqueeze(1)), dim=1)
                c_bar = torch.cat((a, torch.tensor([rho], dtype=a.dtype)))
                fisher_bar = torch.zeros(args.rank + 1, args.rank + 1)
                fisher_bar[: args.rank, : args.rank] = (
                    G - args.damping * identity
                )
                fisher_bar.add_(torch.outer(c_bar, c_bar))
                eigenvalues, eigenvectors = torch.linalg.eigh(fisher_bar)
                top = torch.arange(args.rank, 0, -1)
                rotation = eigenvectors[:, top]
                fisher = torch.diag(
                    eigenvalues[top].clamp_min(0.0)
                ).contiguous()
                U = (U_bar @ rotation).contiguous()
                G = (args.damping * identity + fisher).contiguous()
                L = torch.linalg.cholesky(G).contiguous()
                Aw = whiten(L, fisher)
                refresh_seconds += time.perf_counter() - refresh_start
                refresh_count += 1
                a = U.mT @ g
                residual = g - U @ a
                y_solve = torch.linalg.solve_triangular(
                    L, a.unsqueeze(1), upper=False
                ).squeeze(1)
                x = torch.linalg.solve_triangular(
                    L.mT, y_solve.unsqueeze(1), upper=True
                ).squeeze(1)
            else:
                core_start = time.perf_counter()
                if args.method == "ours":
                    assert fast_cholesky is not None
                    Aw, vz = fast_cholesky.fast_incremental_update(
                        L, Aw, a.clone(), a.clone()
                    )
                    G.add_(torch.outer(a, a))
                    x = torch.linalg.solve_triangular(
                        L.mT, vz.unsqueeze(1), upper=True
                    ).squeeze(1)
                else:
                    G.add_(torch.outer(a, a))
                    L = torch.linalg.cholesky(G).contiguous()
                    Aw = whiten(L, G - args.damping * identity)
                    y_solve = torch.linalg.solve_triangular(
                        L, a.unsqueeze(1), upper=False
                    ).squeeze(1)
                    x = torch.linalg.solve_triangular(
                        L.mT, y_solve.unsqueeze(1), upper=True
                    ).squeeze(1)
                core_seconds += time.perf_counter() - core_start

            attempted_steps += 1
            if (
                args.method == "ours"
                and args.refactor_every > 0
                and attempted_steps % args.refactor_every == 0
            ):
                reset_start = time.perf_counter()
                L_exact = torch.linalg.cholesky(G).contiguous()
                Aw_exact = whiten(L_exact, G - args.damping * identity)
                drift = float(
                    (
                        torch.linalg.norm(Aw - Aw_exact)
                        / torch.linalg.norm(Aw_exact).clamp_min(1e-12)
                    ).item()
                )
                reset_drift.append(drift)
                L = L_exact
                Aw = Aw_exact
                reset_seconds += time.perf_counter() - reset_start

            direction = residual / args.damping + U @ x
            with torch.no_grad():
                first_velocity.mul_(args.momentum).add_(direction)
                apply_first_layer_step(
                    model, first_velocity, args.learning_rate
                )
                for layer_index in (2, 4, 6):
                    layer = model[layer_index]
                    velocity = other_velocities[layer_index]
                    velocity["weight"].mul_(args.momentum).add_(
                        layer.weight.grad
                    )
                    velocity["bias"].mul_(args.momentum).add_(
                        layer.bias.grad
                    )
                    layer.weight.add_(
                        velocity["weight"], alpha=-args.learning_rate
                    )
                    layer.bias.add_(
                        velocity["bias"], alpha=-args.learning_rate
                    )

            training_seconds += time.perf_counter() - step_start
            if budget is not None and training_seconds > budget:
                budget_crossed = True
                break

            # Snapshot copying is intentionally outside measured training
            # time.  It exists solely to recover the last full batch under
            # the predeclared budget.
            selected_state = _snapshot(model)
            selected_training_seconds = training_seconds
            selected_step = attempted_steps
            selected_epoch = epoch + 1
            selected_batch_loss = float(loss.detach())
            selected_refreshes = refresh_count
            selected_core_seconds = core_seconds
            selected_refresh_seconds = refresh_seconds
            selected_reset_seconds = reset_seconds
            selected_max_reset_drift = max(reset_drift, default=None)
            if selected_step % max(1, args.progress_every) == 0:
                progress.append(
                    {
                        "step": selected_step,
                        "training_seconds": selected_training_seconds,
                        "batch_mse": selected_batch_loss,
                    }
                )

        if budget_crossed:
            break

    if selected_step == 0:
        raise RuntimeError("budget expired before one complete training batch")
    _restore(model, selected_state)

    # Pilot runs never evaluate the test set. Evaluation runs touch it exactly
    # once, after the checkpoint is selected solely by the frozen budget.
    with torch.no_grad():
        final_train_mse = float(
            torch.nn.functional.mse_loss(
                model(X_train).squeeze(-1), y_train
            ).item()
        )
        final_test_mse = (
            float(
                torch.nn.functional.mse_loss(
                    model(X_test).squeeze(-1), y_test
                ).item()
            )
            if args.phase == "eval"
            else None
        )
    if final_test_mse is not None and not torch.isfinite(
        torch.tensor(final_test_mse)
    ):
        raise RuntimeError("training produced a non-finite test MSE")

    payload = {
        "experiment": "e2_budget_matched_protein",
        "phase": args.phase,
        "campaign_tag": args.campaign_tag,
        "method": args.method,
        "rank": args.rank,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "budget_seconds": budget,
        "selection_rule": "last completed batch at or below frozen budget",
        "test_evaluations": 1 if args.phase == "eval" else 0,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "damping": args.damping,
        "momentum": args.momentum,
        "refresh_threshold": args.refresh_threshold,
        "refactor_every": args.refactor_every,
        "threads": args.threads,
        "architecture": [X_train.shape[1], 512, 512, 128, 1],
        "d_param": d_param,
        "train_examples": X_train.shape[0],
        "test_examples": X_test.shape[0],
        "steps_per_epoch": steps_per_epoch,
        "selected_steps": selected_step,
        "attempted_steps": attempted_steps,
        "completed_epochs": selected_step // steps_per_epoch,
        "partial_epoch_steps": selected_step % steps_per_epoch,
        "selected_training_seconds": selected_training_seconds,
        "attempted_training_seconds": training_seconds,
        "budget_crossed": budget_crossed,
        "selected_batch_mse": selected_batch_loss,
        "final_train_mse": final_train_mse,
        "final_test_mse": final_test_mse,
        "refresh_count": selected_refreshes,
        "refresh_seconds": selected_refresh_seconds,
        "core_seconds": selected_core_seconds,
        "reset_seconds": selected_reset_seconds,
        "max_reset_drift": selected_max_reset_drift,
        "mean_unexplained_energy": (
            statistics.mean(unexplained_values[:selected_step])
            if selected_step
            else None
        ),
        "progress_training_only": progress,
        "data": {
            "path": str(data_path),
            "sha256": _sha256(data_path),
            "normalization": "fit on fixed 80% training partition only",
        },
        "implementation": {
            "incremental_extension": extension_path,
            "incremental_source": (
                str((HERE / "fast_update.cpp").resolve())
                if args.method == "ours"
                else None
            ),
            "incremental_source_sha256": (
                _sha256(HERE / "fast_update.cpp")
                if args.method == "ours"
                else None
            ),
            "positive_update_kernel": (
                "row-major virtual-padding SIMD fast path"
                if args.method == "ours"
                else None
            ),
            "baseline": (
                "fresh Cholesky plus two triangular solves and rewhitening"
                if args.method == "refact"
                else None
            ),
        },
        "environment": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "python_threads": torch.get_num_threads(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    output = _output_path(
        Path(args.results_dir),
        args.campaign_tag,
        args.phase,
        args.method,
        args.rank,
        args.seed,
    )
    atomic_json_dump(payload, output)
    print(
        f"wrote {output} method={args.method} r={args.rank} seed={args.seed} "
        f"steps={selected_step} time={selected_training_seconds:.3f}s "
        f"test_mse={final_test_mse if final_test_mse is not None else 'not-evaluated'}",
        flush=True,
    )
    return payload


def calibrate_budget(args: argparse.Namespace) -> dict:
    directory = Path(args.results_dir) / args.campaign_tag
    pilot_paths = [
        directory / f"pilot_ours_r1024_seed{seed}.json"
        for seed in PILOT_SEEDS
    ]
    payloads = []
    for path in pilot_paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing pilot result {path}")
        with path.open(encoding="utf-8") as handle:
            payloads.append(json.load(handle))
    durations = [item["selected_training_seconds"] for item in payloads]
    budget = statistics.median(durations)
    payload = {
        "experiment": "e2_budget_calibration",
        "campaign_tag": args.campaign_tag,
        "rule": (
            "median 10-epoch Ours r=1024 training time over excluded "
            f"pilot seeds {list(PILOT_SEEDS)}"
        ),
        "pilot_seeds": list(PILOT_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "pilot_seconds": durations,
        "budget_seconds": budget,
        "submitted_reference_budget_seconds": 198.0,
        "test_values_used": False,
    }
    output = directory / "budget.json"
    atomic_json_dump(payload, output)
    print(f"froze budget={budget:.6f}s in {output}", flush=True)
    return payload


def aggregate_results(args: argparse.Namespace) -> dict:
    directory = Path(args.results_dir) / args.campaign_tag
    with (directory / "budget.json").open(encoding="utf-8") as handle:
        budget_payload = json.load(handle)
    rows = []
    for method, rank in METHOD_RANKS:
        paths = [
            directory / f"eval_{method}_r{rank}_seed{seed}.json"
            for seed in EVALUATION_SEEDS
        ]
        payloads = []
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"missing evaluation result {path}")
            with path.open(encoding="utf-8") as handle:
                payloads.append(json.load(handle))
        seeds = [item["seed"] for item in payloads]
        if seeds != list(EVALUATION_SEEDS):
            raise ValueError(
                f"unexpected seeds for {method} r={rank}: {seeds}"
            )
        test_values = [item["final_test_mse"] for item in payloads]
        steps = [item["selected_steps"] for item in payloads]
        seconds = [item["selected_training_seconds"] for item in payloads]
        rows.append(
            {
                "method": method,
                "rank": rank,
                "test_mse_mean": statistics.mean(test_values),
                "test_mse_sample_std": statistics.stdev(test_values),
                "test_mse_values": test_values,
                "steps_mean": statistics.mean(steps),
                "steps_sample_std": statistics.stdev(steps),
                "steps_values": steps,
                "selected_seconds_mean": statistics.mean(seconds),
                "selected_seconds_sample_std": statistics.stdev(seconds),
                "selected_seconds_values": seconds,
            }
        )
    payload = {
        "experiment": "e2_budget_matched_protein_summary",
        "campaign_tag": args.campaign_tag,
        "budget": budget_payload,
        "selection_rule": "last completed batch at or below frozen budget",
        "test_evaluations_per_run": 1,
        "rows": rows,
    }
    output = directory / "summary.json"
    atomic_json_dump(payload, output)
    print(json.dumps(payload, indent=2), flush=True)
    print(f"wrote {output}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=("run", "calibrate-budget", "aggregate"),
        default="run",
    )
    parser.add_argument("--phase", choices=("pilot", "eval"), default="eval")
    parser.add_argument("--method", choices=("ours", "refact"))
    parser.add_argument("--rank", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--campaign-tag", required=True)
    parser.add_argument("--data-path", default=str(DEFAULT_DATA))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--budget-seconds", type=float)
    parser.add_argument("--budget-file")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--refresh-threshold", type=float, default=0.1)
    parser.add_argument("--refactor-every", type=int, default=50)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    if args.action == "run":
        if args.method is None or args.rank is None or args.seed is None:
            parser.error("--method, --rank, and --seed are required for run")
        if args.budget_file:
            with Path(args.budget_file).open(encoding="utf-8") as handle:
                args.budget_seconds = json.load(handle)["budget_seconds"]
        if args.phase == "pilot" and args.budget_seconds is not None:
            parser.error("pilot runs must complete all epochs without a budget")
        if args.phase == "eval" and args.budget_seconds is None:
            parser.error("eval runs require --budget-seconds or --budget-file")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.action == "run":
        run_training(arguments)
    elif arguments.action == "calibrate-budget":
        calibrate_budget(arguments)
    else:
        aggregate_results(arguments)
