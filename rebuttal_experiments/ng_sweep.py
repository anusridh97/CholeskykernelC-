from __future__ import annotations

import argparse
import os
import platform
import socket
import time

import torch

from .common import (
    add_student_pkg,
    atomic_json_dump,
    configure_threads,
    load_regression_dataset,
    result_path,
    set_reproducible_seed,
    split_80_20,
)


def whiten(L: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    left = torch.linalg.solve_triangular(L, M, upper=False)
    return torch.linalg.solve_triangular(L, left.mT, upper=False).mT.contiguous()


def build_model(d_in: int) -> torch.nn.Module:
    # This gives 9*512 + 512 = 5,120 preconditioned parameters on UCI Protein,
    # matching Section 4.6.2 of the submitted paper.
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
    )


def flatten_first_layer_gradient(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat(
        (model[0].weight.grad.reshape(-1), model[0].bias.grad.reshape(-1))
    ).detach()


def apply_first_layer_step(
    model: torch.nn.Module, direction: torch.Tensor, learning_rate: float
) -> None:
    first = model[0]
    weight_count = first.weight.numel()
    first.weight.add_(
        direction[:weight_count].reshape_as(first.weight), alpha=-learning_rate
    )
    first.bias.add_(direction[weight_count:], alpha=-learning_rate)


def run(args: argparse.Namespace) -> dict:
    configure_threads(args.threads)
    set_reproducible_seed(args.seed)
    add_student_pkg()
    if args.method == "ours":
        import fast_cholesky
    else:
        fast_cholesky = None

    X, y = load_regression_dataset("protein")
    if args.smoke:
        X = X[:4096]
        y = y[:4096]
    X_train, y_train, X_test, y_test = split_80_20(X, y)

    model = build_model(X.shape[1])
    d_param = model[0].weight.numel() + model[0].bias.numel()
    if args.rank > d_param:
        raise ValueError(f"rank {args.rank} exceeds first-layer dimension {d_param}")

    # The copied Cholesky-main prototype initializes C=I and Aw=I.  That is
    # inconsistent with F=lambda*I + U C U^T: the initial accumulated Fisher
    # is zero, hence G=lambda*I, M=G-lambda*I=0, and Aw=0.  The corrected
    # initialization and preconditioner below match the later "faithful §3.5"
    # implementation available in this repository's curvature branch.
    U, _ = torch.linalg.qr(torch.randn(d_param, args.rank), mode="reduced")
    identity = torch.eye(args.rank)
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

    refresh_count = 0
    core_seconds = 0.0
    refresh_seconds = 0.0
    reset_seconds = 0.0
    total_steps = 0
    history = []
    reset_drift = []
    unexplained_values = []
    start = time.perf_counter()

    for epoch in range(args.epochs):
        generator = torch.Generator().manual_seed(args.seed * 100003 + epoch)
        permutation = torch.randperm(X_train.shape[0], generator=generator)

        for offset in range(0, X_train.shape[0] - args.batch_size + 1, args.batch_size):
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
            refreshed = unexplained > args.tau and residual_energy > 1e-24

            if refreshed:
                refresh_start = time.perf_counter()
                rho = residual_energy**0.5
                U_bar = torch.cat((U, (residual / rho).unsqueeze(1)), dim=1)
                c_bar = torch.cat((a, torch.tensor([rho], dtype=a.dtype)))
                # Rotate/truncate the accumulated Fisher, not the damped metric.
                fisher_bar = torch.zeros(args.rank + 1, args.rank + 1)
                fisher_bar[: args.rank, : args.rank] = G - args.damping * identity
                fisher_bar.add_(torch.outer(c_bar, c_bar))
                eigenvalues, eigenvectors = torch.linalg.eigh(fisher_bar)
                top = torch.arange(args.rank, 0, -1)
                R = eigenvectors[:, top]
                fisher = torch.diag(eigenvalues[top].clamp_min(0.0)).contiguous()
                U = (U_bar @ R).contiguous()
                G = (args.damping * identity + fisher).contiguous()
                L = torch.linalg.cholesky(G).contiguous()
                Aw = whiten(L, fisher)
                refresh_seconds += time.perf_counter() - refresh_start
                refresh_count += 1
                # Coordinates changed during refresh; reproject before preconditioning.
                a = U.mT @ g
                residual = g - U @ a
                y = torch.linalg.solve_triangular(
                    L, a.unsqueeze(1), upper=False
                ).squeeze(1)
                x = torch.linalg.solve_triangular(
                    L.mT, y.unsqueeze(1), upper=True
                ).squeeze(1)
            else:
                core_start = time.perf_counter()
                if args.method == "ours":
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
                    y = torch.linalg.solve_triangular(
                        L, a.unsqueeze(1), upper=False
                    ).squeeze(1)
                    x = torch.linalg.solve_triangular(
                        L.mT, y.unsqueeze(1), upper=True
                    ).squeeze(1)
                core_seconds += time.perf_counter() - core_start

            total_steps += 1
            if (
                args.method == "ours"
                and args.refactor_every > 0
                and total_steps % args.refactor_every == 0
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

            # On range(U), F acts as G; off the chart it acts as lambda*I.
            # Thus F^-1 g = residual/lambda + U G^-1 a.
            direction = residual / args.damping + U @ x

            with torch.no_grad():
                first_velocity.mul_(args.momentum).add_(direction)
                apply_first_layer_step(model, first_velocity, args.learning_rate)
                for layer_index in (2, 4, 6):
                    layer = model[layer_index]
                    velocity = other_velocities[layer_index]
                    velocity["weight"].mul_(args.momentum).add_(layer.weight.grad)
                    velocity["bias"].mul_(args.momentum).add_(layer.bias.grad)
                    layer.weight.add_(velocity["weight"], alpha=-args.learning_rate)
                    layer.bias.add_(velocity["bias"], alpha=-args.learning_rate)

        with torch.no_grad():
            test_mse = float(
                torch.nn.functional.mse_loss(model(X_test).squeeze(-1), y_test).item()
            )
            train_mse = float(
                torch.nn.functional.mse_loss(model(X_train).squeeze(-1), y_train).item()
            )
        history.append(
            {
                "epoch": epoch + 1,
                "steps": total_steps,
                "wall_seconds": time.perf_counter() - start,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "refreshes": refresh_count,
            }
        )
        print(
            f"epoch={epoch + 1} method={args.method} r={args.rank} tau={args.tau:g} "
            f"seed={args.seed} test_mse={test_mse:.6g} refreshes={refresh_count}",
            flush=True,
        )
        if not torch.isfinite(torch.tensor(test_mse)):
            raise RuntimeError("training produced a non-finite test MSE")

    wall_seconds = time.perf_counter() - start
    refresh_cycle = total_steps / refresh_count if refresh_count else None
    payload = {
        "experiment": "ng_sweep",
        "method": args.method,
        "rank": args.rank,
        "tau": args.tau,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "damping": args.damping,
        "momentum": args.momentum,
        "refactor_every": args.refactor_every,
        "threads": args.threads,
        "smoke": args.smoke,
        "d_param": d_param,
        "steps": total_steps,
        "refresh_count": refresh_count,
        "refresh_cycle": refresh_cycle,
        "refresh_seconds": refresh_seconds,
        "core_seconds": core_seconds,
        "reset_seconds": reset_seconds,
        "max_reset_drift": max(reset_drift, default=None),
        "mean_reset_drift": (
            sum(reset_drift) / len(reset_drift) if reset_drift else None
        ),
        "mean_unexplained_energy": (
            sum(unexplained_values) / len(unexplained_values)
            if unexplained_values
            else None
        ),
        "wall_seconds": wall_seconds,
        "refresh_time_share": refresh_seconds / wall_seconds,
        "core_time_share": core_seconds / wall_seconds,
        "final_train_mse": history[-1]["train_mse"],
        "final_test_mse": history[-1]["test_mse"],
        "history": history,
        "environment": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "backend_provenance": (
            "reconstructed from the corrected faithful-section-3.5 implementation "
            "in git commit eb99659; the copied Cholesky-main prototype is incomplete"
        ),
    }
    stem = (
        f"{args.method}_r{args.rank}_tau{args.tau:g}_seed{args.seed}"
        + ("_smoke" if args.smoke else "")
    )
    output = result_path("ng", stem)
    atomic_json_dump(payload, output)
    print(f"wrote {output}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("ours", "refact"), required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--refactor-every", type=int, default=50)
    parser.add_argument(
        "--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "10"))
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
