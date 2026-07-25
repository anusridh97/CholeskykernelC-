from __future__ import annotations

import argparse
from itertools import product
from types import SimpleNamespace


def ng_tasks() -> list[dict]:
    tasks = []
    for rank, tau, seed in product(
        (256, 1024), (0.01, 0.02, 0.05, 0.1, 0.2, 0.5), (0, 1, 2)
    ):
        tasks.append({"method": "ours", "rank": rank, "tau": tau, "seed": seed})
    for seed in (0, 1, 2):
        tasks.append({"method": "ours", "rank": 512, "tau": 0.1, "seed": seed})
    for rank, seed in product((256, 512, 1024), (0, 1, 2)):
        tasks.append({"method": "refact", "rank": rank, "tau": 0.1, "seed": seed})
    return tasks


def synthetic_tasks() -> list[dict]:
    tasks = [
        {
            "kind": "representation",
            "rank": 256,
            "widths": [1, 4, 8, 16, 32, 64, 128, 256, 512],
        },
        {
            "kind": "representation",
            "rank": 1024,
            "widths": [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        },
        {"kind": "rank_k", "rank": 256},
        {"kind": "rank_k", "rank": 1024},
    ]
    for condition, precision, reset_interval, seed in product(
        (1e2, 1e4, 1e6, 1e8),
        ("float32", "float64"),
        (0, 50, 200),
        (0, 1, 2),
    ):
        tasks.append(
            {
                "kind": "stability",
                "rank": 128,
                "condition": condition,
                "precision": precision,
                "reset_interval": reset_interval,
                "seed": seed,
            }
        )
    for rank, seed in product((64, 256), (0, 1, 2)):
        tasks.append({"kind": "downdate", "rank": rank, "seed": seed})
    for precision, operator in product(
        ("float32", "float64"), ("cumulative", "random")
    ):
        tasks.append(
            {
                "kind": "kernel_stability",
                "rank": 128,
                "precision": precision,
                "operator": operator,
            }
        )
    return tasks


def gpu_tasks() -> list[dict]:
    tasks = []
    # The fused kernel keeps the padded core on chip and is only supported at
    # these ranks. Larger ranks use the tuned fully parallel HBM fallback.
    for rank, batch in product((64, 128), (1, 4, 16, 64)):
        tasks.append(
            {"impl": "fused", "rank": rank, "batch": batch, "num_warps": 4}
        )
    tuned = {
        256: {
            "tile_rows": 2,
            "tile_cols": 8,
            "num_warps": 8,
            "canonical_filename": (
                "cuda_parallel_r256_b1_w8_sw8_tr2_tc8_seed0_"
                "parallel-tune393048-final0.json"
            ),
        },
        512: {
            "tile_rows": 1,
            "tile_cols": 8,
            "num_warps": 4,
            "canonical_filename": (
                "cuda_parallel_r512_b1_w4_sw8_tr1_tc8_seed0_"
                "parallel-tune393048-final1.json"
            ),
        },
        1024: {
            "tile_rows": 4,
            "tile_cols": 8,
            "num_warps": 4,
            "canonical_filename": (
                "cuda_parallel_r1024_b1_w4_sw8_tr4_tc8_seed0_"
                "parallel-tune393048-final0.json"
            ),
        },
    }
    for rank, config in tuned.items():
        tasks.append(
            {
                "impl": "parallel",
                "rank": rank,
                "batch": 1,
                "sweep_num_warps": 8,
                **config,
            }
        )
    return tasks


def run_ng(config: dict) -> None:
    from .ng_sweep import run

    run(
        SimpleNamespace(
            **config,
            epochs=10,
            batch_size=256,
            learning_rate=0.1,
            damping=1.0,
            momentum=0.9,
            refactor_every=50,
            threads=10,
            smoke=False,
        )
    )


def run_synthetic(config: dict) -> None:
    kind = config["kind"]
    if kind == "representation":
        from .representation_crossover import run

        run(
            SimpleNamespace(
                rank=config["rank"],
                seed=0,
                widths=config["widths"],
                repeats=30,
                threads=1,
            )
        )
    elif kind == "rank_k":
        from .rank_k import run

        run(
            SimpleNamespace(
                rank=config["rank"],
                ks=[1, 2, 4, 8, 16, 32],
                seed=0,
                repeats=20,
                threads=1,
            )
        )
    elif kind == "stability":
        from .stability import run

        run(
            SimpleNamespace(
                **{key: value for key, value in config.items() if key != "kind"},
                steps=200,
                update_scale=0.01,
                fail_on_numerical_failure=False,
                threads=1,
            )
        )
    elif kind == "downdate":
        from .downdate_amplification import run

        run(
            SimpleNamespace(
                rank=config["rank"],
                seed=config["seed"],
                leverages=[0.1, 0.5, 0.9, 0.99, 0.999],
                threads=1,
            )
        )
    elif kind == "kernel_stability":
        from .kernel_stability import run

        run(
            SimpleNamespace(
                rank=config["rank"],
                precision=config["precision"],
                operator=config["operator"],
                deltas=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                seeds=[0, 1, 2],
                jitter=1e-6,
                steps=100,
                reset_interval=50,
                update_scale=0.01,
                threads=1,
            )
        )
    else:
        raise ValueError(f"unknown synthetic task: {kind}")


def run_gpu(config: dict) -> None:
    from .cuda_scaling import run

    repeats = 50 if config["rank"] <= 256 else 20 if config["rank"] <= 512 else 10
    run(
        SimpleNamespace(
            **config,
            seed=0,
            warmup=5,
            repeats=repeats,
            tolerance=2e-4,
            threads=4,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("ng", "synthetic", "gpu"), required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--count", action="store_true")
    args = parser.parse_args()

    matrices = {
        "ng": ng_tasks,
        "synthetic": synthetic_tasks,
        "gpu": gpu_tasks,
    }
    tasks = matrices[args.group]()
    if args.count:
        print(len(tasks))
        return
    if args.index is None:
        parser.error("--index is required unless --count is used")
    if args.index < 0 or args.index >= len(tasks):
        raise IndexError(f"index {args.index} outside [0, {len(tasks)})")
    config = tasks[args.index]
    print(f"group={args.group} index={args.index} config={config}", flush=True)
    if args.group == "ng":
        run_ng(config)
    elif args.group == "synthetic":
        run_synthetic(config)
    else:
        run_gpu(config)


if __name__ == "__main__":
    main()
