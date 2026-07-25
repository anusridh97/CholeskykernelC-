from __future__ import annotations

import argparse
import os
import traceback
from types import SimpleNamespace

from .common import atomic_json_dump, result_path
from .cuda_scaling import run


def _config(
    impl: str,
    setup_num_warps: int,
    sweep_num_warps: int,
    tile_rows: int,
    tile_cols: int,
) -> dict:
    return {
        "impl": impl,
        "num_warps": setup_num_warps,
        "sweep_num_warps": sweep_num_warps,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
    }


def initial_configs(rank: int, impl: str) -> list[dict]:
    default_tile = {256: 8, 512: 4, 1024: 4}.get(
        rank,
        max(1, min(8, 2048 // rank)),
    )
    default_cols = 8
    configs = []
    if impl != "inverse_factor":
        configs.extend(
            _config(impl, warps, 8, default_tile, default_cols)
            for warps in (1, 2, 4, 8)
        )
    configs.extend(
        _config(impl, 4, warps, default_tile, default_cols)
        for warps in (1, 2, 4, 8)
    )
    configs.extend(
        _config(impl, 4, 8, tile_rows, default_cols)
        for tile_rows in (1, 2, 4, 8, 16)
    )
    configs.extend(
        _config(impl, 4, 8, default_tile, tile_cols)
        for tile_cols in (1, 2, 4, 8, 16)
    )
    unique = []
    seen = set()
    for config in configs:
        key = tuple(sorted(config.items()))
        if key not in seen:
            seen.add(key)
            unique.append(config)
    return unique


def run_config(
    rank: int,
    config: dict,
    args: argparse.Namespace,
    suffix: str,
    repeats: int,
    warmup: int,
) -> dict:
    return run(
        SimpleNamespace(
            rank=rank,
            batch=1,
            seed=args.seed,
            warmup=warmup,
            repeats=repeats,
            tolerance=args.tolerance,
            threads=args.threads,
            output_suffix=suffix,
            **config,
        )
    )


def tune(args: argparse.Namespace) -> dict:
    rows = []
    confirmation_rows = []
    failures = []
    job_id = os.environ.get("SLURM_JOB_ID", "local")

    for rank in args.ranks:
        base_configs = initial_configs(rank, args.impl)
        rank_rows = []
        for index, config in enumerate(base_configs):
            print(
                f"===== initial r={rank} config={config} =====",
                flush=True,
            )
            try:
                row = run_config(
                    rank,
                    config,
                    args,
                    f"{args.impl}-tune{job_id}-initial{index}",
                    args.repeats,
                    args.warmup,
                )
                rows.append(row)
                rank_rows.append(row)
            except Exception as exc:
                failures.append(
                    {
                        "rank": rank,
                        **config,
                        "stage": "initial",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

        if not rank_rows:
            continue

        # Each initial sweep varies one launch dimension. Cross the best two
        # observed values of every dimension so interactions are measured.
        dimension_values = {}
        for field in (
            "num_warps",
            "sweep_num_warps",
            "tile_rows",
            "tile_cols",
        ):
            by_value = {}
            for row in rank_rows:
                value = row[field]
                best = by_value.get(value)
                if best is None or row["triton_event_seconds"] < best["triton_event_seconds"]:
                    by_value[value] = row
            dimension_values[field] = [
                item[0]
                for item in sorted(
                    by_value.items(),
                    key=lambda item: item[1]["triton_event_seconds"],
                )[:2]
            ]

        existing = {
            tuple(
                row[field]
                for field in (
                    "num_warps",
                    "sweep_num_warps",
                    "tile_rows",
                    "tile_cols",
                )
            )
            for row in rank_rows
        }
        cross_index = 0
        for setup_warps in dimension_values["num_warps"]:
            for sweep_warps in dimension_values["sweep_num_warps"]:
                for tile_rows in dimension_values["tile_rows"]:
                    for tile_cols in dimension_values["tile_cols"]:
                        key = (setup_warps, sweep_warps, tile_rows, tile_cols)
                        if key in existing:
                            continue
                        existing.add(key)
                        config = _config(args.impl, *key)
                        print(
                            f"===== crossed r={rank} config={config} =====",
                            flush=True,
                        )
                        try:
                            row = run_config(
                                rank,
                                config,
                                args,
                                f"{args.impl}-tune{job_id}-cross{cross_index}",
                                args.repeats,
                                args.warmup,
                            )
                            rows.append(row)
                            rank_rows.append(row)
                        except Exception as exc:
                            failures.append(
                                {
                                    "rank": rank,
                                    **config,
                                    "stage": "cross",
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                    "traceback": traceback.format_exc(),
                                }
                            )
                        cross_index += 1

        for confirm_index, candidate in enumerate(
            sorted(
                rank_rows,
                key=lambda row: row["triton_event_seconds"],
            )[: args.confirm_top]
        ):
            config = {
                "impl": args.impl,
                "num_warps": candidate["num_warps"],
                "sweep_num_warps": candidate["sweep_num_warps"],
                "tile_rows": candidate["tile_rows"],
                "tile_cols": candidate["tile_cols"],
            }
            print(
                f"===== confirming r={rank} config={config} =====",
                flush=True,
            )
            try:
                confirmation_rows.append(
                    run_config(
                        rank,
                        config,
                        args,
                        f"{args.impl}-tune{job_id}-final{confirm_index}",
                        args.confirm_repeats,
                        max(5, args.warmup),
                    )
                )
            except Exception as exc:
                failures.append(
                    {
                        "rank": rank,
                        **config,
                        "stage": "confirmation",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

    best_by_rank = {}
    for rank in args.ranks:
        confirmed = [row for row in confirmation_rows if row["rank"] == rank]
        candidates = confirmed or [row for row in rows if row["rank"] == rank]
        if candidates:
            best_by_rank[str(rank)] = min(
                candidates,
                key=lambda row: row["triton_event_seconds"],
            )

    payload = {
        "experiment": f"cuda_{args.impl}_tuning",
        "ranks": args.ranks,
        "rows": rows,
        "confirmation_rows": confirmation_rows,
        "best_by_rank": best_by_rank,
        "failures": failures,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    output = result_path(
        "cuda_tuning",
        f"cuda_{args.impl}_tuning_{job_id}",
    )
    atomic_json_dump(payload, output)
    print(f"wrote {output}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--impl",
        choices=("parallel", "inverse", "inverse_factor"),
        default="parallel",
    )
    parser.add_argument("--ranks", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--confirm-top", type=int, default=3)
    parser.add_argument("--confirm-repeats", type=int, default=40)
    parser.add_argument("--tolerance", type=float, default=2e-4)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    tune(parse_args())
