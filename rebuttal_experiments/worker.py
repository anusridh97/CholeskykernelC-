from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .common import RESULTS, ROOT, atomic_json_dump
from .task_matrix import gpu_tasks, ng_tasks, synthetic_tasks


LOG_ROOT = ROOT / "rebuttal_logs" / "workers"
LOG_ROOT.mkdir(parents=True, exist_ok=True)


def expected_output(group: str, index: int) -> Path:
    if group == "ng":
        task = ng_tasks()[index]
        return (
            RESULTS
            / "ng"
            / (
                f"{task['method']}_r{task['rank']}_tau{task['tau']:g}"
                f"_seed{task['seed']}.json"
            )
        )
    if group == "synthetic":
        task = synthetic_tasks()[index]
        kind = task["kind"]
        if kind == "representation":
            return RESULTS / "representation" / f"crossover_r{task['rank']}_seed0.json"
        if kind == "rank_k":
            return RESULTS / "rank_k" / f"rank_k_r{task['rank']}_seed0.json"
        if kind == "downdate":
            return (
                RESULTS
                / "stability"
                / f"downdate_r{task['rank']}_seed{task['seed']}.json"
            )
        if kind == "kernel_stability":
            return (
                RESULTS
                / "stability"
                / (
                    f"kernel_r{task['rank']}_{task['precision']}_"
                    f"{task['operator']}.json"
                )
            )
        return (
            RESULTS
            / "stability"
            / (
                f"stability_r{task['rank']}_cond{task['condition']:g}"
                f"_{task['precision']}_reset{task['reset_interval']}"
                f"_seed{task['seed']}.json"
            )
        )
    task = gpu_tasks()[index]
    if "canonical_filename" in task:
        return RESULTS / "cuda" / task["canonical_filename"]
    stem = (
        f"cuda_{task['impl']}_r{task['rank']}_b{task['batch']}"
        f"_w{task['num_warps']}"
    )
    if task["impl"] == "parallel":
        stem += (
            f"_sw{task['sweep_num_warps']}"
            f"_tr{task['tile_rows']}_tc{task['tile_cols']}"
            "_rhs2"
        )
    return RESULTS / "cuda" / f"{stem}_seed0.json"


def interleave(left: list[tuple[str, int]], right: list[tuple[str, int]]):
    output = []
    while left or right:
        if left:
            output.append(left.pop(0))
        if right:
            output.append(right.pop(0))
    return output


def run_one(
    group: str,
    index: int,
    retries: int,
    gpu_id: int | None = None,
) -> dict:
    output = expected_output(group, index)
    if output.is_file():
        return {
            "group": group,
            "index": index,
            "status": "skipped_complete",
            "output": str(output),
        }

    log = LOG_ROOT / f"{group}_{index}.log"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{ROOT / 'rebuttal_deps'}:{ROOT}"
    if group == "ng":
        environment["OMP_NUM_THREADS"] = "10"
        environment["MKL_NUM_THREADS"] = "10"
        environment["OPENBLAS_NUM_THREADS"] = "10"
    else:
        environment["OMP_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        environment["OPENBLAS_NUM_THREADS"] = "1"
    if gpu_id is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    command = [
        sys.executable,
        "-m",
        "rebuttal_experiments.task_matrix",
        "--group",
        group,
        "--index",
        str(index),
    ]
    last_returncode = None
    for attempt in range(1, retries + 2):
        with log.open("a", encoding="utf-8") as handle:
            handle.write(
                f"\n===== attempt {attempt} started {time.ctime()} "
                f"gpu={gpu_id} command={command} =====\n"
            )
            handle.flush()
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
            last_returncode = completed.returncode
            handle.write(
                f"===== attempt {attempt} returncode={completed.returncode} "
                f"ended {time.ctime()} =====\n"
            )
        if completed.returncode == 0 and output.is_file():
            return {
                "group": group,
                "index": index,
                "status": "completed",
                "attempts": attempt,
                "output": str(output),
                "gpu_id": gpu_id,
            }
        time.sleep(min(30, 5 * attempt))
    return {
        "group": group,
        "index": index,
        "status": "failed",
        "attempts": retries + 1,
        "returncode": last_returncode,
        "output": str(output),
        "gpu_id": gpu_id,
        "log": str(log),
    }


def run_cpu(args: argparse.Namespace) -> list[dict]:
    groups = tuple(part.strip() for part in args.groups.split(",") if part.strip())
    invalid = sorted(set(groups) - {"ng", "synthetic"})
    if invalid:
        raise ValueError(f"unsupported CPU worker groups: {invalid}")
    jobs = interleave(
        [("ng", index) for index in range(len(ng_tasks()))] if "ng" in groups else [],
        [("synthetic", index) for index in range(len(synthetic_tasks()))]
        if "synthetic" in groups
        else [],
    )
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, group, index, args.retries): (group, index)
            for group, index in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    return results


def run_gpu(args: argparse.Namespace) -> list[dict]:
    jobs = list(range(len(gpu_tasks())))
    lanes = [jobs[gpu_id :: args.gpus] for gpu_id in range(args.gpus)]

    def run_lane(gpu_id: int, indices: list[int]) -> list[dict]:
        return [
            run_one("gpu", index, args.retries, gpu_id=gpu_id) for index in indices
        ]

    results = []
    with ThreadPoolExecutor(max_workers=args.gpus) as pool:
        futures = {
            pool.submit(run_lane, gpu_id, indices): gpu_id
            for gpu_id, indices in enumerate(lanes)
            if indices
        }
        for future in as_completed(futures):
            for result in future.result():
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--groups",
        default="ng,synthetic",
        help="comma-separated CPU groups to run",
    )
    args = parser.parse_args()
    started = time.time()
    results = run_cpu(args) if args.mode == "cpu" else run_gpu(args)
    summary = {
        "mode": args.mode,
        "started_unix": started,
        "finished_unix": time.time(),
        "counts": {
            status: sum(result["status"] == status for result in results)
            for status in sorted({result["status"] for result in results})
        },
        "results": sorted(results, key=lambda item: (item["group"], item["index"])),
    }
    output = RESULTS / f"worker_{args.mode}_summary.json"
    atomic_json_dump(summary, output)
    # Keep the human-readable report current even when CPU and GPU workers
    # finish in a different order.
    from .aggregate import main as aggregate_results

    aggregate_results()
    if any(result["status"] == "failed" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
