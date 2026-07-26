"""Frozen synthetic pilot for an explicit E1 same-vector CPU specialization.

This is intentionally separate from the source-locked E1 confirmation
campaign.  It never loads Protein or evaluates a model.  Its only purpose is
to establish whether an allocation-free implementation of the common

    G+ = G + z z^T,  M+ = M + z z^T

kernel reflects the expected quadratic-versus-cubic timing advantage under
equal FP32, affinity, hardware, and thread budgets.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import math
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CORRECTED = ROOT / "Cholesky-main-2" / "corrected_incremental_cholesky"
PROTOCOL_PATH = HERE / "e1_positive_candidate_protocol.json"
DEFAULT_RESULTS = ROOT / "rebuttal_results" / "e1_positive_candidate"
CAMPAIGN = "e1-positive-candidate-synthetic-v2"
RANKS = (256, 512, 1024)
THREAD_COUNTS = (1, 10)
PILOT_SEEDS = (6100, 6101, 6102)
TUNING_SEED = 7777
REPEATS = 30
WARMUPS = 3
PHASE_NAMES = (
    "workspace_reset",
    "cholesky_rank_one_update",
    "recover_vz",
    "operator_left_transport",
    "operator_right_transport_and_outer",
)
TOLERANCES = {"L": 5e-6, "Aw": 5e-5, "vz": 5e-5}

# Both extensions are imported only after establishing affinity.  Importing
# PyTorch initializes OpenMP, and OMP_PROC_BIND may immediately narrow the
# main thread to its first place; selecting the CPU set first is therefore
# necessary for honest ten-thread provenance.
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(CORRECTED))
sys.path.insert(2, str(ROOT))

torch: Any = None
e1_positive_candidate: Any = None
fast_cholesky: Any = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(tuple(contiguous.shape)).encode())
    digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(
        ("git",) + args,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _read_protocol() -> tuple[dict[str, Any], str]:
    with PROTOCOL_PATH.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    expected = {
        "campaign": CAMPAIGN,
        "ranks": list(RANKS),
        "thread_counts": list(THREAD_COUNTS),
        "pilot_seeds": list(PILOT_SEEDS),
        "tuning_seed_not_in_pilot": TUNING_SEED,
        "repeats": REPEATS,
        "warmups": WARMUPS,
        "dtype": "float32",
        "relative_error_tolerances": TOLERANCES,
        "protein_or_test_outcomes_used": False,
    }
    actual = {key: protocol.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            f"frozen protocol does not match driver constants: "
            f"expected {expected}, got {actual}"
        )
    return protocol, _sha256(PROTOCOL_PATH)


def _write_once(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
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
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _cpu_topology(cpu: int) -> dict[str, int]:
    base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    return {
        "logical_cpu": cpu,
        "physical_package": int(
            (base / "physical_package_id").read_text().strip()
        ),
        "core_id": int((base / "core_id").read_text().strip()),
    }


def _parse_parallel_value(report: str, label: str) -> int:
    match = re.search(rf"{re.escape(label)}\s*:\s*(\d+)", report)
    if match is None:
        raise RuntimeError(f"thread report omitted {label!r}")
    return int(match.group(1))


def _task_affinities() -> dict[str, list[int]]:
    affinities: dict[str, list[int]] = {}
    for task in sorted(Path("/proc/self/task").iterdir()):
        try:
            affinities[task.name] = sorted(os.sched_getaffinity(int(task.name)))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return affinities


def configure_and_verify_threads(threads: int) -> dict[str, Any]:
    global torch, e1_positive_candidate, fast_cholesky
    required = {
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
    }
    for name, expected in required.items():
        actual = os.environ.get(name)
        if actual is None or actual.lower() != expected.lower():
            raise RuntimeError(
                f"{name} must be {expected!r} before Python starts, got {actual!r}"
            )

    affinity_before = sorted(os.sched_getaffinity(0))
    topology = [_cpu_topology(cpu) for cpu in affinity_before]
    by_package: dict[int, list[dict[str, int]]] = {}
    for item in topology:
        by_package.setdefault(item["physical_package"], []).append(item)
    choices: list[tuple[int, list[dict[str, int]]]] = []
    for package, entries in by_package.items():
        unique = {
            (entry["physical_package"], entry["core_id"]) for entry in entries
        }
        if len(unique) >= threads:
            choices.append((package, entries))
    if not choices:
        raise RuntimeError(
            f"cannot select {threads} physical cores on one socket from "
            f"affinity {affinity_before}"
        )
    package, entries = min(choices, key=lambda item: item[0])
    selected_entries: list[dict[str, int]] = []
    seen: set[int] = set()
    for entry in sorted(
        entries, key=lambda item: (item["core_id"], item["logical_cpu"])
    ):
        if entry["core_id"] in seen:
            continue
        seen.add(entry["core_id"])
        selected_entries.append(entry)
        if len(selected_entries) == threads:
            break
    selected = sorted(item["logical_cpu"] for item in selected_entries)
    existing_tasks = list(Path("/proc/self/task").iterdir())
    for task in existing_tasks:
        try:
            os.sched_setaffinity(int(task.name), set(selected))
        except (FileNotFoundError, ProcessLookupError):
            continue
    if sorted(os.sched_getaffinity(0)) != selected:
        raise RuntimeError("failed to establish exact main-thread affinity")

    # Import only after affinity is exact so that libgomp constructs its
    # OMP_PLACES from this one-socket CPU set.
    torch = importlib.import_module("torch")
    e1_positive_candidate = importlib.import_module(
        "e1_positive_candidate"
    )
    fast_cholesky = importlib.import_module("fast_cholesky")
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as error:
        raise RuntimeError("interop threads initialized before pinning") from error
    report = torch.__config__.parallel_info()
    runtime = {
        "aten_intraop": _parse_parallel_value(
            report, "at::get_num_threads()"
        ),
        "aten_interop": _parse_parallel_value(
            report, "at::get_num_interop_threads()"
        ),
        "openmp_max": _parse_parallel_value(
            report, "omp_get_max_threads()"
        ),
        "mkl_max": _parse_parallel_value(report, "mkl_get_max_threads()"),
    }
    if runtime != {
        "aten_intraop": threads,
        "aten_interop": 1,
        "openmp_max": threads,
        "mkl_max": threads,
    }:
        raise RuntimeError(f"thread runtime mismatch: {runtime}")
    return {
        "requested_threads": threads,
        "required_environment": required,
        "affinity_before": affinity_before,
        "affinity_before_topology": topology,
        "selected_package": package,
        "selected_cpus": selected,
        "selected_cpu_topology": selected_entries,
        "runtime_values": runtime,
        "torch_parallel_info": report,
        "tasks_immediately_after_configuration": _task_affinities(),
    }


def whiten(L: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    left = torch.linalg.solve_triangular(L, M, upper=False)
    return torch.linalg.solve_triangular(
        L, left.mT, upper=False
    ).mT.contiguous()


def relative_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = torch.linalg.norm(value - reference)
    denominator = torch.linalg.norm(reference).clamp_min(1e-30)
    return float((numerator / denominator).item())


def make_case(rank: int, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed * 100_003 + rank)
    width = min(rank, 64)
    B = (
        torch.randn(rank, width, generator=generator, dtype=torch.float32)
        * 0.05
    )
    identity = torch.eye(rank, dtype=torch.float32)
    G = (identity + B @ B.mT).contiguous()
    M = (G - identity).contiguous()
    L = torch.linalg.cholesky(G).contiguous()
    Aw = whiten(L, M)
    z = (
        torch.randn(rank, generator=generator, dtype=torch.float32) * 0.01
    ).contiguous()
    outer = torch.outer(z, z).contiguous()
    return {
        "G": G,
        "M": M,
        "L": L,
        "Aw": Aw,
        "z": z,
        "outer": outer,
    }


def fresh_outputs(case: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    G_plus = case["G"] + torch.outer(case["z"], case["z"])
    M_plus = case["M"] + torch.outer(case["z"], case["z"])
    L_plus = torch.linalg.cholesky(G_plus).contiguous()
    Aw_plus = whiten(L_plus, M_plus)
    vz = torch.linalg.solve_triangular(
        L_plus, case["z"].unsqueeze(1), upper=False
    ).squeeze(1)
    return L_plus, Aw_plus, vz


def validate_case(
    case: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], tuple[torch.Tensor, ...]]:
    rank = case["L"].shape[0]
    reference = fresh_outputs(case)
    L_fresh, Aw_fresh, vz_fresh = reference

    L_candidate = case["L"].clone()
    Aw_candidate = torch.empty_like(case["Aw"])
    vz_candidate = torch.empty(rank, dtype=torch.float32)
    workspace = torch.empty(4 * rank, dtype=torch.float32)
    e1_positive_candidate.positive_metric_operator_same_vector_out(
        L_candidate,
        case["Aw"],
        case["z"],
        Aw_candidate,
        vz_candidate,
        workspace,
    )

    L_locked = case["L"].clone()
    Aw_locked, vz_locked = fast_cholesky.fast_incremental_update(
        L_locked,
        case["Aw"],
        case["z"].clone(),
        case["z"].clone(),
        metric_sign=1,
        operator_sign=1,
    )
    methods = {
        "candidate": (L_candidate, Aw_candidate, vz_candidate),
        "locked": (L_locked, Aw_locked, vz_locked),
    }
    errors: dict[str, dict[str, float]] = {}
    for method, (L_value, Aw_value, vz_value) in methods.items():
        method_errors = {
            "L": relative_error(L_value, L_fresh),
            "Aw": relative_error(Aw_value, Aw_fresh),
            "vz": relative_error(vz_value, vz_fresh),
        }
        for name, value in method_errors.items():
            if not math.isfinite(value) or value > TOLERANCES[name]:
                raise RuntimeError(
                    f"{method} rank {rank} {name} error {value:.6g} "
                    f"exceeds frozen tolerance {TOLERANCES[name]:.6g}"
                )
        errors[method] = method_errors
    errors["candidate_vs_locked"] = {
        "L": relative_error(L_candidate, L_locked),
        "Aw": relative_error(Aw_candidate, Aw_locked),
        "vz": relative_error(vz_candidate, vz_locked),
    }
    return {
        "relative_errors": errors,
        "frozen_tolerances": TOLERANCES,
        "passed": True,
    }, reference


def summarize(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean_seconds": statistics.mean(values),
        "sample_sd_seconds": statistics.stdev(values),
        "median_seconds": statistics.median(values),
        "minimum_seconds": ordered[0],
        "maximum_seconds": ordered[-1],
    }


def benchmark_case(
    case: dict[str, torch.Tensor],
    reference: tuple[torch.Tensor, ...],
) -> dict[str, Any]:
    rank = case["L"].shape[0]
    candidate_L = torch.empty_like(case["L"])
    candidate_Aw = torch.empty_like(case["Aw"])
    candidate_vz = torch.empty(rank, dtype=torch.float32)
    candidate_workspace = torch.empty(4 * rank, dtype=torch.float32)
    locked_L = torch.empty_like(case["L"])

    def prepare_candidate() -> Callable[[], Any]:
        candidate_L.copy_(case["L"])

        def call() -> None:
            e1_positive_candidate.positive_metric_operator_same_vector_out(
                candidate_L,
                case["Aw"],
                case["z"],
                candidate_Aw,
                candidate_vz,
                candidate_workspace,
            )

        return call

    def prepare_locked() -> Callable[[], Any]:
        locked_L.copy_(case["L"])

        def call() -> tuple[torch.Tensor, torch.Tensor]:
            # Preserve the exact locked E1 call shape, including its public
            # caller clones and unchanged internal allocation behavior.
            return fast_cholesky.fast_incremental_update(
                locked_L,
                case["Aw"],
                case["z"].clone(),
                case["z"].clone(),
                metric_sign=1,
                operator_sign=1,
            )

        return call

    def prepare_fresh() -> Callable[[], Any]:
        return lambda: fresh_outputs(case)

    preparers = {
        "candidate": prepare_candidate,
        "locked": prepare_locked,
        "fresh": prepare_fresh,
    }
    for _ in range(WARMUPS):
        for name in ("candidate", "locked", "fresh"):
            preparers[name]()()

    permutations = list(itertools.permutations(preparers))
    samples = {name: [] for name in preparers}
    schedule: list[list[str]] = []
    for repeat in range(REPEATS):
        order = permutations[repeat % len(permutations)]
        schedule.append(list(order))
        for name in order:
            call = preparers[name]()
            started = time.perf_counter()
            call()
            samples[name].append(time.perf_counter() - started)

    phase_output = torch.empty(len(PHASE_NAMES), dtype=torch.float64)
    phase_samples: list[list[float]] = []
    for _ in range(REPEATS):
        candidate_L.copy_(case["L"])
        e1_positive_candidate.positive_metric_operator_same_vector_profile_out(
            candidate_L,
            case["Aw"],
            case["z"],
            candidate_Aw,
            candidate_vz,
            candidate_workspace,
            phase_output,
        )
        phase_samples.append([float(value) for value in phase_output.tolist()])
    phase_by_name = {
        name: [sample[index] for sample in phase_samples]
        for index, name in enumerate(PHASE_NAMES)
    }

    # Revalidate the persistent output buffers after all timed calls.
    L_fresh, Aw_fresh, vz_fresh = reference
    prepare_candidate()()
    post_timing_errors = {
        "L": relative_error(candidate_L, L_fresh),
        "Aw": relative_error(candidate_Aw, Aw_fresh),
        "vz": relative_error(candidate_vz, vz_fresh),
    }
    for name, value in post_timing_errors.items():
        if value > TOLERANCES[name]:
            raise RuntimeError(
                f"post-timing candidate {name} error {value:.6g} exceeds "
                f"{TOLERANCES[name]:.6g}"
            )

    summaries = {name: summarize(values) for name, values in samples.items()}
    candidate_median = summaries["candidate"]["median_seconds"]
    locked_median = summaries["locked"]["median_seconds"]
    fresh_median = summaries["fresh"]["median_seconds"]
    return {
        "rank": rank,
        "raw_seconds": samples,
        "method_order_schedule": schedule,
        "summaries": summaries,
        "speedups": {
            "locked_over_candidate": locked_median / candidate_median,
            "fresh_over_candidate": fresh_median / candidate_median,
            "fresh_over_locked": fresh_median / locked_median,
        },
        "candidate_phase_raw_seconds": phase_by_name,
        "candidate_phase_summaries": {
            name: summarize(values) for name, values in phase_by_name.items()
        },
        "candidate_profile_sum_raw_seconds": [
            sum(sample) for sample in phase_samples
        ],
        "post_timing_relative_errors": post_timing_errors,
        "candidate_storage": {
            "workspace_elements": 4 * rank,
            "workspace_bytes": 4 * rank * 4,
            "vz_output_bytes": rank * 4,
            "ping_pong_Aw_output_bytes": rank * rank * 4,
            "total_caller_extra_bytes": (rank * rank + 5 * rank) * 4,
            "timed_dynamic_allocations": 0,
        },
    }


def run_pilot(threads: int, results_dir: Path) -> Path:
    if threads not in THREAD_COUNTS:
        raise ValueError(f"threads must be one of {THREAD_COUNTS}")
    protocol, protocol_sha = _read_protocol()
    thread_report = configure_and_verify_threads(threads)
    rows: list[dict[str, Any]] = []
    input_hashes: list[dict[str, Any]] = []
    for seed in PILOT_SEEDS:
        for rank in RANKS:
            case = make_case(rank, seed)
            validation, reference = validate_case(case)
            benchmark = benchmark_case(case, reference)
            rows.append(
                {
                    "seed": seed,
                    "rank": rank,
                    "validation": validation,
                    "benchmark": benchmark,
                }
            )
            input_hashes.append(
                {
                    "seed": seed,
                    "rank": rank,
                    "tensor_sha256": {
                        name: _tensor_sha256(case[name])
                        for name in ("G", "M", "L", "Aw", "z")
                    },
                }
            )

    task_affinities = _task_affinities()
    selected_set = set(thread_report["selected_cpus"])
    if any(
        not set(affinity).issubset(selected_set)
        for affinity in task_affinities.values()
    ):
        raise RuntimeError("a runtime worker escaped the selected CPU set")
    worker_union = set().union(*(set(value) for value in task_affinities.values()))
    if worker_union != selected_set:
        raise RuntimeError(
            f"runtime worker affinity union {sorted(worker_union)} does not "
            f"match selected CPUs {sorted(selected_set)}"
        )
    thread_report["tasks_after_benchmark"] = task_affinities
    thread_report["worker_affinity_union"] = sorted(worker_union)

    candidate_binary = Path(e1_positive_candidate.__file__).resolve()
    locked_binary = Path(fast_cholesky.__file__).resolve()
    payload = {
        "experiment": "e1_positive_same_vector_synthetic_pilot",
        "campaign": CAMPAIGN,
        "status": "complete",
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "threads": threads,
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "git_head_at_run": _git("rev-parse", "HEAD"),
            "thread_verification": thread_report,
        },
        "provenance": {
            "driver": str(Path(__file__).resolve()),
            "driver_sha256": _sha256(Path(__file__).resolve()),
            "protocol": str(PROTOCOL_PATH.resolve()),
            "protocol_sha256": protocol_sha,
            "candidate_source": str(
                (HERE / "e1_positive_candidate_ext.cpp").resolve()
            ),
            "candidate_source_sha256": _sha256(
                HERE / "e1_positive_candidate_ext.cpp"
            ),
            "candidate_binary": str(candidate_binary),
            "candidate_binary_sha256": _sha256(candidate_binary),
            "locked_source": str(
                (CORRECTED / "fast_update.cpp").resolve()
            ),
            "locked_source_sha256": _sha256(
                CORRECTED / "fast_update.cpp"
            ),
            "locked_binary": str(locked_binary),
            "locked_binary_sha256": _sha256(locked_binary),
        },
        "scientific_guardrails": {
            "protein_loaded": False,
            "model_or_test_outcomes_used": False,
            "tuning_seed": TUNING_SEED,
            "tuning_seed_disjoint_from_pilot": TUNING_SEED not in PILOT_SEEDS,
            "candidate_contract": (
                "explicit equal positive rank-one terms in G and M; "
                "general signed/different-vector calls retain locked fallback"
            ),
        },
        "input_hashes": input_hashes,
        "rows": rows,
    }
    destination = results_dir / CAMPAIGN / f"pilot_t{threads}.json"
    _write_once(payload, destination)
    return destination


def aggregate(results_dir: Path) -> Path:
    campaign_dir = results_dir / CAMPAIGN
    payloads: dict[int, dict[str, Any]] = {}
    for threads in THREAD_COUNTS:
        path = campaign_dir / f"pilot_t{threads}.json"
        with path.open(encoding="utf-8") as handle:
            payloads[threads] = json.load(handle)
    protocol_sha = payloads[1]["protocol_sha256"]
    source_sha = payloads[1]["provenance"]["candidate_source_sha256"]
    for threads, payload in payloads.items():
        if payload["threads"] != threads or payload["status"] != "complete":
            raise RuntimeError(f"invalid pilot artifact for t={threads}")
        if payload["protocol_sha256"] != protocol_sha:
            raise RuntimeError("mixed protocol hashes across thread modes")
        if payload["provenance"]["candidate_source_sha256"] != source_sha:
            raise RuntimeError("mixed candidate sources across thread modes")
        observed = {
            (row["seed"], row["rank"]) for row in payload["rows"]
        }
        expected = {
            (seed, rank) for seed in PILOT_SEEDS for rank in RANKS
        }
        if observed != expected:
            raise RuntimeError(f"missing or extra t={threads} pilot rows")

    summary_rows = []
    for rank in RANKS:
        item: dict[str, Any] = {"rank": rank, "threads": {}}
        medians: dict[int, dict[str, list[float]]] = {}
        for threads, payload in payloads.items():
            rank_rows = [
                row for row in payload["rows"] if row["rank"] == rank
            ]
            medians[threads] = {
                method: [
                    row["benchmark"]["summaries"][method]["median_seconds"]
                    for row in rank_rows
                ]
                for method in ("candidate", "locked", "fresh")
            }
            item["threads"][str(threads)] = {
                method: summarize(values)
                for method, values in medians[threads].items()
            }
            candidate = statistics.median(medians[threads]["candidate"])
            locked = statistics.median(medians[threads]["locked"])
            fresh = statistics.median(medians[threads]["fresh"])
            item["threads"][str(threads)]["median_seed_speedups"] = {
                "locked_over_candidate": locked / candidate,
                "fresh_over_candidate": fresh / candidate,
                "fresh_over_locked": fresh / locked,
            }
        item["candidate_thread_scaling_t1_over_t10"] = (
            statistics.median(medians[1]["candidate"])
            / statistics.median(medians[10]["candidate"])
        )
        item["fresh_thread_scaling_t1_over_t10"] = (
            statistics.median(medians[1]["fresh"])
            / statistics.median(medians[10]["fresh"])
        )
        summary_rows.append(item)

    summary = {
        "experiment": "e1_positive_same_vector_synthetic_pilot_summary",
        "campaign": CAMPAIGN,
        "status": "complete",
        "protocol_sha256": protocol_sha,
        "candidate_source_sha256": source_sha,
        "rows": summary_rows,
        "interpretation_scope": (
            "synthetic kernel pilot only; no Protein data or held-out model "
            "outcome was used, and this does not replace E1 confirmation"
        ),
    }
    destination = campaign_dir / "summary.json"
    _write_once(summary, destination)
    return destination


def self_test() -> None:
    _read_protocol()
    configure_and_verify_threads(int(os.environ["OMP_NUM_THREADS"]))
    for rank in (8, 17):
        case = make_case(rank, TUNING_SEED)
        validation, _ = validate_case(case)
        if not validation["passed"]:
            raise RuntimeError("candidate self-test unexpectedly failed")
    rank = 8
    case = make_case(rank, TUNING_SEED)
    try:
        e1_positive_candidate.positive_metric_operator_same_vector_out(
            case["L"].clone(),
            case["Aw"],
            case["z"],
            case["Aw"],
            torch.empty(rank),
            torch.empty(4 * rank),
        )
    except RuntimeError:
        pass
    else:
        raise RuntimeError("candidate accepted aliased Aw/Aw_out")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action", choices=("self-test", "run", "aggregate"), required=True
    )
    parser.add_argument("--threads", type=int, choices=THREAD_COUNTS)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "self-test":
        self_test()
        print("e1 positive candidate self-test: PASS")
    elif args.action == "run":
        if args.threads is None:
            raise ValueError("--threads is required for --action run")
        print(run_pilot(args.threads, args.results_dir.resolve()))
    else:
        print(aggregate(args.results_dir.resolve()))


if __name__ == "__main__":
    main()
