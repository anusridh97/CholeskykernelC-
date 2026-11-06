"""Reproduce the recovered corrected CPU BO/NG experiments.

These drivers do not claim to reproduce the original submitted paper tables.
Real input files are required unless --smoke or --dataset synthetic is explicit.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from benchmarks.reproduce import ROOT, sha256
from benchmarks.summarize import write_csv

PROTOCOL = "recovered-corrected-cpu-v1"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("experiment", choices=("bo", "ng"))
    parser.add_argument("--dataset", choices=("gb1", "protein", "synthetic"))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ranks", type=int, nargs="+")
    parser.add_argument("--backend", choices=("extension", "torch"), default="extension")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--smoke", action="store_true",
                        help="Explicit synthetic, small-rank check; never a paper result")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--initial", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--bo-seeds", type=int, help="BO seed count: seeds 0 through N-1")
    parser.add_argument("--lengthscale", type=float, default=5.0)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--refactor-every", type=int, default=50, help="BO only")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--ng-seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--hidden-dims", type=int, nargs="+")
    parser.add_argument("--chart-modes", choices=("inverse", "sandwich"), nargs="+",
                        default=["inverse", "sandwich"])
    args = parser.parse_args(argv)
    provided = {token.split("=")[0] for token in (argv if argv is not None else sys.argv[1:])}
    bo_only = {"--initial", "--rounds", "--batch-size", "--candidate-count", "--bo-seeds",
               "--lengthscale", "--noise-std", "--refactor-every"}
    ng_only = {"--epochs", "--ng-seeds", "--hidden-dims", "--chart-modes"}
    wrong = provided & (ng_only if args.experiment == "bo" else bo_only)
    if wrong:
        parser.error("Options for the other experiment: " + ", ".join(sorted(wrong)))
    if args.smoke and args.dataset not in (None, "synthetic"):
        parser.error("--smoke uses synthetic data; omit --smoke for a real dataset")
    args.dataset = "synthetic" if args.smoke else (args.dataset or
                    ("gb1" if args.experiment == "bo" else "protein"))
    if (args.experiment == "bo" and args.dataset == "protein") or (
            args.experiment == "ng" and args.dataset == "gb1"):
        parser.error("BO supports gb1/synthetic; NG supports protein/synthetic")
    defaults = dict(ranks=[8] if args.smoke else [64],
                    initial=16 if args.smoke else 96,
                    rounds=2 if args.smoke else 30,
                    batch_size=4 if args.smoke else 96,
                    candidate_count=64 if args.smoke else 5000,
                    bo_seeds=1 if args.smoke else 5,
                    epochs=1 if args.smoke else 5,
                    hidden_dims=[16] if args.smoke else [128])
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    positive = [args.threads, args.initial, args.rounds, args.batch_size,
                args.candidate_count, args.bo_seeds, args.epochs, *args.ranks, *args.hidden_dims]
    if min(positive) < 1 or args.lengthscale <= 0 or args.noise_std <= 0:
        parser.error("Sizes, thread count, lengthscale, and noise standard deviation must be positive")
    if args.refactor_every < 0 or min(args.ng_seeds) < 0:
        parser.error("Seed and refactor interval must be nonnegative")
    for name in ("ranks", "ng_seeds", "chart_modes"):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must not contain duplicates")
    if args.experiment == "bo" and max(args.ranks) > args.initial:
        parser.error("BO rank must not exceed --initial; ranks are never silently clipped")
    args.output_dir = (args.output_dir or ROOT / "results/local" /
                       (args.experiment + "-" +
                        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))).resolve()
    args.data_dir = args.data_dir.resolve()
    return args


def input_files(args):
    names = (["gb1_X.npy", "gb1_y.npy"] if args.dataset == "gb1" else
             ["protein.mat"] if args.dataset == "protein" else [])
    return [args.data_dir / name for name in names]


def configuration(args):
    common = {"backend": args.backend, "dtype": args.dtype, "threads": args.threads,
              "dataset": args.dataset, "ranks": args.ranks, "device": "cpu"}
    if args.experiment == "bo":
        common.update(initial=args.initial, rounds=args.rounds, batch_size=args.batch_size,
                      candidate_count=args.candidate_count, seeds=list(range(args.bo_seeds)),
                      lengthscale=args.lengthscale, noise_std=args.noise_std,
                      refactor_every=args.refactor_every)
    else:
        common.update(epochs=args.epochs, hidden_dims=args.hidden_dims,
                      chart_modes=args.chart_modes, seeds=args.ng_seeds,
                      batch_size=256, learning_rate=0.01, lam=1.0, refresh_threshold=0.1,
                      refactor_every=50, sandwich_beta=0.9, train_fraction=0.8)
    return common


def main(argv=None):
    args = arguments(argv)
    config = configuration(args)
    if args.dry_run:
        print(json.dumps({"protocol": PROTOCOL, "profile": "smoke" if args.smoke else "full",
                          "configuration": config, "required_inputs": list(map(str, input_files(args))),
                          "output_dir": str(args.output_dir)}, indent=2))
        return
    if args.output_dir.exists():
        raise FileExistsError("Output already exists; choose a fresh --output-dir")
    for path in input_files(args):
        if not path.is_file():
            raise FileNotFoundError(f"Required real dataset missing: {path}. "
                                    "Supply it, or use --smoke for a labeled synthetic check.")
    import numpy as np
    import torch
    from ._corrected import core, experiments
    from ._corrected.data_utils import load_gb1, load_uci_mat
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.backend == "extension" and not core.extension_available():
        raise RuntimeError("Signed CPU backend missing. Run bash scripts/setup.sh applications; "
                           "or explicitly choose --backend torch for reference-only execution.")
    dtype = getattr(torch, args.dtype)
    dataset_provenance = None
    if args.dataset == "gb1":
        from .prepare_gb1 import MANIFEST, verify
        if (args.data_dir / MANIFEST).exists():
            dataset_provenance = verify(args.data_dir)
        X, y = load_gb1(data_dir=args.data_dir, dtype=dtype)
        if X.ndim != 2 or not torch.isfinite(X).all() or not torch.isfinite(y).all():
            raise ValueError("Expected finite GB1 X[N,D] and y[N]")
        if len(y) < args.initial + args.rounds * args.batch_size:
            raise ValueError("Dataset too small for the requested distinct observations")
        del X, y
    elif args.dataset == "protein":
        parts = load_uci_mat("protein", data_dir=args.data_dir, dtype=dtype)
        if not all(torch.isfinite(part).all() for part in parts):
            raise ValueError("Protein data contains nonfinite values")
        if max(args.ranks) > (parts[0].shape[1] + 1) * args.hidden_dims[0]:
            raise ValueError("NG rank exceeds the number of first-layer parameters")
        del parts
    elif args.dataset == "synthetic":
        if args.experiment == "bo" and 20000 < args.initial + args.rounds * args.batch_size:
            raise ValueError("Synthetic landscape has 20000 examples; requested observations exceed it")
        if args.experiment == "ng" and max(args.ranks) > 10 * args.hidden_dims[0]:
            raise ValueError("NG rank exceeds the number of first-layer parameters")
    args.output_dir.mkdir(parents=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    files = sorted(Path(__file__).parent.rglob("*.py"))
    files += sorted(Path(__file__).parent.rglob("*.cpp"))
    native = getattr(core, "_fast_cholesky", None)
    report = {"protocol": PROTOCOL, "experiment": args.experiment,
              "scope": "Corrected CPU experiment; original submitted tables not asserted",
              "profile": "smoke" if args.smoke else "full", "configuration": config,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "git_commit": commit, "git_dirty": dirty,
              "environment": {"python": sys.version, "torch": torch.__version__,
                              "numpy": np.__version__, "platform": platform.platform()},
              "input_sha256": {p.name: sha256(p) for p in input_files(args)},
              "dataset_provenance": dataset_provenance,
              "input_manifest_sha256": sha256(args.data_dir / MANIFEST) if dataset_provenance else None,
              "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in files},
              "native_sha256": sha256(Path(native.__file__)) if native else None,
              "ng_function_defaults": {k: str(v.default) for k, v in inspect.signature(
                  experiments.transported_ng_experiment).parameters.items()
                  if v.default is not inspect.Parameter.empty},
              "status": "running", "runs": []}
    manifest = args.output_dir / "run.json"
    def save():
        manifest.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    rows = []
    try:
        for rank in args.ranks:
            jobs = [(None, None)] if args.experiment == "bo" else [
                (mode, seed) for mode in args.chart_modes for seed in args.ng_seeds]
            for mode, seed in jobs:
                name = f"r{rank}" if mode is None else f"{mode}-r{rank}-seed{seed}"
                directory = args.output_dir / name
                directory.mkdir()
                experiments.RESULTS_DIR = directory
                print(f"Running {args.experiment} {name} ({args.dataset}, CPU {args.backend}); "
                      f"log: {directory / 'run.log'}", flush=True)
                entry = {"name": name, "status": "running"}
                report["runs"].append(entry)
                save()
                start = time.monotonic()
                with (directory / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
                    if args.experiment == "bo":
                        result = experiments.run_protein_bo(
                            landscape=args.dataset, data_dir=str(args.data_dir), rank=rank,
                            initial=args.initial, rounds=args.rounds, batch_size=args.batch_size,
                            candidate_count=args.candidate_count, seeds=args.bo_seeds,
                            backend=args.backend, dtype=dtype, lengthscale=args.lengthscale,
                            noise_std=args.noise_std, refactor_every=args.refactor_every)
                        for method in ("ours", "refact"):
                            summary = result["summary"][method]
                            for s, (elapsed, curve, fallback) in enumerate(zip(
                                    summary["raw_times"], summary["raw_best_curves"], summary["fallbacks"])):
                                rows.append(dict(rank=rank, seed=s, method=method,
                                                 wall_seconds=elapsed, best=curve[-1], fallbacks=fallback))
                    else:
                        result = experiments.run_ng(
                            dataset=args.dataset, data_dir=str(args.data_dir), rank=rank,
                            epochs=args.epochs, hidden_dims=args.hidden_dims, chart_mode=mode,
                            backend=args.backend, seed=seed, dtype=dtype)
                        for method, measured in result["results"].items():
                            rows.append(dict(rank=rank, seed=seed, chart_mode=mode, method=method,
                                             wall_seconds=measured["wall_times"][-1],
                                             core_seconds=measured["core_times"][-1],
                                             test_mse=measured["test_mse"], steps=measured["steps"]))
                entry.update(status="passed", wall_seconds=time.monotonic() - start)
                save()
                write_csv(args.output_dir / "summary.csv", rows)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        if report["runs"]:
            report["runs"][-1]["status"] = "failed"
        raise
    finally:
        save()
    print(f"Finished. Raw results, configuration and summary.csv: {args.output_dir}")


if __name__ == "__main__":
    main()
