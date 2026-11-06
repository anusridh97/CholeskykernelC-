"""Run an explicitly reconstructed BO/Protein NG sweep on CPU.

Uses the corrected models with documented resolutions of ambiguous paper
settings. This is not an exact reproduction claim for submitted Tables 4–6.
GB1 inputs must match prepare_gb1; Protein must match the recovered MAT hash.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmarks.reproduce import ROOT, sha256
from .prepare_gb1 import verify

PROTEIN_SHA256 = "5ce538fa05f6834de3acc727e220e777a02fb1ebc3a7bfebcb5aaad9769fed13"
PROTOCOL = "corrected-applications-explicit-sweep-v1"


def plan(data_dir, output_dir, threads=1, experiments=("bo", "ng"),
         ng_ranks=(64, 128, 256, 512, 1024)):
    common = [sys.executable, "-m", "benchmarks.applications.run"]
    shared = ["--data-dir", str(data_dir), "--threads", str(threads),
              "--dtype", "float32", "--backend", "extension"]
    jobs = []
    if "bo" in experiments:
        for rank, initial in [(64, 96), (128, 128)]:
            name = f"bo-r{rank}"
            jobs.append((name, common + ["bo"] + shared + [
                "--dataset", "gb1", "--ranks", str(rank), "--initial", str(initial),
                "--rounds", "15", "--batch-size", "96", "--candidate-count", "5000",
                "--bo-seeds", "5", "--lengthscale", "5", "--noise-std", "0.1",
                "--refactor-every", "50", "--output-dir", str(output_dir / name)]))
    if "ng" in experiments:
        for rank in ng_ranks:
            name = f"ng-protein-r{rank}"
            jobs.append((name, common + ["ng"] + shared + [
                "--dataset", "protein", "--ranks", str(rank), "--epochs", "10",
                "--hidden-dims", "512", "512", "128", "--ng-seeds", "42",
                "--chart-modes", "inverse", "--output-dir", str(output_dir / name)]))
    return jobs


def require_completed(path):
    report = json.loads(path.read_text())
    if report.get("status") != "passed" or not report.get("runs") or any(
            run.get("status") != "passed" for run in report["runs"]):
        raise RuntimeError(f"Application run incomplete: {path}")
    summary = path.parent / "summary.csv"
    if not summary.is_file() or summary.stat().st_size == 0:
        raise RuntimeError(f"Application summary missing: {summary}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--experiments", choices=("bo", "ng"), nargs="+", default=["bo", "ng"])
    parser.add_argument("--ng-ranks", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1 or not set(args.ng_ranks) <= {64, 128, 256, 512, 1024}:
        parser.error("Use positive threads and NG ranks from 64,128,256,512,1024")
    if len(set(args.ng_ranks)) != len(args.ng_ranks) or len(set(args.experiments)) != len(args.experiments):
        parser.error("Duplicate experiments or ranks are not allowed")
    args.data_dir = args.data_dir.resolve()
    output = (args.output_dir or ROOT / "results/local" / (
        "applications-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))).resolve()
    jobs = plan(args.data_dir, output, args.threads, args.experiments, args.ng_ranks)
    record = {"protocol": PROTOCOL, "scope": "Corrected reconstruction; not original table reproduction",
              "threads": args.threads, "experiments": args.experiments,
              "ng_ranks": args.ng_ranks if "ng" in args.experiments else [],
              "status": "planned", "jobs": [{"name": n, "command": c} for n, c in jobs],
              "deviations": ["BO rank128 uses initial128, not the impossible initial96/rank128 pairing",
                             "NG uses input9 -> 512 -> 512 -> 128 -> 1 and seed42",
                             "Corrected models; Table5 Elevators/EMA and E2 budget study are separate",
                             "The thread count applies to all Torch work; native update remains sequential"]}
    if args.dry_run:
        print(json.dumps(record, indent=2))
        return
    if output.exists():
        raise FileExistsError("Choose a new --output-dir")
    if "bo" in args.experiments:
        verify(args.data_dir)
    if "ng" in args.experiments and sha256(args.data_dir / "protein.mat") != PROTEIN_SHA256:
        raise ValueError("Protein MAT hash differs from the recovered input")
    output.mkdir(parents=True)
    record["status"] = "running"
    def save():
        (output / "campaign.json").write_text(json.dumps(record, indent=2) + "\n")
    save()
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads))
    try:
        for job in record["jobs"]:
            name = job["name"]
            print(f"Running {name}; log: {output / (name + '.log')}", flush=True)
            with (output / (name + ".log")).open("w") as log:
                completed = subprocess.run(job["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            job["returncode"] = completed.returncode
            if completed.returncode:
                raise RuntimeError(f"{name} failed: see {output / (name + '.log')}")
            require_completed(output / name / "run.json")
            job["status"] = "passed"
            job["summary_sha256"] = sha256(output / name / "summary.csv")
            save()
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        save()
    print(f"Completed {len(jobs)} CPU jobs: {output}")


if __name__ == "__main__":
    main()
