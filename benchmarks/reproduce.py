"""Run the documented reproduction commands and retain complete per-step logs.

This module uses only the standard library; --dry-run needs no installed Torch.
Run from a checkout. Installation is explicit and separate from experiments.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import statistics
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
WELL_RANKS = (2048, 4096, 8192, 16384, 32768, 49152, 65536)
SYMMETRIC_RANKS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
DEFAULT_TRACE = ROOT / "data/the_well/processed/post_neutron_star_merger_scenario_4_dct65536.npy"


def module(name, *args):
    return [sys.executable, "-u", "-m", name, *map(str, args)]


def plan(suite, output, trace, smoke=False, workers=32):
    """Return isolated processes in the same order as the validated commands."""
    if suite == "prepare-well":
        return [
            ("download", module("benchmarks.the_well.download")),
            ("features", module("benchmarks.the_well.prepare_features", "--workers", workers)),
        ]
    if suite == "cpu":
        return [
            ("tests", module("pytest", "-m", "not cuda and not large", "-x", "-q", "-rs",
                             "--junitxml", output / "tests.xml")),
            ("cpu-synthetic", module("benchmarks.cpu_synthetic")),
        ]
    if suite != "b200":
        raise ValueError(f"Unknown suite: {suite}")
    symmetric = (128, 1024) if smoke else SYMMETRIC_RANKS
    asymmetric = (8192,) if smoke else (8192, 32768, 65536)
    well = (2048,) if smoke else WELL_RANKS
    tests = ["-m", "not large"] if smoke else []
    steps = [
        ("tests", module("pytest", *tests, "-x", "-q", "-rs",
                         "--junitxml", output / "tests.xml")),
        ("symmetric", module("benchmarks.synthetic_symmetric", "--seed", 0,
                             "--ranks", *symmetric, "--output", output / "symmetric")),
        ("asymmetric", module("benchmarks.synthetic_asymmetric", "--seed", 0,
                              "--ranks", *asymmetric, "--warmups", 2, "--repeats", 5,
                              "--output", output / "asymmetric")),
    ]
    for method in ("tiled", "inplace"):
        directory = output / ("well-" + method)
        for rank in well:
            steps.append((f"well-{method}-{rank}", module(
                "benchmarks.the_well.rank_ladder", "--method", method,
                "--ranks", rank, "--trace", trace, "--output-dir", directory,
                "--run-tag", "reproduction-" + method)))
        steps.append((f"aggregate-{method}", module(
            "benchmarks.the_well.aggregate", directory)))
    return steps


def require_test_pass(path):
    """Pytest exits zero when selected tests skip; that is not reproduction."""
    cases = list(ET.parse(path).getroot().iter("testcase"))
    if not cases:
        raise RuntimeError("No test cases recorded")
    bad = [c.attrib.get("name") for c in cases
           if any(c.find(tag) is not None for tag in ("skipped", "failure", "error"))]
    if bad:
        raise RuntimeError(f"Selected tests did not all pass: {bad}")


def require_ladder_pass(path, expected_rank, expected_method):
    data = json.loads(path.read_text())
    if data["rank"] != expected_rank or data["method"] != expected_method:
        raise RuntimeError("Unexpected rank/method in ladder artifact")
    cases = data["cases"]
    if len(cases) != 10 or data["case_count"] != 10:
        raise RuntimeError("The Well protocol requires ten cases per rank")
    if {case["case_index"] for case in cases} != set(range(10)):
        raise RuntimeError("Missing or duplicate case indices")
    for case in cases:
        if case["cross_method_probe_comparison"]["passed"] is not True:
            raise RuntimeError("Cross-method comparison failed")
        for name in ("inverse", "fresh_refact"):
            method = case["methods"][name]
            if method["validation"]["passed"] is not True:
                raise RuntimeError(f"{name} correctness failed")
            samples = method["event_seconds_samples"]
            if not samples or len(samples) != method["repeats"]:
                raise RuntimeError("Incomplete timing sample array")
            if not all(math.isfinite(x) and x > 0 for x in samples):
                raise RuntimeError("Invalid timing samples")
            if not math.isclose(statistics.median(samples), method["event_seconds_median"], rel_tol=1e-12):
                raise RuntimeError("Median disagrees with raw samples")


def verify_step(name, output):
    if name == "tests":
        require_test_pass(output / "tests.xml")
    elif name.startswith("well-"):
        _, method, rank = name.split("-")
        require_ladder_pass(output / f"well-{method}/ladder_r{rank}.json", int(rank), method)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(suite, trace, smoke):
    if suite == "prepare-well":
        for package in ("numpy", "scipy", "h5py", "huggingface_hub"):
            __import__(package)
        return
    import torch
    from doublesided_cholesky import _C_cpu  # Fail rather than silently time fallback.
    if suite == "cpu":
        return
    from doublesided_cholesky import _C
    if not torch.cuda.is_available():
        raise RuntimeError("The B200 suite requires a CUDA GPU")
    props = torch.cuda.get_device_properties(0)
    if not smoke and "B200" not in props.name:
        raise RuntimeError("Use B200 hardware for the full B200 timing reproduction")
    if props.major < 8:
        raise RuntimeError("CUDA kernels require compute capability 8.0 or newer")
    free, _ = torch.cuda.mem_get_info()
    minimum = (4 if smoke else 128) * 2**30
    if free < minimum:
        raise RuntimeError(f"Need at least {minimum / 2**30:g} GiB free GPU memory")
    protocol = json.loads((ROOT / "benchmarks/the_well/protocol.json").read_text())
    # The consumer performs its own hash/shape/scale checks too.
    expected = protocol["trace"]["sha256"]
    if not trace.is_file() or sha256(trace) != expected:
        raise RuntimeError("Prepared trace missing or hash mismatch; run scripts/prepare_well.sh first")


def execute(steps, output, suite, smoke):
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    if suite in ("cpu", "b200"):
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if suite == "b200" and not smoke:
        env["DSC_RUN_LARGE"] = "1"
    # Prevent inherited opt-in large flags changing the smoke profile.
    if suite == "b200" and smoke:
        env.pop("DSC_RUN_LARGE", None)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    report = {"suite": suite, "profile": "smoke" if smoke else "full",
              "git_commit": commit, "status": "running", "steps": [],
              "created_utc": datetime.now(timezone.utc).isoformat()}
    manifest = output / "run.json"
    def save():
        manifest.write_text(json.dumps(report, indent=2) + "\n")
    save()
    try:
        for index, (name, command) in enumerate(steps, 1):
            log = output / f"{name}.log"
            print(f"[{index}/{len(steps)}] {name} — log: {log}", flush=True)
            start = time.monotonic()
            row = {"name": name, "command": command, "status": "running"}
            report["steps"].append(row)
            save()
            with log.open("w") as handle:
                result = subprocess.run(command, cwd=ROOT, env=env,
                                        stdout=handle, stderr=subprocess.STDOUT, check=False)
            row.update(returncode=result.returncode, wall_seconds=time.monotonic() - start)
            if result.returncode:
                row["status"] = "failed"
                raise RuntimeError(f"{name} failed with exit {result.returncode}; see {log}")
            verify_step(name, output)
            row["status"] = "passed"
            save()
        report["status"] = "passed"
    except BaseException:
        report["status"] = "failed"
        if report["steps"] and report["steps"][-1]["status"] == "running":
            report["steps"][-1]["status"] = "failed"
        raise
    finally:
        save()
    print(f"Completed {suite} ({report['profile']}). Evidence: {output}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("cpu", "b200", "prepare-well"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--smoke", action="store_true",
                        help="B200 only: small selected ranks; not full reproduction")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without imports, allocation, downloads or writes")
    args = parser.parse_args(argv)
    if args.smoke and args.suite != "b200":
        parser.error("--smoke applies only to b200")
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = (args.output_dir or ROOT / "results/local" /
              (args.suite + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))).resolve()
    steps = plan(args.suite, output, args.trace.resolve(), args.smoke, args.workers)
    if args.dry_run:
        for name, command in steps:
            print(name + ": " + shlex.join(command))
        return
    if output.exists():
        parser.error("Output directory already exists; choose a fresh directory to preserve evidence")
    preflight(args.suite, args.trace.resolve(), args.smoke)
    execute(steps, output, args.suite, args.smoke)


if __name__ == "__main__":
    main()
