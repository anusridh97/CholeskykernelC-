#!/usr/bin/env python
"""Run both GPU experiments end-to-end and auto-generate the figure report.

No scheduler required — just run it on whatever machine has the GPU (e.g. a
Nebius instance):

    python run_experiments.py                 # both experiments + figures
    python run_experiments.py --only gp       # gp | curvature
    python run_experiments.py --figures-only  # re-render plots from existing JSON

Results JSON land in results/, figures and an HTML report in figures/. The
benchmarks pick CUDA automatically when available and fall back to CPU (small
sizes) otherwise, so this is safe to smoke-test anywhere.
"""

import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["gp", "curvature"], default=None,
                    help="run just one experiment")
    ap.add_argument("--figures-only", action="store_true",
                    help="skip benchmarks, just re-render figures from results/")
    args = ap.parse_args()

    if not args.figures_only:
        if args.only in (None, "curvature"):
            print("=" * 70)
            print(" Experiment 1: incremental curvature optimizer")
            print("=" * 70)
            from tests import bench_curvature_opt
            bench_curvature_opt.run()
            print()
        if args.only in (None, "gp"):
            print("=" * 70)
            print(" Experiment 2: massively parallel GP Thompson sampling")
            print("=" * 70)
            from tests import bench_gp_thompson
            bench_gp_thompson.run()
            print()

    print("=" * 70)
    print(" Rendering figures")
    print("=" * 70)
    from experiments.viz import render_all
    report = render_all()
    if report is not None:
        print(f"\nOpen {report} to view the report.")


if __name__ == "__main__":
    main()
