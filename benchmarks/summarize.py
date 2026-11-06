"""Regenerate CSV tables from saved raw samples without Torch or a GPU."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from .reproduce import require_ladder_pass
from .the_well.aggregate import aggregate_csv, aggregate_json


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(source, output):
    """Accept the canonical filenames produced by reproduce_b200.sh."""
    files = list(source.rglob("*.json"))
    if not files:
        raise ValueError("No JSON evidence found")
    if output.exists():
        raise FileExistsError("Choose a fresh output directory")
    output.mkdir(parents=True)
    manifest = {}
    def record(path):
        manifest[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for stem in ("symmetric", "asymmetric"):
        path = source / f"{stem}.json"
        if not path.exists():
            continue
        record(path)
        data = json.loads(path.read_text())
        rows = []
        for entry in data["rows"]:
            medians = {}
            for method, measured in entry["methods"].items():
                samples = measured["samples_ms"]
                if len(samples) != entry["repeats"] or not samples:
                    raise ValueError(f"Incomplete samples: {stem}, {entry['rank']}, {method}")
                if not all(math.isfinite(x) and x > 0 for x in samples):
                    raise ValueError("Invalid timing samples")
                medians[method] = statistics.median(samples)
                if medians[method] != measured["median_ms"]:
                    raise ValueError("Stored median disagrees with raw samples")
            row = {"rank": entry["rank"], **{key + "_ms": val for key, val in medians.items()}}
            if stem == "symmetric":
                row["fresh_over_incremental"] = medians["fresh"] / medians["incremental"]
            else:
                row["default_inplace_over_tiled"] = medians["inplace_default"] / medians["tiled"]
                row["tuned_inplace_over_tiled"] = medians["inplace_tuned"] / medians["tiled_tuned"]
            rows.append(row)
        write_csv(output / f"{stem}.csv", rows)
    for method in ("tiled", "inplace"):
        directory = source / ("well-" + method)
        paths = sorted(directory.glob("ladder_r*.json"))
        if not paths:
            continue
        for path in paths:
            rank = int(path.stem.split("_r")[1])
            require_ladder_pass(path, rank, method)
            record(path)
        (output / f"well-{method}.csv").write_bytes(aggregate_csv(str(directory)))
        (output / f"well-{method}.json").write_text(
            json.dumps(aggregate_json(str(directory)), indent=2) + "\n")
    if not manifest:
        raise ValueError("No recognized raw benchmark artifacts")
    (output / "sources.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Regenerated tables from {len(manifest)} raw artifacts in {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/reproduced/b200-20261010"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/local/tables"))
    args = parser.parse_args(argv)
    summarize(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
