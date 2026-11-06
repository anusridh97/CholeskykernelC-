"""Compare corrected GP methods on a fixed real-GB1 stream, outside BO timing.

This reports numerical differences; it does not assert identical closed-loop
acquisition decisions or certify the submitted paper's table values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.reproduce import ROOT, sha256
from .prepare_gb1 import verify


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Choose a fresh diagnostic output file")
    data = verify(args.data_dir)
    import torch
    from ._corrected.data_utils import load_gb1
    from ._corrected.adaptive_svgp import AdaptiveInducingGaussian
    from ._corrected.core import extension_available, relative_error
    if not extension_available():
        raise RuntimeError("Build the native application backend first")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    X, y = load_gb1(data_dir=args.data_dir)
    records = []
    for rank, initial in [(64, 96), (128, 128)]:
        generator = torch.Generator().manual_seed(2)
        order = torch.randperm(len(y), generator=generator)
        train = order[:initial]
        replay = order[initial:initial + 1440]
        probes = X[order[-5000:]]
        models = [AdaptiveInducingGaussian(
            X[train[:rank]], method=method, backend="extension", lengthscale=5,
            noise_std=0.1, refactor_every=50) for method in ("ours", "refact")]
        for model in models:
            for index in train:
                model.condition(X[index], y[index])
        metrics = []
        for step, index in enumerate(replay, 1):
            for model in models:
                model.replace_and_condition(X[index], y[index])
            if step % 96 == 0:
                predictions = [model.predict(probes) for model in models]
                metrics.append({"step": step,
                    "mean_relative": relative_error(predictions[0][0], predictions[1][0]),
                    "variance_relative": relative_error(predictions[0][1], predictions[1][1]),
                    "validation": models[0].validate()})
        record = dict(rank=rank, initial=initial, seed=2, replayed_updates=1440,
                      mean_relative_max=max(x["mean_relative"] for x in metrics),
                      variance_relative_max=max(x["variance_relative"] for x in metrics),
                      checkpoints=metrics)
        records.append(record)
        print(f"rank {rank}: mean {record['mean_relative_max']:.3g}, "
              f"variance {record['variance_relative_max']:.3g}", flush=True)
    source_dir = Path(__file__).parent
    from ._corrected import core
    payload = dict(protocol="gb1-shared-stream-diagnostic-v1", status="completed",
                   scope="Fixed-input numerical diagnostic; no acquisition or timing comparison",
                   torch=torch.__version__, dataset=data, dtype="float32", threads=1,
                   source_sha256={str(p.relative_to(ROOT)): sha256(p)
                                  for p in sorted(source_dir.rglob("*.py"))},
                   native_sha256=sha256(Path(core._fast_cholesky.__file__)), ranks=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        out.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
