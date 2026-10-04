"""Time descriptor, north scan, west scan and apply phases in milliseconds."""
import argparse
from pathlib import Path
import statistics
import json
import torch
from doublesided_cholesky import layouts
from ._env import configure_cuda, metadata

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, default=65536)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path, default=Path('results/local/phases.json'))
    args = parser.parse_args()
    from doublesided_cholesky import _C
    from doublesided_cholesky.cuda import SymmetricTiledWorkspace
    configure_cuda()
    ws = SymmetricTiledWorkspace(args.rank)
    A = torch.zeros(layouts.tile_elements(args.rank), device='cuda')
    ws.cs.fill_(1)
    for tensor in [ws.h, ws.d, ws.vz]:
        tensor.zero_()
    inputs = (A, ws.cs, ws.h, ws.d, ws.vz, ws.south, ws.east0, ws.upper, ws.north, ws.west, args.rank)
    _C.sym_descriptor(*inputs)
    torch.cuda.synchronize()
    samples = [_C.sym_descriptor_profile(*inputs) for _ in range(args.repeats)]
    payload = {'rank': args.rank, 'environment': metadata(), 'samples_ms': samples, 'median_ms': dict(zip(['descriptor', 'scan_north', 'scan_west', 'apply'], [statistics.median((row[i] for row in samples)) for i in range(4)]))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
if __name__ == '__main__':
    main()
