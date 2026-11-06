"""Regression check for the three batch-16 cells that exposed the Metal race."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '1'

import numpy as np
import mlx.core as mx
import validate_comparison as validation
import kernel_snapshot as kernel

mx.set_default_device(mx.gpu)
ROOT = Path(__file__).resolve().parent
output = ROOT.parent / 'mlx-fixed-repeat-checks.json'
if output.exists():
    raise SystemExit('Refusing to overwrite existing evidence')
record = {'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
          'device': dict(mx.metal.device_info()), 'purpose': 'same-input race regression',
          'tolerance': validation.TOL, 'repeats_per_cell': 20,
          'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [Path(__file__), ROOT/'kernel_snapshot.py', ROOT/'validate_comparison.py']},
          'cells': []}
for rank in [512, 768, 1024]:
    seed = 1009 * rank + 16
    p = validation.problem(np, rank, 16, seed)
    reference = validation.make_reference(np, p)[2:]
    inputs = [mx.array(p[k]) for k in ['L', 'Aw', 'z', 'w']]
    mx.eval(*inputs)
    cell = {'rank': rank, 'batch': 16, 'seed': seed, 'calls': []}
    for repeat in range(20):
        out = kernel.incremental_update_mlx_batched(*inputs)
        mx.eval(*out)
        mx.synchronize()
        checks = {name: validation.errors(np, actual, expected)
                  for name, actual, expected in zip(['L', 'Aw', 'vz'], out, reference)}
        cell['calls'].append({'repeat': repeat, 'checks': checks,
                              'passed': all(c['passed'] for c in checks.values())})
    cell['passed'] = all(c['passed'] for c in cell['calls'])
    record['cells'].append(cell)
    validation.write_json(output, record)
    print(f'r={rank}, batch=16: {sum(c["passed"] for c in cell["calls"])}/20 calls passed', flush=True)
record['status'] = 'passed' if all(c['passed'] for c in record['cells']) else 'failed'
record['finished_utc'] = datetime.now(timezone.utc).isoformat()
validation.write_json(output, record)
raise SystemExit(0 if record['status'] == 'passed' else 1)
