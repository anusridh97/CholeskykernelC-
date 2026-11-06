"""Validate all paper MLX cells; optionally collect new matched-output timings.

Default is correctness only. Nothing imports MLX or executes numerical work on
--help. Run in an environment with NumPy, Torch and MLX installed. This is a new
protocol, not retroactive validation of the May 2026 timing records.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone


RANKS = [64, 128, 256, 512, 768, 1024]
BATCHES = [1, 4, 16]
TOL = 5e-5


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, obj):
    pending = path.with_suffix(path.suffix + '.tmp')
    pending.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True)
    except OSError as exc:
        return {'command': args, 'returncode': -1, 'stdout': '', 'stderr': str(exc)}
    return {'command': args, 'returncode': result.returncode,
            'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}


def whiten(np, L, M):
    left = np.linalg.solve(L, M)
    return np.linalg.solve(L, left.swapaxes(-1, -2)).swapaxes(-1, -2)


def problem(np, rank, batch, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((batch, rank, rank), dtype=np.float32)
    G = A @ A.swapaxes(-1, -2) + rank * np.eye(rank, dtype=np.float32)
    L = np.linalg.cholesky(G.astype(np.float64)).astype(np.float32)
    M = rng.standard_normal((batch, rank, rank), dtype=np.float32)
    Aw = whiten(np, L.astype(np.float64), M.astype(np.float64)).astype(np.float32)
    z = rng.standard_normal((batch, rank), dtype=np.float32) * np.float32(.1)
    w = rng.standard_normal((batch, rank), dtype=np.float32) * np.float32(.1)
    return dict(G=G, L=L, M=M, Aw=Aw, z=z, w=w)


def errors(np, actual, expected):
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape:
        return {'passed': False, 'shape': list(a.shape), 'expected_shape': list(b.shape)}
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:
        return {'passed': False, 'finite': False}
    axes = tuple(range(1, a.ndim))
    num = np.sqrt(np.sum((a - b) ** 2, axis=axes))
    den = np.sqrt(np.sum(b ** 2, axis=axes))
    rel = num / np.maximum(den, np.finfo(np.float64).tiny)
    return {'passed': bool(np.all(rel <= TOL)), 'finite': True,
            'per_item_relative_frobenius': rel.tolist(),
            'max_relative_frobenius': float(rel.max()),
            'max_absolute_error': float(np.max(np.abs(a - b)))}


def make_reference(np, p):
    g = p['G'].astype(np.float64)
    m = p['M'].astype(np.float64)
    z = p['z'].astype(np.float64)
    w = p['w'].astype(np.float64)
    gp = g + z[..., :, None] * z[..., None, :]
    mp = m + z[..., :, None] * w[..., None, :]
    lp = np.linalg.cholesky(gp)
    return gp, mp, lp, whiten(np, lp, mp), np.linalg.solve(lp, z[..., None])[..., 0]


def validate(np, outputs, reference):
    gp, mp, expected_l, expected_aw, expected_vz = reference
    reports = {}
    converted = {}
    for name, out in outputs.items():
        L, Aw = [np.asarray(x, dtype=np.float64) for x in out[:2]]
        converted[name] = (L, Aw)
        checks = {'factor_vs_fp64': errors(np, L, expected_l),
                  'operator_vs_fp64': errors(np, Aw, expected_aw),
                  'factor_identity': errors(np, L @ L.swapaxes(-1, -2), gp),
                  'operator_identity': errors(np, L @ Aw @ L.swapaxes(-1, -2), mp)}
        checks['lower_positive_factor'] = {'passed': bool(
            np.all(np.diagonal(L, axis1=-2, axis2=-1) > 0) and
            np.all(np.triu(L, k=1) == 0))}
        if len(out) == 3:
            checks['auxiliary_vz'] = errors(np, out[2], expected_vz)
        reports[name] = {'passed': all(x['passed'] for x in checks.values()), 'checks': checks}
    mlx_l, mlx_aw = converted['mlx_batched_update']
    paired = {name: {'factor': errors(np, mlx_l, pair[0]),
                     'operator': errors(np, mlx_aw, pair[1])}
              for name, pair in converted.items() if name != 'mlx_batched_update'}
    passed = all(x['passed'] for x in reports.values()) and all(
        x['passed'] for pair in paired.values() for x in pair.values())
    return {'passed': passed, 'methods': reports, 'cross_method': paired}


def one_cell(np, torch, mx, kernel, args, rank, batch):
    seed = args.seed + 1009 * rank + batch
    p = problem(np, rank, batch, seed)
    L64, Aw64, M64, G64 = [p[k].astype(np.float64) for k in ('L', 'Aw', 'M', 'G')]
    initial = {'factor': errors(np, L64 @ L64.swapaxes(-1, -2), G64),
               'operator': errors(np, L64 @ Aw64 @ L64.swapaxes(-1, -2), M64)}
    record = {'rank': rank, 'batch': batch, 'seed': seed, 'dtype': 'float32',
              'input_hashes': {k: {'sha256': digest(v.tobytes()), 'shape': list(v.shape),
                                   'dtype': str(v.dtype)} for k, v in p.items()},
              'initial_consistency': initial, 'status': 'running'}
    if not all(x['passed'] for x in initial.values()):
        record['status'] = 'failed_initial_consistency'
        return record
    reference = make_reference(np, p)
    inputs = [mx.array(p[k]) for k in ('L', 'Aw', 'z', 'w')]
    mx.eval(*inputs)
    torch_p = {k: torch.from_numpy(p[k].copy()) for k in ('G', 'M', 'z', 'w')}

    def mlx_call():
        out = kernel.incremental_update_mlx_batched(*inputs)
        mx.eval(*out)
        mx.synchronize()
        return out

    def numpy_call():
        gp = p['G'] + p['z'][..., :, None] * p['z'][..., None, :]
        mp = p['M'] + p['z'][..., :, None] * p['w'][..., None, :]
        lp = np.linalg.cholesky(gp)
        inv = np.linalg.inv(lp)
        return lp, inv @ mp @ inv.swapaxes(-1, -2)

    def torch_call():
        z, w = torch_p['z'], torch_p['w']
        gp = torch_p['G'] + z.unsqueeze(-1) * z.unsqueeze(-2)
        mp = torch_p['M'] + z.unsqueeze(-1) * w.unsqueeze(-2)
        lp = torch.linalg.cholesky(gp)
        inv = torch.linalg.inv(lp)
        return lp, inv @ mp @ inv.transpose(-1, -2)

    calls = {'mlx_batched_update': mlx_call, 'numpy_lapack_refactor': numpy_call,
             'torch_linalg_refactor_cpu': torch_call}
    record['correctness'] = validate(np, {k: f() for k, f in calls.items()}, reference)
    record['status'] = 'passed' if record['correctness']['passed'] else 'failed_correctness'
    if not args.time or record['status'] != 'passed':
        return record
    names = list(calls)
    for i in range(args.warmups):
        for name in names[i % 3:] + names[:i % 3]:
            calls[name]()
    samples = {name: [] for name in names}
    last = {}
    # Balance all six method permutations. GPU synchronization precedes each
    # interval; MLX materialization and completion remain inside its interval.
    import itertools
    orders = list(itertools.permutations(names))
    for i in range(args.repeats):
        for name in orders[i % len(orders)]:
            mx.synchronize()
            t0 = time.perf_counter_ns()
            out = calls[name]()
            samples[name].append((time.perf_counter_ns() - t0) / 1e6)
            last[name] = out
    record['post_timing_correctness'] = validate(np, last, reference)
    if not record['post_timing_correctness']['passed']:
        record['status'] = 'failed_post_timing_correctness'
    record['timing'] = {name: {'samples_ms': values, 'mean_ms': statistics.mean(values),
                              'median_ms': statistics.median(values),
                              'sample_sd_ms': statistics.stdev(values)}
                        for name, values in samples.items()}
    record['timing']['eligible_for_comparison'] = record['status'] == 'passed'
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--ranks', type=int, nargs='+', default=RANKS)
    parser.add_argument('--batches', type=int, nargs='+', default=BATCHES)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--time', action='store_true', help='Collect new timings after validation')
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--warmups', type=int, default=5)
    args = parser.parse_args()
    if (args.threads < 1 or args.repeats < 2 or args.warmups < 1 or
            not set(args.ranks) <= set(RANKS) or not set(args.batches) <= set(BATCHES) or
            len(set(args.ranks)) != len(args.ranks) or len(set(args.batches)) != len(args.batches)):
        parser.error('Use unique paper grid entries, positive threads/warmups, and >=2 repeats')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    thread_vars = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                   'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']
    for key in thread_vars:
        os.environ[key] = str(args.threads)
    # Imports deliberately occur after thread controls and CLI parsing.
    import numpy as np
    import torch
    import mlx.core as mx
    if not mx.metal.is_available():
        raise RuntimeError('A working Metal device is required; CPU fallback is not permitted')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    mx.set_default_device(mx.gpu)
    kernel_path = Path(__file__).resolve().parents[2] / 'src/doublesided_cholesky/mlx.py'
    spec = importlib.util.spec_from_file_location('matched_mlx_snapshot', kernel_path)
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    env = {'python': sys.version, 'platform': platform.platform(),
           'executable': sys.executable, 'command': sys.argv, 'cwd': os.getcwd(),
           'numpy': np.__version__, 'torch': torch.__version__,
           'mlx': importlib.metadata.version('mlx'),
           'numpy_config': config.getvalue(), 'torch_config': torch.__config__.show(),
           'thread_environment': {k: os.environ[k] for k in thread_vars},
           'torch_num_threads': torch.get_num_threads(),
           'torch_num_interop_threads': torch.get_num_interop_threads(),
           'cpu': command(['sysctl', '-n', 'machdep.cpu.brand_string']),
           'memory': command(['sysctl', '-n', 'hw.memsize']),
           'model': command(['sysctl', '-n', 'hw.model']),
           'metal': dict(mx.metal.device_info())}
    # Do not preserve a machine serial number or unrelated environment secrets.
    write_json(args.output_dir / 'environment.json', env)
    source = {}
    for label, path in [('driver', Path(__file__)), ('kernel', kernel_path)]:
        data = path.read_bytes()
        saved = args.output_dir / ('executed_' + path.name)
        saved.write_bytes(data)
        source[label] = {'original_path': str(path.resolve()), 'sha256': digest(data),
                         'snapshot': saved.name}
    write_json(args.output_dir / 'source.json', source)
    manifest = {'protocol': 'mlx-consistent-dense-output-v1',
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'status': 'running', 'seed': args.seed, 'ranks': args.ranks,
                'batches': args.batches, 'tolerance_per_item': TOL,
                'full_paper_grid': set(args.ranks) == set(RANKS) and set(args.batches) == set(BATCHES),
                'timing_requested': args.time, 'repeats': args.repeats, 'warmups': args.warmups,
                'timing_statistic': 'arithmetic mean; sample SD and median also retained',
                'contract': 'Return L+ and Aw+ for G+zzT and M+zwT; auxiliary MLX vz also checked',
                'reference': 'Independent FP64 Cholesky and two solves on common quantized G/M/z/w',
                'timing_excludes': ['construction', 'input transfers', 'correctness', 'compilation warmup'],
                'timing_includes': ['output allocation', 'method computation', 'MLX eval/synchronization'],
                'baseline': 'Original inverse-plus-two-matmuls baseline, CPU NumPy and Torch',
                'historical_runs_validated_retroactively': False, 'cells': []}
    write_json(args.output_dir / 'manifest.json', manifest)
    for rank in args.ranks:
        for batch in args.batches:
            try:
                cell = one_cell(np, torch, mx, kernel, args, rank, batch)
            except Exception as exc:
                cell = {'rank': rank, 'batch': batch, 'status': 'error',
                        'error': repr(exc)}
            filename = f'r{rank}_b{batch}.json'
            write_json(args.output_dir / filename, cell)
            manifest['cells'].append({'rank': rank, 'batch': batch, 'file': filename,
                                      'sha256': digest((args.output_dir / filename).read_bytes()),
                                      'status': cell['status']})
            write_json(args.output_dir / 'manifest.json', manifest)
            print(f'r={rank} batch={batch}: {cell["status"]}', flush=True)
    manifest['status'] = 'passed' if all(c['status'] == 'passed' for c in manifest['cells']) else 'failed'
    manifest['hardware_record_complete'] = bool(env['metal']) and all(
        env[k]['returncode'] == 0 and bool(env[k]['stdout']) for k in ('cpu', 'memory', 'model'))
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(args.output_dir / 'manifest.json', manifest)
    return 0 if manifest['status'] == 'passed' and manifest['hardware_record_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
