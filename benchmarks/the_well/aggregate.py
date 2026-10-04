"""Aggregate per-transition medians, preserving the archived B200 CSV byte format."""
import argparse
import csv
import glob
import io
import json
from pathlib import Path
import statistics
import math
HEADER = ['rank', 'incremental_median_ms', 'refactor_median_ms', 'median_transition_speedup', 'min_transition_speedup', 'max_transition_speedup', 'all_correctness_passed', 'max_cross_operator_relative_error', 'max_cross_factor_relative_error']

def aggregate_csv(ladder_dir: str) -> bytes:
    """Rebuild the archived CSV, including default csv.writer CRLF line endings."""
    buf = io.StringIO(newline='')
    writer = csv.writer(buf)
    writer.writerow(HEADER)
    paths = sorted(glob.glob(f'{ladder_dir}/ladder_r*.json'), key=lambda p: int(p.rsplit('_r', 1)[1][:-5]))
    for path in paths:
        with open(path) as handle:
            data = json.load(handle)
        cases = data['cases']
        inv = [c['methods']['inverse']['event_seconds_median'] * 1000.0 for c in cases]
        fresh = [c['methods']['fresh_refact']['event_seconds_median'] * 1000.0 for c in cases]
        speedups = [f / i for i, f in zip(inv, fresh)]
        passed = all((c['methods'][m]['validation']['passed'] for c in cases for m in ('inverse', 'fresh_refact'))) and all((c['cross_method_probe_comparison']['passed'] for c in cases))
        writer.writerow([data['rank'], statistics.median(inv), statistics.median(fresh), statistics.median(speedups), min(speedups), max(speedups), passed, max((c['cross_method_probe_comparison']['operator']['max_relative_error'] for c in cases)), max((c['cross_method_probe_comparison']['factor']['max_relative_error'] for c in cases))])
    return buf.getvalue().encode()

def aggregate_json(ladder_dir: str) -> dict:
    """Add mean, sample SD and Student-t interval for ten transition speedups."""
    flat = {int(row['rank']): row for row in csv.DictReader(io.StringIO(aggregate_csv(ladder_dir).decode()))}
    rows = []
    for path in sorted(Path(ladder_dir).glob('ladder_r*.json'), key=lambda p: int(p.stem.split('_r')[1])):
        data = json.loads(path.read_text())
        cases = data['cases']
        if len(cases) != 10:
            raise ValueError('frozen protocol requires ten transitions for t(9) intervals')
        speedups = [c['methods']['fresh_refact']['event_seconds_median'] / c['methods']['inverse']['event_seconds_median'] for c in cases]
        mean = statistics.mean(speedups)
        sd = statistics.stdev(speedups)
        margin = 2.2621571627 * sd / math.sqrt(10)
        row = {key: int(value) if key == 'rank' else value == 'True' if key == 'all_correctness_passed' else float(value) for key, value in flat[data['rank']].items()}
        row.update(mean_transition_speedup=mean, sample_sd_transition_speedup=sd, ci95_transition_speedup=[mean - margin, mean + margin])
        for metric in ['operator', 'factor']:
            errors = []
            for case in cases:
                validation = case['methods']['inverse']['validation']
                errors.append(validation['probe'][metric]['max_relative_error'])
                if validation.get('full_dense'):
                    errors.append(validation['full_dense'][metric + '_relative_error'])
            row['max_incremental_' + metric + '_relative_error'] = max(errors)
        rows.append(row)
    return {'rows': rows, 'statistics': 'ratios formed per transition before aggregation; t(9), n=10'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ladder_dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not list(args.ladder_dir.glob('ladder_r*.json')):
        raise ValueError('no ladder JSONs found')
    output = args.output_dir or args.ladder_dir
    output.mkdir(parents=True, exist_ok=True)
    payload = aggregate_json(str(args.ladder_dir))
    (output / 'aggregate.csv').write_bytes(aggregate_csv(str(args.ladder_dir)))
    (output / 'aggregate.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
if __name__ == '__main__':
    main()
