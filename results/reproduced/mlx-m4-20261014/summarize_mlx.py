"""Generate a summary and standalone heatmaps only from a complete passing run."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
run = ROOT / 'mlx-matched-fixed'
manifest = json.loads((run/'manifest.json').read_text())
assert manifest['status'] == 'passed' and manifest['full_paper_grid']
assert manifest['hardware_record_complete'] and manifest['timing_requested']
assert len(manifest['cells']) == 18
environment = json.loads((run/'environment.json').read_text())
summary = {'hardware': environment['metal']['device_name'],
           'memory_bytes': int(environment['memory']['stdout']),
           'library_versions': {k: environment[k] for k in ['numpy','torch','mlx']},
           'source': json.loads((run/'source.json').read_text()),
           'statistic': 'ratio of arithmetic mean latencies; not mean of paired ratios',
           'cpu_thread_limit_requested': 1, 'repeats': manifest['repeats'],
           'warmups': manifest['warmups'], 'cells': [], 'maximum_errors': {}}
rows = ['rank batch  MLX_mean_ms  NumPy_mean_ms  Torch_mean_ms  NumPy/MLX  Torch/MLX']
for c in manifest['cells']:
    path = run/c['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == c['sha256']
    d = json.loads(path.read_text())
    original = json.loads((ROOT/'mlx-matched'/c['file']).read_text())
    assert d['input_hashes'] == original['input_hashes']
    assert d['status'] == 'passed' and d['timing']['eligible_for_comparison']
    latency = {k:v for k,v in d['timing'].items() if isinstance(v,dict)}
    for v in latency.values():
        assert len(v['samples_ms']) == manifest['repeats']
    means = {k:v['mean_ms'] for k,v in latency.items()}
    mlx = means['mlx_batched_update']
    np_speed = means['numpy_lapack_refactor'] / mlx
    th_speed = means['torch_linalg_refactor_cpu'] / mlx
    summary['cells'].append({'rank': d['rank'], 'batch': d['batch'],
        'mean_ms': means, 'numpy_over_mlx': np_speed, 'torch_over_mlx': th_speed,
        'raw_file': str(path.relative_to(ROOT))})
    rows.append(f"{d['rank']:4} {d['batch']:5} {mlx:12.5f} {means['numpy_lapack_refactor']:14.5f} {means['torch_linalg_refactor_cpu']:14.5f} {np_speed:10.3f} {th_speed:10.3f}")
    for phase in ['correctness','post_timing_correctness']:
        assert d[phase]['passed']
        for method, result in d[phase]['methods'].items():
            for check, metric in result['checks'].items():
                assert metric['passed']
                if 'max_relative_frobenius' in metric:
                    key = method + '/' + check
                    summary['maximum_errors'][key] = max(
                        summary['maximum_errors'].get(key,0),metric['max_relative_frobenius'])
(ROOT/'mlx_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(ROOT/'mlx_timing_table.txt').write_text('\n'.join(rows)+'\n')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ranks, batches = [64,128,256,512,768,1024], [1,4,16]
values = []
for key in ['numpy_over_mlx','torch_over_mlx']:
    grid = np.array([[next(c[key] for c in summary['cells'] if c['rank']==r and c['batch']==b)
                      for b in batches] for r in ranks])
    values.append(grid)
logs = [np.log2(v) for v in values]
norm = TwoSlopeNorm(vmin=min(-.2,min(v.min() for v in logs)),vcenter=0,
                    vmax=max(.2,max(v.max() for v in logs)))
fig, axes = plt.subplots(1,2,figsize=(9,7))
fig.subplots_adjust(left=.09,right=.84,bottom=.24,top=.82,wspace=.32)
for ax, grid, lg, label in zip(axes, values, logs, ['NumPy / MLX','PyTorch CPU / MLX']):
    im = ax.imshow(lg,cmap='RdBu',norm=norm,aspect='auto')
    ax.set_xticks(range(3),batches); ax.set_yticks(range(6),ranks)
    ax.set_xlabel('Batch size'); ax.set_ylabel('Rank'); ax.set_title(label)
    for i in range(6):
        for j in range(3):
            scaled = norm(lg[i,j])
            ax.text(j,i,f'{grid[i,j]:.2f}×',ha='center',va='center',fontsize=11,
                    color='white' if scaled<.18 or scaled>.82 else 'black')
cb = fig.colorbar(im,cax=fig.add_axes([.88,.24,.025,.51]))
ticks = [n for n in range(-6,13) if norm.vmin <= n <= norm.vmax]
cb.set_ticks(ticks,labels=[f'{2**n:g}×' for n in ticks]); cb.set_label('CPU baseline time / MLX time')
fig.suptitle('Matched-output Cholesky update on Apple M4\nCorrected synchronization • all 18 cells passed',fontsize=15,y=.95)
fig.text(.5,.055,'FP32 • mean of 100 samples • 5 warmups • CPU thread limit requested: 1\n24 GiB • MLX 0.29.3 • NumPy 2.0.2 • PyTorch 2.8.0\nNew measurements; not a reproduction of the historical M4 Pro heatmaps.',ha='center',fontsize=10)
fig.savefig(ROOT/'mlx_speedups.png',dpi=180)
fig.savefig(ROOT/'mlx_speedups.svg')
print('\n'.join(rows))
print('Largest MLX errors:',{k:v for k,v in summary['maximum_errors'].items() if k.startswith('mlx_')})
