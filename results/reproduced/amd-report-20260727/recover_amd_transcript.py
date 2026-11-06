"""Recover supplied historical numbers; performs no experiments or network I/O."""
from pathlib import Path
import hashlib
import json
import re
import statistics

ROOT = Path(__file__).resolve().parent
source = ROOT / 'amd_report_excerpts.txt'
text = source.read_text()
pattern = r'rank (\d+)\s+original:\s*(\[[^\]]+\])\s+optimized:\s*(\[[^\]]+\])'
rows = []
for rank, original, optimized in re.findall(pattern, text):
    old, new = json.loads(original), json.loads(optimized)
    a, b = statistics.median(old), statistics.median(new)
    rows.append(dict(rank=int(rank), original_ms_samples=old, optimized_ms_samples=new,
                     original_median_ms=a, optimized_median_ms=b,
                     median_ratio_speedup=a / b, sample_count=len(old)))
assert [r['rank'] for r in rows] == [2048, 4096, 8192, 16384, 24576, 28672]
assert round(rows[-1]['optimized_median_ms'], 3) == 624.674
assert round(rows[-1]['median_ratio_speedup'], 2) == 10.99

record = {
    'provenance': {
        'kind': 'user_supplied_historical_chat_transcript',
        'received_date': '2026-10-14', 'source_file': source.name,
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'unabridged_supplied_transcript_sha256': 'b8a98d49e3cd2870297e232c243f468af1e7cfc15f073877de4f4aef4119ed9d',
        'original_archive_recovered': False, 'source_code_recovered': False,
        'source_sha256_values_recovered': False, 'experiment_rerun': False,
        'note': 'Samples and settings were reported in the supplied chat. Original JSON/source/IR remain missing.'},
    'hardware_software_reported': {
        'gpu': 'AMD Radeon Pro V520 / Navi12', 'target': 'gfx1011:xnack-',
        'vram_bytes': 8422162432, 'vram_gib': 8422162432 / 2**30,
        'compute_units_rocminfo': 36, 'multiprocessors_torch': 18,
        'wavefront_lanes': 32, 'l2_bytes': 4 * 2**20,
        'os': 'Ubuntu 26.04 LTS Resolute', 'kernel': '7.0.0-1006-aws',
        'host_rocm': '7.1', 'host_hip_runtime': '7.1.52801-9999',
        'rocm_llvm': '21.1.8', 'python': '3.12.13',
        'torch': '2.6.0+rocm6.2.4', 'torch_version_hip': '6.2.41134-65d174c3e',
        'triton': '3.2.0', 'host_ram_gib_reported': 61, 'swap': 'none',
        'git_commit': None, 'git_metadata_present': False},
    'measurement_reported': {
        'region': 'incremental Linv/P and Aw update only; excludes G/M outer products',
        'timer': 'time.perf_counter with torch.cuda.synchronize around launches',
        'reason_host_timer': 'GPU event elapsed_time returned exactly zero',
        'statistic': 'median of saved samples; speedup is ratio of method medians',
        'baseline': 'original Triton transport with pure-Triton setup replacing unavailable ATen matmul/cumsum',
        'fresh_refactor_speedup': None, 'warmup_count': None,
        'unknown_settings': ['exact driver/source hashes', 'warmup count', 'complete input generation/seed and restoration protocol']},
    'timing': rows,
    'correctness_reported': {
        'tests_passed': 4, 'tests_total': 4, 'distinct_z_w': True,
        'one_step': [
            {'rank': 129, 'P_relative_frobenius': 5.1413e-8, 'Aw_relative_frobenius': 6.7946e-8,
             'vz_relative_l2': 7.5952e-8, 'factor_residual': 1.0469e-7, 'operator_residual': 8.3709e-8},
            {'rank': 257, 'P_relative_frobenius': 4.7106e-8, 'Aw_relative_frobenius': 6.6822e-8,
             'vz_relative_l2': 1.0383e-7, 'factor_residual': 9.5351e-8, 'operator_residual': 7.8525e-8},
            {'rank': 1025, 'P_relative_frobenius': 4.8762e-8, 'Aw_relative_frobenius': 6.7973e-8,
             'vz_relative_l2': 1.1062e-7, 'factor_residual': 9.9157e-8, 'operator_residual': 8.1300e-8}],
        'trajectory': {'rank': 129, 'steps': 100, 'P_difference': 5.3774e-7,
                       'Aw_difference': 1.0520e-6, 'factor_residual': 1.0667e-6,
                       'operator_residual': 5.2013e-7, 'vz_difference': 5.0497e-7,
                       'all_states_finite': True},
        'does_not_establish_per_rank_correctness_for_timing_ladder': True},
    'fresh_baseline_reported': {
        'status': 'blocked; no valid GPU times', 'cpu_sequence_relative_error': 1.79e-16,
        'cpu_unit_tests_passed': 2, 'gpu_probe_ranks': [129, 2048, 4096, 8192],
        'gpu_failure': 'HIP invalid device function; rocBLAS lacks gfx1011',
        'cpu_fallback_used_for_gpu_timings': False, 'source_build_performed': False},
    'implementation_reported': {
        'general_asymmetric': True, 'pure_triton': True, 'Aw_tile': 64,
        'P_tile_columns': 16, 'P_chunk': 32, 'matrix_address_bits': 64,
        'packed_symmetric_implemented': False, 'hardware_profiler_counters_collected': False},
    'original_paths_reported_relative_to_remote_repository': [
        'rebuttal_results/amd_triton/final_20260727T054333Z/report.md',
        'rebuttal_results/amd_triton/final_20260727T054333Z/aggregate.json',
        'rebuttal_results/amd_triton/final_20260727T054333Z/aggregate.csv',
        'rebuttal_results/amd_triton/final_20260727T054333Z/environment.json',
        'rebuttal_results/amd_triton/final_20260727T054333Z/correctness.json',
        'rebuttal_results/amd_triton/final_20260727T054333Z/ablations.json',
        'student_pkg/doublesided_cholesky_triton_amd.py',
        'student_pkg/tests/test_triton_amd_correctness.py',
        'rebuttal_results/amd_triton/ir_snapshot_20260727T054000Z',
        'student_pkg/tests/bench_pytorch_refactor_amd.py',
        'student_pkg/tests/test_pytorch_refactor_baseline.py',
        'rebuttal_results/amd_triton/pytorch_refactor_20260727T062506Z/aggregate.json']}
(ROOT / 'amd_transcript_evidence.json').write_text(json.dumps(record, indent=2) + '\n')
for r in rows:
    print(f"r={r['rank']}: {r['original_median_ms']:.6f} / {r['optimized_median_ms']:.6f} = {r['median_ratio_speedup']:.6f}x, n={r['sample_count']}")
