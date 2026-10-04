"""The Well rank ladder with frozen numerical protocol and portable execution."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import time
import warnings
from typing import Any, Callable, Sequence
import numpy as np
import torch
from benchmarks._env import metadata
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = Path(__file__).with_name('protocol.json')
DEFAULT_TRACE = ROOT / 'data/the_well/processed/post_neutron_star_merger_scenario_4_dct65536.npy'
SCHEMA_VERSION = 'cuda_blackhole_rank_scaling_v1'
PROTOCOL_VERSION = 'blackhole_s4_ladder_v1'
DATASET_REPOSITORY = 'polymathic-ai/post_neutron_star_merger'
DATASET_REVISION = '1fd582f3e29749ac4d0c14bc816bddb2394258cc'
DATASET_RELATIVE_PATH = 'data/train/post_neutron_star_merger_scenario_4.hdf5'
DATASET_BYTES = 14109638656
DATASET_SHA256 = '21e587c90c006c4b7d37e2b04899bd7e928967ae506e9ceb5afd05dfca6909fd'
N_SNAPSHOTS = 181
MAX_RANK = 65536
RANKS = (2048, 4096, 8192, 16384, 32768, 49152, 65536)
TILES = {2048: 8, 4096: 8, 8192: 8, 16384: 16, 32768: 32, 49152: 32, 65536: 32}
REPEATS = {2048: 10, 4096: 10, 8192: 10, 16384: 5, 32768: 5, 49152: 3, 65536: 3}
INITIAL_TRANSITIONS = tuple(range(64))
WARMUP_TRANSITION = 64
EVALUATION_TRANSITIONS = (80, 91, 102, 113, 124, 135, 146, 157, 168, 179)
RIDGE = 1.0
SETUP_NUM_WARPS = 4
SWEEP_NUM_WARPS = 1
CORRECTNESS_MAX_RANK = 32768
PROBE_COUNT = 16
PROBE_SEED = 29421
CORRECTNESS_THRESHOLD = 0.001
CHANNEL_ORDER = ('density', 'internal_energy', 'pressure', 'temperature', 'electron_fraction', 'entropy', 'magnetic_field/log_r', 'magnetic_field/theta', 'magnetic_field/phi', 'velocity/log_r', 'velocity/theta', 'velocity/phi')
TIMING_REGION = {'includes': ['G += z z^T', 'M += z w^T', 'method core'], 'excludes': ['feature I/O', 'host-to-device feature transfer', 'common-state construction', 'state restoration', 'CUDA synchronization before timing', 'correctness checks'], 'cuda_event_unit': 'seconds', 'wall_unit': 'seconds'}

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            block = handle.read(16 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()

def _require_equal(field: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f'{field}: expected {expected!r}, got {actual!r}')

def _validate_trace(trace_path: Path, manifest: dict[str, Any]) -> tuple[np.memmap, dict[str, float]]:
    trace = np.load(trace_path, mmap_mode='r', allow_pickle=False)
    if not isinstance(trace, np.memmap):
        raise ValueError('feature trace must load as a memory-mapped .npy array')
    _require_equal('trace.shape', list(trace.shape), [N_SNAPSHOTS, MAX_RANK])
    _require_equal('trace.dtype', str(trace.dtype), 'float32')
    if not trace.flags.c_contiguous or trace.flags.f_contiguous:
        raise ValueError('feature trace must be exclusively C-contiguous')
    if not bool(np.isfinite(trace).all()):
        raise ValueError('feature trace contains a non-finite value')
    calibration = np.asarray(trace[:64, :], dtype=np.float64)
    cumulative_energy = np.cumsum(calibration * calibration, axis=1, dtype=np.float64)
    recorded_scales = manifest['rank_scaling']['scales']
    verified_scales: dict[str, float] = {}
    for rank in RANKS:
        recomputed = math.sqrt(float(np.mean(cumulative_energy[:, rank - 1], dtype=np.float64)))
        recorded = float(recorded_scales[str(rank)])
        if not math.isclose(recomputed, recorded, rel_tol=1e-13, abs_tol=0.0):
            raise ValueError(f'rank {rank} scale mismatch: manifest={recorded:.17g}, trace={recomputed:.17g}')
        verified_scales[str(rank)] = recorded
    return (trace, verified_scales)

def _features_for_rank(trace: np.memmap, rank: int, scale: float) -> tuple[torch.Tensor, dict[int, int]]:
    needed_snapshots = tuple(sorted(set(range(66)) | set(EVALUATION_TRANSITIONS) | {transition + 1 for transition in EVALUATION_TRANSITIONS}))
    values = np.array(trace[np.asarray(needed_snapshots), :rank], dtype=np.float32, order='C', copy=True)
    values /= np.float32(scale)
    if not bool(np.isfinite(values).all()):
        raise ValueError('scaled feature vectors contain a non-finite value')
    indices = {snapshot: offset for offset, snapshot in enumerate(needed_snapshots)}
    return (torch.from_numpy(values).to(device='cuda'), indices)

def _condition_from_rows(rows: np.ndarray) -> dict[str, float]:
    if rows.ndim != 2 or rows.shape[1] <= rows.shape[0]:
        raise ValueError('small-Gram condition calculation requires width > rows')
    gram = rows.astype(np.float64, copy=False) @ rows.astype(np.float64, copy=False).T
    eigenvalues = np.linalg.eigvalsh(gram)
    if not bool(np.isfinite(eigenvalues).all()):
        raise ValueError('small Gram eigenspectrum is non-finite')
    lambda_min = RIDGE
    lambda_max = RIDGE + max(0.0, float(eigenvalues[-1]))
    return {'small_gram_rows': int(rows.shape[0]), 'small_gram_columns': int(rows.shape[1]), 'lambda_min_exact': lambda_min, 'lambda_max_from_small_gram': lambda_max, 'condition_number_exact': lambda_max / lambda_min, 'small_gram_min_eigenvalue': float(eigenvalues[0]), 'small_gram_max_eigenvalue': float(eigenvalues[-1])}

def _finite_full(tensor: torch.Tensor, chunk_elements: int=1 << 26) -> bool:
    rows = tensor.shape[-2]
    elements_per_row = tensor.numel() // rows
    rows_per_chunk = max(1, chunk_elements // elements_per_row)
    for start in range(0, rows, rows_per_chunk):
        chunk = torch.isfinite(tensor[start:start + rows_per_chunk])
        if not bool(chunk.all().item()):
            return False
    return True

def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(expected).clamp_min(torch.finfo(expected.dtype).tiny)
    return float((torch.linalg.vector_norm(actual - expected) / denominator).item())

def _column_errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    denominator = torch.linalg.vector_norm(expected, dim=0).clamp_min(torch.finfo(expected.dtype).tiny)
    errors_tensor = torch.linalg.vector_norm(actual - expected, dim=0) / denominator
    errors = [float(value) for value in errors_tensor.cpu().tolist()]
    frobenius = _relative_error(actual, expected)
    return {'per_probe_relative_errors': errors, 'max_relative_error': max(errors), 'median_relative_error': statistics.median(errors), 'frobenius_relative_error': frobenius}

def _make_probes(rank: int) -> torch.Tensor:
    generator = np.random.Generator(np.random.PCG64(PROBE_SEED))
    values = generator.integers(0, 2, size=(rank, PROBE_COUNT), dtype=np.int8)
    probes = values.astype(np.float32)
    probes *= 2.0
    probes -= 1.0
    probes /= math.sqrt(rank)
    return torch.from_numpy(probes).to(device='cuda')

class BenchmarkState:
    """Four-matrix GPU state with layout views for both compared methods."""

    def __init__(self, *, rank: int, initial_z: torch.Tensor, initial_w: torch.Tensor) -> None:
        self.rank = rank
        self.initial_z = initial_z
        self.initial_w = initial_w
        self.G = torch.empty((rank, rank), device='cuda', dtype=torch.float32)
        self.M = torch.empty_like(self.G)
        self.factor_storage = torch.empty(rank * rank, device='cuda', dtype=torch.float32)
        self.Linv = self.factor_storage.view(rank, rank)
        self.L = torch.as_strided(self.factor_storage, (rank, rank), (1, rank))
        self.Aw = torch.empty_like(self.G)
        self.info = torch.empty((), device='cuda', dtype=torch.int32)
        self.Linv_initial_cpu: torch.Tensor | None = None
        self.Aw_initial_cpu: torch.Tensor | None = None

    def restore_G_M(self) -> None:
        torch.mm(self.initial_z.mT, self.initial_z, out=self.G)
        self.G.diagonal().add_(RIDGE)
        torch.mm(self.initial_z.mT, self.initial_w, out=self.M)

    def construct_common_state(self) -> dict[str, Any]:
        started = time.perf_counter()
        self.restore_G_M()
        self.info.zero_()
        torch.linalg.cholesky_ex(self.G, check_errors=False, out=(self.L, self.info))
        torch.cuda.synchronize()
        info_value = int(self.info.item())
        if info_value != 0:
            raise RuntimeError(f'common-state Cholesky failed with info={info_value}')
        torch.linalg.solve_triangular(self.L, self.M, upper=False, left=True, out=self.Aw)
        torch.linalg.solve_triangular(self.L.mT, self.Aw, upper=True, left=False, out=self.Aw)
        torch.cuda.synchronize()
        if not _finite_full(self.Aw):
            raise RuntimeError('common whitened operator is non-finite')
        self.Aw_initial_cpu = self.Aw.cpu()
        self.M.zero_()
        self.M.diagonal().fill_(1.0)
        torch.linalg.solve_triangular(self.L, self.M, upper=False, left=True, out=self.Aw)
        self.Aw.tril_()
        torch.cuda.synchronize()
        if not _finite_full(self.Aw):
            raise RuntimeError('common inverse factor is non-finite')
        self.Linv_initial_cpu = self.Aw.cpu()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return {'construction_wall_seconds': elapsed, 'inverse_factor_host_backup_bytes': self.Linv_initial_cpu.numel() * self.Linv_initial_cpu.element_size(), 'operator_host_backup_bytes': self.Aw_initial_cpu.numel() * self.Aw_initial_cpu.element_size(), 'factor_storage_stride': list(self.Linv.stride()), 'cholesky_output_stride': list(self.L.stride()), 'operator_stride': list(self.Aw.stride())}

    def prepare(self, method: str) -> float:
        started = time.perf_counter()
        self.restore_G_M()
        if method == 'inverse':
            if self.Linv_initial_cpu is None or self.Aw_initial_cpu is None:
                raise RuntimeError('common inverse state has not been constructed')
            self.Linv.copy_(self.Linv_initial_cpu)
            self.Aw.copy_(self.Aw_initial_cpu)
        elif method == 'fresh_refact':
            self.info.zero_()
        else:
            raise ValueError(f'unknown method {method!r}')
        torch.cuda.synchronize()
        return time.perf_counter() - started

def _update_inverse(state: BenchmarkState, z: torch.Tensor, w: torch.Tensor, workspace: Any, update_function: Callable[..., Any], tile: int) -> None:
    torch.addr(state.G, z, z, out=state.G)
    torch.addr(state.M, z, w, out=state.M)
    update_function(state.Linv, state.Aw, z, w, tile_rows=tile, tile_cols=tile, setup_num_warps=SETUP_NUM_WARPS, sweep_num_warps=SWEEP_NUM_WARPS, workspace=workspace)

def _update_refact(state: BenchmarkState, z: torch.Tensor, w: torch.Tensor) -> None:
    torch.addr(state.G, z, z, out=state.G)
    torch.addr(state.M, z, w, out=state.M)
    torch.linalg.cholesky_ex(state.G, check_errors=False, out=(state.L, state.info))
    torch.linalg.solve_triangular(state.L, state.M, upper=False, left=True, out=state.Aw)
    torch.linalg.solve_triangular(state.L.mT, state.Aw, upper=True, left=False, out=state.Aw)

def _run_untimed_update(*, method: str, state: BenchmarkState, z: torch.Tensor, w: torch.Tensor, workspace: Any, update_function: Callable[..., Any], tile: int) -> float:
    state.prepare(method)
    started = time.perf_counter()
    if method == 'inverse':
        _update_inverse(state, z, w, workspace, update_function, tile)
    else:
        _update_refact(state, z, w)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if method == 'fresh_refact' and int(state.info.item()) != 0:
        raise RuntimeError(f'fresh Cholesky failed with info={int(state.info.item())}')
    return elapsed

def _time_method(*, method: str, state: BenchmarkState, z: torch.Tensor, w: torch.Tensor, workspace: Any, update_function: Callable[..., Any], tile: int, repeats: int) -> dict[str, Any]:
    event_samples: list[float] = []
    wall_samples: list[float] = []
    restoration_samples: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        restoration_samples.append(state.prepare(method))
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_started = time.perf_counter()
        start_event.record()
        if method == 'inverse':
            _update_inverse(state, z, w, workspace, update_function, tile)
        else:
            _update_refact(state, z, w)
        end_event.record()
        torch.cuda.synchronize()
        wall_samples.append(time.perf_counter() - wall_started)
        event_samples.append(start_event.elapsed_time(end_event) / 1000.0)
        if method == 'fresh_refact' and int(state.info.item()) != 0:
            raise RuntimeError(f'fresh Cholesky failed with info={int(state.info.item())}')
    return {'mode': method, 'repeats': repeats, 'event_seconds_samples': event_samples, 'wall_seconds_samples': wall_samples, 'state_restoration_seconds_samples': restoration_samples, 'event_seconds_median': statistics.median(event_samples), 'wall_seconds_median': statistics.median(wall_samples), 'state_restoration_seconds_median': statistics.median(restoration_samples), 'updates_per_second_from_event_median': 1.0 / statistics.median(event_samples), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'includes_G_M_outer_updates': True, 'state_restoration_timed': False}

def _probe_validation(*, method: str, state: BenchmarkState, probes: torch.Tensor) -> dict[str, Any]:
    if method == 'inverse':
        transformed = state.Linv.mT @ probes
        expected_factor = probes
        actual_factor = state.Linv @ (state.G @ transformed)
        expected_operator = state.Linv @ (state.M @ transformed)
        actual_operator = state.Aw @ probes
        factor_identity = 'Linv G Linv^T Q = Q'
        operator_identity = 'Aw Q = Linv M Linv^T Q'
    else:
        transformed = state.L.mT @ probes
        expected_factor = state.G @ probes
        actual_factor = state.L @ transformed
        expected_operator = state.M @ probes
        actual_operator = state.L @ (state.Aw @ transformed)
        factor_identity = 'L L^T Q = G Q'
        operator_identity = 'L Aw L^T Q = M Q'
    factor = _column_errors(actual_factor, expected_factor)
    operator = _column_errors(actual_operator, expected_operator)
    return {'seed': PROBE_SEED, 'count': PROBE_COUNT, 'distribution': 'NumPy PCG64 Rademacher (+/-1/sqrt(rank)), one fixed rank-by-16 matrix reused for every transition and method', 'factor_identity': factor_identity, 'operator_identity': operator_identity, 'factor': factor, 'operator': operator}

def _full_dense_validation(*, method: str, state: BenchmarkState) -> dict[str, Any]:
    if method == 'inverse':
        L_exact = torch.linalg.cholesky(state.G)
        identity = torch.eye(state.rank, device='cuda', dtype=torch.float32)
        Linv_exact = torch.linalg.solve_triangular(L_exact, identity, upper=False)
        del identity
        left = torch.linalg.solve_triangular(L_exact, state.M, upper=False, left=True)
        Aw_exact = torch.linalg.solve_triangular(L_exact.mT, left, upper=True, left=False)
        payload = {'kind': 'exact_fresh_factor_and_whiten', 'factor_metric': 'relative Frobenius error in Linv', 'factor_relative_error': _relative_error(state.Linv, Linv_exact), 'operator_metric': 'relative Frobenius error in Aw', 'operator_relative_error': _relative_error(state.Aw, Aw_exact)}
        del L_exact, Linv_exact, left, Aw_exact
    else:
        reconstructed_G = state.L @ state.L.mT
        factor_error = _relative_error(reconstructed_G, state.G)
        del reconstructed_G
        left = state.L @ state.Aw
        reconstructed_M = left @ state.L.mT
        operator_error = _relative_error(reconstructed_M, state.M)
        del left, reconstructed_M
        payload = {'kind': 'full_backward_residual', 'factor_metric': '||L L^T - G||_F / ||G||_F', 'factor_relative_error': factor_error, 'operator_metric': '||L Aw L^T - M||_F / ||M||_F', 'operator_relative_error': operator_error}
    torch.cuda.empty_cache()
    return payload

def _validate_method(*, method: str, state: BenchmarkState, z: torch.Tensor, w: torch.Tensor, workspace: Any, update_function: Callable[..., Any], tile: int, probes: torch.Tensor) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    _run_untimed_update(method=method, state=state, z=z, w=w, workspace=workspace, update_function=update_function, tile=tile)
    tensors = (state.G, state.M, state.Aw)
    if method == 'inverse':
        tensors += (state.Linv,)
    else:
        tensors += (state.L,)
    finite = all((_finite_full(tensor) for tensor in tensors))
    probe = _probe_validation(method=method, state=state, probes=probes)
    full_dense = _full_dense_validation(method=method, state=state) if state.rank <= CORRECTNESS_MAX_RANK else None
    errors = [probe['factor']['max_relative_error'], probe['operator']['max_relative_error']]
    if full_dense is not None:
        errors.extend([full_dense['factor_relative_error'], full_dense['operator_relative_error']])
    passed = finite and max(errors) <= CORRECTNESS_THRESHOLD
    payload = {'threshold': CORRECTNESS_THRESHOLD, 'passed': passed, 'finite_full': finite, 'probe': probe, 'full_dense': full_dense}
    if method == 'inverse':
        factor_action = state.Linv @ probes
    else:
        factor_action = torch.linalg.solve_triangular(state.L, probes, upper=False)
    actions = {'factor': factor_action, 'operator': state.Aw @ probes}
    return (payload, actions)

def _cross_method_probe_validation(*, inverse_actions: dict[str, torch.Tensor], fresh_actions: dict[str, torch.Tensor]) -> dict[str, Any]:
    factor = _column_errors(inverse_actions['factor'], fresh_actions['factor'])
    operator = _column_errors(inverse_actions['operator'], fresh_actions['operator'])
    passed = factor['max_relative_error'] <= CORRECTNESS_THRESHOLD and operator['max_relative_error'] <= CORRECTNESS_THRESHOLD
    payload = {'threshold': CORRECTNESS_THRESHOLD, 'passed': passed, 'seed': PROBE_SEED, 'count': PROBE_COUNT, 'distribution': 'NumPy PCG64 Rademacher (+/-1/sqrt(rank))', 'factor_identity': 'Linv_incremental P vs solve(L_fresh, P)', 'operator_identity': 'Aw_incremental P vs Aw_fresh P', 'factor': factor, 'operator': operator}
    return payload

def _atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

def _validate_protocol(protocol: dict) -> None:
    """Check frozen numerical settings without source, machine or campaign gates."""
    expected = {
        "ranks": list(RANKS), "ridge": RIDGE,
        "initial_transitions": list(INITIAL_TRANSITIONS),
        "warmup_transition": WARMUP_TRANSITION,
        "evaluation_transitions": list(EVALUATION_TRANSITIONS),
        "channel_order": list(CHANNEL_ORDER),
        "tiles": {str(k): v for k,v in TILES.items()},
        "repeats": {str(k): v for k,v in REPEATS.items()},
        "setup_num_warps": SETUP_NUM_WARPS,
        "sweep_num_warps": SWEEP_NUM_WARPS,
        "correctness_threshold": CORRECTNESS_THRESHOLD,
        "full_dense_correctness_max_rank": CORRECTNESS_MAX_RANK,
    }
    for key, value in expected.items():
        _require_equal("protocol."+key, protocol[key], value)
    _require_equal("protocol.trace.shape", protocol["trace"]["shape"], [N_SNAPSHOTS,MAX_RANK])
    _require_equal("protocol.trace.dtype", protocol["trace"]["dtype"], "float32")
    _require_equal("protocol.trace.sha256", protocol["trace"]["sha256"], "bdf1619a70af70585af317d25aa69452fcb8e94d8417e525de9d50f878e530b4")
    _require_equal("protocol.probe.seed", protocol["probe"]["seed"], PROBE_SEED)
    _require_equal("protocol.probe.count", protocol["probe"]["count"], PROBE_COUNT)
    for key,value in {"repository":DATASET_REPOSITORY,"revision":DATASET_REVISION,"relative_path":DATASET_RELATIVE_PATH,"bytes":DATASET_BYTES,"sha256":DATASET_SHA256,"scenario":4,"scenario_name":"torus_MBH_10"}.items():
        _require_equal("protocol.dataset."+key, protocol["dataset"][key], value)

def _repo_relative(path: Path) -> str:
    """Never write absolute input paths into result metadata."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path.name

@torch.no_grad()
def run(args: argparse.Namespace) -> Path:
    rank = args.rank
    tile = TILES[rank]
    repeats = REPEATS[rank]
    output_path = args.output_dir / f'ladder_r{rank}.json'
    if output_path.exists() and (not args.overwrite):
        raise FileExistsError(f'refusing to overwrite {output_path}')
    protocol = json.loads(PROTOCOL_PATH.read_text())
    _validate_protocol(protocol)
    trace_path = args.trace
    if _sha256_file(trace_path) != protocol['trace']['sha256']:
        raise ValueError('trace SHA-256 does not match frozen protocol; stop')
    manifest = {'rank_scaling': {'scales': protocol['rank_scales']}, 'trace': protocol['trace']}
    trace, scales = _validate_trace(trace_path, manifest)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    hardware = metadata()
    expected = 'B200' if args.method == 'tiled' else 'H100'
    if expected not in hardware.get('gpu', {}).get('name', ''):
        warnings.warn(f'Archived {args.method} timings used {expected}; recording current hardware')
    source = {'package_version': hardware['package_version'], 'git_commit': hardware['git_commit'], 'kernel_sha256': hardware['kernel_sha256']}
    feature_rows, snapshot_indices = _features_for_rank(trace, rank, scales[str(rank)])
    del trace
    feature_rows_cpu = feature_rows.cpu().numpy().astype(np.float64, copy=True)
    initial_z = feature_rows[[snapshot_indices[transition] for transition in INITIAL_TRANSITIONS]].contiguous()
    initial_w = feature_rows[[snapshot_indices[transition + 1] for transition in INITIAL_TRANSITIONS]].contiguous()
    initial_z_cpu = feature_rows_cpu[[snapshot_indices[transition] for transition in INITIAL_TRANSITIONS]]
    initial_condition = _condition_from_rows(initial_z_cpu)
    from doublesided_cholesky.cuda import InplaceConfig, InplaceWorkspace, TiledWorkspace, update_inplace, update_tiled
    workspace_type = TiledWorkspace if args.method == 'tiled' else InplaceWorkspace

    def update_function(P, A, z, w, *, tile_rows, tile_cols, setup_num_warps, sweep_num_warps, workspace):
        if args.method == 'tiled':
            return update_tiled(P, A, z, w, workspace=workspace, p_tile_cols=tile_cols, p_num_warps=sweep_num_warps)
        return update_inplace(P, A, z, w, workspace=workspace, config=InplaceConfig(tile_cols, setup_num_warps, sweep_num_warps))
    state = BenchmarkState(rank=rank, initial_z=initial_z, initial_w=initial_w)
    initial_state_metadata = state.construct_common_state()
    workspace = workspace_type(state.Aw)
    probes = _make_probes(rank)

    def vectors(transition: int) -> tuple[torch.Tensor, torch.Tensor]:
        z = feature_rows[snapshot_indices[transition]]
        w = feature_rows[snapshot_indices[transition + 1]]
        return (z, w)
    cases: list[dict[str, Any]] = []
    for case_index, transition in enumerate(EVALUATION_TRANSITIONS):
        z_cpu = feature_rows_cpu[snapshot_indices[transition]]
        w_cpu = feature_rows_cpu[snapshot_indices[transition + 1]]
        condition = _condition_from_rows(np.concatenate((initial_z_cpu, z_cpu[None, :]), axis=0))
        method_order = ['inverse', 'fresh_refact'] if case_index % 2 == 0 else ['fresh_refact', 'inverse']
        cases.append({'case_index': case_index, 'transition_index': transition, 'snapshot_pair': [transition, transition + 1], 'z_l2_norm': float(np.linalg.norm(z_cpu)), 'w_l2_norm': float(np.linalg.norm(w_cpu)), **condition, 'method_order': method_order, 'methods': {}})
    del feature_rows_cpu
    warmup_methods: dict[str, Any] = {}
    warmup_z, warmup_w = vectors(WARMUP_TRANSITION)
    for method in ('inverse', 'fresh_refact'):
        warmup_methods[method] = {'execution_wall_seconds': _run_untimed_update(method=method, state=state, z=warmup_z, w=warmup_w, workspace=workspace, update_function=update_function, tile=tile)}
    for case in cases:
        transition = int(case['transition_index'])
        z, w = vectors(transition)
        for method in case['method_order']:
            case['methods'][method] = _time_method(method=method, state=state, z=z, w=w, workspace=workspace, update_function=update_function, tile=tile, repeats=repeats)
    for case in cases:
        transition = int(case['transition_index'])
        z, w = vectors(transition)
        action_outputs: dict[str, dict[str, torch.Tensor]] = {}
        for method in ('inverse', 'fresh_refact'):
            validation, actions = _validate_method(method=method, state=state, z=z, w=w, workspace=workspace, update_function=update_function, tile=tile, probes=probes)
            case['methods'][method]['validation'] = validation
            action_outputs[method] = actions
        case['cross_method_probe_comparison'] = _cross_method_probe_validation(inverse_actions=action_outputs['inverse'], fresh_actions=action_outputs['fresh_refact'])
        del action_outputs
    payload = {'schema_version': SCHEMA_VERSION, 'experiment': 'cuda_blackhole_rank_scaling', 'protocol_version': PROTOCOL_VERSION, 'run_tag': args.run_tag, 'method': args.method, 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'rank': rank, 'repeats_per_method_per_case': repeats, 'case_count': len(cases), 'method_order_policy': 'zero-based even case indices run inverse then fresh_refact; odd case indices reverse that order', 'tile_config': {'tile_rows': tile, 'tile_cols': tile, 'setup_num_warps': SETUP_NUM_WARPS, 'sweep_num_warps': SWEEP_NUM_WARPS}, 'timing_region': TIMING_REGION, 'protocol': {'path': 'benchmarks/the_well/protocol.json', 'sha256': _sha256_file(PROTOCOL_PATH)}, 'trace': {'path': _repo_relative(trace_path), 'sha256': manifest['trace']['sha256'], 'shape': manifest['trace']['shape'], 'dtype': manifest['trace']['dtype'], 'rank_scale': scales[str(rank)], 'consumer_operation': 'trace[t, :rank] / rank_scale', 'consumer_scaling_dtype': 'float32'}, 'dataset': {'repository': DATASET_REPOSITORY, 'revision': DATASET_REVISION, 'relative_path': DATASET_RELATIVE_PATH, 'bytes': DATASET_BYTES, 'sha256': DATASET_SHA256, 'scenario': 4, 'scenario_name': 'torus_MBH_10'}, 'initial_state': {'ridge': RIDGE, 'transitions': list(INITIAL_TRANSITIONS), 'definition_G': 'I + sum_{t=0}^{63} z_t z_t^T', 'definition_M': 'sum_{t=0}^{63} z_t w_t^T', 'condition_number_method': 'FP64 eigvalsh of the 64-by-64 small Gram; because rank>64, lambda_min(G0)=ridge exactly', **initial_condition, **initial_state_metadata}, 'warmup': {'transition_index': WARMUP_TRANSITION, 'snapshot_pair': [WARMUP_TRANSITION, WARMUP_TRANSITION + 1], 'excluded_from_timing': True, 'method_order': ['inverse', 'fresh_refact'], 'methods': warmup_methods}, 'probe': {'seed': PROBE_SEED, 'count': PROBE_COUNT, 'distribution': 'NumPy PCG64 Rademacher (+/-1/sqrt(rank))', 'threshold': CORRECTNESS_THRESHOLD, 'cross_method_comparison_required': True, 'cross_method_factor_identity': 'Linv_incremental P vs solve(L_fresh, P)', 'cross_method_operator_identity': 'Aw_incremental P vs Aw_fresh P'}, 'full_dense_correctness_max_rank': CORRECTNESS_MAX_RANK, 'source': source, 'hardware': hardware, 'cases': cases}
    _atomic_json_dump(payload, output_path)
    print(json.dumps(payload, sort_keys=True), flush=True)
    print(f'wrote {output_path}', flush=True)
    return output_path

def main(argv: Sequence[str] | None=None) -> None:
    """Run selected ranks; numerical protocol remains fixed in protocol.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['tiled', 'inplace'], default='tiled')
    parser.add_argument('--ranks', type=int, nargs='+', choices=RANKS, default=list(RANKS))
    parser.add_argument('--trace', type=Path, default=DEFAULT_TRACE)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/local/the_well')
    parser.add_argument('--run-tag', default='release')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    for rank in args.ranks:
        args.rank = rank
        run(args)
        torch.cuda.empty_cache()
if __name__ == '__main__':
    main()
