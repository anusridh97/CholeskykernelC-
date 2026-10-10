"""Asymmetric core timings with both default and tuned original-kernel baselines."""
import argparse
from pathlib import Path
import torch
from ._env import configure_cuda, metadata, measure, write_results

@torch.no_grad()
def run_rank(rank, repeats, warmups, seed, fresh):
    """Restore archived P0=r^-1/2 I and A0=0 outside every timed sample."""
    from doublesided_cholesky.cuda import InplaceWorkspace, InplaceConfig, TiledWorkspace, update_inplace, update_tiled
    configure_cuda()
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    z = (torch.randn(rank, device='cuda') / rank ** 0.5).contiguous()
    w = (torch.randn(rank, device='cuda') / rank ** 0.5).contiguous()
    P0 = torch.zeros((rank, rank), device='cuda')
    P0.diagonal().fill_(rank ** (-0.5))
    A0 = torch.zeros_like(P0)
    P = torch.empty_like(P0)
    A = torch.empty_like(A0)
    iw = InplaceWorkspace(A)
    tw = TiledWorkspace(A)

    def prepare():
        P.copy_(P0)
        A.copy_(A0)
    methods = {}
    methods['inplace_default'] = measure(lambda: update_inplace(P, A, z, w, workspace=iw), prepare, repeats, warmups)
    methods['inplace_tuned'] = measure(lambda: update_inplace(P, A, z, w, workspace=iw, config=InplaceConfig.for_rank(rank)), prepare, repeats, warmups)
    methods['tiled'] = measure(lambda: update_tiled(P, A, z, w, workspace=tw), prepare, repeats, warmups)
    # Preserve the archived default column and also compare the rank-ladder
    # inverse-factor settings used by the tuned in-place baseline and The Well.
    tuned = InplaceConfig.for_rank(rank)
    methods['tiled_tuned'] = measure(
        lambda: update_tiled(
            P, A, z, w, workspace=tw,
            p_tile_cols=tuned.tile, p_num_warps=tuned.sweep_num_warps,
        ),
        prepare, repeats, warmups,
    )
    method_configs = {
        'inplace_default': {'tile': 8, 'setup_num_warps': 4, 'sweep_num_warps': 4},
        'inplace_tuned': {
            'tile': tuned.tile, 'setup_num_warps': tuned.setup_num_warps,
            'sweep_num_warps': tuned.sweep_num_warps,
        },
        'tiled': {'p_tile_cols': 8, 'p_num_warps': 4, 'descriptor_tile': 64},
        'tiled_tuned': {
            'p_tile_cols': tuned.tile, 'p_num_warps': tuned.sweep_num_warps,
            'descriptor_tile': 64,
        },
    }
    if fresh:
        info = torch.zeros((), device='cuda', dtype=torch.int32)

        def prepare_fresh():
            P0.zero_()
            P0.diagonal().fill_(rank)
            A0.zero_()
            info.zero_()

        def fresh_update():
            torch.addr(P0, z, z, out=P0)
            torch.addr(A0, z, w, out=A0)
            torch.linalg.cholesky_ex(P0, check_errors=False, out=(P, info))
            torch.linalg.solve_triangular(P, A0, upper=False, left=True, out=A)
            torch.linalg.solve_triangular(P.mT, A, upper=True, left=False, out=A)
        methods['fresh'] = measure(fresh_update, prepare_fresh, repeats, warmups)
        if info.item():
            raise RuntimeError('fresh Cholesky failed')
    return {'rank': rank, 'seed': seed, 'repeats': repeats, 'warmups': warmups, 'workspace_bytes': tw.nbytes, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'methods': methods, 'method_configs': method_configs, 'incremental_region': 'core only (archived B2 protocol)', 'fresh_region': 'G/M outer updates and refactorization' if fresh else None}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranks', nargs='+', type=int, default=[8192, 32768, 65536])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('results/local/asymmetric'))
    args = parser.parse_args()
    rows = [run_rank(r, args.repeats, args.warmups, args.seed, args.fresh) for r in args.ranks]
    write_results({'environment': metadata(), 'rows': rows}, args.output)
if __name__ == '__main__':
    main()
