"""Symmetric synthetic benchmark: packed/tiled state versus dense fresh refactorization."""
import argparse
from pathlib import Path
import torch
from doublesided_cholesky import layouts
from ._env import configure_cuda, metadata, measure, write_results

def memory_bytes(rank: int, method: str) -> dict:
    """Exact persistent tensor sizes; solver/runtime allocations are additional."""
    n = layouts.tile_count(rank)
    nt = n * (n + 1) // 2
    split = 1 if rank <= 8192 else 16 if rank <= 32768 else 64
    incremental = (2 * layouts.packed_size(rank) + 2 * layouts.tile_elements(rank)) * 4
    workspace = (5 * nt * 64 + (7 + split) * rank) * 4
    fresh = 4 * rank * rank * 4 + 4
    return {'incremental_state_bytes': incremental if method != 'fresh' else 0, 'workspace_bytes': workspace if method != 'fresh' else 0, 'fresh_state_bytes': fresh if method != 'incremental' else 0, 'vector_bytes': rank * 4}

@torch.no_grad()
def run_rank(rank: int, method: str, repeats: int, warmups: int, seed: int) -> dict:
    """Preserve archived state construction and timed operations, with selectable allocation."""
    configure_cuda()
    torch.manual_seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    budget = memory_bytes(rank, method)
    if sum(budget.values()) > torch.cuda.mem_get_info()[0]:
        raise RuntimeError(f'Persistent buffers need {sum(budget.values()) / 2 ** 30:.2f} GiB; use --method incremental and --method fresh separately')
    z = (0.05 * torch.randn(rank, device='cuda')).contiguous()
    methods = {}
    if method != 'fresh':
        from doublesided_cholesky.cuda import SymmetricTiledWorkspace, update_symmetric_tiled, packed_symmetric_rank1_update_
        i = torch.arange(rank, device='cuda', dtype=torch.int64)
        rd = i * (i + 3) // 2
        td = layouts.tile_diagonal_offsets(rank)
        G = torch.empty(layouts.packed_size(rank), device='cuda')
        M = torch.empty_like(G)
        P = torch.empty(layouts.tile_elements(rank), device='cuda')
        A = torch.empty_like(P)
        ws = SymmetricTiledWorkspace(rank)

        def prepare_incremental():
            G.zero_()
            G[rd] = rank
            M.zero_()
            M[rd] = 1
            P.zero_()
            P[td] = rank ** (-0.5)
            A.zero_()
            A[td] = 1 / rank

        def incremental():
            packed_symmetric_rank1_update_(G, M, z, rank=rank)
            update_symmetric_tiled(P, A, z, rank=rank, workspace=ws)
        methods['incremental'] = measure(incremental, prepare_incremental, repeats, warmups)
        assert ws.nbytes == budget['workspace_bytes']
    if method != 'incremental':
        Gi = torch.empty((rank, rank), device='cuda')
        Mi = torch.empty_like(Gi)
        L = torch.empty_like(Gi)
        Aw = torch.empty_like(Gi)
        info = torch.zeros((), device='cuda', dtype=torch.int32)

        def prepare_fresh():
            Gi.zero_()
            Gi.diagonal().fill_(rank)
            Mi.zero_()
            Mi.diagonal().fill_(1)
            info.zero_()

        def fresh():
            torch.addr(Gi, z, z, out=Gi)
            torch.addr(Mi, z, z, out=Mi)
            torch.linalg.cholesky_ex(Gi, check_errors=False, out=(L, info))
            torch.linalg.solve_triangular(L, Mi, upper=False, left=True, out=Aw)
            torch.linalg.solve_triangular(L.mT, Aw, upper=True, left=False, out=Aw)
        methods['fresh'] = measure(fresh, prepare_fresh, repeats, warmups)
        if info.item() != 0:
            raise RuntimeError('fresh Cholesky failed')
    row = {'rank': rank, 'seed': seed, 'repeats': repeats, 'warmups': warmups, **budget, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'methods': methods, 'status': 'unvalidated-measurement'}
    if len(methods) == 2:
        row['speedup'] = methods['fresh']['median_ms'] / methods['incremental']['median_ms']
    return row

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranks', type=int, nargs='+', default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    parser.add_argument('--method', choices=['incremental', 'fresh', 'both'], default='both')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--output', type=Path, default=Path('results/local/symmetric'))
    args = parser.parse_args()
    rows = [run_rank(r, args.method, args.repeats or (7 if r <= 8192 else 5 if r <= 32768 else 3), args.warmups, args.seed) for r in args.ranks]
    write_results({'environment': metadata(), 'rows': rows}, args.output)
if __name__ == '__main__':
    main()
