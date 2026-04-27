"""Debug the test_tiled_triton_matches_fused_triton failure at r=64."""
import torch

from tests.test_correctness import random_problem, ground_truth
from doublesided_cholesky import (
    incremental_update_triton,
    incremental_update_triton_tiled,
)

assert torch.cuda.is_available()

for r in [16, 64, 256]:
    print(f"=== r = {r} ===")
    G, L, M, Aw, z, w = random_problem(r, dtype=torch.float32)
    G, L, M, Aw, z, w = [t.cuda() for t in (G, L, M, Aw, z, w)]
    L_exact, Aw_exact = ground_truth(G, M, z, w)

    L_fused = L.clone()
    Aw_fused, vz_fused = incremental_update_triton(
        L_fused, Aw.clone(), z.clone(), w.clone()
    )

    L_tiled = L.clone()
    Aw_tiled, vz_tiled = incremental_update_triton_tiled(
        L_tiled, Aw.clone(), z.clone(), w.clone()
    )

    fused_vs_exact = (torch.norm(Aw_fused - Aw_exact) / torch.norm(Aw_exact)).item()
    tiled_vs_exact = (torch.norm(Aw_tiled - Aw_exact) / torch.norm(Aw_exact)).item()
    fused_vs_tiled = (torch.norm(Aw_fused - Aw_tiled) / torch.norm(Aw_fused)).item()
    fused_norm = torch.norm(Aw_fused).item()
    tiled_norm = torch.norm(Aw_tiled).item()
    exact_norm = torch.norm(Aw_exact).item()

    print(f"  ||Aw_exact|| = {exact_norm:.4e}")
    print(f"  ||Aw_fused|| = {fused_norm:.4e}")
    print(f"  ||Aw_tiled|| = {tiled_norm:.4e}")
    print(f"  fused vs exact (rel) = {fused_vs_exact:.4e}")
    print(f"  tiled vs exact (rel) = {tiled_vs_exact:.4e}")
    print(f"  fused vs tiled (rel) = {fused_vs_tiled:.4e}")

    # Where are the largest disagreements?
    diff = (Aw_fused - Aw_tiled).abs()
    flat_diff = diff.flatten()
    topk = torch.topk(flat_diff, k=min(5, flat_diff.numel()))
    print(f"  top-5 abs diffs: {topk.values.tolist()}")
    idx0 = topk.indices[0].item()
    i0, j0 = idx0 // r, idx0 % r
    print(f"    @[{i0},{j0}]: fused={Aw_fused[i0, j0].item():.6e}, "
          f"tiled={Aw_tiled[i0, j0].item():.6e}, "
          f"exact={Aw_exact[i0, j0].item():.6e}")

    # Now: try CALLING THEM IN OPPOSITE ORDER. Does it still fail?
    L_tiled2 = L.clone()
    Aw_tiled2, _ = incremental_update_triton_tiled(
        L_tiled2, Aw.clone(), z.clone(), w.clone()
    )
    L_fused2 = L.clone()
    Aw_fused2, _ = incremental_update_triton(
        L_fused2, Aw.clone(), z.clone(), w.clone()
    )
    fused2_vs_tiled2 = (torch.norm(Aw_fused2 - Aw_tiled2) / torch.norm(Aw_fused2)).item()
    print(f"  swapped order: fused vs tiled (rel) = {fused2_vs_tiled2:.4e}")

    # Fused twice in a row
    L_a = L.clone()
    Aw_a, _ = incremental_update_triton(L_a, Aw.clone(), z.clone(), w.clone())
    L_b = L.clone()
    Aw_b, _ = incremental_update_triton(L_b, Aw.clone(), z.clone(), w.clone())
    a_vs_b = (torch.norm(Aw_a - Aw_b) / torch.norm(Aw_a)).item()
    print(f"  fused vs fused (rel) = {a_vs_b:.4e}")
    print()
