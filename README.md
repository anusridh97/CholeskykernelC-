# Double-sided Cholesky

Apply the rank-one edits `G+ = G + zzᵀ` and `M+ = M + zwᵀ` while maintaining
`Aw = L⁻¹ M L⁻ᵀ`, where `G = LLᵀ`. The update takes O(r²) work instead of a
fresh O(r³) Cholesky factorization and two triangular solves. This repository
combines the original reference/CPU/Triton implementations and the tiled CUDA
implementations tuned on B200.

**Release candidate:** CPU golden equivalence and all 71 existing tests on B200,
including the rank-65,536 overflow regression, have passed. GPU snapshot
equivalence, complete reproduction and final timing-artifact import remain
pending. The archived symmetric 239.55× row predates the int64 offset fix and
remains flagged until the replacement result is finalized.
See [validation](docs/VALIDATION.md).
The owner must select a license before publication.

## Install

Use Python 3.10 or newer. Install PyTorch for your CPU or CUDA environment first,
then build against that installation:

```bash
python -m pip install 'torch>=2.8' 'setuptools>=69' wheel numpy pytest
python -m pip install --no-build-isolation -e .
```

CPU-only installation skips CUDA and does not import Triton. On CUDA, use the
Triton version compatible with your PyTorch wheel and install the CUDA toolkit
with nvcc. `FORCE_CUDA=1` enables cross-building without a visible GPU. Unless
`TORCH_CUDA_ARCH_LIST` is set, CUDA >=12.8 targets `8.0;9.0;10.0+PTX`; older
nvcc targets `8.0;9.0+PTX`. CPU compilation uses native architecture flags and
does not force AVX-512. Rebuild native extensions for each target environment.

## Quickstart (requires CUDA and the native CUDA extension)

```python
import torch
from doublesided_cholesky import reference
from doublesided_cholesky.cuda import TiledWorkspace, update_tiled

torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
rank = 129
with torch.no_grad():
    torch.manual_seed(0)
    Q = torch.randn(rank, 32, device="cuda") / rank**0.5
    G = torch.eye(rank, device="cuda") + Q @ Q.T
    M = torch.randn(rank, rank, device="cuda") / rank**0.5
    Linv, Aw = reference.inverse_state(G, M)
    z = torch.randn(rank, device="cuda") / rank**0.5
    w = torch.randn(rank, device="cuda") / rank**0.5
    workspace = TiledWorkspace(Aw)
    Aw, vz = update_tiled(Linv, Aw, z, w, workspace=workspace)
    G += torch.outer(z, z)
    M += torch.outer(z, w)
    _, exact_Aw = reference.refactor_dense(G.double(), M.double())
    relative_error = (Aw.double() - exact_Aw).norm() / exact_Aw.norm()
    assert relative_error < 5e-6
```

## Choose a path

| API | State | Use |
|---|---|---|
| `reference.update` | dense `(L, Aw)` | Slow correctness reference, any device; FP64 supported |
| `cpu.update` / `cpu.update_batched` | dense `(L, Aw)` | CPU FP32/FP64; factor updated in place |
| `cuda.update_inplace` | dense `(Linv, Aw)` | Original portable GPU baseline; batches supported |
| `cuda.update_tiled` | dense `(Linv, Aw)` | Large-rank asymmetric operators; CUDA/Triton |
| `cuda.update_symmetric_tiled` | lower-tile `(P, A)` | Symmetric `M`, `w=z`; approximately half dense-state memory |

Small ranks can favor refactorization. The archived symmetric B200 sweep crosses
break-even near rank 512; The Well asymmetric workload gives 2.23× at rank 2,048.
Use `InplaceConfig.for_rank(r)` for the tuned baseline; omitting the configuration
preserves the original library defaults.

## State and mutation contracts

- Factor gauge: `L` is the lower Cholesky factor and **must be row-major
  contiguous**. Call `.contiguous()` on a newly computed Cholesky factor.
  CPU functions preserve `Aw`, `z`, and `w` and return new `Aw` and `vz`.
- Dense inverse gauge: `Linv=L⁻¹`, contiguous with an **explicitly zero upper
  triangle**, as produced by `reference.inverse_state`.
- Symmetric inverse gauge: one-dimensional lower-tile buffers of
  `layouts.tile_elements(r)` elements. Tiles are 64×64, ordered by
  `(i*(i+1)//2+j)*4096`. Diagonal operator tiles store both triangles. Factor
  diagonal tiles have zero upper triangle. All padding beyond rank is zero.
- Row-packed G/M use offset `i*(i+1)//2+j`, `j<=i`, and length `r*(r+1)//2`.
- Every GPU path requires contiguous FP32 tensors on one CUDA device and updates
  its factor and operator **in place**. Returned `vz=(L+)⁻¹z` aliases workspace
  memory and is overwritten on reuse; clone it if retaining it.
- G/M bookkeeping belongs to the caller. Symmetric and The Well timing regions
  include that bookkeeping. The asymmetric synthetic benchmark preserves the
  archived **core-only** comparison and labels the optional fresh region.
- Disable TF32 and use matmul precision `highest`. GPU paths support positive
  rank-one updates only, with no downdates, autograd, mixed precision or batched
  tiled updates. Tile size is fixed at 64.
- FP32 drift accumulates. Periodically refresh using `reference.inverse_state`
  according to application needs; the library does not impose a refresh policy.

## Archived results

| Workload, rank 65,536 | GPU / implementation | Statistic | Speedup |
|---|---|---|---:|
| The Well | B200 / tiled asymmetric | Median of transition speedups | 112.27× |
| The Well | H100 / original in-place | Mean of transition speedups | 71.75× |
| Symmetric synthetic | B200 / symmetric tiled | Ratio of sample medians | 239.55× — **superseded-F1; rerun required** |

The symmetric rank-32,768 archived ratio is 115.98×. Symmetric publication requires
one-step, trajectory and large-rank correctness gates plus a new B200 sweep.
Comparing the two The Well rows changes both hardware and kernel; a new original
kernel run on B200 separates those effects. Full tables, environments and
baseline caveats are in [results/README.md](results/README.md).

## Tests and reproduction

```bash
python -m pytest -m 'not cuda and not large'
# Requires CUDA; tiled tests require the native CUDA extension.
python -m pytest -m 'not large'
# Requires about 40 GB for symmetric, 80 GB for asymmetric tests.
DSC_RUN_LARGE=1 python -m pytest tests/test_large_rank.py
```

See [REPRODUCING.md](docs/REPRODUCING.md) for exact commands, memory requirements,
The Well preparation and hardware gates, and [ALGORITHM.md](docs/ALGORITHM.md)
for equations, memory formulas and migration names. Tests import the installed
`src/` package; no source file modifies `sys.path`.

## Citation and license

Package authorship and paper citation: **owner to supply**. `LICENSE` is a
placeholder and grants no open-source license yet.

Dataset: Polymathic AI, *The Well*, NeurIPS 2024 Datasets & Benchmarks.
[Post-neutron-star-merger dataset](https://huggingface.co/datasets/polymathic-ai/post_neutron_star_merger).
The default workflow regenerates features locally; no dataset or derived trace
is redistributed here.
