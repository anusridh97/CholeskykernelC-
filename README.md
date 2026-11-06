# Double-sided Cholesky

Apply the rank-one edits `G+ = G + zzᵀ` and `M+ = M + zwᵀ` while maintaining
`Aw = L⁻¹ M L⁻ᵀ`, where `G = LLᵀ`. The update takes O(r²) work instead of a
fresh O(r³) Cholesky factorization and two triangular solves. This repository
combines the original reference/CPU/Triton implementations and the tiled CUDA
implementations tuned on B200, plus the validated MLX/Metal update for Apple Silicon.

**Release candidate:** the completed B200 run passed 73 tests, including both
rank-65,536 regressions, and all 140 The Well cases across both methods.
Twenty-two tested CPU/GPU snapshot cases were bitwise equal. Raw rerun evidence
and source revisions are in [results/reproduced/b200-20261010](results/reproduced/b200-20261010).
See [validation](docs/VALIDATION.md) for scope and remaining checks.
The owner must select a license before publication.

## Simple reproduction scripts

- Tables from saved B200 samples, no GPU: `bash scripts/summarize_results.sh`
- CPU kernels and tests: `bash scripts/reproduce_cpu.sh`
- Fresh B200 measurements: `bash scripts/reproduce_b200.sh`
- CPU Bayesian optimization: `bash scripts/reproduce_bo.sh`
- CPU natural-gradient/chart experiments: `bash scripts/reproduce_ng.sh`
- Apple Silicon matched-output validation: `bash scripts/reproduce_mlx.sh --output-dir results/local/mlx`

Start with [the setup and reproduction instructions](docs/REPRODUCING.md).
The [MLX instructions and evidence](docs/MLX.md) cover the synchronization fix,
all 18 validated rank/batch cells, and the new Apple M4 measurements.
The BO/NG drivers preserve the recovered **corrected** implementation; they are
not a claim to reproduce the original submitted tables. Real GB1 arrays can now be regenerated with `bash scripts/prepare_gb1.sh`. Both application scripts support `--smoke` for explicitly synthetic
checks, and the CPU/CUDA experiment scripts support `--dry-run`. The MLX runner
checks correctness by default and collects timings only with `--time`.

The standalone package supersedes the earlier research layout. Historical
scripts and results remain available at
[the previous main revision](https://github.com/anusridh97/CholeskykernelC-/tree/6e3e2261cac6ec7d2f41179cca64cad45023bf44).

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
| Symmetric synthetic | B200 / symmetric tiled | Ratio of sample medians | 239.55× — **superseded-F1; historical only** |

The symmetric rank-32,768 archived ratio is 115.98×. The completed rerun measured
239.65× at rank 65,536 after the overflow fix. New The Well measurements on the
same B200 give 113.40× for tiled and 80.04× for in-place.
[New raw measurements](results/reproduced/b200-20261010) remain separate from the
[historical tables and baseline caveats](results/README.md).

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

### Real-data BO/NG reproduction

The CPU application handoff now includes a pinned GB1 data-preparation recipe:

```bash
bash scripts/setup.sh applications
bash scripts/prepare_gb1.sh
bash scripts/reproduce_applications.sh --dry-run
```

Run the printed sweep with `bash scripts/reproduce_applications.sh` after supplying
the recovered Protein MAT (included in the private handoff). See
[the application audit](docs/APPLICATION_AUDIT.md) for the settings, input hashes
and explicit differences from the historical paper tables. No B200 is needed.
