# Double-sided Cholesky

**Quadratic-time updates for operators expressed in a changing Cholesky-whitened basis.**

Many scientific algorithms work with two matrices: a positive-definite metric or Gram matrix `G`, and an operator or cross-moment matrix `M`. Rather than repeatedly forming the Cholesky factor and transforming the full operator when new data arrive, this package updates the whitened operator directly.

For `G=LLᵀ`, the maintained state is

\[
A=L^{-1}ML^{-T}.
\]

Given a new pair of vectors `z,w`, the supported update is

\[
G^+=G+zz^T,\qquad M^+=M+zw^T,
\qquad A^+=(L^+)^{-1}M^+(L^+)^{-T}.
\]

The Cholesky update induces a change of orthonormal coordinates. Instead of recomputing a dense factorization and applying its inverse from both sides, the algorithm transports the operator through structured rotations on the left and right. That **double-sided transport** is the central operation: the package maintains both the metric's factor state and the correctly transformed operator in **O(r²)** work per supported rank-one edit rather than the **O(r³)** work of dense recomputation.

This is useful when a **dense whitened operator must be maintained through many compatible updates**. It is not a drop-in replacement for arbitrary matrix changes, an eigensolver, or a complete scientific simulation. See the application mappings below before choosing a kernel.

**Implementations:** reference PyTorch, optimized CPU, original CUDA baseline, tiled CUDA (general and symmetric), and MLX/Metal for Apple Silicon. The repository also includes The Well and synthetic benchmarks, BO/NG application experiments, tests, and reproduction records.

**Release candidate:** the completed B200 run passed 73 tests, including both rank-65,536 regressions, and all 140 The Well cases across both methods. Twenty-two tested CPU/GPU snapshot cases were bitwise equal. Raw rerun evidence and source revisions are in [results/reproduced/b200-20261010](results/reproduced/b200-20261010). See [validation](docs/VALIDATION.md) for scope and remaining checks. The owner must select a license before publication.

### Navigate this README

- [Scientific applications](#scientific-applications-where-this-kernel-fits): Koopman/EDMD, Hamiltonians, electronic structure/DFT, reduced-order modeling, and an integration checklist.
- [Repository guide](#repository-guide): the purpose of each major source, benchmark, test, and documentation directory.
- [Simple reproduction scripts](#simple-reproduction-scripts) and [Install](#install): start from a verified environment.
- [Quickstart](#quickstart-requires-cuda-and-the-native-cuda-extension) and [Choose a path](#choose-a-path): run an update and select the appropriate API.
- [State and mutation contracts](#state-and-mutation-contracts), [Archived results](#archived-results), and [Tests and reproduction](#tests-and-reproduction): correct integration and evidence.

## Scientific applications: where this kernel fits

The same operator-transport pattern appears in streaming estimation and scientific computing. The examples below show how to map each problem onto the kernel and distinguish **exact update matches** from **conditional integration opportunities**. The repository is a kernel library, not a completed DFT, EDMD, or PDE solver.

### The required update pattern

Maintain an SPD matrix `G=LLᵀ` and a possibly nonsymmetric `M`, with whitened operator

\[
A=L^{-1}ML^{-T}.
\]

Each supported step supplies `z,w ∈ R^r` such that

\[
G^+=G+zz^T,\qquad M^+=M+zw^T.
\]

It is worth checking **both** equalities before integrating the kernel. Merely having a low-rank change in `G`, or a symmetric `M`, is insufficient. In particular, the specialized symmetric tiled implementation assumes `w=z`, rather than handling all possible symmetric changes to `M`.

The updates apply to real-valued tensors. Positive additions retain an SPD metric; deleting old observations, sliding windows, decays and arbitrary basis changes are not directly implemented.

### 1. Koopman / extended dynamic mode decomposition (direct)

Let `φ(x) ∈ R^r` be a dictionary of observables. For time-ordered state pairs `(x_i,y_i)` with `y_i` the one-step successor, define the unnormalized moments

\[
G_n=\lambda I+\sum_{i=1}^n z_i z_i^T,
\qquad M_n=\sum_{i=1}^n z_iw_i^T,
\qquad z_i=\phi(x_i),\;w_i=\phi(y_i).
\]

Adding one pair gives precisely `G⁺=G+zzᵀ`, `M⁺=M+zwᵀ`. This is the simplest way to use the asymmetric kernel on scientific time-series data, for example trajectory snapshots from simulations. A positive `λ` gives an initially invertible Gram matrix. If you use an averaged version of the moments, keep the scaling convention consistent (the formula above deliberately uses sums).

EDMD usually represents the least-squares map as `K=G⁻¹M` (subject to feature orientation). This library instead maintains `A=L⁻¹ML⁻ᵀ`. Since `A=LᵀKL⁻ᵀ`, the two are similar in exact arithmetic; eigenvalues agree but eigenvectors and the action on coordinates must be transformed properly. Do not simply substitute `A` for `K` in a feature-space forecasting formula.

**Pick:** `cuda.update_tiled` for large real nonsymmetric operators; `cpu.update` for CPU or float64 work. **Not included:** feature construction, Koopman mode decoding, spectrum tracking, sliding-window downdates or a production EDMD pipeline.

### 2. Generalized eigenproblems and Hamiltonian operators (conditional)

In a nonorthogonal basis the generalized eigenproblem is

\[
Hc=\varepsilon Sc,\qquad S=LL^T.
\]

By setting `u=Lᵀc`, this is equivalent to

\[
(L^{-1}HL^{-T})u=\varepsilon u.
\]

In this interpretation `G=S`, `M=H`, `A=L⁻¹HL⁻ᵀ`. The package maintains the expensive orthogonalized representation if the *pair* `(S,H)` has the required coupled edit. It does not diagonalize the transformed Hamiltonian; eigensolvers remain separate. A generic change in the Hamiltonian is not necessarily expressible as `zwᵀ`, and a generic symmetric Hamiltonian update is not necessarily `zzᵀ`.

**Pick:** general asymmetric kernel only when its mathematical edit form actually applies; if `H` must remain symmetric, verify that the edit preserves symmetry. The symmetric tiled kernel is valid for the specialized `ΔH=zzᵀ` case. **Not included:** a generalized eigensolver, eigenvector tracking, arbitrary Hermitian complex operators or arbitrary SPD downdates.

### 3. Nonorthogonal-basis electronic structure and DFT (conditional)

Atom-centered basis calculations use an overlap `S_ij=⟨φ_i|φ_j⟩` and projected Hamiltonian `H_ij=⟨φ_i|Ĥ|φ_j⟩`. To solve the resulting generalized eigenproblem one often orthogonalizes with a two-sided transformation. Cholesky whitening uses `L⁻¹HL⁻ᵀ` (not the identical matrix representation as symmetric Löwdin `S⁻¹/² H S⁻¹/²`, although both lead to an orthonormal representation).

This kernel is a candidate for *incrementally assembled or projected* Hamiltonian workflows when a sequence of contributions gives compatible updates to both matrices. It can reduce the cost of maintaining the orthogonalized **dense operator**, but cannot replace integral evaluation, Hamiltonian construction, self-consistent field iterations or the eigensolver. Ordinary DFT self-consistency updates typically modify `H` while `S` stays fixed; that is **not** the kernel's supported edit. Accordingly, this is a plausible integration point, **not a demonstrated linear-scaling DFT implementation**.

### 4. Reduced modeling, Galerkin methods and scientific machine learning (conditional)

In projected simulation, reduced mass/Gram matrices define a metric, while a reduced stiffness, dynamics or cross-moment matrix defines an operator. If contributions are accumulated from individual samples, basis evaluations or weighted observations and fit the coupled form, double-sided transport can save repeated dense refactor-and-rewhiten operations. It does **not** update an arbitrary Galerkin basis, differentiate through the solver, or solve the underlying ODE/PDE.

Similar bookkeeping appears in streaming regression and covariance-normalized models. However, if a workflow needs only the solution of `Gx=b`, a matrix-free operator, or a few selected eigenpairs, maintaining a dense `A` may be unnecessary. Benchmark the *whole application* before adopting the kernel.

### Integration checklist

1. Identify `G`, `M` and the desired downstream coordinates. Is the required maintained state really `A=L⁻¹ML⁻ᵀ`?
2. Prove that **each step** has `ΔG=zzᵀ` and `ΔM=zwᵀ` (or split a supported sum into sequential rank-one steps). If the metric edit is negative, independent or complex, this implementation does not apply directly.
3. Choose `reference` for an initial correctness test, CPU for FP64, tiled asymmetric CUDA for large general operators, or the symmetric tiled special case only for `w=z`.
4. Initialize a correctly gauged state, use contiguous tensors and disable TF32 on GPU.
5. For small ranks and representative streams, compare `A` against a fresh double-precision factorization; monitor residuals over many updates and plan periodic refreshes.
6. Include the costs of feature generation, matrix bookkeeping, eigen/singular-value computations, transfers and refreshes in application-level timings.

### Further reading and scientific references

- Nonorthogonal atomic-orbital Hamiltonian and overlap matrices: [Deep-learning density functional theory Hamiltonian for efficient *ab initio* electronic-structure calculation](https://pmc.ncbi.nlm.nih.gov/articles/PMC11499279/).
- Transforming a generalized Kohn–Sham eigenproblem by a two-sided congruence: [Quantum Perturbation Theory Using Tensor Cores and a Deep Neural Network](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00274).
- EDMD Gram / cross-covariance and finite dictionary Koopman operators: [Extended dynamic mode decomposition with dictionary learning](https://pubs.aip.org/aip/cha/article/27/10/103111/151485/Extended-dynamic-mode-decomposition-with).
- Kernel derivation and actual implementation: [ALGORITHM.md](docs/ALGORITHM.md).
- Testing, benchmark methods and hardware conditions: [REPRODUCING.md](docs/REPRODUCING.md).

## Repository guide

| Location | What you will find |
|---|---|
| `src/doublesided_cholesky/reference.py` | Dense, readable reference operations and initialization; use to check numerical correctness. |
| `src/doublesided_cholesky/cpu.py` | Native CPU updates and batched CPU entry points, including FP64 paths. |
| `src/doublesided_cholesky/cuda/inplace.py` | Original in-place CUDA baseline and tunable rank-dependent configuration. |
| `src/doublesided_cholesky/cuda/tiled.py` | Tiled CUDA update for general (possibly asymmetric) operators. |
| `src/doublesided_cholesky/cuda/symmetric_tiled.py` | Specialized tiled CUDA path for `w=z` and symmetric operator state. |
| `src/doublesided_cholesky/mlx.py` | Apple Silicon implementation via MLX/Metal. |
| `src/doublesided_cholesky/layouts.py` | Packed and tiled storage layouts shared by the implementations. |
| `benchmarks/` | CPU, CUDA, synthetic, and The Well experiments; result summarization and timing harnesses. |
| `benchmarks/applications/` | Bayesian optimization / natural-gradient experiment drivers and GB1 data handling; not turnkey DFT or EDMD code. |
| `benchmarks/mlx/` | Matched-output checks and Apple Silicon measurement harness. |
| `scripts/` | Reproduction entry points, setup commands, and data preparation helpers. |
| `tests/` | Unit, numerical, reproducibility, and large-rank regression tests. |
| `results/` | Archived and reproduced measurements, with methodology caveats in `results/README.md`. |
| `docs/ALGORITHM.md` | Derivation, state representations, complexity, memory, and migration notes. |
| `docs/REPRODUCING.md`, `docs/VALIDATION.md`, `docs/MLX.md`, `docs/APPLICATION_AUDIT.md` | Setup, evidence, hardware requirements, and application reproduction limits. |

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

Package authorship and citation: 

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
