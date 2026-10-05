# Algorithm and storage

Let `P=L⁻¹`, `u=Pz`, and `y=Pw`. Math symbols in the kernels use `c`, `s`, and
`q` for rotation cosine, sine, and prefix cosine; `r` is rank, `B=64` is tile
width, and `i,j,k` are indices. `A` or `Aw` denotes the whitened operator.

## Rotation parameters and factor update

For row k, define `a_k = 1 + sum_{j<k} u_j²` and `b_k=a_k+u_k²`:

- `c_k = sqrt(a_k/b_k)`
- `s_k = u_k/sqrt(b_k)`
- `q_k = 1/sqrt(a_k)`
- `vz_k = u_k/sqrt(a_k*b_k) = ((L+)⁻¹z)_k`
- `vw_k = c_k*(y_k-u_k*sum_{j<k}(u_j*y_j)/a_k) = ((L+)⁻¹w)_k`

Two inclusive scans supply these quantities for asymmetric updates; one suffices
when `w=z`, for which `vw=vz`. The inverse factor updates as

`P+_{i,:} = c_i*(P_{i,:} - u_i*q_i²*sum_{j<i}(u_j*P_{j,:}))`.

The lower triangle is preserved. The reference and CPU variants instead apply
explicit Givens rotations to `[L | z]`: phase 1 updates L and records rotations;
phase 3 recovers vz; phase 4 forward-solves `L+ vw=w`; phase 2 sweeps the operator
using virtual padded row/column buffers and adds `vz vwᵀ`.

## Operator transport

Define `h_k=s_k*q_k` and `d_k=-s_k/(q_k*c_k)`. The two passes are

`X_ij = c_i*A_ij + h_i*sum_{k<i}(d_k*A_kj)`

`A+_ij = c_j*X_ij + h_j*sum_{k<j}(d_k*X_ik) + vz_i*vw_j`.

The original GPU path assigns independent columns to serial left recurrences,
fusing the inverse-factor update, followed by independent serial row recurrences.
It reads and writes A twice.

The tiled path reconstructs the carry into each 64×64 tile from linear descriptors.
For tile `(I,J)` and local indices `a,b`:

- `south[I,J,b] = sum_a d[IB+a]*A[IB+a,JB+b]`, using original A.
- `north[I,J] = sum_{K<I} south[K,J]`.
- `east0[I,J,a] = sum_b d[JB+b]*X0[IB+a,JB+b]`, where X0 uses zero carry.
- `west[I,J,a] = sum_{K<J}(east0[I,K,a] + h[IB+a]*sum_b d[KB+b]*north[I,K,b])`.

The descriptor, north scan, west scan, and apply kernels read A twice and write
it once. Symmetric storage retains `nt=n(n+1)/2` lower tiles, where
`n=ceil(r/64)`. The `upper` descriptor contributes mirrored tiles to the north
scan, diagonal operator tiles are re-symmetrized on write, and split-K computes
Pz from factor tiles. Shared memory uses a 64×65 array to reduce bank conflicts.
The tests emulate this transport in float64 NumPy, including ragged tiles.

## Addressing correction F1

The original Triton expression multiplied the tile index by 4096 in int32 before
adding int64 local offsets. At rank 65,536 the first overflowing tile is
`(1023,512)`, at element offset `2³¹`; 512 tiles are affected. The first affected
rank is 65,473. Conversion kernels had the same pattern.

The release casts tile indices to int64 **before multiplication** and uses int64
local indices in conversions. Floating-point expressions are preserved. The
archived rank-65,536 symmetric measurement is historical evidence only and must
be replaced after the large-rank correctness gate and B200 remeasurement.

## Exact FP32 memory accounting

| Allocation | Bytes |
|---|---:|
| Dense inverse factor + operator | `8*r²` |
| Tiled factor + operator | `2*nt*4096*4` |
| Packed G and M bookkeeping | `2*(r*(r+1)/2)*4` |
| Original in-place workspace | `9*batch*r*4` |
| Symmetric tiled workspace | `(5*nt*64 + (7+split_k)*r)*4` |
| Asymmetric tiled workspace | `(4*n²*64 + 11*r)*4` |

The asymmetric vector count is **11r**, comprising two `(r,2)` buffers and seven
length-r vectors. The symmetric workspace is exactly 0.6429443359375 GiB at
rank 65,536, including split-K partials. The default splits are 1 through 8,192,
16 through 32,768, then 64. Workspace `nbytes` sums actual tensors.

B1's incremental state (including packed G/M) uses about 34.4 GB at rank 65,536;
the dense fresh path uses four matrices, about 68.7 GB, plus scalar info. Holding
both needs about 103.1 GB plus 0.69 GB workspace and additional solver/runtime
memory. `--method incremental` and `--method fresh` permit separate processes.
The benchmark records exact persistent sizes and measured peak allocations;
these are different quantities. B3 shares a single factor buffer between a
column-major Cholesky view and a row-major inverse view, retaining four dense
matrices total (about 68.7 GB at rank 65,536), plus C4 workspace (about 1.08 GB).

Probe helpers apply packed/tiled matrices without a full dense materialization.
The tile helper uses O(64*r) strip scratch; the packed helper uses a selectable
row-block size. Transpose factor actions are supported by the tile helper.

## Historical optimization and roofline

Historical B200 measurements progressed from 64.008 to 60.178 to 43.598 ms at
rank 65,536. Descriptor-transport phases fell from 32.562 to 16.017 ms after
padding shared memory. These are archived observations, not validation of the
corrected release. Phase data and rejected-variant measurements are in
[results/README.md](../results/README.md); the rejected implementations are omitted.

The update's leading work and memory traffic are quadratic; fresh factorization
and whitening are cubic. The plan's approximate 0.64 flop/byte arithmetic
intensity is a qualitative reference for the original streaming update. It is
not a measured intensity for every release path. C4/C5 operator transport moves
12 bytes per stored FP32 element (two reads and one write), plus descriptor
traffic; C5 has approximately half as many stored matrix elements. For a
bandwidth-limited update against a compute-limited refactorization, speedup
therefore grows roughly linearly with rank. No new B200 roofline measurement
has been made. The separate `flop_analysis.md` input was unavailable.

## Migration

| Original symbol | Public replacement |
|---|---|
| `incremental_update_reference` | `reference.update`, now returning `(Aw,vz)` |
| `fast_cholesky.fast_incremental_update` | `cpu.update` |
| `fast_incremental_update_batched` | `cpu.update_batched`; now updates caller L |
| `TritonInverseInplaceWorkspace` | `cuda.InplaceWorkspace` |
| `incremental_update_triton_inverse_factor_inplace` | `cuda.update_inplace` |
| `B200AsymmetricWorkspace` | `cuda.TiledWorkspace` |
| `incremental_update_inverse_asymmetric_b200` | `cuda.update_tiled` |
| `T2Workspace` | `cuda.SymmetricTiledWorkspace` |
| `symmetric_update_t2` | `cuda.update_symmetric_tiled` |
| `packed_symmetric_metric_operator_outer_update_` | `cuda.packed_symmetric_rank1_update_` |
