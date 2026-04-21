# TODO

Ranked from easiest to hardest. Each item has an expected speedup if the
change works, measured on the bench_timing harness at r = 128 float32
single-threaded. Check the baseline first, then re-run after your change.

## 1. Fix the batched ternary (warm-up)

Difficulty: trivial. Expected speedup: none, this is a correctness /
build fix.

In `fast_update.cpp`, the original version of `fast_incremental_update_batched`
used a ternary expression to select between `update_impl<double>` and
`update_impl<float>`. That does not compile because the two instantiations
return different `std::tuple` types. The current file has an if/else fix
in place; verify it still builds on your compiler and that the batched
correctness test passes.

## 2. Inline the Phase 4 triangular solve

Difficulty: easy. Expected speedup: 10 to 25 percent at small r.

Phase 4 currently calls `torch::linalg_solve_triangular`, which dispatches
into LAPACK for r^2 / 2 flops. At r = 128 that is a function call for
about 8K flops, and worse, LAPACK may spawn threads that collide with
`at::parallel_for` in the batched path.

Replace the call with an inline forward substitution loop. The function
is solving L+ * v_w = w for v_w given lower-triangular L+.

## 3. Remove the column-major round trip in Phase 1

Difficulty: easy. Expected speedup: 10 to 20 percent at r from 64 to 256.

The Phase 1 code copies L from row-major to `Lcm` in column-major, runs
the rotation loop, then copies back. That is 2 * r^2 of pointless memory
traffic. The inner rotation loop only touches one column of L at a time,
which is already a contiguous slice in row-major if you transpose the
indexing.

Work directly on `L_rm`. Use the index `L_rm[i * r + k]` where the old
code used `Lcm[k * r + i]`. Verify correctness first (the tests will
catch it), then measure.

## 4. Hand-vectorize the rotation inner loops

Difficulty: medium. Expected speedup: 1.3 to 1.5x overall.

The three hot inner loops in Phase 1 and Phase 2 have the shape

    for (int i = 0; i < r; i++) {
        scalar_t lv = X[i], zv = Y[i];
        X[i] = c * lv + s * zv;
        Y[i] = -s * lv + c * zv;
    }

With `-O3 -ffast-math -march=native` the compiler auto-vectorizes this,
but usually not as well as hand-written intrinsics (it tends to reload
c, s on every iteration instead of keeping them in registers).

On x86 with AVX-512, write these with `_mm512_loadu_ps` and
`_mm512_fmadd_ps`, keeping `c`, `s`, and `-s` broadcast into vector
registers across the loop. The rotation loops are about 60 percent of
the total runtime, so this pays off.

Portability note: guard the intrinsics on `__AVX512F__` with a fallback
scalar version for other machines.

## 5. Fix the Triton right sweep

Difficulty: hard. Expected speedup: significant on the GPU path;
currently the right sweep is O(r^2 * BLOCK) memory traffic instead of
O(r^2).

See the KNOWN ISSUE note in `doublesided_cholesky.py`. The current
implementation loads a full row from `Ah_ptr`, patches two entries, and
writes the row back, once per (k, i) pair. The intended plan is:

  a. Transpose Ah into a second (RP x RP) scratch buffer. You need to
     allocate this scratch from Python and pass a pointer in.
  b. Run the rotation sweep over rows of the transpose, which are the
     original columns. This reuses the Phase 2 left-sweep code almost
     verbatim.
  c. Transpose back into Ah_ptr.

Alternatively, restructure Phase 2 entirely to operate on a single
SRAM-resident tile whose layout supports both row and column sweeps
efficiently. That is more ambitious but cleaner.

Verify with `test_triton_matches_ground_truth` after each change.

## 6. Batched GPU kernel (stretch)

Difficulty: hard. Expected speedup: depends on batch size.

The current Triton kernel launches with grid `(1,)` and works on a single
update. Extend it to process a batch of T independent updates in parallel,
one per program instance. Each program instance reads its own slice of
L, Aw, z, w and writes its own slice of Aw_new, vz. The existing SRAM
working set per program is fine; just add a `pid = tl.program_id(0)`
and offset all pointers by `pid * r * r` or `pid * r`.

Useful when running many parallel Bayesian optimization replicates or
ensemble updates.
