#include <torch/extension.h>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>
#ifdef __AVX512F__
#include <immintrin.h>
#endif

template <typename scalar_t>
inline void givens_contiguous(scalar_t* __restrict__ X, scalar_t* __restrict Y, scalar_t c, scalar_t s, int len) {
    int i = 0;
#ifdef __AVX512F__
    if constexpr(std::is_same_v<scalar_t, float>) {
        __m512 vc = _mm512_set1_ps(c);
        __m512 vs = _mm512_set1_ps(s);
        __m512 vms = _mm512_set1_ps(-s);
        for (; i+15 < len; i+=16) {
            __m512 vx = _mm512_loadu_ps(X + i);
            __m512 vy = _mm512_loadu_ps(Y + i);
            __m512 nx = _mm512_fmadd_ps(vc, vx, _mm512_mul_ps(vs, vy));
            __m512 ny = _mm512_fmadd_ps(vms, vx, _mm512_mul_ps(vc, vy));
            _mm512_storeu_ps(X + i, nx);
            _mm512_storeu_ps(Y + i, ny);
        }
    } else if constexpr(std::is_same_v<scalar_t, double>) {
        __m512d vc = _mm512_set1_pd(c);
        __m512d vs = _mm512_set1_pd(s);
        __m512d vms = _mm512_set1_pd(-s);
        for (; i + 7 < len; i += 8) {
            __m512d vx = _mm512_loadu_pd(X + i);
            __m512d vy = _mm512_loadu_pd(Y + i);
            __m512d nx = _mm512_fmadd_pd(vc, vx, _mm512_mul_pd(vs, vy));
            __m512d ny = _mm512_fmadd_pd(vms, vx, _mm512_mul_pd(vc, vy));
            _mm512_storeu_pd(X + i, nx);
            _mm512_storeu_pd(Y + i, ny);
        }
    }
#endif
    // fallback (scalar)
    for (; i<len; i++) {
        scalar_t lv = X[i];
        scalar_t zv = Y[i];
        X[i] = c*lv + s*zv;
        Y[i] = -s*lv + c*zv;
    }
}

template <typename scalar_t>
inline void givens_from_input_contiguous(
    scalar_t* __restrict__ X,
    const scalar_t* __restrict__ X_in,
    scalar_t* __restrict__ Y,
    scalar_t c,
    scalar_t s,
    int len
) {
    int i = 0;
#ifdef __AVX512F__
    if constexpr(std::is_same_v<scalar_t, float>) {
        __m512 vc = _mm512_set1_ps(c);
        __m512 vs = _mm512_set1_ps(s);
        __m512 vms = _mm512_set1_ps(-s);
        for (; i + 15 < len; i += 16) {
            __m512 vx = _mm512_loadu_ps(X_in + i);
            __m512 vy = _mm512_loadu_ps(Y + i);
            __m512 nx = _mm512_fmadd_ps(vc, vx, _mm512_mul_ps(vs, vy));
            __m512 ny = _mm512_fmadd_ps(vms, vx, _mm512_mul_ps(vc, vy));
            _mm512_storeu_ps(X + i, nx);
            _mm512_storeu_ps(Y + i, ny);
        }
    } else if constexpr(std::is_same_v<scalar_t, double>) {
        __m512d vc = _mm512_set1_pd(c);
        __m512d vs = _mm512_set1_pd(s);
        __m512d vms = _mm512_set1_pd(-s);
        for (; i + 7 < len; i += 8) {
            __m512d vx = _mm512_loadu_pd(X_in + i);
            __m512d vy = _mm512_loadu_pd(Y + i);
            __m512d nx = _mm512_fmadd_pd(vc, vx, _mm512_mul_pd(vs, vy));
            __m512d ny = _mm512_fmadd_pd(vms, vx, _mm512_mul_pd(vc, vy));
            _mm512_storeu_pd(X + i, nx);
            _mm512_storeu_pd(Y + i, ny);
        }
    }
#endif
    for (; i < len; i++) {
        scalar_t lv = X_in[i];
        scalar_t zv = Y[i];
        X[i] = c * lv + s * zv;
        Y[i] = -s * lv + c * zv;
    }
}

template <typename scalar_t>
inline void add_scaled_contiguous(
    scalar_t* __restrict__ X,
    const scalar_t* __restrict__ Y,
    scalar_t a,
    int len
) {
    int i = 0;
#ifdef __AVX512F__
    if constexpr(std::is_same_v<scalar_t, float>) {
        __m512 va = _mm512_set1_ps(a);
        for (; i + 15 < len; i += 16) {
            __m512 vx = _mm512_loadu_ps(X + i);
            __m512 vy = _mm512_loadu_ps(Y + i);
            _mm512_storeu_ps(X + i, _mm512_fmadd_ps(va, vy, vx));
        }
    } else if constexpr(std::is_same_v<scalar_t, double>) {
        __m512d va = _mm512_set1_pd(a);
        for (; i + 7 < len; i += 8) {
            __m512d vx = _mm512_loadu_pd(X + i);
            __m512d vy = _mm512_loadu_pd(Y + i);
            _mm512_storeu_pd(X + i, _mm512_fmadd_pd(va, vy, vx));
        }
    }
#endif
    for (; i < len; i++)
        X[i] += a * Y[i];
}

template <typename scalar_t>
inline scalar_t dot_contiguous(
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ Y,
    int len
) {
    int i = 0;
    scalar_t total = scalar_t(0);
#ifdef __AVX512F__
    if constexpr(std::is_same_v<scalar_t, float>) {
        __m512 vacc = _mm512_setzero_ps();
        for (; i + 15 < len; i += 16) {
            __m512 vx = _mm512_loadu_ps(X + i);
            __m512 vy = _mm512_loadu_ps(Y + i);
            vacc = _mm512_fmadd_ps(vx, vy, vacc);
        }
        alignas(64) float tmp[16];
        _mm512_store_ps(tmp, vacc);
        for (int j = 0; j < 16; j++)
            total += tmp[j];
    } else if constexpr(std::is_same_v<scalar_t, double>) {
        __m512d vacc = _mm512_setzero_pd();
        for (; i + 7 < len; i += 8) {
            __m512d vx = _mm512_loadu_pd(X + i);
            __m512d vy = _mm512_loadu_pd(Y + i);
            vacc = _mm512_fmadd_pd(vx, vy, vacc);
        }
        alignas(64) double tmp[8];
        _mm512_store_pd(tmp, vacc);
        for (int j = 0; j < 8; j++)
            total += tmp[j];
    }
#endif
    for (; i < len; i++)
        total += X[i] * Y[i];
    return total;
}

template <typename scalar_t>
inline scalar_t dot_and_add_scaled_contiguous(
    scalar_t* __restrict__ prefix,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ u,
    scalar_t value,
    int len
) {
    int i = 0;
    scalar_t total = scalar_t(0);
#ifdef __AVX512F__
    if constexpr(std::is_same_v<scalar_t, float>) {
        const __m512 vvalue = _mm512_set1_ps(value);
        if (len == 32) {
            const __m512 vp0 = _mm512_loadu_ps(prefix);
            const __m512 vp1 = _mm512_loadu_ps(prefix + 16);
            const __m512 vq0 = _mm512_loadu_ps(q);
            const __m512 vq1 = _mm512_loadu_ps(q + 16);
            const __m512 vu0 = _mm512_loadu_ps(u);
            const __m512 vu1 = _mm512_loadu_ps(u + 16);
            _mm512_storeu_ps(
                prefix,
                _mm512_fmadd_ps(vvalue, vu0, vp0)
            );
            _mm512_storeu_ps(
                prefix + 16,
                _mm512_fmadd_ps(vvalue, vu1, vp1)
            );
            return _mm512_reduce_add_ps(
                _mm512_add_ps(
                    _mm512_mul_ps(vq0, vp0),
                    _mm512_mul_ps(vq1, vp1)
                )
            );
        }
        __m512 vacc = _mm512_setzero_ps();
        for (; i + 15 < len; i += 16) {
            const __m512 vprefix = _mm512_loadu_ps(prefix + i);
            const __m512 vq = _mm512_loadu_ps(q + i);
            const __m512 vu = _mm512_loadu_ps(u + i);
            vacc = _mm512_fmadd_ps(vq, vprefix, vacc);
            _mm512_storeu_ps(
                prefix + i,
                _mm512_fmadd_ps(vvalue, vu, vprefix)
            );
        }
        alignas(64) float tmp[16];
        _mm512_store_ps(tmp, vacc);
        for (int j = 0; j < 16; j++)
            total += tmp[j];
    } else if constexpr(std::is_same_v<scalar_t, double>) {
        const __m512d vvalue = _mm512_set1_pd(value);
        if (len == 32) {
            const __m512d vp0 = _mm512_loadu_pd(prefix);
            const __m512d vp1 = _mm512_loadu_pd(prefix + 8);
            const __m512d vp2 = _mm512_loadu_pd(prefix + 16);
            const __m512d vp3 = _mm512_loadu_pd(prefix + 24);
            const __m512d vq0 = _mm512_loadu_pd(q);
            const __m512d vq1 = _mm512_loadu_pd(q + 8);
            const __m512d vq2 = _mm512_loadu_pd(q + 16);
            const __m512d vq3 = _mm512_loadu_pd(q + 24);
            const __m512d vu0 = _mm512_loadu_pd(u);
            const __m512d vu1 = _mm512_loadu_pd(u + 8);
            const __m512d vu2 = _mm512_loadu_pd(u + 16);
            const __m512d vu3 = _mm512_loadu_pd(u + 24);
            _mm512_storeu_pd(
                prefix,
                _mm512_fmadd_pd(vvalue, vu0, vp0)
            );
            _mm512_storeu_pd(
                prefix + 8,
                _mm512_fmadd_pd(vvalue, vu1, vp1)
            );
            _mm512_storeu_pd(
                prefix + 16,
                _mm512_fmadd_pd(vvalue, vu2, vp2)
            );
            _mm512_storeu_pd(
                prefix + 24,
                _mm512_fmadd_pd(vvalue, vu3, vp3)
            );
            return _mm512_reduce_add_pd(
                _mm512_add_pd(
                    _mm512_add_pd(
                        _mm512_mul_pd(vq0, vp0),
                        _mm512_mul_pd(vq1, vp1)
                    ),
                    _mm512_add_pd(
                        _mm512_mul_pd(vq2, vp2),
                        _mm512_mul_pd(vq3, vp3)
                    )
                )
            );
        }
        __m512d vacc = _mm512_setzero_pd();
        for (; i + 7 < len; i += 8) {
            const __m512d vprefix = _mm512_loadu_pd(prefix + i);
            const __m512d vq = _mm512_loadu_pd(q + i);
            const __m512d vu = _mm512_loadu_pd(u + i);
            vacc = _mm512_fmadd_pd(vq, vprefix, vacc);
            _mm512_storeu_pd(
                prefix + i,
                _mm512_fmadd_pd(vvalue, vu, vprefix)
            );
        }
        alignas(64) double tmp[8];
        _mm512_store_pd(tmp, vacc);
        for (int j = 0; j < 8; j++)
            total += tmp[j];
    }
#endif
    for (; i < len; i++) {
        total += q[i] * prefix[i];
        prefix[i] += value * u[i];
    }
    return total;
}

template <typename scalar_t>
inline void dual_add_scaled_contiguous(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ prefix,
    const scalar_t* __restrict__ input,
    scalar_t output_scale,
    scalar_t prefix_scale,
    int len
) {
    int i = 0;
#ifdef __AVX512F__
    if constexpr(std::is_same_v<scalar_t, float>) {
        const __m512 vo_scale = _mm512_set1_ps(output_scale);
        const __m512 vp_scale = _mm512_set1_ps(prefix_scale);
        for (; i + 15 < len; i += 16) {
            const __m512 vo = _mm512_loadu_ps(output + i);
            const __m512 vp = _mm512_loadu_ps(prefix + i);
            const __m512 vx = _mm512_loadu_ps(input + i);
            _mm512_storeu_ps(
                output + i,
                _mm512_fmadd_ps(vo_scale, vp, vo)
            );
            _mm512_storeu_ps(
                prefix + i,
                _mm512_fmadd_ps(vp_scale, vx, vp)
            );
        }
    } else if constexpr(std::is_same_v<scalar_t, double>) {
        const __m512d vo_scale = _mm512_set1_pd(output_scale);
        const __m512d vp_scale = _mm512_set1_pd(prefix_scale);
        for (; i + 7 < len; i += 8) {
            const __m512d vo = _mm512_loadu_pd(output + i);
            const __m512d vp = _mm512_loadu_pd(prefix + i);
            const __m512d vx = _mm512_loadu_pd(input + i);
            _mm512_storeu_pd(
                output + i,
                _mm512_fmadd_pd(vo_scale, vp, vo)
            );
            _mm512_storeu_pd(
                prefix + i,
                _mm512_fmadd_pd(vp_scale, vx, vp)
            );
        }
    }
#endif
    for (; i < len; i++) {
        output[i] += output_scale * prefix[i];
        prefix[i] += prefix_scale * input[i];
    }
}


// Core single-update implementation. Templated on scalar type so we can
// instantiate for float and double without runtime dispatch inside the
// hot loops.
template <typename scalar_t>
std::tuple<torch::Tensor, torch::Tensor> update_impl(
    torch::Tensor L, torch::Tensor Aw, torch::Tensor z, torch::Tensor w
) {
    const int r = L.size(0);
    auto opts = L.options();

    scalar_t* L_rm = L.data_ptr<scalar_t>();
    scalar_t* z_d = z.data_ptr<scalar_t>();
    std::vector<scalar_t> scratch(5 * r, scalar_t(0));
    scalar_t* cs = scratch.data();
    scalar_t* ss = cs + r;
    scalar_t* row_pad = ss + r;
    scalar_t* col_pad = row_pad + r;
    scalar_t* vw_d = col_pad + r;

    // Phase 1: Rank-1 Cholesky update via paired Givens rotations.
    // Each rotation k zeros z_d[k] using L[k,k], then both L's k-th column
    // and z are updated. Fusing pairs keeps z hot across two rotations.
    int k = 0;
    for (; k + 1 < r; k += 2) {
        scalar_t a0 = L_rm[k * r + k], b0 = z_d[k];
        scalar_t rho0 = std::sqrt(a0 * a0 + b0 * b0);
        scalar_t c0 = a0 / rho0, s0 = b0 / rho0;
        cs[k] = c0; ss[k] = s0;

        for (int i = k; i < r; i++) {
            scalar_t lv = L_rm[i * r + k];
            scalar_t zv = z_d[i];
            L_rm[i * r + k] = c0 * lv + s0 * zv;
            z_d[i] = -s0 * lv + c0 * zv;
        }

        scalar_t a1 = L_rm[(k + 1) * r + (k + 1)], b1 = z_d[k + 1];
        scalar_t rho1 = std::sqrt(a1 * a1 + b1 * b1);
        scalar_t c1 = a1 / rho1;
        scalar_t s1 = b1 / rho1;
        cs[k + 1] = c1;
        ss[k + 1] = s1;

        for (int i = k + 1; i < r; i++) {
            scalar_t lv = L_rm[i * r + (k + 1)];
            scalar_t zv = z_d[i];
            L_rm[i * r + (k + 1)] = c1 * lv + s1 * zv;
            z_d[i] = -s1 * lv + c1 * zv;
        }
    }

    if (k < r) {
        scalar_t a = L_rm[k * r + k], b = z_d[k];
        scalar_t rho = std::sqrt(a * a + b * b);
        scalar_t c = a / rho, s = b / rho;
        cs[k] = c;
        ss[k] = s;

        for (int i = k; i < r; i++) {
            scalar_t lv = L_rm[i * r + k];
            scalar_t zv = z_d[i];
            L_rm[i * r + k] = c * lv + s * zv;
            z_d[i] = -s * lv + c * zv;
        }
    }

    // Phase 3: recover v_z = (L+)^{-1} z from the rotation sequence.
    auto vz = torch::empty({r}, opts);
    scalar_t* vz_d = vz.data_ptr<scalar_t>();
    scalar_t e_last = 1.0;
    for (int kk = 0; kk < r; kk++) {
        vz_d[kk] = ss[kk] * e_last;
        e_last = cs[kk] * e_last;
    }

    // Phase 4: solve (L+) v_w = w by forward substitution.
    const scalar_t* w_d = w.data_ptr<scalar_t>();
    for (int i = 0; i < r; i++) {
        scalar_t sum = w_d[i] - dot_contiguous(&L_rm[i * r], vw_d, i);
        vw_d[i] = sum / L_rm[i*r+i];
    }

    // Phase 2: Apply Q^T [Aw 0; 0 0] Q without materializing the padded
    // (r+1) x (r+1) matrix. Aw_out is the top-left block, while row_pad and
    // col_pad represent the virtual padded row and column.
    auto Aw_out = torch::empty_like(Aw);
    const scalar_t* Aw_in = Aw.data_ptr<scalar_t>();
    scalar_t* Aw_d = Aw_out.data_ptr<scalar_t>();

    // Left sweep: Q_k^T mixes row k with the virtual padded row.
    for (int kk = 0; kk < r - 1; kk += 2) {
        givens_from_input_contiguous(
            &Aw_d[kk * r], &Aw_in[kk * r], row_pad, cs[kk], ss[kk], r
        );
        givens_from_input_contiguous(
            &Aw_d[(kk + 1) * r], &Aw_in[(kk + 1) * r], row_pad,
            cs[kk + 1], ss[kk + 1], r
        );
    }

    if (r % 2 == 1) {
        givens_from_input_contiguous(
            &Aw_d[(r - 1) * r], &Aw_in[(r - 1) * r], row_pad,
            cs[r - 1], ss[r - 1], r
        );
    }

    // Right sweep: Q_k mixes column k with the virtual padded column.
    // Process one row at a time to keep Aw_out accesses contiguous.
    for (int i = 0; i < r; i++) {
        scalar_t cr = col_pad[i];
        scalar_t* row = &Aw_d[i * r];
        int kk = 0;
        for (; kk < r - 1; kk += 2) {
            scalar_t c0 = cs[kk], s0 = ss[kk];
            scalar_t c1 = cs[kk + 1], s1 = ss[kk + 1];

            scalar_t ck0 = row[kk];
            row[kk] = c0 * ck0 + s0 * cr;
            cr = -s0 * ck0 + c0 * cr;

            scalar_t ck1 = row[kk + 1];
            row[kk + 1] = c1 * ck1 + s1 * cr;
            cr = -s1 * ck1 + c1 * cr;
        }

        if (kk < r) {
            scalar_t c = cs[kk], s = ss[kk];
            scalar_t ck = row[kk];
            row[kk] = c * ck + s * cr;
            cr = -s * ck + c * cr;
        }

        col_pad[i] = cr;
        add_scaled_contiguous(row, vw_d, vz_d[i], r);
    }

    return {Aw_out, vz};
}

// Native sequential rank-k path.  Unlike k Python calls to update_impl, this
// path copies each Z/W column into reusable vector storage and updates Aw in
// place.  The left sweep is safe in place: it consumes each source row once,
// and the only state carried between rows is row_pad.  Thus the arithmetic is
// identical to the out-of-place single-update path but no r-by-r temporary is
// allocated or swept solely to exchange buffers.
//
// Workspace layout (all length r):
//   cs | ss | row_pad | col_pad | vw | z_work | w_work
// vz_out has shape (k, r).  PROFILE=false compiles out all clock reads.
template <typename scalar_t, bool PROFILE>
void multi_update_inplace_impl(
    scalar_t* __restrict__ L_rm,
    scalar_t* __restrict__ Aw_d,
    const scalar_t* __restrict__ Z_d,
    const scalar_t* __restrict__ W_d,
    scalar_t* __restrict__ workspace,
    scalar_t* __restrict__ vz_out,
    int r,
    int update_count,
    double* phase_seconds
) {
    using clock = std::chrono::steady_clock;
    scalar_t* cs = workspace;
    scalar_t* ss = cs + r;
    scalar_t* row_pad = ss + r;
    scalar_t* col_pad = row_pad + r;
    scalar_t* vw_d = col_pad + r;
    scalar_t* z_work = vw_d + r;
    scalar_t* w_work = z_work + r;

    for (int update = 0; update < update_count; update++) {
        typename clock::time_point phase_start;
        if constexpr(PROFILE)
            phase_start = clock::now();

        // Z and W are row-major (r, k), so one native copy replaces the two
        // non-contiguous Python column clones used by the legacy loop.
        for (int i = 0; i < r; i++) {
            z_work[i] = Z_d[i * update_count + update];
            w_work[i] = W_d[i * update_count + update];
        }
        std::fill(row_pad, row_pad + r, scalar_t(0));
        std::fill(col_pad, col_pad + r, scalar_t(0));

        if constexpr(PROFILE) {
            phase_seconds[0] += std::chrono::duration<double>(
                clock::now() - phase_start
            ).count();
            phase_start = clock::now();
        }

        // Phase 1: the same Givens dependency graph as update_impl, reordered
        // across independent rows.  For row i, rotations 0..i are still
        // applied in exactly the original order, and rotation i is generated
        // only after its predecessors have transformed z_i.  Traversing the
        // row-major lower triangle by row instead of by column changes no
        // arithmetic dependency but replaces r-strided loads/stores with a
        // contiguous sweep.
        for (int i = 0; i < r; i++) {
            scalar_t zi = z_work[i];
            scalar_t* row = &L_rm[i * r];
            for (int k = 0; k < i; k++) {
                scalar_t lv = row[k];
                scalar_t c = cs[k], s = ss[k];
                row[k] = c * lv + s * zi;
                zi = -s * lv + c * zi;
            }

            scalar_t a = row[i];
            scalar_t rho = std::sqrt(a * a + zi * zi);
            scalar_t c = a / rho;
            scalar_t s = zi / rho;
            cs[i] = c;
            ss[i] = s;
            row[i] = c * a + s * zi;
            z_work[i] = -s * a + c * zi;
        }

        if constexpr(PROFILE) {
            phase_seconds[1] += std::chrono::duration<double>(
                clock::now() - phase_start
            ).count();
            phase_start = clock::now();
        }

        // Phase 2: recover v_z for this update.
        scalar_t* vz_d = vz_out + update * r;
        scalar_t e_last = scalar_t(1);
        for (int kk = 0; kk < r; kk++) {
            vz_d[kk] = ss[kk] * e_last;
            e_last = cs[kk] * e_last;
        }

        if constexpr(PROFILE) {
            phase_seconds[2] += std::chrono::duration<double>(
                clock::now() - phase_start
            ).count();
            phase_start = clock::now();
        }

        // Phase 3: (L+) v_w = w.
        for (int i = 0; i < r; i++) {
            scalar_t sum = w_work[i] - dot_contiguous(
                &L_rm[i * r], vw_d, i
            );
            vw_d[i] = sum / L_rm[i * r + i];
        }

        if constexpr(PROFILE) {
            phase_seconds[3] += std::chrono::duration<double>(
                clock::now() - phase_start
            ).count();
            phase_start = clock::now();
        }

        // Phase 4: Q^T [Aw 0; 0 0], now in place.  A row is never read after
        // it is transformed, so this has exactly the same dependencies as
        // the legacy Aw_in -> Aw_out sweep.
        for (int kk = 0; kk < r - 1; kk += 2) {
            givens_contiguous(
                &Aw_d[kk * r], row_pad, cs[kk], ss[kk], r
            );
            givens_contiguous(
                &Aw_d[(kk + 1) * r], row_pad,
                cs[kk + 1], ss[kk + 1], r
            );
        }
        if (r % 2 == 1) {
            givens_contiguous(
                &Aw_d[(r - 1) * r], row_pad,
                cs[r - 1], ss[r - 1], r
            );
        }

        if constexpr(PROFILE) {
            phase_seconds[4] += std::chrono::duration<double>(
                clock::now() - phase_start
            ).count();
            phase_start = clock::now();
        }

        // Phase 5: right sweep and v_z v_w^T correction, also in place.
        for (int i = 0; i < r; i++) {
            scalar_t cr = col_pad[i];
            scalar_t* row = &Aw_d[i * r];
            int kk = 0;
            for (; kk < r - 1; kk += 2) {
                scalar_t c0 = cs[kk], s0 = ss[kk];
                scalar_t c1 = cs[kk + 1], s1 = ss[kk + 1];

                scalar_t ck0 = row[kk];
                scalar_t updated0 = c0 * ck0 + s0 * cr;
                row[kk] = std::fma(vz_d[i], vw_d[kk], updated0);
                cr = -s0 * ck0 + c0 * cr;

                scalar_t ck1 = row[kk + 1];
                scalar_t updated1 = c1 * ck1 + s1 * cr;
                row[kk + 1] = std::fma(
                    vz_d[i], vw_d[kk + 1], updated1
                );
                cr = -s1 * ck1 + c1 * cr;
            }
            if (kk < r) {
                scalar_t c = cs[kk], s = ss[kk];
                scalar_t ck = row[kk];
                scalar_t updated = c * ck + s * cr;
                row[kk] = std::fma(vz_d[i], vw_d[kk], updated);
                cr = -s * ck + c * cr;
            }
            col_pad[i] = cr;
        }

        if constexpr(PROFILE) {
            phase_seconds[5] += std::chrono::duration<double>(
                clock::now() - phase_start
            ).count();
        }
    }
}

void validate_multi_update_inputs(
    const torch::Tensor& L,
    const torch::Tensor& Aw,
    const torch::Tensor& Z,
    const torch::Tensor& W,
    const torch::Tensor& workspace,
    const torch::Tensor& vz_out
) {
    TORCH_CHECK(
        L.device().is_cpu() && Aw.device().is_cpu()
            && Z.device().is_cpu() && W.device().is_cpu()
            && workspace.device().is_cpu() && vz_out.device().is_cpu(),
        "all multi-update tensors must be on CPU"
    );
    TORCH_CHECK(
        L.scalar_type() == torch::kFloat32
            || L.scalar_type() == torch::kFloat64,
        "multi-update supports only float32 and float64"
    );
    TORCH_CHECK(
        Aw.scalar_type() == L.scalar_type()
            && Z.scalar_type() == L.scalar_type()
            && W.scalar_type() == L.scalar_type()
            && workspace.scalar_type() == L.scalar_type()
            && vz_out.scalar_type() == L.scalar_type(),
        "all multi-update tensors must have the same dtype"
    );
    TORCH_CHECK(
        L.dim() == 2 && L.size(0) == L.size(1),
        "L must be square 2D"
    );
    const int64_t r = L.size(0);
    TORCH_CHECK(Aw.sizes() == L.sizes(), "Aw must match L shape");
    TORCH_CHECK(
        Z.dim() == 2 && Z.size(0) == r && Z.size(1) > 0,
        "Z must have shape (r, k) with k > 0"
    );
    TORCH_CHECK(W.sizes() == Z.sizes(), "W must match Z shape");
    const int64_t update_count = Z.size(1);
    TORCH_CHECK(
        workspace.dim() == 1 && workspace.numel() >= 7 * r,
        "workspace must be a vector with at least 7*r elements"
    );
    TORCH_CHECK(
        vz_out.dim() == 2
            && vz_out.size(0) == update_count
            && vz_out.size(1) == r,
        "vz_out must have shape (k, r)"
    );
    TORCH_CHECK(
        L.is_contiguous() && Aw.is_contiguous()
            && Z.is_contiguous() && W.is_contiguous()
            && workspace.is_contiguous() && vz_out.is_contiguous(),
        "all multi-update tensors must be contiguous"
    );
}

template <bool PROFILE>
std::tuple<torch::Tensor, torch::Tensor> fast_incremental_update_multi_out_impl(
    torch::Tensor L,
    torch::Tensor Aw,
    torch::Tensor Z,
    torch::Tensor W,
    torch::Tensor workspace,
    torch::Tensor vz_out,
    torch::Tensor phase_seconds
) {
    validate_multi_update_inputs(L, Aw, Z, W, workspace, vz_out);
    double* phase_d = nullptr;
    if constexpr(PROFILE) {
        TORCH_CHECK(
            phase_seconds.device().is_cpu()
                && phase_seconds.scalar_type() == torch::kFloat64
                && phase_seconds.is_contiguous()
                && phase_seconds.dim() == 1
                && phase_seconds.numel() >= 6,
            "phase_seconds must be a contiguous CPU float64 vector of length >= 6"
        );
        phase_seconds.zero_();
        phase_d = phase_seconds.data_ptr<double>();
    }

    const int r = static_cast<int>(L.size(0));
    const int update_count = static_cast<int>(Z.size(1));
    if (L.scalar_type() == torch::kFloat64) {
        multi_update_inplace_impl<double, PROFILE>(
            L.data_ptr<double>(),
            Aw.data_ptr<double>(),
            Z.data_ptr<double>(),
            W.data_ptr<double>(),
            workspace.data_ptr<double>(),
            vz_out.data_ptr<double>(),
            r,
            update_count,
            phase_d
        );
    } else {
        multi_update_inplace_impl<float, PROFILE>(
            L.data_ptr<float>(),
            Aw.data_ptr<float>(),
            Z.data_ptr<float>(),
            W.data_ptr<float>(),
            workspace.data_ptr<float>(),
            vz_out.data_ptr<float>(),
            r,
            update_count,
            phase_d
        );
    }
    return {Aw, vz_out};
}

// Allocation-free hot-path entry point.  L and Aw are updated in place;
// workspace and vz_out are caller-owned and reusable across benchmark calls.
std::tuple<torch::Tensor, torch::Tensor> fast_incremental_update_multi_out(
    torch::Tensor L,
    torch::Tensor Aw,
    torch::Tensor Z,
    torch::Tensor W,
    torch::Tensor workspace,
    torch::Tensor vz_out
) {
    return fast_incremental_update_multi_out_impl<false>(
        L, Aw, Z, W, workspace, vz_out, torch::Tensor()
    );
}

// Identical arithmetic with six accumulated native phase timings:
// vector setup, factor update, v_z recovery, forward solve, left sweep,
// and right sweep plus rank-one correction.
std::tuple<torch::Tensor, torch::Tensor> fast_incremental_update_multi_profile_out(
    torch::Tensor L,
    torch::Tensor Aw,
    torch::Tensor Z,
    torch::Tensor W,
    torch::Tensor workspace,
    torch::Tensor vz_out,
    torch::Tensor phase_seconds
) {
    return fast_incremental_update_multi_out_impl<true>(
        L, Aw, Z, W, workspace, vz_out, phase_seconds
    );
}

// Convenience API matching the legacy ownership convention: L follows the
// old contiguous/mutate behavior, while Aw is copied once and returned.
std::tuple<torch::Tensor, torch::Tensor> fast_incremental_update_multi(
    torch::Tensor L,
    torch::Tensor Aw,
    torch::Tensor Z,
    torch::Tensor W
) {
    L = L.contiguous();
    Aw = Aw.contiguous().clone();
    Z = Z.contiguous();
    W = W.contiguous();
    TORCH_CHECK(
        L.dim() == 2 && L.size(0) == L.size(1),
        "L must be square 2D"
    );
    TORCH_CHECK(
        Z.dim() == 2 && Z.size(0) == L.size(0) && Z.size(1) > 0,
        "Z must have shape (r, k) with k > 0"
    );
    auto workspace = torch::empty({7 * L.size(0)}, L.options());
    auto vz_out = torch::empty({Z.size(1), L.size(0)}, L.options());
    return fast_incremental_update_multi_out(
        L, Aw, Z, W, workspace, vz_out
    );
}

// Construct the compact rank-k Cholesky representation and update L in
// place.  For U=L^{-1}Z, let
//
//   H = I + U U^T = C C^T.
//
// With S_i=I+U[:i]^T U[:i], a_i=S_i^{-1}u_i, and
// d_i=sqrt(1+u_i^T a_i), the lower Cholesky factor is
//
//   C_ii=d_i,  C_ij=u_i^T q_j (i>j),  q_j=a_j/d_j.
//
// The product L+ = L C is evaluated rowwise with a k-vector suffix state:
//
//   L+_ij = d_j L_ij
//            + (sum_{ell=j+1}^i L_i,ell u_ell)^T q_j.
//
// This is O(r k^2 + r^2 k), stores only U,Q,d and O(k^2) workspace, and
// produces the ordinary lower-triangular Cholesky factor expected by all
// legacy callers.
template <typename scalar_t, bool PROFILE>
void compact_factor_impl(
    scalar_t* __restrict__ L,
    const scalar_t* __restrict__ U,
    scalar_t* __restrict__ d,
    scalar_t* __restrict__ Q,
    scalar_t* __restrict__ workspace,
    int r,
    int update_count,
    double* phase_seconds
) {
    using clock = std::chrono::steady_clock;
    scalar_t* Sinv = workspace;
    scalar_t* a = Sinv + update_count * update_count;
    scalar_t* suffix = a + update_count;

    std::fill(
        Sinv,
        Sinv + update_count * update_count,
        scalar_t(0)
    );
    for (int p = 0; p < update_count; p++)
        Sinv[p * update_count + p] = scalar_t(1);

    typename clock::time_point phase_start;
    if constexpr(PROFILE)
        phase_start = clock::now();

    for (int i = 0; i < r; i++) {
        const scalar_t* u = &U[i * update_count];
        for (int p = 0; p < update_count; p++) {
            a[p] = dot_contiguous(
                &Sinv[p * update_count],
                u,
                update_count
            );
        }
        scalar_t d_squared = scalar_t(1)
            + dot_contiguous(u, a, update_count);
        scalar_t diagonal = std::sqrt(d_squared);
        d[i] = diagonal;
        scalar_t* q = &Q[i * update_count];
        for (int p = 0; p < update_count; p++)
            q[p] = a[p] / diagonal;

        scalar_t inverse_d_squared = scalar_t(1) / d_squared;
        for (int p = 0; p < update_count; p++) {
            scalar_t coefficient = -a[p] * inverse_d_squared;
            add_scaled_contiguous(
                &Sinv[p * update_count],
                a,
                coefficient,
                update_count
            );
        }
    }

    if constexpr(PROFILE) {
        phase_seconds[0] += std::chrono::duration<double>(
            clock::now() - phase_start
        ).count();
        phase_start = clock::now();
    }

    for (int i = 0; i < r; i++) {
        std::fill(suffix, suffix + update_count, scalar_t(0));
        scalar_t* row = &L[i * r];
        for (int j = i; j >= 0; j--) {
            scalar_t original = row[j];
            row[j] = d[j] * original + dot_contiguous(
                suffix,
                &Q[j * update_count],
                update_count
            );
            add_scaled_contiguous(
                suffix,
                &U[j * update_count],
                original,
                update_count
            );
        }
    }

    if constexpr(PROFILE) {
        phase_seconds[1] += std::chrono::duration<double>(
            clock::now() - phase_start
        ).count();
    }
}

void validate_compact_factor_inputs(
    const torch::Tensor& L,
    const torch::Tensor& U,
    const torch::Tensor& d,
    const torch::Tensor& Q,
    const torch::Tensor& workspace
) {
    TORCH_CHECK(
        L.device().is_cpu() && U.device().is_cpu()
            && d.device().is_cpu() && Q.device().is_cpu()
            && workspace.device().is_cpu(),
        "all compact-factor tensors must be on CPU"
    );
    TORCH_CHECK(
        L.scalar_type() == torch::kFloat32
            || L.scalar_type() == torch::kFloat64,
        "compact-factor supports only float32 and float64"
    );
    TORCH_CHECK(
        U.scalar_type() == L.scalar_type()
            && d.scalar_type() == L.scalar_type()
            && Q.scalar_type() == L.scalar_type()
            && workspace.scalar_type() == L.scalar_type(),
        "all compact-factor tensors must have the same dtype"
    );
    TORCH_CHECK(
        L.dim() == 2 && L.size(0) == L.size(1),
        "L must be square 2D"
    );
    const int64_t r = L.size(0);
    TORCH_CHECK(
        U.dim() == 2 && U.size(0) == r && U.size(1) > 0,
        "U must have shape (r, k) with k > 0"
    );
    const int64_t update_count = U.size(1);
    TORCH_CHECK(
        d.dim() == 1 && d.size(0) == r,
        "d must have shape (r,)"
    );
    TORCH_CHECK(Q.sizes() == U.sizes(), "Q must match U shape");
    TORCH_CHECK(
        workspace.dim() == 1
            && workspace.numel()
                >= update_count * update_count + 2 * update_count,
        "workspace must have at least k*k+2*k elements"
    );
    TORCH_CHECK(
        L.is_contiguous() && U.is_contiguous() && d.is_contiguous()
            && Q.is_contiguous() && workspace.is_contiguous(),
        "all compact-factor tensors must be contiguous"
    );
}

template <bool PROFILE>
torch::Tensor fast_compact_factor_out_impl(
    torch::Tensor L,
    torch::Tensor U,
    torch::Tensor d,
    torch::Tensor Q,
    torch::Tensor workspace,
    torch::Tensor phase_seconds
) {
    validate_compact_factor_inputs(L, U, d, Q, workspace);
    double* phase_d = nullptr;
    if constexpr(PROFILE) {
        TORCH_CHECK(
            phase_seconds.device().is_cpu()
                && phase_seconds.scalar_type() == torch::kFloat64
                && phase_seconds.is_contiguous()
                && phase_seconds.dim() == 1
                && phase_seconds.numel() >= 2,
            "phase_seconds must be a contiguous CPU float64 vector of length >= 2"
        );
        phase_seconds.zero_();
        phase_d = phase_seconds.data_ptr<double>();
    }
    const int r = static_cast<int>(L.size(0));
    const int update_count = static_cast<int>(U.size(1));
    if (L.scalar_type() == torch::kFloat64) {
        compact_factor_impl<double, PROFILE>(
            L.data_ptr<double>(),
            U.data_ptr<double>(),
            d.data_ptr<double>(),
            Q.data_ptr<double>(),
            workspace.data_ptr<double>(),
            r,
            update_count,
            phase_d
        );
    } else {
        compact_factor_impl<float, PROFILE>(
            L.data_ptr<float>(),
            U.data_ptr<float>(),
            d.data_ptr<float>(),
            Q.data_ptr<float>(),
            workspace.data_ptr<float>(),
            r,
            update_count,
            phase_d
        );
    }
    return L;
}

torch::Tensor fast_compact_factor_out(
    torch::Tensor L,
    torch::Tensor U,
    torch::Tensor d,
    torch::Tensor Q,
    torch::Tensor workspace
) {
    return fast_compact_factor_out_impl<false>(
        L, U, d, Q, workspace, torch::Tensor()
    );
}

torch::Tensor fast_compact_factor_profile_out(
    torch::Tensor L,
    torch::Tensor U,
    torch::Tensor d,
    torch::Tensor Q,
    torch::Tensor workspace,
    torch::Tensor phase_seconds
) {
    return fast_compact_factor_out_impl<true>(
        L, U, d, Q, workspace, phase_seconds
    );
}

// Apply the compact inverse factor P=C^{-1} to a caller-supplied r-by-n
// matrix without materializing P.  For row i,
//
//   (PX)_i = X_i/d_i - q_i^T (U[:i]^T X[:i]).
//
// The k-by-n prefix is kept in workspace.  This path is especially useful
// for n=k: P U and P Y let the rank-k correction be accumulated only once,
// directly into the final dense output.
template <typename scalar_t>
void compact_apply_left_impl(
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ U,
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ d,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ prefix,
    int r,
    int update_count,
    int columns
) {
    std::fill(
        prefix,
        prefix + update_count * columns,
        scalar_t(0)
    );
    for (int i = 0; i < r; i++) {
        const scalar_t* x = &X[i * columns];
        scalar_t* out = &output[i * columns];
        const scalar_t inverse_diagonal = scalar_t(1) / d[i];
        for (int column = 0; column < columns; column++)
            out[column] = x[column] * inverse_diagonal;

        const scalar_t* q = &Q[i * update_count];
        const scalar_t* u = &U[i * update_count];
        for (int p = 0; p < update_count; p++) {
            scalar_t* prefix_row = &prefix[p * columns];
            dual_add_scaled_contiguous(
                out,
                prefix_row,
                x,
                -q[p],
                u[p],
                columns
            );
        }
    }
}

torch::Tensor fast_compact_apply_left_out(
    torch::Tensor X,
    torch::Tensor U,
    torch::Tensor Q,
    torch::Tensor d,
    torch::Tensor output,
    torch::Tensor workspace
) {
    TORCH_CHECK(
        X.device().is_cpu() && U.device().is_cpu()
            && Q.device().is_cpu() && d.device().is_cpu()
            && output.device().is_cpu() && workspace.device().is_cpu(),
        "all compact-apply tensors must be on CPU"
    );
    TORCH_CHECK(
        X.scalar_type() == torch::kFloat32
            || X.scalar_type() == torch::kFloat64,
        "compact apply supports only float32 and float64"
    );
    TORCH_CHECK(
        U.scalar_type() == X.scalar_type()
            && Q.scalar_type() == X.scalar_type()
            && d.scalar_type() == X.scalar_type()
            && output.scalar_type() == X.scalar_type()
            && workspace.scalar_type() == X.scalar_type(),
        "all compact-apply tensors must have the same dtype"
    );
    TORCH_CHECK(X.dim() == 2 && X.size(0) > 0 && X.size(1) > 0,
                "X must have shape (r, n) with r,n > 0");
    const int64_t r = X.size(0);
    const int64_t columns = X.size(1);
    TORCH_CHECK(
        U.dim() == 2 && U.size(0) == r && U.size(1) > 0,
        "U must have shape (r, k) with k > 0"
    );
    const int64_t update_count = U.size(1);
    TORCH_CHECK(Q.sizes() == U.sizes(), "Q must match U shape");
    TORCH_CHECK(d.dim() == 1 && d.size(0) == r, "d must have shape (r,)");
    TORCH_CHECK(output.sizes() == X.sizes(), "output must match X shape");
    TORCH_CHECK(
        workspace.dim() == 1
            && workspace.numel() >= update_count * columns,
        "workspace must have at least k*n elements"
    );
    TORCH_CHECK(
        X.is_contiguous() && U.is_contiguous() && Q.is_contiguous()
            && d.is_contiguous() && output.is_contiguous()
            && workspace.is_contiguous(),
        "all compact-apply tensors must be contiguous"
    );
    TORCH_CHECK(
        X.data_ptr() != output.data_ptr(),
        "compact apply does not support input/output aliasing"
    );

    if (X.scalar_type() == torch::kFloat64) {
        compact_apply_left_impl<double>(
            X.data_ptr<double>(),
            U.data_ptr<double>(),
            Q.data_ptr<double>(),
            d.data_ptr<double>(),
            output.data_ptr<double>(),
            workspace.data_ptr<double>(),
            static_cast<int>(r),
            static_cast<int>(update_count),
            static_cast<int>(columns)
        );
    } else {
        compact_apply_left_impl<float>(
            X.data_ptr<float>(),
            U.data_ptr<float>(),
            Q.data_ptr<float>(),
            d.data_ptr<float>(),
            output.data_ptr<float>(),
            workspace.data_ptr<float>(),
            static_cast<int>(r),
            static_cast<int>(update_count),
            static_cast<int>(columns)
        );
    }
    return output;
}

// Apply the same compact inverse factor from the right without transposing a
// dense r-by-r operand.  Since P=C^{-1} is lower triangular,
//
//   (X P^T)_{a,i}
//     = X_{a,i}/d_i - q_i^T sum_{j<i} u_j X_{a,j}.
//
// Each row of X is therefore an independent forward recurrence with only a
// k-vector prefix.  This removes the two dense transpose/copy passes that
// were previously needed to reuse the left-application routine.
template <typename scalar_t>
void compact_apply_right_impl(
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ U,
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ inverse_d,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ prefix,
    int r,
    int update_count
) {
    for (int row = 0; row < r; row++) {
        std::fill(prefix, prefix + update_count, scalar_t(0));
        const scalar_t* x_row = &X[row * r];
        scalar_t* output_row = &output[row * r];
        for (int i = 0; i < r; i++) {
            const scalar_t value = x_row[i];
            output_row[i] = value * inverse_d[i]
                - dot_and_add_scaled_contiguous(
                prefix,
                &Q[i * update_count],
                &U[i * update_count],
                value,
                update_count
            );
        }
    }
}

torch::Tensor fast_compact_apply_right_out(
    torch::Tensor X,
    torch::Tensor U,
    torch::Tensor Q,
    torch::Tensor d,
    torch::Tensor output,
    torch::Tensor workspace
) {
    TORCH_CHECK(
        X.device().is_cpu() && U.device().is_cpu()
            && Q.device().is_cpu() && d.device().is_cpu()
            && output.device().is_cpu() && workspace.device().is_cpu(),
        "all compact-right-apply tensors must be on CPU"
    );
    TORCH_CHECK(
        X.scalar_type() == torch::kFloat32
            || X.scalar_type() == torch::kFloat64,
        "compact right apply supports only float32 and float64"
    );
    TORCH_CHECK(
        U.scalar_type() == X.scalar_type()
            && Q.scalar_type() == X.scalar_type()
            && d.scalar_type() == X.scalar_type()
            && output.scalar_type() == X.scalar_type()
            && workspace.scalar_type() == X.scalar_type(),
        "all compact-right-apply tensors must have the same dtype"
    );
    TORCH_CHECK(
        X.dim() == 2 && X.size(0) > 0 && X.size(0) == X.size(1),
        "X must be a nonempty square matrix"
    );
    const int64_t r = X.size(0);
    TORCH_CHECK(
        U.dim() == 2 && U.size(0) == r && U.size(1) > 0,
        "U must have shape (r, k) with k > 0"
    );
    const int64_t update_count = U.size(1);
    TORCH_CHECK(Q.sizes() == U.sizes(), "Q must match U shape");
    TORCH_CHECK(d.dim() == 1 && d.size(0) == r, "d must have shape (r,)");
    TORCH_CHECK(output.sizes() == X.sizes(), "output must match X shape");
    TORCH_CHECK(
        workspace.dim() == 1
            && workspace.numel() >= r + update_count,
        "workspace must have at least r+k elements"
    );
    TORCH_CHECK(
        X.is_contiguous() && U.is_contiguous() && Q.is_contiguous()
            && d.is_contiguous() && output.is_contiguous()
            && workspace.is_contiguous(),
        "all compact-right-apply tensors must be contiguous"
    );
    TORCH_CHECK(
        X.data_ptr() != output.data_ptr(),
        "compact right apply does not support input/output aliasing"
    );

    if (X.scalar_type() == torch::kFloat64) {
        double* workspace_data = workspace.data_ptr<double>();
        const double* d_data = d.data_ptr<double>();
        for (int64_t i = 0; i < r; i++)
            workspace_data[i] = 1.0 / d_data[i];
        compact_apply_right_impl<double>(
            X.data_ptr<double>(),
            U.data_ptr<double>(),
            Q.data_ptr<double>(),
            workspace_data,
            output.data_ptr<double>(),
            workspace_data + r,
            static_cast<int>(r),
            static_cast<int>(update_count)
        );
    } else {
        float* workspace_data = workspace.data_ptr<float>();
        const float* d_data = d.data_ptr<float>();
        for (int64_t i = 0; i < r; i++)
            workspace_data[i] = 1.0f / d_data[i];
        compact_apply_right_impl<float>(
            X.data_ptr<float>(),
            U.data_ptr<float>(),
            Q.data_ptr<float>(),
            workspace_data,
            output.data_ptr<float>(),
            workspace_data + r,
            static_cast<int>(r),
            static_cast<int>(update_count)
        );
    }
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> fast_incremental_update(
    torch::Tensor L, torch::Tensor Aw, torch::Tensor z, torch::Tensor w
) {
    L = L.contiguous(); Aw = Aw.contiguous();
    z = z.contiguous(); w = w.contiguous();
    TORCH_CHECK(L.dim() == 2 && L.size(0) == L.size(1), "L must be square 2D");
    TORCH_CHECK(Aw.sizes() == L.sizes(), "Aw must match L shape");
    TORCH_CHECK(z.dim() == 1 && z.size(0) == L.size(0), "z must match L rows");
    TORCH_CHECK(w.dim() == 1 && w.size(0) == L.size(0), "w must match L rows");
    if (L.scalar_type() == torch::kFloat64)
        return update_impl<double>(L, Aw, z, w);
    else
        return update_impl<float>(L, Aw, z, w);
}

// Batched entry point. Each slice t is an independent update, so we run
// them in parallel across threads. Inputs are (T, r, r) and (T, r).
std::tuple<torch::Tensor, torch::Tensor> fast_incremental_update_batched(
    torch::Tensor L, torch::Tensor Aw, torch::Tensor z, torch::Tensor w
) {
    L = L.contiguous(); Aw = Aw.contiguous();
    z = z.contiguous(); w = w.contiguous();
    const int T = L.size(0);
    const int r = L.size(1);
    auto opts = L.options();
    auto vz_out = torch::zeros({T, r}, opts);
    auto Aw_out = torch::zeros({T, r, r}, opts);

    const bool is_double = (L.scalar_type() == torch::kFloat64);

    at::parallel_for(0, T, 1, [&](int begin, int end) {
        for (int t = begin; t < end; t++) {
            auto L_t = L[t].clone();
            auto z_t = z[t].clone();
            auto Aw_t = Aw[t].contiguous();
            auto w_t = w[t].contiguous();
            // NOTE: the two update_impl instantiations return distinct
            // std::tuple types, so a ternary over them does not compile.
            // Branch on dtype with an if/else instead.
            if (is_double) {
                auto [aw, vz] = update_impl<double>(L_t, Aw_t, z_t, w_t);
                Aw_out[t] = aw;
                vz_out[t] = vz;
            } else {
                auto [aw, vz] = update_impl<float>(L_t, Aw_t, z_t, w_t);
                Aw_out[t] = aw;
                vz_out[t] = vz;
            }
        }
    });
    return {Aw_out, vz_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_incremental_update", &fast_incremental_update);
    m.def("fast_incremental_update_batched", &fast_incremental_update_batched);
    m.def(
        "fast_incremental_update_multi",
        &fast_incremental_update_multi
    );
    m.def(
        "fast_incremental_update_multi_out",
        &fast_incremental_update_multi_out
    );
    m.def(
        "fast_incremental_update_multi_profile_out",
        &fast_incremental_update_multi_profile_out
    );
    m.def("fast_compact_factor_out", &fast_compact_factor_out);
    m.def(
        "fast_compact_factor_profile_out",
        &fast_compact_factor_profile_out
    );
    m.def(
        "fast_compact_apply_left_out",
        &fast_compact_apply_left_out
    );
    m.def(
        "fast_compact_apply_right_out",
        &fast_compact_apply_right_out
    );
}
