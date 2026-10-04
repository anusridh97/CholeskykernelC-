#include <torch/extension.h>
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
    std::vector<scalar_t> scratch(6 * r, scalar_t(0));
    scalar_t* cs = scratch.data();
    scalar_t* ss = cs + r;
    scalar_t* row_pad = ss + r;
    scalar_t* col_pad = row_pad + r;
    scalar_t* vw_d = col_pad + r;
    scalar_t* z_d = vw_d + r;
    std::copy_n(z.data_ptr<scalar_t>(), r, z_d);

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


void check_inputs(torch::Tensor L, torch::Tensor Aw, torch::Tensor z, torch::Tensor w, bool batched) {
    TORCH_CHECK(L.is_contiguous(), "L must be row-major contiguous; call contiguous() before storing the state");
    TORCH_CHECK(L.dim() == (batched ? 3 : 2), "incorrect L dimensions");
    TORCH_CHECK(L.size(-1) >= 1 && L.size(-2) == L.size(-1), "L must be nonempty and square");
    TORCH_CHECK(!batched || L.size(0) >= 1, "batch must be nonempty");
    TORCH_CHECK(L.scalar_type() == torch::kFloat32 || L.scalar_type() == torch::kFloat64, "CPU update requires float32 or float64");
    for (const auto& tensor : {L, Aw, z, w}) {
        TORCH_CHECK(tensor.device().is_cpu(), "all inputs must be on CPU");
        TORCH_CHECK(tensor.scalar_type() == L.scalar_type(), "all inputs must share dtype");
    }
    TORCH_CHECK(Aw.sizes() == L.sizes(), "Aw must match L");
    TORCH_CHECK(z.dim() == L.dim()-1 && w.sizes() == z.sizes(), "z and w shapes must match");
    TORCH_CHECK(z.size(-1) == L.size(-1), "vector rank mismatch");
    TORCH_CHECK(!batched || z.size(0) == L.size(0), "vector batch mismatch");
}

std::tuple<torch::Tensor, torch::Tensor> fast_incremental_update(
    torch::Tensor L, torch::Tensor Aw, torch::Tensor z, torch::Tensor w
) {
    torch::NoGradGuard no_grad;
    check_inputs(L, Aw, z, w, false);
    Aw = Aw.contiguous();
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
    torch::NoGradGuard no_grad;
    check_inputs(L, Aw, z, w, true);
    Aw = Aw.contiguous();
    z = z.contiguous(); w = w.contiguous();
    const int T = L.size(0);
    const int r = L.size(1);
    auto opts = L.options();
    auto vz_out = torch::zeros({T, r}, opts);
    auto Aw_out = torch::zeros({T, r, r}, opts);

    const bool is_double = (L.scalar_type() == torch::kFloat64);

    at::parallel_for(0, T, 1, [&](int begin, int end) {
        torch::NoGradGuard thread_no_grad;
        for (int t = begin; t < end; t++) {
            auto L_t = L[t];
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
}
