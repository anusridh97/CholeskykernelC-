#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <vector>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

namespace {

void check_cpu_float_matrix(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 ||
                x.scalar_type() == torch::kFloat64,
                name, " must have dtype float32 or float64");
    TORCH_CHECK(x.dim() == 2, name, " must be a matrix, got ", x.sizes());
}

void check_square_state(
    const torch::Tensor& L,
    const torch::Tensor& Aw,
    const torch::Tensor& z,
    const torch::Tensor& w
) {
    check_cpu_float_matrix(L, "L");
    TORCH_CHECK(L.size(0) == L.size(1), "L must be square, got ", L.sizes());
    const int64_t r = L.size(0);
    TORCH_CHECK(Aw.device() == L.device() && Aw.scalar_type() == L.scalar_type(),
                "Aw must match L's device and dtype");
    TORCH_CHECK(Aw.dim() == 2 && Aw.size(0) == r && Aw.size(1) == r,
                "Aw must have shape (", r, ", ", r, "), got ", Aw.sizes());
    for (const auto& item : std::vector<std::pair<const torch::Tensor*, const char*>>{
             {&z, "z"}, {&w, "w"}}) {
        const auto& x = *item.first;
        TORCH_CHECK(x.device() == L.device() && x.scalar_type() == L.scalar_type(),
                    item.second, " must match L's device and dtype");
        TORCH_CHECK(x.dim() == 1 && x.size(0) == r,
                    item.second, " must have shape (", r, "), got ", x.sizes());
    }
}

template <typename scalar_t>
inline void positive_givens_from_input_contiguous(
    scalar_t* __restrict__ x,
    const scalar_t* __restrict__ x_in,
    scalar_t* __restrict__ y,
    scalar_t c,
    scalar_t s,
    int64_t len
) {
    int64_t i = 0;
#ifdef __AVX512F__
    if constexpr (std::is_same_v<scalar_t, float>) {
        const __m512 vc = _mm512_set1_ps(c);
        const __m512 vs = _mm512_set1_ps(s);
        const __m512 vms = _mm512_set1_ps(-s);
        for (; i + 15 < len; i += 16) {
            const __m512 vx = _mm512_loadu_ps(x_in + i);
            const __m512 vy = _mm512_loadu_ps(y + i);
            const __m512 nx =
                _mm512_fmadd_ps(vc, vx, _mm512_mul_ps(vs, vy));
            const __m512 ny =
                _mm512_fmadd_ps(vms, vx, _mm512_mul_ps(vc, vy));
            _mm512_storeu_ps(x + i, nx);
            _mm512_storeu_ps(y + i, ny);
        }
    } else if constexpr (std::is_same_v<scalar_t, double>) {
        const __m512d vc = _mm512_set1_pd(c);
        const __m512d vs = _mm512_set1_pd(s);
        const __m512d vms = _mm512_set1_pd(-s);
        for (; i + 7 < len; i += 8) {
            const __m512d vx = _mm512_loadu_pd(x_in + i);
            const __m512d vy = _mm512_loadu_pd(y + i);
            const __m512d nx =
                _mm512_fmadd_pd(vc, vx, _mm512_mul_pd(vs, vy));
            const __m512d ny =
                _mm512_fmadd_pd(vms, vx, _mm512_mul_pd(vc, vy));
            _mm512_storeu_pd(x + i, nx);
            _mm512_storeu_pd(y + i, ny);
        }
    }
#endif
    for (; i < len; ++i) {
        const scalar_t xv = x_in[i];
        const scalar_t yv = y[i];
        x[i] = c * xv + s * yv;
        y[i] = -s * xv + c * yv;
    }
}

template <typename scalar_t>
inline void positive_add_scaled_contiguous(
    scalar_t* __restrict__ x,
    const scalar_t* __restrict__ y,
    scalar_t a,
    int64_t len
) {
    int64_t i = 0;
#ifdef __AVX512F__
    if constexpr (std::is_same_v<scalar_t, float>) {
        const __m512 va = _mm512_set1_ps(a);
        for (; i + 15 < len; i += 16) {
            const __m512 vx = _mm512_loadu_ps(x + i);
            const __m512 vy = _mm512_loadu_ps(y + i);
            _mm512_storeu_ps(x + i, _mm512_fmadd_ps(va, vy, vx));
        }
    } else if constexpr (std::is_same_v<scalar_t, double>) {
        const __m512d va = _mm512_set1_pd(a);
        for (; i + 7 < len; i += 8) {
            const __m512d vx = _mm512_loadu_pd(x + i);
            const __m512d vy = _mm512_loadu_pd(y + i);
            _mm512_storeu_pd(x + i, _mm512_fmadd_pd(va, vy, vx));
        }
    }
#endif
    for (; i < len; ++i) {
        x[i] += a * y[i];
    }
}

template <typename scalar_t>
inline scalar_t positive_dot_contiguous(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ y,
    int64_t len
) {
    int64_t i = 0;
    scalar_t total = scalar_t(0);
#ifdef __AVX512F__
    if constexpr (std::is_same_v<scalar_t, float>) {
        __m512 acc = _mm512_setzero_ps();
        for (; i + 15 < len; i += 16) {
            const __m512 vx = _mm512_loadu_ps(x + i);
            const __m512 vy = _mm512_loadu_ps(y + i);
            acc = _mm512_fmadd_ps(vx, vy, acc);
        }
        alignas(64) float tmp[16];
        _mm512_store_ps(tmp, acc);
        for (int j = 0; j < 16; ++j) {
            total += tmp[j];
        }
    } else if constexpr (std::is_same_v<scalar_t, double>) {
        __m512d acc = _mm512_setzero_pd();
        for (; i + 7 < len; i += 8) {
            const __m512d vx = _mm512_loadu_pd(x + i);
            const __m512d vy = _mm512_loadu_pd(y + i);
            acc = _mm512_fmadd_pd(vx, vy, acc);
        }
        alignas(64) double tmp[8];
        _mm512_store_pd(tmp, acc);
        for (int j = 0; j < 8; ++j) {
            total += tmp[j];
        }
    }
#endif
    for (; i < len; ++i) {
        total += x[i] * y[i];
    }
    return total;
}

// Fast path for the common positive metric/operator update.  It implements
// the same Q^T [Aw 0; 0 0] Q transport as signed_update_impl, but represents
// the padded row and column virtually instead of allocating an (r+1)^2
// matrix.  L stays row-major and the contiguous O(r) sweeps vectorize.
template <typename scalar_t>
std::tuple<torch::Tensor, torch::Tensor> positive_update_impl(
    torch::Tensor L,
    const torch::Tensor& Aw,
    torch::Tensor z,
    const torch::Tensor& w
) {
    const int64_t r = L.size(0);
    const auto opts = L.options();

    scalar_t* L_rm = L.data_ptr<scalar_t>();
    scalar_t* z_d = z.data_ptr<scalar_t>();
    std::vector<scalar_t> scratch(
        static_cast<size_t>(5 * r), scalar_t(0)
    );
    scalar_t* cs = scratch.data();
    scalar_t* ss = cs + r;
    scalar_t* row_pad = ss + r;
    scalar_t* col_pad = row_pad + r;
    scalar_t* vw_d = col_pad + r;

    // Rank-one Cholesky update via paired Givens rotations.
    int64_t k = 0;
    for (; k + 1 < r; k += 2) {
        const scalar_t a0 = L_rm[k * r + k];
        const scalar_t b0 = z_d[k];
        const scalar_t rho0 = std::hypot(a0, b0);
        TORCH_CHECK(rho0 > scalar_t(0), "zero Givens norm at k=", k);
        const scalar_t c0 = a0 / rho0;
        const scalar_t s0 = b0 / rho0;
        cs[k] = c0;
        ss[k] = s0;

        for (int64_t i = k; i < r; ++i) {
            const scalar_t lv = L_rm[i * r + k];
            const scalar_t zv = z_d[i];
            L_rm[i * r + k] = c0 * lv + s0 * zv;
            z_d[i] = -s0 * lv + c0 * zv;
        }

        const scalar_t a1 = L_rm[(k + 1) * r + (k + 1)];
        const scalar_t b1 = z_d[k + 1];
        const scalar_t rho1 = std::hypot(a1, b1);
        TORCH_CHECK(rho1 > scalar_t(0), "zero Givens norm at k=", k + 1);
        const scalar_t c1 = a1 / rho1;
        const scalar_t s1 = b1 / rho1;
        cs[k + 1] = c1;
        ss[k + 1] = s1;

        for (int64_t i = k + 1; i < r; ++i) {
            const scalar_t lv = L_rm[i * r + (k + 1)];
            const scalar_t zv = z_d[i];
            L_rm[i * r + (k + 1)] = c1 * lv + s1 * zv;
            z_d[i] = -s1 * lv + c1 * zv;
        }
    }

    if (k < r) {
        const scalar_t a = L_rm[k * r + k];
        const scalar_t b = z_d[k];
        const scalar_t rho = std::hypot(a, b);
        TORCH_CHECK(rho > scalar_t(0), "zero Givens norm at k=", k);
        const scalar_t c = a / rho;
        const scalar_t s = b / rho;
        cs[k] = c;
        ss[k] = s;

        for (int64_t i = k; i < r; ++i) {
            const scalar_t lv = L_rm[i * r + k];
            const scalar_t zv = z_d[i];
            L_rm[i * r + k] = c * lv + s * zv;
            z_d[i] = -s * lv + c * zv;
        }
    }

    // The final column of Q gives (L+)^{-1} z.
    auto vz = torch::empty({r}, opts);
    scalar_t* vz_d = vz.data_ptr<scalar_t>();
    scalar_t e_last = scalar_t(1);
    for (int64_t kk = 0; kk < r; ++kk) {
        vz_d[kk] = ss[kk] * e_last;
        e_last = cs[kk] * e_last;
    }

    // Forward solve (L+) v_w = w.
    const scalar_t* w_d = w.data_ptr<scalar_t>();
    for (int64_t i = 0; i < r; ++i) {
        const scalar_t sum =
            w_d[i] - positive_dot_contiguous(&L_rm[i * r], vw_d, i);
        vw_d[i] = sum / L_rm[i * r + i];
    }

    auto Aw_out = torch::empty_like(Aw);
    const scalar_t* Aw_in = Aw.data_ptr<scalar_t>();
    scalar_t* Aw_d = Aw_out.data_ptr<scalar_t>();

    // Left sweep: Q_k^T mixes row k with the virtual padded row.
    for (int64_t kk = 0; kk < r - 1; kk += 2) {
        positive_givens_from_input_contiguous(
            &Aw_d[kk * r], &Aw_in[kk * r], row_pad,
            cs[kk], ss[kk], r
        );
        positive_givens_from_input_contiguous(
            &Aw_d[(kk + 1) * r], &Aw_in[(kk + 1) * r], row_pad,
            cs[kk + 1], ss[kk + 1], r
        );
    }
    if (r % 2 == 1) {
        positive_givens_from_input_contiguous(
            &Aw_d[(r - 1) * r], &Aw_in[(r - 1) * r], row_pad,
            cs[r - 1], ss[r - 1], r
        );
    }

    // Right sweep: Q_k mixes column k with the virtual padded column.
    // Walking each row contiguously avoids a strided O(r^2) traversal.
    for (int64_t i = 0; i < r; ++i) {
        scalar_t cr = col_pad[i];
        scalar_t* row = &Aw_d[i * r];
        int64_t kk = 0;
        for (; kk < r - 1; kk += 2) {
            const scalar_t c0 = cs[kk];
            const scalar_t s0 = ss[kk];
            const scalar_t c1 = cs[kk + 1];
            const scalar_t s1 = ss[kk + 1];

            const scalar_t ck0 = row[kk];
            row[kk] = c0 * ck0 + s0 * cr;
            cr = -s0 * ck0 + c0 * cr;

            const scalar_t ck1 = row[kk + 1];
            row[kk + 1] = c1 * ck1 + s1 * cr;
            cr = -s1 * ck1 + c1 * cr;
        }
        if (kk < r) {
            const scalar_t c = cs[kk];
            const scalar_t s = ss[kk];
            const scalar_t ck = row[kk];
            row[kk] = c * ck + s * cr;
            cr = -s * ck + c * cr;
        }

        col_pad[i] = cr;
        positive_add_scaled_contiguous(row, vw_d, vz_d[i], r);
    }

    return {Aw_out, vz};
}

template <typename scalar_t>
scalar_t forward_solve_leverage(
    const scalar_t* L,
    const scalar_t* z,
    int64_t r,
    std::vector<scalar_t>& y
) {
    scalar_t tau = scalar_t(0);
    y.assign(static_cast<size_t>(r), scalar_t(0));
    for (int64_t i = 0; i < r; ++i) {
        scalar_t acc = z[i];
        for (int64_t j = 0; j < i; ++j) {
            acc -= L[i * r + j] * y[static_cast<size_t>(j)];
        }
        const scalar_t diag = L[i * r + i];
        TORCH_CHECK(diag > scalar_t(0), "L must have a positive diagonal");
        const scalar_t yi = acc / diag;
        y[static_cast<size_t>(i)] = yi;
        tau += yi * yi;
    }
    return tau;
}

template <typename scalar_t>
std::tuple<torch::Tensor, torch::Tensor> signed_update_impl(
    torch::Tensor L,
    const torch::Tensor& Aw,
    torch::Tensor z,
    const torch::Tensor& w,
    int64_t metric_sign,
    int64_t operator_sign,
    double downdate_tol
) {
    const int64_t r = L.size(0);
    const int64_t rp = r + 1;
    const auto opts = L.options();

    scalar_t* L_rm = L.data_ptr<scalar_t>();
    scalar_t* z_d = z.data_ptr<scalar_t>();
    const scalar_t* w_d = w.data_ptr<scalar_t>();

    TORCH_CHECK(metric_sign == 1 || metric_sign == -1,
                "metric_sign must be +1 or -1");
    TORCH_CHECK(operator_sign == -1 || operator_sign == 0 || operator_sign == 1,
                "operator_sign must be -1, 0, or +1");

    if (metric_sign < 0) {
        std::vector<scalar_t> y;
        const scalar_t tau = forward_solve_leverage(L_rm, z_d, r, y);
        TORCH_CHECK(
            static_cast<double>(tau) < 1.0 - downdate_tol,
            "inadmissible or poorly conditioned Cholesky downdate: ",
            "z^T G^{-1} z = ", static_cast<double>(tau),
            " must be below ", 1.0 - downdate_tol
        );
    }

    // Column-major scratch gives contiguous access to the active Cholesky
    // column while the public tensor remains ordinary row-major contiguous.
    std::vector<scalar_t> Lcm(static_cast<size_t>(r * r));
    for (int64_t i = 0; i < r; ++i) {
        for (int64_t j = 0; j < r; ++j) {
            Lcm[static_cast<size_t>(j * r + i)] = L_rm[i * r + j];
        }
    }

    std::vector<scalar_t> cs(static_cast<size_t>(r));
    std::vector<scalar_t> ss(static_cast<size_t>(r));

    // Positive update: ordinary Givens rotations Q_k.
    // Downdate: hyperbolic rotations H_k with H_k J H_k^T = J.
    for (int64_t k = 0; k < r; ++k) {
        const scalar_t a = Lcm[static_cast<size_t>(k * r + k)];
        const scalar_t b = z_d[k];
        scalar_t c;
        scalar_t s;

        if (metric_sign > 0) {
            const scalar_t rho = std::hypot(a, b);
            TORCH_CHECK(rho > scalar_t(0), "zero Givens norm at k=", k);
            c = a / rho;
            s = b / rho;
        } else {
            // sqrt((a-b)(a+b)) is more accurate near the SPD boundary than
            // directly forming a*a-b*b.
            const scalar_t rad = (a - b) * (a + b);
            TORCH_CHECK(rad > scalar_t(0),
                        "downdate left the SPD cone at rotation k=", k);
            const scalar_t rho = std::sqrt(rad);
            c = a / rho;
            s = -b / rho;
        }

        cs[static_cast<size_t>(k)] = c;
        ss[static_cast<size_t>(k)] = s;

        for (int64_t i = k; i < r; ++i) {
            const scalar_t lv = Lcm[static_cast<size_t>(k * r + i)];
            const scalar_t zv = z_d[i];
            Lcm[static_cast<size_t>(k * r + i)] = c * lv + s * zv;
            if (metric_sign > 0) {
                z_d[i] = -s * lv + c * zv;
            } else {
                z_d[i] = s * lv + c * zv;
            }
        }
    }

    for (int64_t i = 0; i < r; ++i) {
        for (int64_t j = 0; j < r; ++j) {
            L_rm[i * r + j] = Lcm[static_cast<size_t>(j * r + i)];
        }
    }

    // Pad the current whitened operator.
    std::vector<scalar_t> Ah(static_cast<size_t>(rp * rp), scalar_t(0));
    const scalar_t* Aw_d = Aw.data_ptr<scalar_t>();
    for (int64_t i = 0; i < r; ++i) {
        for (int64_t j = 0; j < r; ++j) {
            Ah[static_cast<size_t>(i * rp + j)] = Aw_d[i * r + j];
        }
    }

    constexpr int64_t NB = 32;

    // Positive metric update transports by Q^T A Q.
    // Downdate transports by S A S^T, S=H^{-1}.
    for (int64_t j0 = 0; j0 < rp; j0 += NB) {
        const int64_t j1 = std::min(j0 + NB, rp);
        for (int64_t k = 0; k < r; ++k) {
            const scalar_t c = cs[static_cast<size_t>(k)];
            const scalar_t s = ss[static_cast<size_t>(k)];
            for (int64_t j = j0; j < j1; ++j) {
                const scalar_t rk = Ah[static_cast<size_t>(k * rp + j)];
                const scalar_t rr = Ah[static_cast<size_t>(r * rp + j)];
                if (metric_sign > 0) {
                    Ah[static_cast<size_t>(k * rp + j)] = c * rk + s * rr;
                } else {
                    Ah[static_cast<size_t>(k * rp + j)] = c * rk - s * rr;
                }
                Ah[static_cast<size_t>(r * rp + j)] = -s * rk + c * rr;
            }
        }
    }

    for (int64_t i0 = 0; i0 < rp; i0 += NB) {
        const int64_t i1 = std::min(i0 + NB, rp);
        for (int64_t k = 0; k < r; ++k) {
            const scalar_t c = cs[static_cast<size_t>(k)];
            const scalar_t s = ss[static_cast<size_t>(k)];
            for (int64_t i = i0; i < i1; ++i) {
                const scalar_t ck = Ah[static_cast<size_t>(i * rp + k)];
                const scalar_t cr = Ah[static_cast<size_t>(i * rp + r)];
                if (metric_sign > 0) {
                    Ah[static_cast<size_t>(i * rp + k)] = c * ck + s * cr;
                } else {
                    Ah[static_cast<size_t>(i * rp + k)] = c * ck - s * cr;
                }
                Ah[static_cast<size_t>(i * rp + r)] = -s * ck + c * cr;
            }
        }
    }

    auto vz = torch::empty({r}, opts);
    scalar_t* vz_d = vz.data_ptr<scalar_t>();
    scalar_t e_last = scalar_t(1);
    for (int64_t k = 0; k < r; ++k) {
        const scalar_t s = ss[static_cast<size_t>(k)];
        vz_d[k] = (metric_sign > 0 ? s : -s) * e_last;
        e_last = cs[static_cast<size_t>(k)] * e_last;
    }

    if (operator_sign != 0) {
        std::vector<scalar_t> vw(static_cast<size_t>(r), scalar_t(0));
        for (int64_t i = 0; i < r; ++i) {
            scalar_t acc = w_d[i];
            for (int64_t j = 0; j < i; ++j) {
                acc -= L_rm[i * r + j] * vw[static_cast<size_t>(j)];
            }
            vw[static_cast<size_t>(i)] = acc / L_rm[i * r + i];
        }

        const scalar_t op = static_cast<scalar_t>(operator_sign);
        for (int64_t i = 0; i < r; ++i) {
            const scalar_t vzi = op * vz_d[i];
            for (int64_t j = 0; j < r; ++j) {
                Ah[static_cast<size_t>(i * rp + j)] +=
                    vzi * vw[static_cast<size_t>(j)];
            }
        }
    }

    auto Aw_out = torch::empty({r, r}, opts);
    scalar_t* out_d = Aw_out.data_ptr<scalar_t>();
    for (int64_t i = 0; i < r; ++i) {
        for (int64_t j = 0; j < r; ++j) {
            out_d[i * r + j] = Ah[static_cast<size_t>(i * rp + j)];
        }
    }

    return {Aw_out, vz};
}

template <typename scalar_t>
std::tuple<torch::Tensor, torch::Tensor> cyclic_shift_impl(
    torch::Tensor L,
    const torch::Tensor& Aw,
    const torch::Tensor& mw
) {
    const int64_t r = L.size(0);
    const auto opts = L.options();
    scalar_t* L_d = L.data_ptr<scalar_t>();
    const scalar_t* Aw_d = Aw.data_ptr<scalar_t>();
    const scalar_t* mw_d = mw.data_ptr<scalar_t>();

    std::vector<scalar_t> B(static_cast<size_t>(r * r), scalar_t(0));
    for (int64_t i = 0; i < r; ++i) {
        const int64_t old_i = (i + 1) % r;
        for (int64_t j = 0; j < r; ++j) {
            B[static_cast<size_t>(i * r + j)] = L_d[old_i * r + j];
        }
    }

    std::vector<scalar_t> cs(static_cast<size_t>(std::max<int64_t>(r - 1, 0)));
    std::vector<scalar_t> ss(static_cast<size_t>(std::max<int64_t>(r - 1, 0)));

    // The cyclic row permutation makes B lower Hessenberg. Adjacent right
    // Givens rotations remove the one superdiagonal in O(r^2).
    for (int64_t k = 0; k + 1 < r; ++k) {
        const scalar_t a = B[static_cast<size_t>(k * r + k)];
        const scalar_t b = B[static_cast<size_t>(k * r + k + 1)];
        const scalar_t rho = std::hypot(a, b);
        TORCH_CHECK(rho > scalar_t(0), "degenerate cyclic-shift rotation at k=", k);
        const scalar_t c = a / rho;
        const scalar_t s = b / rho;
        cs[static_cast<size_t>(k)] = c;
        ss[static_cast<size_t>(k)] = s;

        for (int64_t i = 0; i < r; ++i) {
            const scalar_t x = B[static_cast<size_t>(i * r + k)];
            const scalar_t y = B[static_cast<size_t>(i * r + k + 1)];
            B[static_cast<size_t>(i * r + k)] = c * x + s * y;
            B[static_cast<size_t>(i * r + k + 1)] = -s * x + c * y;
        }
    }

    std::vector<scalar_t> Anew(static_cast<size_t>(r * r));
    std::copy(Aw_d, Aw_d + r * r, Anew.begin());
    std::vector<scalar_t> mnew(static_cast<size_t>(r));
    std::copy(mw_d, mw_d + r, mnew.begin());

    // Q^T Aw Q and Q^T mw for the same adjacent rotations.
    for (int64_t k = 0; k + 1 < r; ++k) {
        const scalar_t c = cs[static_cast<size_t>(k)];
        const scalar_t s = ss[static_cast<size_t>(k)];
        for (int64_t j = 0; j < r; ++j) {
            const scalar_t x = Anew[static_cast<size_t>(k * r + j)];
            const scalar_t y = Anew[static_cast<size_t>((k + 1) * r + j)];
            Anew[static_cast<size_t>(k * r + j)] = c * x + s * y;
            Anew[static_cast<size_t>((k + 1) * r + j)] = -s * x + c * y;
        }
        const scalar_t x = mnew[static_cast<size_t>(k)];
        const scalar_t y = mnew[static_cast<size_t>(k + 1)];
        mnew[static_cast<size_t>(k)] = c * x + s * y;
        mnew[static_cast<size_t>(k + 1)] = -s * x + c * y;
    }

    for (int64_t k = 0; k + 1 < r; ++k) {
        const scalar_t c = cs[static_cast<size_t>(k)];
        const scalar_t s = ss[static_cast<size_t>(k)];
        for (int64_t i = 0; i < r; ++i) {
            const scalar_t x = Anew[static_cast<size_t>(i * r + k)];
            const scalar_t y = Anew[static_cast<size_t>(i * r + k + 1)];
            Anew[static_cast<size_t>(i * r + k)] = c * x + s * y;
            Anew[static_cast<size_t>(i * r + k + 1)] = -s * x + c * y;
        }
    }

    // QR-like triangularization may leave negative diagonal entries. Multiplying
    // the corresponding columns by -1 preserves G and restores the unique
    // positive-diagonal Cholesky convention. The same diagonal orthogonal map
    // must be applied to Aw and mw.
    std::vector<scalar_t> signs(static_cast<size_t>(r), scalar_t(1));
    for (int64_t j = 0; j < r; ++j) {
        if (B[static_cast<size_t>(j * r + j)] < scalar_t(0)) {
            signs[static_cast<size_t>(j)] = scalar_t(-1);
        }
    }

    for (int64_t i = 0; i < r; ++i) {
        for (int64_t j = 0; j < r; ++j) {
            B[static_cast<size_t>(i * r + j)] *= signs[static_cast<size_t>(j)];
            Anew[static_cast<size_t>(i * r + j)] *=
                signs[static_cast<size_t>(i)] * signs[static_cast<size_t>(j)];
            if (j > i) {
                B[static_cast<size_t>(i * r + j)] = scalar_t(0);
            }
        }
        mnew[static_cast<size_t>(i)] *= signs[static_cast<size_t>(i)];
    }

    std::copy(B.begin(), B.end(), L_d);

    auto Aw_out = torch::empty({r, r}, opts);
    auto mw_out = torch::empty({r}, opts);
    std::copy(Anew.begin(), Anew.end(), Aw_out.data_ptr<scalar_t>());
    std::copy(mnew.begin(), mnew.end(), mw_out.data_ptr<scalar_t>());
    return {Aw_out, mw_out};
}

std::tuple<torch::Tensor, torch::Tensor> dispatch_signed_update(
    torch::Tensor L,
    torch::Tensor Aw,
    torch::Tensor z,
    torch::Tensor w,
    int64_t metric_sign,
    int64_t operator_sign,
    double downdate_tol,
    bool copy_back
) {
    check_square_state(L, Aw, z, w);
    auto L_original = L;
    auto L_work = L.contiguous();
    auto Aw_work = Aw.contiguous();
    auto z_work = z.clone().contiguous();
    auto w_work = w.contiguous();

    std::tuple<torch::Tensor, torch::Tensor> result;
    if (L.scalar_type() == torch::kFloat64) {
        if (metric_sign == 1 && operator_sign == 1) {
            result = positive_update_impl<double>(
                L_work, Aw_work, z_work, w_work
            );
        } else {
            result = signed_update_impl<double>(
                L_work, Aw_work, z_work, w_work,
                metric_sign, operator_sign, downdate_tol
            );
        }
    } else {
        if (metric_sign == 1 && operator_sign == 1) {
            result = positive_update_impl<float>(
                L_work, Aw_work, z_work, w_work
            );
        } else {
            result = signed_update_impl<float>(
                L_work, Aw_work, z_work, w_work,
                metric_sign, operator_sign, downdate_tol
            );
        }
    }

    if (copy_back) {
        L_original.copy_(L_work);
    }
    return result;
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> fast_incremental_update(
    torch::Tensor L,
    torch::Tensor Aw,
    torch::Tensor z,
    torch::Tensor w,
    int64_t metric_sign = 1,
    int64_t operator_sign = 1,
    double downdate_tol = 1e-8
) {
    at::NoGradGuard no_grad;
    return dispatch_signed_update(
        L, Aw, z, w, metric_sign, operator_sign, downdate_tol, true
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fast_incremental_update_functional(
    torch::Tensor L,
    torch::Tensor Aw,
    torch::Tensor z,
    torch::Tensor w,
    int64_t metric_sign = 1,
    int64_t operator_sign = 1,
    double downdate_tol = 1e-8
) {
    at::NoGradGuard no_grad;
    check_square_state(L, Aw, z, w);
    auto L_new = L.contiguous().clone();
    auto result = dispatch_signed_update(
        L_new, Aw, z, w, metric_sign, operator_sign, downdate_tol, true
    );
    return {L_new, std::get<0>(result), std::get<1>(result)};
}

std::tuple<torch::Tensor, torch::Tensor> fast_cyclic_shift(
    torch::Tensor L,
    torch::Tensor Aw,
    torch::Tensor mw
) {
    at::NoGradGuard no_grad;
    check_cpu_float_matrix(L, "L");
    TORCH_CHECK(L.size(0) == L.size(1), "L must be square");
    const int64_t r = L.size(0);
    TORCH_CHECK(Aw.device() == L.device() && Aw.scalar_type() == L.scalar_type(),
                "Aw must match L's device and dtype");
    TORCH_CHECK(Aw.dim() == 2 && Aw.size(0) == r && Aw.size(1) == r,
                "Aw must have shape (r,r)");
    TORCH_CHECK(mw.device() == L.device() && mw.scalar_type() == L.scalar_type(),
                "mw must match L's device and dtype");
    TORCH_CHECK(mw.dim() == 1 && mw.size(0) == r,
                "mw must have shape (r,)");

    auto L_original = L;
    auto L_work = L.contiguous();
    auto Aw_work = Aw.contiguous();
    auto mw_work = mw.contiguous();

    std::tuple<torch::Tensor, torch::Tensor> result;
    if (L.scalar_type() == torch::kFloat64) {
        result = cyclic_shift_impl<double>(L_work, Aw_work, mw_work);
    } else {
        result = cyclic_shift_impl<float>(L_work, Aw_work, mw_work);
    }
    L_original.copy_(L_work);
    return result;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fast_cyclic_shift_functional(
    torch::Tensor L,
    torch::Tensor Aw,
    torch::Tensor mw
) {
    at::NoGradGuard no_grad;
    auto L_new = L.contiguous().clone();
    auto result = fast_cyclic_shift(L_new, Aw, mw);
    return {L_new, std::get<0>(result), std::get<1>(result)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fast_incremental_update",
        &fast_incremental_update,
        py::arg("L"), py::arg("Aw"), py::arg("z"), py::arg("w"),
        py::arg("metric_sign") = 1,
        py::arg("operator_sign") = 1,
        py::arg("downdate_tol") = 1e-8,
        "Signed double-sided Cholesky update; updates L in place"
    );
    m.def(
        "fast_incremental_update_functional",
        &fast_incremental_update_functional,
        py::arg("L"), py::arg("Aw"), py::arg("z"), py::arg("w"),
        py::arg("metric_sign") = 1,
        py::arg("operator_sign") = 1,
        py::arg("downdate_tol") = 1e-8,
        "Functional signed double-sided Cholesky update"
    );
    m.def(
        "fast_cyclic_shift",
        &fast_cyclic_shift,
        py::arg("L"), py::arg("Aw"), py::arg("mw"),
        "Move coordinate zero to the end and transport L, Aw, and mw"
    );
    m.def(
        "fast_cyclic_shift_functional",
        &fast_cyclic_shift_functional,
        py::arg("L"), py::arg("Aw"), py::arg("mw"),
        "Functional cyclic coordinate shift"
    );
}
