#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

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
        result = signed_update_impl<double>(
            L_work, Aw_work, z_work, w_work,
            metric_sign, operator_sign, downdate_tol
        );
    } else {
        result = signed_update_impl<float>(
            L_work, Aw_work, z_work, w_work,
            metric_sign, operator_sign, downdate_tol
        );
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
