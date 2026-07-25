#include <torch/extension.h>

#include <cmath>
#include <tuple>
#include <vector>


template <typename scalar_t>
std::tuple<torch::Tensor, torch::Tensor> update_impl(
    torch::Tensor L, torch::Tensor Vb, torch::Tensor Vc, torch::Tensor z
) {
    const int64_t r = L.size(0);
    const int64_t p = Vb.size(1);
    scalar_t* l = L.data_ptr<scalar_t>();
    scalar_t* vb = Vb.data_ptr<scalar_t>();
    scalar_t* vc = Vc.data_ptr<scalar_t>();
    scalar_t* z_work = z.data_ptr<scalar_t>();
    std::vector<scalar_t> cosines(r);
    std::vector<scalar_t> sines(r);

    for (int64_t k = 0; k < r; ++k) {
        const scalar_t a = l[k * r + k];
        const scalar_t b = z_work[k];
        const scalar_t rho = std::hypot(a, b);
        const scalar_t c = rho == scalar_t(0) ? scalar_t(1) : a / rho;
        const scalar_t s = rho == scalar_t(0) ? scalar_t(0) : b / rho;
        cosines[k] = c;
        sines[k] = s;
        for (int64_t i = k; i < r; ++i) {
            const scalar_t lv = l[i * r + k];
            const scalar_t zv = z_work[i];
            l[i * r + k] = c * lv + s * zv;
            z_work[i] = -s * lv + c * zv;
        }
    }

    std::vector<scalar_t> tail_b(p, scalar_t(0));
    std::vector<scalar_t> tail_c(p, scalar_t(0));
    for (int64_t k = 0; k < r; ++k) {
        const scalar_t c = cosines[k];
        const scalar_t s = sines[k];
        scalar_t* row_b = vb + k * p;
        scalar_t* row_c = vc + k * p;
        for (int64_t j = 0; j < p; ++j) {
            const scalar_t old_b = row_b[j];
            const scalar_t old_c = row_c[j];
            row_b[j] = c * old_b + s * tail_b[j];
            tail_b[j] = -s * old_b + c * tail_b[j];
            row_c[j] = c * old_c + s * tail_c[j];
            tail_c[j] = -s * old_c + c * tail_c[j];
        }
    }
    return {Vb, Vc};
}


std::tuple<torch::Tensor, torch::Tensor> one_sided_update(
    torch::Tensor L, torch::Tensor Vb, torch::Tensor Vc, torch::Tensor z
) {
    L = L.contiguous();
    Vb = Vb.contiguous();
    Vc = Vc.contiguous();
    z = z.contiguous();
    TORCH_CHECK(L.device().is_cpu(), "CPU tensors required");
    TORCH_CHECK(L.dim() == 2 && L.size(0) == L.size(1), "L must be square");
    TORCH_CHECK(Vb.dim() == 2 && Vb.size(0) == L.size(0), "Vb shape mismatch");
    TORCH_CHECK(Vc.sizes() == Vb.sizes(), "Vc shape mismatch");
    TORCH_CHECK(z.dim() == 1 && z.size(0) == L.size(0), "z shape mismatch");
    TORCH_CHECK(
        L.scalar_type() == Vb.scalar_type() &&
        L.scalar_type() == Vc.scalar_type() &&
        L.scalar_type() == z.scalar_type(),
        "all tensors must have the same dtype"
    );
    if (L.scalar_type() == torch::kFloat64) {
        return update_impl<double>(L, Vb, Vc, z);
    }
    if (L.scalar_type() == torch::kFloat32) {
        return update_impl<float>(L, Vb, Vc, z);
    }
    TORCH_CHECK(false, "only float32 and float64 are supported");
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("one_sided_update", &one_sided_update);
}
