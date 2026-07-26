#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <type_traits>

#ifdef __AVX512F__
#include <immintrin.h>
#endif


template <typename scalar_t>
inline void rotate_two_rows(
    scalar_t* __restrict__ row_b,
    scalar_t* __restrict__ tail_b,
    scalar_t* __restrict__ row_c,
    scalar_t* __restrict__ tail_c,
    scalar_t cosine,
    scalar_t sine,
    int64_t width
) {
    int64_t j = 0;
#ifdef __AVX512F__
    if constexpr (std::is_same_v<scalar_t, double>) {
        const __m512d vc = _mm512_set1_pd(cosine);
        const __m512d vs = _mm512_set1_pd(sine);
        const __m512d vms = _mm512_set1_pd(-sine);
        for (; j + 7 < width; j += 8) {
            const __m512d xb = _mm512_loadu_pd(row_b + j);
            const __m512d yb = _mm512_loadu_pd(tail_b + j);
            const __m512d xc = _mm512_loadu_pd(row_c + j);
            const __m512d yc = _mm512_loadu_pd(tail_c + j);
            _mm512_storeu_pd(
                row_b + j,
                _mm512_fmadd_pd(vc, xb, _mm512_mul_pd(vs, yb))
            );
            _mm512_storeu_pd(
                tail_b + j,
                _mm512_fmadd_pd(vms, xb, _mm512_mul_pd(vc, yb))
            );
            _mm512_storeu_pd(
                row_c + j,
                _mm512_fmadd_pd(vc, xc, _mm512_mul_pd(vs, yc))
            );
            _mm512_storeu_pd(
                tail_c + j,
                _mm512_fmadd_pd(vms, xc, _mm512_mul_pd(vc, yc))
            );
        }
    } else if constexpr (std::is_same_v<scalar_t, float>) {
        const __m512 vc = _mm512_set1_ps(cosine);
        const __m512 vs = _mm512_set1_ps(sine);
        const __m512 vms = _mm512_set1_ps(-sine);
        for (; j + 15 < width; j += 16) {
            const __m512 xb = _mm512_loadu_ps(row_b + j);
            const __m512 yb = _mm512_loadu_ps(tail_b + j);
            const __m512 xc = _mm512_loadu_ps(row_c + j);
            const __m512 yc = _mm512_loadu_ps(tail_c + j);
            _mm512_storeu_ps(
                row_b + j,
                _mm512_fmadd_ps(vc, xb, _mm512_mul_ps(vs, yb))
            );
            _mm512_storeu_ps(
                tail_b + j,
                _mm512_fmadd_ps(vms, xb, _mm512_mul_ps(vc, yb))
            );
            _mm512_storeu_ps(
                row_c + j,
                _mm512_fmadd_ps(vc, xc, _mm512_mul_ps(vs, yc))
            );
            _mm512_storeu_ps(
                tail_c + j,
                _mm512_fmadd_ps(vms, xc, _mm512_mul_ps(vc, yc))
            );
        }
    }
#endif
    for (; j < width; ++j) {
        const scalar_t xb = row_b[j];
        const scalar_t yb = tail_b[j];
        const scalar_t xc = row_c[j];
        const scalar_t yc = tail_c[j];
        row_b[j] = cosine * xb + sine * yb;
        tail_b[j] = -sine * xb + cosine * yb;
        row_c[j] = cosine * xc + sine * yc;
        tail_c[j] = -sine * xc + cosine * yc;
    }
}


// Development candidate retained for the frozen implementation-selection
// pilot.  It writes the dense product into caller-owned storage using
// contiguous row dots.  The confirmation protocol may use it only if it
// satisfies the predeclared development selection rule.
template <typename scalar_t>
inline scalar_t dot_rows_candidate(
    const scalar_t* __restrict__ left,
    const scalar_t* __restrict__ right,
    int64_t width
) {
    int64_t k = 0;
    scalar_t total = scalar_t(0);
#ifdef __AVX512F__
    if constexpr (std::is_same_v<scalar_t, double>) {
        __m512d accumulator = _mm512_setzero_pd();
        for (; k + 7 < width; k += 8) {
            accumulator = _mm512_fmadd_pd(
                _mm512_loadu_pd(left + k),
                _mm512_loadu_pd(right + k),
                accumulator
            );
        }
        alignas(64) double lanes[8];
        _mm512_store_pd(lanes, accumulator);
        for (int lane = 0; lane < 8; ++lane) {
            total += lanes[lane];
        }
    } else if constexpr (std::is_same_v<scalar_t, float>) {
        __m512 accumulator = _mm512_setzero_ps();
        for (; k + 15 < width; k += 16) {
            accumulator = _mm512_fmadd_ps(
                _mm512_loadu_ps(left + k),
                _mm512_loadu_ps(right + k),
                accumulator
            );
        }
        alignas(64) float lanes[16];
        _mm512_store_ps(lanes, accumulator);
        for (int lane = 0; lane < 16; ++lane) {
            total += lanes[lane];
        }
    }
#endif
    for (; k < width; ++k) {
        total = std::fma(left[k], right[k], total);
    }
    return total;
}


void validate_update_inputs(
    const torch::Tensor& L,
    const torch::Tensor& Vb,
    const torch::Tensor& Vc,
    const torch::Tensor& z,
    const torch::Tensor& workspace
) {
    TORCH_CHECK(
        L.device().is_cpu() && Vb.device().is_cpu()
            && Vc.device().is_cpu() && z.device().is_cpu()
            && workspace.device().is_cpu(),
        "all tensors must be on CPU"
    );
    TORCH_CHECK(
        L.scalar_type() == torch::kFloat32
            || L.scalar_type() == torch::kFloat64,
        "only float32 and float64 are supported"
    );
    TORCH_CHECK(
        Vb.scalar_type() == L.scalar_type()
            && Vc.scalar_type() == L.scalar_type()
            && z.scalar_type() == L.scalar_type()
            && workspace.scalar_type() == L.scalar_type(),
        "all tensors must have the same dtype"
    );
    TORCH_CHECK(
        L.dim() == 2 && L.size(0) == L.size(1),
        "L must be square"
    );
    TORCH_CHECK(
        Vb.dim() == 2 && Vb.size(0) == L.size(0),
        "Vb shape mismatch"
    );
    TORCH_CHECK(Vc.sizes() == Vb.sizes(), "Vc shape mismatch");
    TORCH_CHECK(
        z.dim() == 1 && z.size(0) == L.size(0),
        "z shape mismatch"
    );
    TORCH_CHECK(
        workspace.dim() == 1
            && workspace.numel() >= 2 * L.size(0) + 2 * Vb.size(1),
        "workspace must contain at least 2*r + 2*p entries"
    );
    TORCH_CHECK(
        L.is_contiguous() && Vb.is_contiguous() && Vc.is_contiguous()
            && z.is_contiguous() && workspace.is_contiguous(),
        "all tensors must be contiguous"
    );
}


template <typename scalar_t>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> update_impl(
    torch::Tensor L,
    torch::Tensor Vb,
    torch::Tensor Vc,
    torch::Tensor z,
    torch::Tensor workspace
) {
    const int64_t r = L.size(0);
    const int64_t p = Vb.size(1);
    scalar_t* const l = L.data_ptr<scalar_t>();
    scalar_t* const vb = Vb.data_ptr<scalar_t>();
    scalar_t* const vc = Vc.data_ptr<scalar_t>();
    scalar_t* const z_work = z.data_ptr<scalar_t>();
    scalar_t* const scratch = workspace.data_ptr<scalar_t>();
    scalar_t* const cosines = scratch;
    scalar_t* const sines = cosines + r;
    scalar_t* const tail_b = sines + r;
    scalar_t* const tail_c = tail_b + p;

    // Traverse the row-major Cholesky factor contiguously while preserving the
    // exact dependency order of the rank-one Givens update.
    for (int64_t i = 0; i < r; ++i) {
        scalar_t zi = z_work[i];
        scalar_t* const row = l + i * r;
        for (int64_t k = 0; k < i; ++k) {
            const scalar_t value = row[k];
            const scalar_t cosine = cosines[k];
            const scalar_t sine = sines[k];
            row[k] = cosine * value + sine * zi;
            zi = -sine * value + cosine * zi;
        }
        const scalar_t diagonal = row[i];
        const scalar_t rho = std::hypot(diagonal, zi);
        const scalar_t cosine = rho == scalar_t(0)
            ? scalar_t(1)
            : diagonal / rho;
        const scalar_t sine = rho == scalar_t(0)
            ? scalar_t(0)
            : zi / rho;
        cosines[i] = cosine;
        sines[i] = sine;
        row[i] = cosine * diagonal + sine * zi;
        z_work[i] = -sine * diagonal + cosine * zi;
    }

    std::fill(tail_b, tail_b + p, scalar_t(0));
    std::fill(tail_c, tail_c + p, scalar_t(0));
    for (int64_t k = 0; k < r; ++k) {
        rotate_two_rows(
            vb + k * p,
            tail_b,
            vc + k * p,
            tail_c,
            cosines[k],
            sines[k],
            p
        );
    }
    return {L, Vb, Vc};
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
one_sided_update_(
    torch::Tensor L,
    torch::Tensor Vb,
    torch::Tensor Vc,
    torch::Tensor z,
    torch::Tensor workspace
) {
    validate_update_inputs(L, Vb, Vc, z, workspace);
    if (L.scalar_type() == torch::kFloat64) {
        return update_impl<double>(L, Vb, Vc, z, workspace);
    }
    return update_impl<float>(L, Vb, Vc, z, workspace);
}


void validate_materialize_inputs(
    const torch::Tensor& Vb,
    const torch::Tensor& Vc,
    const torch::Tensor& output
) {
    TORCH_CHECK(
        Vb.device().is_cpu() && Vc.device().is_cpu()
            && output.device().is_cpu(),
        "all tensors must be on CPU"
    );
    TORCH_CHECK(
        Vb.scalar_type() == torch::kFloat32
            || Vb.scalar_type() == torch::kFloat64,
        "only float32 and float64 are supported"
    );
    TORCH_CHECK(
        Vc.scalar_type() == Vb.scalar_type()
            && output.scalar_type() == Vb.scalar_type(),
        "all tensors must have the same dtype"
    );
    TORCH_CHECK(
        Vb.dim() == 2 && Vc.sizes() == Vb.sizes(),
        "Vb and Vc must have identical matrix shapes"
    );
    TORCH_CHECK(
        output.dim() == 2 && output.size(0) == Vb.size(0)
            && output.size(1) == Vb.size(0),
        "output must be an r by r matrix"
    );
    TORCH_CHECK(
        Vb.is_contiguous() && Vc.is_contiguous() && output.is_contiguous(),
        "all tensors must be contiguous"
    );
}


template <typename scalar_t>
void materialize_candidate_impl(
    const torch::Tensor& Vb,
    const torch::Tensor& Vc,
    torch::Tensor output
) {
    const int64_t r = Vb.size(0);
    const int64_t p = Vb.size(1);
    const scalar_t* const vb = Vb.data_ptr<scalar_t>();
    const scalar_t* const vc = Vc.data_ptr<scalar_t>();
    scalar_t* const dense = output.data_ptr<scalar_t>();
    for (int64_t i = 0; i < r; ++i) {
        const scalar_t* const left = vb + i * p;
        scalar_t* const row = dense + i * r;
        for (int64_t j = 0; j < r; ++j) {
            row[j] = dot_rows_candidate(left, vc + j * p, p);
        }
    }
}


torch::Tensor materialize_candidate_(
    const torch::Tensor& Vb,
    const torch::Tensor& Vc,
    torch::Tensor output
) {
    validate_materialize_inputs(Vb, Vc, output);
    if (Vb.scalar_type() == torch::kFloat64) {
        materialize_candidate_impl<double>(Vb, Vc, output);
    } else {
        materialize_candidate_impl<float>(Vb, Vc, output);
    }
    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "one_sided_update_",
        &one_sided_update_,
        "In-place one-sided Cholesky update with caller-owned workspace"
    );
    module.def(
        "materialize_candidate_",
        &materialize_candidate_,
        "Development-only caller-owned dense materialization candidate"
    );
}
