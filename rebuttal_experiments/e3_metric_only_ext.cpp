#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <type_traits>

#ifdef __AVX512F__
#include <immintrin.h>
#endif


template <typename scalar_t>
inline void rotate_contiguous(
    scalar_t* __restrict__ row,
    scalar_t* __restrict__ tail,
    scalar_t c,
    scalar_t s,
    int64_t length
) {
    int64_t j = 0;
#ifdef __AVX512F__
    if constexpr (std::is_same_v<scalar_t, double>) {
        const __m512d vc = _mm512_set1_pd(c);
        const __m512d vs = _mm512_set1_pd(s);
        const __m512d vms = _mm512_set1_pd(-s);
        for (; j + 7 < length; j += 8) {
            const __m512d x = _mm512_loadu_pd(row + j);
            const __m512d y = _mm512_loadu_pd(tail + j);
            const __m512d next_x = _mm512_fmadd_pd(vc, x, _mm512_mul_pd(vs, y));
            const __m512d next_y = _mm512_fmadd_pd(vms, x, _mm512_mul_pd(vc, y));
            _mm512_storeu_pd(row + j, next_x);
            _mm512_storeu_pd(tail + j, next_y);
        }
    } else if constexpr (std::is_same_v<scalar_t, float>) {
        const __m512 vc = _mm512_set1_ps(c);
        const __m512 vs = _mm512_set1_ps(s);
        const __m512 vms = _mm512_set1_ps(-s);
        for (; j + 15 < length; j += 16) {
            const __m512 x = _mm512_loadu_ps(row + j);
            const __m512 y = _mm512_loadu_ps(tail + j);
            const __m512 next_x = _mm512_fmadd_ps(vc, x, _mm512_mul_ps(vs, y));
            const __m512 next_y = _mm512_fmadd_ps(vms, x, _mm512_mul_ps(vc, y));
            _mm512_storeu_ps(row + j, next_x);
            _mm512_storeu_ps(tail + j, next_y);
        }
    }
#endif
    for (; j < length; ++j) {
        const scalar_t x = row[j];
        const scalar_t y = tail[j];
        row[j] = c * x + s * y;
        tail[j] = -s * x + c * y;
    }
}


template <typename scalar_t>
inline scalar_t dot_contiguous(
    const scalar_t* __restrict__ left,
    const scalar_t* __restrict__ right,
    int64_t length
) {
    int64_t j = 0;
    scalar_t total = scalar_t(0);
#ifdef __AVX512F__
    if constexpr (std::is_same_v<scalar_t, double>) {
        __m512d accumulator = _mm512_setzero_pd();
        for (; j + 7 < length; j += 8) {
            accumulator = _mm512_fmadd_pd(
                _mm512_loadu_pd(left + j),
                _mm512_loadu_pd(right + j),
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
        for (; j + 15 < length; j += 16) {
            accumulator = _mm512_fmadd_ps(
                _mm512_loadu_ps(left + j),
                _mm512_loadu_ps(right + j),
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
    for (; j < length; ++j) {
        total += left[j] * right[j];
    }
    return total;
}


template <typename scalar_t>
inline void add_scaled_contiguous(
    scalar_t* __restrict__ destination,
    const scalar_t* __restrict__ source,
    scalar_t scale,
    int64_t length
) {
    int64_t j = 0;
#ifdef __AVX512F__
    if constexpr (std::is_same_v<scalar_t, double>) {
        const __m512d vscale = _mm512_set1_pd(scale);
        for (; j + 7 < length; j += 8) {
            const __m512d current = _mm512_loadu_pd(destination + j);
            const __m512d update = _mm512_loadu_pd(source + j);
            _mm512_storeu_pd(
                destination + j,
                _mm512_fmadd_pd(vscale, update, current)
            );
        }
    } else if constexpr (std::is_same_v<scalar_t, float>) {
        const __m512 vscale = _mm512_set1_ps(scale);
        for (; j + 15 < length; j += 16) {
            const __m512 current = _mm512_loadu_ps(destination + j);
            const __m512 update = _mm512_loadu_ps(source + j);
            _mm512_storeu_ps(
                destination + j,
                _mm512_fmadd_ps(vscale, update, current)
            );
        }
    }
#endif
    for (; j < length; ++j) {
        destination[j] = std::fma(scale, source[j], destination[j]);
    }
}


void validate_common(
    const torch::Tensor& L,
    const torch::Tensor& Aw,
    const torch::Tensor& z,
    const torch::Tensor& workspace
) {
    TORCH_CHECK(
        L.device().is_cpu() && Aw.device().is_cpu()
            && z.device().is_cpu() && workspace.device().is_cpu(),
        "all tensors must be on CPU"
    );
    TORCH_CHECK(
        L.scalar_type() == torch::kFloat32
            || L.scalar_type() == torch::kFloat64,
        "only float32 and float64 are supported"
    );
    TORCH_CHECK(
        Aw.scalar_type() == L.scalar_type()
            && z.scalar_type() == L.scalar_type()
            && workspace.scalar_type() == L.scalar_type(),
        "all tensors must have the same dtype"
    );
    TORCH_CHECK(
        L.dim() == 2 && L.size(0) == L.size(1),
        "L must be square"
    );
    TORCH_CHECK(Aw.sizes() == L.sizes(), "Aw must match L shape");
    TORCH_CHECK(
        z.dim() == 1 && z.size(0) == L.size(0),
        "z must match L rows"
    );
    TORCH_CHECK(
        workspace.dim() == 1 && workspace.numel() >= 5 * L.size(0),
        "workspace must be a vector with at least 5*r entries"
    );
    TORCH_CHECK(
        L.is_contiguous() && Aw.is_contiguous()
            && z.is_contiguous() && workspace.is_contiguous(),
        "all tensors must be contiguous"
    );
}


template <typename scalar_t, bool GENERIC_OPERATOR_UPDATE>
std::tuple<torch::Tensor, torch::Tensor> update_impl(
    torch::Tensor L,
    torch::Tensor Aw,
    torch::Tensor z,
    torch::Tensor w,
    torch::Tensor workspace
) {
    const int64_t r = L.size(0);
    scalar_t* const l = L.data_ptr<scalar_t>();
    scalar_t* const aw = Aw.data_ptr<scalar_t>();
    scalar_t* const z_work = z.data_ptr<scalar_t>();
    scalar_t* const scratch = workspace.data_ptr<scalar_t>();
    scalar_t* const cosines = scratch;
    scalar_t* const sines = cosines + r;
    scalar_t* const padded_row = sines + r;
    scalar_t* const v_w = padded_row + r;
    scalar_t* const v_z = v_w + r;

    std::fill(padded_row, padded_row + r, scalar_t(0));

    // Row-major Cholesky update. For row i, rotations 0..i retain exactly
    // their original dependency order, but L is traversed contiguously.
    for (int64_t i = 0; i < r; ++i) {
        scalar_t zi = z_work[i];
        scalar_t* const row = l + i * r;
        for (int64_t k = 0; k < i; ++k) {
            const scalar_t lv = row[k];
            const scalar_t c = cosines[k];
            const scalar_t s = sines[k];
            row[k] = c * lv + s * zi;
            zi = -s * lv + c * zi;
        }

        const scalar_t diagonal = row[i];
        const scalar_t rho = std::sqrt(diagonal * diagonal + zi * zi);
        const scalar_t c = diagonal / rho;
        const scalar_t s = zi / rho;
        cosines[i] = c;
        sines[i] = s;
        row[i] = c * diagonal + s * zi;
        z_work[i] = -s * diagonal + c * zi;
    }

    if constexpr (GENERIC_OPERATOR_UPDATE) {
        scalar_t final_coordinate = scalar_t(1);
        for (int64_t k = 0; k < r; ++k) {
            v_z[k] = sines[k] * final_coordinate;
            final_coordinate *= cosines[k];
        }

        const scalar_t* const w_data = w.data_ptr<scalar_t>();
        for (int64_t i = 0; i < r; ++i) {
            const scalar_t numerator = w_data[i] - dot_contiguous(
                l + i * r, v_w, i
            );
            v_w[i] = numerator / l[i * r + i];
        }
    }

    // Apply Q^T [Aw 0] in place. Each source row is consumed exactly once.
    for (int64_t k = 0; k < r; ++k) {
        rotate_contiguous(
            aw + k * r,
            padded_row,
            cosines[k],
            sines[k],
            r
        );
    }

    // Apply the right rotation sequence. The virtual padded-column value is
    // scalar and private to each output row.
    for (int64_t i = 0; i < r; ++i) {
        scalar_t virtual_column = scalar_t(0);
        scalar_t* const row = aw + i * r;
        int64_t k = 0;
        for (; k + 1 < r; k += 2) {
            const scalar_t old0 = row[k];
            row[k] = cosines[k] * old0 + sines[k] * virtual_column;
            virtual_column = -sines[k] * old0 + cosines[k] * virtual_column;

            const scalar_t old1 = row[k + 1];
            row[k + 1] = (
                cosines[k + 1] * old1 + sines[k + 1] * virtual_column
            );
            virtual_column = (
                -sines[k + 1] * old1 + cosines[k + 1] * virtual_column
            );
        }
        if (k < r) {
            const scalar_t old = row[k];
            row[k] = cosines[k] * old + sines[k] * virtual_column;
        }
    }

    if constexpr (GENERIC_OPERATOR_UPDATE) {
        // This is intentionally a separate dense pass, matching the generic
        // E3 implementation's v_z v_w^T correction. For the pilot w is an
        // externally supplied zero vector, so the pass is mathematically null.
        for (int64_t i = 0; i < r; ++i) {
            add_scaled_contiguous(aw + i * r, v_w, v_z[i], r);
        }
    }
    return {L, Aw};
}


std::tuple<torch::Tensor, torch::Tensor> metric_only_update_(
    torch::Tensor L,
    torch::Tensor Aw,
    torch::Tensor z,
    torch::Tensor workspace
) {
    validate_common(L, Aw, z, workspace);
    if (L.scalar_type() == torch::kFloat64) {
        return update_impl<double, false>(
            L, Aw, z, torch::Tensor(), workspace
        );
    }
    return update_impl<float, false>(
        L, Aw, z, torch::Tensor(), workspace
    );
}


std::tuple<torch::Tensor, torch::Tensor> generic_update_(
    torch::Tensor L,
    torch::Tensor Aw,
    torch::Tensor z,
    torch::Tensor w,
    torch::Tensor workspace
) {
    validate_common(L, Aw, z, workspace);
    TORCH_CHECK(
        w.device().is_cpu()
            && w.scalar_type() == L.scalar_type()
            && w.dim() == 1
            && w.size(0) == L.size(0)
            && w.is_contiguous(),
        "w must be a contiguous CPU vector matching L dtype and rows"
    );
    if (L.scalar_type() == torch::kFloat64) {
        return update_impl<double, true>(L, Aw, z, w, workspace);
    }
    return update_impl<float, true>(L, Aw, z, w, workspace);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "metric_only_update_",
        &metric_only_update_,
        "In-place metric-only dense Cholesky/whitening update"
    );
    module.def(
        "generic_update_",
        &generic_update_,
        "In-place generic dense update used as the w=0 ablation"
    );
}
