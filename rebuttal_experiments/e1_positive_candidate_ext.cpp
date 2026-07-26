#include <torch/extension.h>

#include <ATen/Parallel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <tuple>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

void check_float_contiguous(const torch::Tensor& value, const char* name) {
    TORCH_CHECK(value.device().is_cpu(), name, " must be on the CPU");
    TORCH_CHECK(value.scalar_type() == torch::kFloat32,
                name, " must have dtype float32");
    TORCH_CHECK(value.is_contiguous(), name, " must be row-major contiguous");
}

void check_arguments(
    const torch::Tensor& L,
    const torch::Tensor& Aw,
    const torch::Tensor& z,
    const torch::Tensor& Aw_out,
    const torch::Tensor& vz_out,
    const torch::Tensor& workspace
) {
    check_float_contiguous(L, "L");
    check_float_contiguous(Aw, "Aw");
    check_float_contiguous(z, "z");
    check_float_contiguous(Aw_out, "Aw_out");
    check_float_contiguous(vz_out, "vz_out");
    check_float_contiguous(workspace, "workspace");
    TORCH_CHECK(L.dim() == 2 && L.size(0) == L.size(1),
                "L must be square");
    const int64_t r = L.size(0);
    TORCH_CHECK(Aw.dim() == 2 && Aw.size(0) == r && Aw.size(1) == r,
                "Aw must have shape (r, r)");
    TORCH_CHECK(Aw_out.dim() == 2 && Aw_out.size(0) == r &&
                    Aw_out.size(1) == r,
                "Aw_out must have shape (r, r)");
    TORCH_CHECK(z.dim() == 1 && z.size(0) == r,
                "z must have shape (r,)");
    TORCH_CHECK(vz_out.dim() == 1 && vz_out.size(0) == r,
                "vz_out must have shape (r,)");
    TORCH_CHECK(workspace.dim() == 1 && workspace.numel() >= 4 * r,
                "workspace must have at least 4*r float32 elements");
    TORCH_CHECK(Aw.data_ptr() != Aw_out.data_ptr(),
                "Aw and Aw_out must not alias (use ping-pong buffers)");
    TORCH_CHECK(L.data_ptr() != workspace.data_ptr() &&
                    Aw.data_ptr() != workspace.data_ptr() &&
                    Aw_out.data_ptr() != workspace.data_ptr() &&
                    z.data_ptr() != workspace.data_ptr() &&
                    vz_out.data_ptr() != workspace.data_ptr(),
                "workspace must not alias an input or output");
}

double seconds_since(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// Pilot-only specialization for
//
//   G+ = G + z z^T,       M+ = M + z z^T.
//
// Consequently v_w = (L+)^{-1} z = v_z exactly.  This removes the general
// kernel's second forward solve.  It is an explicit contract, not a
// floating-point comparison between z and w.
//
// The caller owns all storage:
//   * L is updated in place;
//   * Aw is read-only and Aw_out is the next ping-pong buffer;
//   * vz_out receives (L+)^{-1} z;
//   * workspace contains 4*r floats: c, s, transformed z, virtual row.
//
// phase_seconds, when non-null, has five entries:
//   workspace reset, Cholesky update, recover vz, left transport,
//   right transport plus zz^T correction.
void positive_same_vector_core(
    torch::Tensor L,
    const torch::Tensor& Aw,
    const torch::Tensor& z,
    torch::Tensor Aw_out,
    torch::Tensor vz_out,
    torch::Tensor workspace,
    double* phase_seconds
) {
    const int64_t r = L.size(0);
    float* const L_data = L.data_ptr<float>();
    const float* const Aw_data = Aw.data_ptr<float>();
    const float* const z_data = z.data_ptr<float>();
    float* const out_data = Aw_out.data_ptr<float>();
    float* const vz = vz_out.data_ptr<float>();
    float* const work = workspace.data_ptr<float>();
    float* const cs = work;
    float* const ss = cs + r;
    float* const z_work = ss + r;
    float* const virtual_row = z_work + r;

    Clock::time_point started;
    if (phase_seconds != nullptr) {
        started = Clock::now();
    }
    std::copy(z_data, z_data + r, z_work);
    std::fill(virtual_row, virtual_row + r, 0.0f);
    if (phase_seconds != nullptr) {
        phase_seconds[0] = seconds_since(started);
        started = Clock::now();
    }

    // Reorder the ordinary column-wise rank-one Cholesky update by rows.
    // For row i, rotations 0,...,i are still applied in exactly that order.
    // Rotation i is created only after all predecessor rotations have
    // transformed z_i, preserving the original dependency graph while
    // making every factor access contiguous.
    int factor_threads = 1;
#ifdef _OPENMP
    factor_threads = omp_get_max_threads();
#endif
    if (factor_threads == 1 || r < 128) {
        for (int64_t i = 0; i < r; ++i) {
            float zi = z_work[i];
            float* const row = L_data + i * r;
            for (int64_t k = 0; k < i; ++k) {
                const float old_l = row[k];
                const float c = cs[k];
                const float s = ss[k];
                row[k] = c * old_l + s * zi;
                zi = -s * old_l + c * zi;
            }
            const float diagonal = row[i];
            const float rho = std::hypot(diagonal, zi);
            TORCH_CHECK(rho > 0.0f, "zero Givens norm at row ", i);
            cs[i] = diagonal / rho;
            ss[i] = zi / rho;
            row[i] = rho;
            z_work[i] = 0.0f;
        }
    } else {
        // A panel creates a small consecutive group of rotations on the
        // serial diagonal dependency path.  Once created, that entire group
        // can be applied independently to every trailing row.  This replaces
        // r barriers with r/PANEL barriers and gives the ten-thread mode
        // useful factor work without changing operation order within a row.
        constexpr int64_t PANEL = 32;
        for (int64_t begin = 0; begin < r; begin += PANEL) {
            const int64_t end = std::min(begin + PANEL, r);
            for (int64_t i = begin; i < end; ++i) {
                float zi = z_work[i];
                float* const row = L_data + i * r;
                for (int64_t k = begin; k < i; ++k) {
                    const float old_l = row[k];
                    const float c = cs[k];
                    const float s = ss[k];
                    row[k] = c * old_l + s * zi;
                    zi = -s * old_l + c * zi;
                }
                const float diagonal = row[i];
                const float rho = std::hypot(diagonal, zi);
                TORCH_CHECK(rho > 0.0f, "zero Givens norm at row ", i);
                cs[i] = diagonal / rho;
                ss[i] = zi / rho;
                row[i] = rho;
                z_work[i] = 0.0f;
            }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int64_t i = end; i < r; ++i) {
                float zi = z_work[i];
                float* const row = L_data + i * r;
                for (int64_t k = begin; k < end; ++k) {
                    const float old_l = row[k];
                    const float c = cs[k];
                    const float s = ss[k];
                    row[k] = c * old_l + s * zi;
                    zi = -s * old_l + c * zi;
                }
                z_work[i] = zi;
            }
        }
    }
    if (phase_seconds != nullptr) {
        phase_seconds[1] = seconds_since(started);
        started = Clock::now();
    }

    float final_coordinate = 1.0f;
    for (int64_t k = 0; k < r; ++k) {
        vz[k] = ss[k] * final_coordinate;
        final_coordinate *= cs[k];
    }
    if (phase_seconds != nullptr) {
        phase_seconds[2] = seconds_since(started);
        started = Clock::now();
    }

    // Each OpenMP worker owns a disjoint contiguous column interval and the
    // matching virtual-row interval for the entire sweep.  No barrier is
    // needed between rotations because there are no cross-column
    // dependencies.
#ifdef _OPENMP
#pragma omp parallel
    {
        const int thread = omp_get_thread_num();
        const int threads = omp_get_num_threads();
        const int64_t begin = (r * thread) / threads;
        const int64_t end = (r * (thread + 1)) / threads;
        for (int64_t k = 0; k < r; ++k) {
            const float c = cs[k];
            const float s = ss[k];
            const float* const input_row = Aw_data + k * r;
            float* const output_row = out_data + k * r;
#pragma omp simd
            for (int64_t j = begin; j < end; ++j) {
                const float old_a = input_row[j];
                const float old_virtual = virtual_row[j];
                output_row[j] = c * old_a + s * old_virtual;
                virtual_row[j] = -s * old_a + c * old_virtual;
            }
        }
    }
#else
    for (int64_t k = 0; k < r; ++k) {
        const float c = cs[k];
        const float s = ss[k];
        const float* const input_row = Aw_data + k * r;
        float* const output_row = out_data + k * r;
        for (int64_t j = 0; j < r; ++j) {
            const float old_a = input_row[j];
            const float old_virtual = virtual_row[j];
            output_row[j] = c * old_a + s * old_virtual;
            virtual_row[j] = -s * old_a + c * old_virtual;
        }
    }
#endif
    if (phase_seconds != nullptr) {
        phase_seconds[3] = seconds_since(started);
        started = Clock::now();
    }

    // Rows are independent in the right sweep.  The v_z v_z^T correction is
    // written as soon as a column leaves the recurrence; later rotations do
    // not read that column, so this fusion is algebraically identical to a
    // separate outer-product pass.
    // Four independent row recurrences are interleaved.  A single row has a
    // loop-carried virtual-column dependency; interleaving exposes enough
    // independent FMAs to hide that latency while preserving contiguous
    // streams for all four rows.
    constexpr int64_t ROW_BLOCK = 4;
    const int64_t complete_rows = (r / ROW_BLOCK) * ROW_BLOCK;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < complete_rows; i += ROW_BLOCK) {
        float cr0 = 0.0f;
        float cr1 = 0.0f;
        float cr2 = 0.0f;
        float cr3 = 0.0f;
        const float vz0 = vz[i];
        const float vz1 = vz[i + 1];
        const float vz2 = vz[i + 2];
        const float vz3 = vz[i + 3];
        float* const row0 = out_data + i * r;
        float* const row1 = row0 + r;
        float* const row2 = row1 + r;
        float* const row3 = row2 + r;
        for (int64_t k = 0; k < r; ++k) {
            const float c = cs[k];
            const float s = ss[k];
            const float vzk = vz[k];
            const float x0 = row0[k];
            const float x1 = row1[k];
            const float x2 = row2[k];
            const float x3 = row3[k];
            row0[k] = c * x0 + s * cr0 + vz0 * vzk;
            row1[k] = c * x1 + s * cr1 + vz1 * vzk;
            row2[k] = c * x2 + s * cr2 + vz2 * vzk;
            row3[k] = c * x3 + s * cr3 + vz3 * vzk;
            cr0 = -s * x0 + c * cr0;
            cr1 = -s * x1 + c * cr1;
            cr2 = -s * x2 + c * cr2;
            cr3 = -s * x3 + c * cr3;
        }
    }
    for (int64_t i = complete_rows; i < r; ++i) {
        float virtual_column = 0.0f;
        const float vzi = vz[i];
        float* const row = out_data + i * r;
        for (int64_t k = 0; k < r; ++k) {
            const float old_a = row[k];
            row[k] = cs[k] * old_a + ss[k] * virtual_column + vzi * vz[k];
            virtual_column =
                -ss[k] * old_a + cs[k] * virtual_column;
        }
    }
    if (phase_seconds != nullptr) {
        phase_seconds[4] = seconds_since(started);
    }
}

void positive_metric_operator_same_vector_out(
    torch::Tensor L,
    torch::Tensor Aw,
    torch::Tensor z,
    torch::Tensor Aw_out,
    torch::Tensor vz_out,
    torch::Tensor workspace
) {
    at::NoGradGuard no_grad;
    check_arguments(L, Aw, z, Aw_out, vz_out, workspace);
    positive_same_vector_core(
        L, Aw, z, Aw_out, vz_out, workspace, nullptr
    );
}

void positive_metric_operator_same_vector_profile_out(
    torch::Tensor L,
    torch::Tensor Aw,
    torch::Tensor z,
    torch::Tensor Aw_out,
    torch::Tensor vz_out,
    torch::Tensor workspace,
    torch::Tensor phase_seconds
) {
    at::NoGradGuard no_grad;
    check_arguments(L, Aw, z, Aw_out, vz_out, workspace);
    TORCH_CHECK(phase_seconds.device().is_cpu() &&
                    phase_seconds.scalar_type() == torch::kFloat64 &&
                    phase_seconds.is_contiguous() &&
                    phase_seconds.dim() == 1 &&
                    phase_seconds.numel() == 5,
                "phase_seconds must be a contiguous CPU float64 vector of length 5");
    positive_same_vector_core(
        L,
        Aw,
        z,
        Aw_out,
        vz_out,
        workspace,
        phase_seconds.data_ptr<double>()
    );
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "positive_metric_operator_same_vector_out",
        &positive_metric_operator_same_vector_out,
        py::arg("L"),
        py::arg("Aw"),
        py::arg("z"),
        py::arg("Aw_out"),
        py::arg("vz_out"),
        py::arg("workspace"),
        "Pilot-only positive update for G+=G+zz^T and M+=M+zz^T"
    );
    module.def(
        "positive_metric_operator_same_vector_profile_out",
        &positive_metric_operator_same_vector_profile_out,
        py::arg("L"),
        py::arg("Aw"),
        py::arg("z"),
        py::arg("Aw_out"),
        py::arg("vz_out"),
        py::arg("workspace"),
        py::arg("phase_seconds"),
        "Profile the pilot-only same-vector positive update"
    );
}
