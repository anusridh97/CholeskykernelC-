#include <torch/extension.h>
#include <cmath>
#include <vector>
#include <algorithm>

template <typename scalar_t>
std::tuple<torch::Tensor, torch::Tensor> update_impl(
    torch::Tensor L, torch::Tensor Aw, torch::Tensor z, torch::Tensor w
) {
    const int r = L.size(0);
    const int rp = r + 1;
    auto opts = L.options();

    scalar_t* L_rm = L.data_ptr<scalar_t>();
    scalar_t* z_d = z.data_ptr<scalar_t>();
    std::vector<scalar_t> cs(r), ss(r);

    // Column-major L for contiguous column access in Phase 1
    std::vector<scalar_t> Lcm(r * r);
    for (int i = 0; i < r; i++)
        for (int j = 0; j < r; j++)
            Lcm[j * r + i] = L_rm[i * r + j];

    // Phase 1: Cholesky update with rotation fusion
    // Columns are contiguous in Lcm. Fuse pairs of rotations
    // to reuse z (which both rotations touch).
    int k = 0;
    for (; k + 1 < r; k += 2) {
        // Rotation k
        scalar_t a0 = Lcm[k * r + k], b0 = z_d[k];
        scalar_t rho0 = std::sqrt(a0*a0 + b0*b0);
        scalar_t c0 = a0/rho0, s0 = b0/rho0;
        cs[k] = c0; ss[k] = s0;

        for (int i = 0; i < r; i++) {
            scalar_t lv = Lcm[k*r+i], zv = z_d[i];
            Lcm[k*r+i] = c0*lv + s0*zv;
            z_d[i] = -s0*lv + c0*zv;
        }

        // Rotation k+1 (z already updated)
        scalar_t a1 = Lcm[(k+1)*r+(k+1)], b1 = z_d[k+1];
        scalar_t rho1 = std::sqrt(a1*a1 + b1*b1);
        scalar_t c1 = a1/rho1, s1 = b1/rho1;
        cs[k+1] = c1; ss[k+1] = s1;

        for (int i = 0; i < r; i++) {
            scalar_t lv = Lcm[(k+1)*r+i], zv = z_d[i];
            Lcm[(k+1)*r+i] = c1*lv + s1*zv;
            z_d[i] = -s1*lv + c1*zv;
        }
    }
    // Handle odd r
    if (k < r) {
        scalar_t a = Lcm[k*r+k], b = z_d[k];
        scalar_t rho = std::sqrt(a*a + b*b);
        scalar_t c = a/rho, s = b/rho;
        cs[k] = c; ss[k] = s;
        for (int i = 0; i < r; i++) {
            scalar_t lv = Lcm[k*r+i], zv = z_d[i];
            Lcm[k*r+i] = c*lv + s*zv;
            z_d[i] = -s*lv + c*zv;
        }
    }

    // Copy L back to row-major
    for (int i = 0; i < r; i++)
        for (int j = 0; j < r; j++)
            L_rm[i*r+j] = Lcm[j*r+i];

    // Phase 2: Pad Aw into Ah (row-major)
    std::vector<scalar_t> Ah(rp * rp, 0);
    const scalar_t* Aw_d = Aw.data_ptr<scalar_t>();
    for (int i = 0; i < r; i++)
        for (int j = 0; j < r; j++)
            Ah[i*rp+j] = Aw_d[i*r+j];

    // Left sweep with rotation fusion
    // Rows are contiguous in row-major. Fuse pairs: both touch row r.
    constexpr int NB = 32;  // panel width tuned to L1

    // Left sweep: wavefront paneling over columns
    for (int j0 = 0; j0 < rp; j0 += NB) {
        int j1 = std::min(j0 + NB, rp);
        // Apply all rotations to this column panel
        for (int kk = 0; kk < r - 1; kk += 2) {
            scalar_t c0 = cs[kk], s0 = ss[kk];
            scalar_t c1 = cs[kk+1], s1 = ss[kk+1];
            // Rotation kk: mix rows kk and r
            for (int j = j0; j < j1; j++) {
                scalar_t rk = Ah[kk*rp+j];
                scalar_t rr = Ah[r*rp+j];
                Ah[kk*rp+j] = c0*rk + s0*rr;
                Ah[r*rp+j] = -s0*rk + c0*rr;
            }
            // Rotation kk+1: mix rows kk+1 and r (r already updated)
            for (int j = j0; j < j1; j++) {
                scalar_t rk = Ah[(kk+1)*rp+j];
                scalar_t rr = Ah[r*rp+j];
                Ah[(kk+1)*rp+j] = c1*rk + s1*rr;
                Ah[r*rp+j] = -s1*rk + c1*rr;
            }
        }
        if (r % 2 == 1) {
            scalar_t c = cs[r-1], s = ss[r-1];
            for (int j = j0; j < j1; j++) {
                scalar_t rk = Ah[(r-1)*rp+j];
                scalar_t rr = Ah[r*rp+j];
                Ah[(r-1)*rp+j] = c*rk + s*rr;
                Ah[r*rp+j] = -s*rk + c*rr;
            }
        }
    }

    // Right sweep: wavefront paneling over rows
    // Column k and column r are at stride rp. Panel rows for cache locality.
    for (int i0 = 0; i0 < rp; i0 += NB) {
        int i1 = std::min(i0 + NB, rp);
        for (int kk = 0; kk < r - 1; kk += 2) {
            scalar_t c0 = cs[kk], s0 = ss[kk];
            scalar_t c1 = cs[kk+1], s1 = ss[kk+1];
            // Rotation kk: mix cols kk and r for this row panel
            for (int i = i0; i < i1; i++) {
                scalar_t ck = Ah[i*rp+kk];
                scalar_t cr = Ah[i*rp+r];
                Ah[i*rp+kk] = c0*ck + s0*cr;
                Ah[i*rp+r] = -s0*ck + c0*cr;
            }
            // Rotation kk+1: mix cols kk+1 and r
            for (int i = i0; i < i1; i++) {
                scalar_t ck = Ah[i*rp+(kk+1)];
                scalar_t cr = Ah[i*rp+r];
                Ah[i*rp+(kk+1)] = c1*ck + s1*cr;
                Ah[i*rp+r] = -s1*ck + c1*cr;
            }
        }
        if (r % 2 == 1) {
            scalar_t c = cs[r-1], s = ss[r-1];
            for (int i = i0; i < i1; i++) {
                scalar_t ck = Ah[i*rp+(r-1)];
                scalar_t cr = Ah[i*rp+r];
                Ah[i*rp+(r-1)] = c*ck + s*cr;
                Ah[i*rp+r] = -s*ck + c*cr;
            }
        }
    }

    // Phase 3: vz
    auto vz = torch::zeros({r}, opts);
    scalar_t* vz_d = vz.data_ptr<scalar_t>();
    scalar_t e_last = 1.0;
    for (int kk = 0; kk < r; kk++) {
        vz_d[kk] = ss[kk] * e_last;
        e_last = cs[kk] * e_last;
    }

    // Phase 4: forward sub (LAPACK via torch)
    auto vw = torch::linalg_solve_triangular(L, w.unsqueeze(1), false).squeeze(1);
    const scalar_t* vw_d = vw.data_ptr<scalar_t>();

    // Rank-1 correction
    for (int i = 0; i < r; i++) {
        const scalar_t vzi = vz_d[i];
        for (int j = 0; j < r; j++)
            Ah[i*rp+j] += vzi * vw_d[j];
    }

    // Extract
    auto Aw_out = torch::zeros({r, r}, opts);
    scalar_t* out_d = Aw_out.data_ptr<scalar_t>();
    for (int i = 0; i < r; i++)
        for (int j = 0; j < r; j++)
            out_d[i*r+j] = Ah[i*rp+j];

    return {Aw_out, vz};
}

std::tuple<torch::Tensor, torch::Tensor> fast_incremental_update(
    torch::Tensor L, torch::Tensor Aw, torch::Tensor z, torch::Tensor w
) {
    L = L.contiguous(); Aw = Aw.contiguous();
    z = z.contiguous(); w = w.contiguous();
    if (L.scalar_type() == torch::kFloat64)
        return update_impl<double>(L, Aw, z, w);
    else
        return update_impl<float>(L, Aw, z, w);
}

std::tuple<torch::Tensor, torch::Tensor> fast_incremental_update_batched(
    torch::Tensor L, torch::Tensor Aw, torch::Tensor z, torch::Tensor w
) {
    L = L.contiguous(); Aw = Aw.contiguous();
    z = z.contiguous(); w = w.contiguous();
    const int T = L.size(0); const int r = L.size(1);
    auto opts = L.options();
    auto vz_out = torch::zeros({T, r}, opts);
    auto Aw_out = torch::zeros({T, r, r}, opts);
    at::parallel_for(0, T, 1, [&](int begin, int end) {
        for (int t = begin; t < end; t++) {
            auto L_t = L[t].clone(); auto z_t = z[t].clone();
            auto [aw, vz] = (L_t.scalar_type() == torch::kFloat64)
                ? update_impl<double>(L_t, Aw[t], z_t, w[t])
                : update_impl<float>(L_t, Aw[t], z_t, w[t]);
            Aw_out[t] = aw; vz_out[t] = vz;
        }
    });
    return {Aw_out, vz_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_incremental_update", &fast_incremental_update);
    m.def("fast_incremental_update_batched", &fast_incremental_update_batched);
}
