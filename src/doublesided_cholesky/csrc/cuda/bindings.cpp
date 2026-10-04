#include <torch/extension.h>
#include <vector>
namespace dsc::sym { void run(torch::Tensor A,torch::Tensor c,torch::Tensor h,torch::Tensor d,torch::Tensor vz,torch::Tensor south,torch::Tensor east0,torch::Tensor upper,torch::Tensor north,torch::Tensor west,int64_t r); }
namespace dsc::sym { std::vector<float> profile(torch::Tensor A,torch::Tensor c,torch::Tensor h,torch::Tensor d,torch::Tensor vz,torch::Tensor south,torch::Tensor east0,torch::Tensor upper,torch::Tensor north,torch::Tensor west,int64_t r); }
namespace dsc::asym { void run(torch::Tensor A, torch::Tensor c, torch::Tensor h, torch::Tensor d,
         torch::Tensor vz, torch::Tensor vw, torch::Tensor south,
         torch::Tensor east0, torch::Tensor north, torch::Tensor west,
         int64_t r); }
namespace dsc::ptile { void pz(torch::Tensor p,torch::Tensor z,torch::Tensor partial,torch::Tensor out,int64_t r,int64_t split); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
 m.def("sym_descriptor",&dsc::sym::run);
 m.def("sym_descriptor_profile",&dsc::sym::profile);
 m.def("asym_descriptor",&dsc::asym::run);
 m.def("pz_split_k",&dsc::ptile::pz);
}
