#pragma once
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <climits>
namespace dsc {
constexpr int B=64;
constexpr int LDS=65;
__device__ __forceinline__ long long tile_id(int i,int j){return (long long)i*(i+1)/2+j;}
inline int64_t tile_count(int64_t r) {
    TORCH_CHECK(r>=1 && r<=INT_MAX-63, "rank must fit positive CUDA integer indices");
    return (r+B-1)/B;
}
inline void check_tensor(const torch::Tensor& tensor,const torch::Tensor& like,int64_t minimum) {
    TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type()==torch::kFloat32, "CUDA FP32 required");
    TORCH_CHECK(tensor.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(tensor.device()==like.device(), "all tensors must use one device");
    TORCH_CHECK(tensor.numel()>=minimum, "tensor or workspace is too small");
}
inline void check_matrix(const torch::Tensor& A,int64_t elements) {
    check_tensor(A,A,elements);
    TORCH_CHECK(A.numel()==elements, "incorrect matrix storage length");
}
}
