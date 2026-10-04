#include "common.cuh"
namespace dsc::ptile {

__global__ void split_pz(const float *p, const float *z, float *partial, int r,
                         int n, int split) {

  int ti = blockIdx.x, part = blockIdx.y, lr = threadIdx.x;
  __shared__ float tile[B * B];
  __shared__ float zv[B];
  float sum = 0.f;
  int row = ti * B + lr;

  for (int tj = part; tj <= ti; tj += split) {
    long long base = tile_id(ti, tj) * B * B;
    for (int k = threadIdx.x; k < B * B; k += blockDim.x)
      tile[k] = p[base + k];
    if (threadIdx.x < B) {
      int col = tj * B + threadIdx.x;
      zv[threadIdx.x] = (col < r) ? z[col] : 0.f;
    }
    __syncthreads();

    if (lr < B && row < r) {
      int last = (tj == ti) ? lr : B - 1;

#pragma unroll 4
      for (int lc = 0; lc <= last; lc++)
        sum = fmaf(tile[lr * B + lc], zv[lc], sum);
    }
    __syncthreads();
  }

  if (lr < B && row < r)
    partial[(long long)part * r + row] = sum;
}

__global__ void reduce_pz(const float *partial, float *out, int r, int split) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < r) {
    float x = 0;
    for (int k = 0; k < split; k++)
      x += partial[(long long)k * r + i];
    out[i] = x;
  }
}

void pz(torch::Tensor p, torch::Tensor z, torch::Tensor partial,
        torch::Tensor out, int64_t r, int64_t split) {

  auto count = tile_count(r);
  auto tiles = count * (count + 1) / 2;

  TORCH_CHECK(split >= 1 && split <= 65535,
              "split must fit positive CUDA grid y");

  check_matrix(p, tiles * B * B);
  check_tensor(z, p, r);
  check_tensor(partial, p, split * r);
  check_tensor(out, p, r);

  c10::cuda::CUDAGuard guard(p.device());

  int n = (r + B - 1) / B;
  auto st = at::cuda::getCurrentCUDAStream();
  dim3 g(n, split);
  split_pz<<<g, 256, 0, st>>>(p.data_ptr<float>(), z.data_ptr<float>(),
                              partial.data_ptr<float>(), r, n, split);
  reduce_pz<<<(r + 255) / 256, 256, 0, st>>>(partial.data_ptr<float>(),
                                             out.data_ptr<float>(), r, split);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace dsc::ptile
