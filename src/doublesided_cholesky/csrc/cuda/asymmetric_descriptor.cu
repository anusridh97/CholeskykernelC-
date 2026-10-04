#include "common.cuh"
namespace dsc::asym {




__global__ void descriptor(
    const float* A, const float* c, const float* h, const float* d,
    float* south, float* east0, int r, int n) {

  int q = blockIdx.x, ti = q / n, tj = q - ti * n;

  int row0 = ti * B, col0 = tj * B;

  __shared__ float x[B * LDS];

  for (int k = threadIdx.x;
 k < B * B;
 k += blockDim.x) {

    int a = k / B, b = k % B, row = row0 + a, col = col0 + b;

    x[a * LDS + b] = (row < r && col < r) ? A[(long long)row * r + col] : 0.f;

  }

  __syncthreads();

  if (threadIdx.x < B) {

    int b = threadIdx.x, col = col0 + b;

    float z = 0.f;

    if (col < r) for (int a = 0;
 a < B && row0 + a < r;
 ++a) {

      int row = row0 + a;

      float v = x[a * LDS + b];

      x[a * LDS + b] = c[row] * v + h[row] * z;

      z += d[row] * v;

    }

    south[(long long)q * B + b] = z;

  }

  __syncthreads();

  if (threadIdx.x < B) {

    int a = threadIdx.x, row = row0 + a;

    float z = 0.f;

    if (row < r) for (int b = 0;
 b < B && col0 + b < r;
 ++b)
      z += d[col0 + b] * x[a * LDS + b];

    east0[(long long)q * B + a] = z;

  }

}


__global__ void scan_north(const float* south, float* north, int n) {

  int tj = blockIdx.x, b = threadIdx.x;

  if (b >= B) return;

  float z = 0.f;

  for (int ti = 0;
 ti < n;
 ++ti) {

    long long q = (long long)ti * n + tj;

    north[q * B + b] = z;

    z += south[q * B + b];

  }

}


__global__ void scan_west(
    const float* east0, const float* north, const float* h, const float* d,
    float* west, int r, int n) {

  int ti = blockIdx.x, a = threadIdx.x, row = ti * B + a;

  if (a >= B) return;

  float w = 0.f, hr = row < r ? h[row] : 0.f;

  for (int tj = 0;
 tj < n;
 ++tj) {

    long long q = (long long)ti * n + tj;

    west[q * B + a] = w;

    float dot = 0.f;

    for (int b = 0;
 b < B && tj * B + b < r;
 ++b)
      dot += d[tj * B + b] * north[q * B + b];

    w += east0[q * B + a] + hr * dot;

  }

}


__global__ void apply(
    float* A, const float* c, const float* h, const float* d,
    const float* vz, const float* vw, const float* north, const float* west,
    int r, int n) {

  int q = blockIdx.x, ti = q / n, tj = q - ti * n;

  int row0 = ti * B, col0 = tj * B;

  __shared__ float x[B * LDS];

  for (int k = threadIdx.x;
 k < B * B;
 k += blockDim.x) {

    int a = k / B, b = k % B, row = row0 + a, col = col0 + b;

    x[a * LDS + b] = (row < r && col < r) ? A[(long long)row * r + col] : 0.f;

  }

  __syncthreads();

  if (threadIdx.x < B) {

    int b = threadIdx.x, col = col0 + b;

    float z = north[(long long)q * B + b];

    if (col < r) for (int a = 0;
 a < B && row0 + a < r;
 ++a) {

      int row = row0 + a;

      float v = x[a * LDS + b];

      x[a * LDS + b] = c[row] * v + h[row] * z;

      z += d[row] * v;

    }

  }

  __syncthreads();

  if (threadIdx.x < B) {

    int a = threadIdx.x, row = row0 + a;

    float z = west[(long long)q * B + a];

    if (row < r) for (int b = 0;
 b < B && col0 + b < r;
 ++b) {

      int col = col0 + b;

      float v = x[a * LDS + b];

      x[a * LDS + b] = c[col] * v + h[col] * z + vz[row] * vw[col];

      z += d[col] * v;

    }

  }

  __syncthreads();

  for (int k = threadIdx.x;
 k < B * B;
 k += blockDim.x) {

    int a = k / B, b = k % B, row = row0 + a, col = col0 + b;

    if (row < r && col < r) A[(long long)row * r + col] = x[a * LDS + b];

  }

}


void run(torch::Tensor A, torch::Tensor c, torch::Tensor h, torch::Tensor d,
         torch::Tensor vz, torch::Tensor vw, torch::Tensor south,
         torch::Tensor east0, torch::Tensor north, torch::Tensor west,
         int64_t r) {

auto count=tile_count(r);
 auto tiles=count*count;

    TORCH_CHECK(tiles<=INT_MAX, "too many tiles");

    check_matrix(A,r*r);

    for (const auto& v : {
c,h,d,vz,vw}
) check_tensor(v,A,r);

    for (const auto& v : {
south,east0,north,west}
) check_tensor(v,A,tiles*B);

    c10::cuda::CUDAGuard guard(A.device());


  int n = (r + B - 1) / B, nt = n * n;

  auto st = at::cuda::getCurrentCUDAStream();

  descriptor<<<nt, 256, 0, st>>>(A.data_ptr<float>(), c.data_ptr<float>(),
      h.data_ptr<float>(), d.data_ptr<float>(), south.data_ptr<float>(),
      east0.data_ptr<float>(), r, n);

  scan_north<<<n, B, 0, st>>>(south.data_ptr<float>(), north.data_ptr<float>(), n);

  scan_west<<<n, B, 0, st>>>(east0.data_ptr<float>(), north.data_ptr<float>(),
      h.data_ptr<float>(), d.data_ptr<float>(), west.data_ptr<float>(), r, n);

  apply<<<nt, 256, 0, st>>>(A.data_ptr<float>(), c.data_ptr<float>(),
      h.data_ptr<float>(), d.data_ptr<float>(), vz.data_ptr<float>(),
      vw.data_ptr<float>(), north.data_ptr<float>(), west.data_ptr<float>(), r, n);

  C10_CUDA_KERNEL_LAUNCH_CHECK();

}



}
