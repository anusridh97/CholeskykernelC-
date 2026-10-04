#include "common.cuh"
namespace dsc::sym {
  
__global__ void desc(const float*A,const float*c,const float*h,const float*d,float*south,float*east0,float*upper,int r,int n){
long long q=blockIdx.x;
int i=(int)((sqrtf(8.0f*(float)q+1.0f)-1.0f)*0.5f);
while(tile_id(i+1,0)<=q)i++;
while(tile_id(i,0)>q)i--;
int j=(int)(q-tile_id(i,0));
long long base=q*B*B;
__shared__ float x[B*LDS];
for(int k=threadIdx.x;
k<B*B;
k+=blockDim.x){
int a=k/B,b=k%B;
x[a*LDS+b]=(i*B+a<r&&j*B+b<r)?A[base+k]:0;
}
__syncthreads();
if(threadIdx.x<B){
int b=threadIdx.x;
float u=0;
if(i>j&&i*B+b<r)for(int a=0;
a<B&&j*B+a<r;
a++){
int row=j*B+a;
u+=d[row]*x[b*LDS+a];
}
upper[q*B+b]=u;
}
__syncthreads();
if(threadIdx.x<B){
int b=threadIdx.x,col=j*B+b;
float z=0;
if(col<r)for(int a=0;
a<B&&i*B+a<r;
a++){
int row=i*B+a;
float v=x[a*LDS+b],t=d[row]*v;
x[a*LDS+b]=c[row]*v+h[row]*z;
z+=t;
}
south[q*B+b]=z;
}
__syncthreads();
if(threadIdx.x<B){
int a=threadIdx.x,row=i*B+a;
float z=0;
if(row<r){
int last=i==j?a:B-1;
for(int b=0;
b<=last&&j*B+b<r;
b++){
int col=j*B+b;
float v=x[a*LDS+b];
z+=d[col]*v;
}
}
east0[q*B+a]=z;
}
}

__global__ void scan_north(const float*south,const float*upper,float*north,int r,int n){
int j=blockIdx.x,b=threadIdx.x;
if(b>=B)return;
float z=0;
for(int k=0;
k<j;
k++)z+=upper[tile_id(j,k)*B+b];
for(int i=j;
i<n;
i++){
long long q=tile_id(i,j);
north[q*B+b]=z;
z+=south[q*B+b];
}
}

__global__ void scan_west(const float*east0,const float*north,const float*c,const float*h,const float*d,float*west,int r,int n){
int i=blockIdx.x,a=threadIdx.x;
if(a>=B)return;
int row=i*B+a;
float w=0,hr=row<r?h[row]:0;
for(int j=0;
j<=i;
j++){
long long q=tile_id(i,j);
west[q*B+a]=w;
float dot=0;
for(int b=0;
b<B&&j*B+b<r;
b++){
int col=j*B+b;
dot+=d[col]*north[q*B+b];
}
w+=east0[q*B+a]+hr*dot;
}
}

__global__ void apply(float*A,const float*c,const float*h,const float*d,const float*vz,const float*north,const float*west,int r,int n){
long long q=blockIdx.x;
int i=(int)((sqrtf(8.0f*(float)q+1.0f)-1.0f)*0.5f);
while(tile_id(i+1,0)<=q)i++;
while(tile_id(i,0)>q)i--;
int j=(int)(q-tile_id(i,0));
long long base=q*B*B;
__shared__ float x[B*LDS];
for(int k=threadIdx.x;
k<B*B;
k+=blockDim.x){
int a=k/B,b=k%B;
x[a*LDS+b]=(i*B+a<r&&j*B+b<r)?A[base+k]:0;
}
__syncthreads();
if(threadIdx.x<B){
int b=threadIdx.x,col=j*B+b;
float z=north[q*B+b];
if(col<r)for(int a=0;
a<B&&i*B+a<r;
a++){
int row=i*B+a;
float v=x[a*LDS+b],t=d[row]*v;
x[a*LDS+b]=c[row]*v+h[row]*z;
z+=t;
}
}
__syncthreads();
if(threadIdx.x<B){
int a=threadIdx.x,row=i*B+a;
float z=west[q*B+a];
if(row<r){
int last=i==j?a:B-1;
for(int b=0;
b<=last&&j*B+b<r;
b++){
int col=j*B+b;
float v=x[a*LDS+b],t=d[col]*v;
x[a*LDS+b]=c[col]*v+h[col]*z+vz[row]*vz[col];
z+=t;
}
}
}
__syncthreads();
for(int k=threadIdx.x;
k<B*B;
k+=blockDim.x){
int a=k/B,b=k%B,row=i*B+a,col=j*B+b;
if(row<r&&col<r)A[base+k]=(i==j&&b>a)?x[b*LDS+a]:x[a*LDS+b];
}
}

void run(torch::Tensor A,torch::Tensor c,torch::Tensor h,torch::Tensor d,torch::Tensor vz,torch::Tensor south,torch::Tensor east0,torch::Tensor upper,torch::Tensor north,torch::Tensor west,int64_t r){

auto count=tile_count(r);
 auto tiles=count*(count+1)/2;

    TORCH_CHECK(tiles<=INT_MAX, "too many tiles");

    check_matrix(A,tiles*B*B);

    for (const auto& v : {
c,h,d,vz}
) check_tensor(v,A,r);

    for (const auto& v : {
south,east0,upper,north,west}
) check_tensor(v,A,tiles*B);

    c10::cuda::CUDAGuard guard(A.device());

int n=(r+B-1)/B;
auto st=at::cuda::getCurrentCUDAStream();
int nt=n*(n+1)/2;
dim3 g(nt);
desc<<<g,256,0,st>>>(A.data_ptr<float>(),c.data_ptr<float>(),h.data_ptr<float>(),d.data_ptr<float>(),south.data_ptr<float>(),east0.data_ptr<float>(),upper.data_ptr<float>(),r,n);
scan_north<<<n,B,0,st>>>(south.data_ptr<float>(),upper.data_ptr<float>(),north.data_ptr<float>(),r,n);
scan_west<<<n,B,0,st>>>(east0.data_ptr<float>(),north.data_ptr<float>(),c.data_ptr<float>(),h.data_ptr<float>(),d.data_ptr<float>(),west.data_ptr<float>(),r,n);
apply<<<g,256,0,st>>>(A.data_ptr<float>(),c.data_ptr<float>(),h.data_ptr<float>(),d.data_ptr<float>(),vz.data_ptr<float>(),north.data_ptr<float>(),west.data_ptr<float>(),r,n);
C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<float> profile(torch::Tensor A,torch::Tensor c,torch::Tensor h,torch::Tensor d,torch::Tensor vz,torch::Tensor south,torch::Tensor east0,torch::Tensor upper,torch::Tensor north,torch::Tensor west,int64_t r){

auto count=tile_count(r);
 auto tiles=count*(count+1)/2;

    TORCH_CHECK(tiles<=INT_MAX, "too many tiles");

    check_matrix(A,tiles*B*B);

    for (const auto& v : {
c,h,d,vz}
) check_tensor(v,A,r);

    for (const auto& v : {
south,east0,upper,north,west}
) check_tensor(v,A,tiles*B);

    c10::cuda::CUDAGuard guard(A.device());

int n=(r+B-1)/B,nt=n*(n+1)/2;
auto st=at::cuda::getCurrentCUDAStream();
cudaEvent_t e[5];
for(int k=0;
k<5;
k++)C10_CUDA_CHECK(cudaEventCreate(&e[k]));
C10_CUDA_CHECK(cudaEventRecord(e[0],st));
desc<<<nt,256,0,st>>>(A.data_ptr<float>(),c.data_ptr<float>(),h.data_ptr<float>(),d.data_ptr<float>(),south.data_ptr<float>(),east0.data_ptr<float>(),upper.data_ptr<float>(),r,n);
C10_CUDA_CHECK(cudaEventRecord(e[1],st));
scan_north<<<n,B,0,st>>>(south.data_ptr<float>(),upper.data_ptr<float>(),north.data_ptr<float>(),r,n);
C10_CUDA_CHECK(cudaEventRecord(e[2],st));
scan_west<<<n,B,0,st>>>(east0.data_ptr<float>(),north.data_ptr<float>(),c.data_ptr<float>(),h.data_ptr<float>(),d.data_ptr<float>(),west.data_ptr<float>(),r,n);
C10_CUDA_CHECK(cudaEventRecord(e[3],st));
apply<<<nt,256,0,st>>>(A.data_ptr<float>(),c.data_ptr<float>(),h.data_ptr<float>(),d.data_ptr<float>(),vz.data_ptr<float>(),north.data_ptr<float>(),west.data_ptr<float>(),r,n);
C10_CUDA_CHECK(cudaEventRecord(e[4],st));
C10_CUDA_CHECK(cudaEventSynchronize(e[4]));
C10_CUDA_KERNEL_LAUNCH_CHECK();
std::vector<float> out(4);
for(int k=0;
k<4;
k++){
C10_CUDA_CHECK(cudaEventElapsedTime(&out[k],e[k],e[k+1]));
cudaEventDestroy(e[k]);
}
cudaEventDestroy(e[4]);
return out;
}


}
