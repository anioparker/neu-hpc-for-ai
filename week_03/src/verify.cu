#include "common.cuh"
#include <cmath>
#include <cstdio>

__global__ void max_abs_diff_kernel(const float* a, const float* b, int n, float* out) {
  __shared__ float smax[256];
  int tid = threadIdx.x;
  float m = 0.0f;

  for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
    float d = fabsf(a[i] - b[i]);
    if (d > m) m = d;
  }
  smax[tid] = m;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      if (smax[tid + stride] > smax[tid]) smax[tid] = smax[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) atomicMax((int*)out, __float_as_int(smax[0]));
}

float max_abs_diff(const float* dA, const float* dB, int n, cudaStream_t stream) {
  float zero = 0.0f;
  float* dOut = nullptr;
  CUDA_CHECK(cudaMalloc(&dOut, sizeof(float)));
  CUDA_CHECK(cudaMemcpyAsync(dOut, &zero, sizeof(float), cudaMemcpyHostToDevice, stream));

  int blocks = 256;
  max_abs_diff_kernel<<<blocks, 256, 0, stream>>>(dA, dB, n, dOut);
  CUDA_CHECK(cudaGetLastError());

  float hOut = 0.0f;
  CUDA_CHECK(cudaMemcpyAsync(&hOut, dOut, sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaFree(dOut));
  return hOut;
}

