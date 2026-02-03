#include "common.cuh"
#include "kernels.cuh"
#include <cuda_runtime.h>

__global__ void k1_naive(const float* A, const float* B, float* C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;

  float acc = 0.0f;
  for (int kk = 0; kk < K; ++kk) {
    acc += A[row * K + kk] * B[kk * N + col];
  }
  C[row * N + col] = acc;
}

void launch_kernel_1(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
  dim3 block(16, 16);
  dim3 grid(CEIL_DIV(N, block.x), CEIL_DIV(M, block.y));
  k1_naive<<<grid, block, 0, s>>>(A, B, C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}
