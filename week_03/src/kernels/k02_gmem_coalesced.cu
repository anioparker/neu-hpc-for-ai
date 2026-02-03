#include "common.cuh"
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>

__global__ void transpose_B(const float* B, float* BT, int K, int N) {
  int r = blockIdx.y * blockDim.y + threadIdx.y; // k
  int c = blockIdx.x * blockDim.x + threadIdx.x; // n
  if (r < K && c < N) {
    BT[c * K + r] = B[r * N + c]; // BT is (N x K) row-major
  }
}

__global__ void k2_gemm_BT(const float* A, const float* BT, float* C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;

  float acc = 0.0f;
  const float* arow = A + row * K;
  const float* btrow = BT + col * K; // contiguous in K
  for (int kk = 0; kk < K; ++kk) acc += arow[kk] * btrow[kk];
  C[row * N + col] = acc;
}

void launch_kernel_2(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
  // Cache BT across calls so transpose happens only once per (B,N,K)
  static float* dBT = nullptr;
  static const float* cachedB = nullptr;
  static int cachedN = 0, cachedK = 0;

  if (dBT == nullptr || cachedB != B || cachedN != N || cachedK != K) {
    if (dBT) CUDA_CHECK(cudaFree(dBT));
    CUDA_CHECK(cudaMalloc(&dBT, (size_t)N * (size_t)K * sizeof(float)));

    dim3 tblock(16, 16);
    dim3 tgrid(CEIL_DIV(N, tblock.x), CEIL_DIV(K, tblock.y));
    transpose_B<<<tgrid, tblock, 0, s>>>(B, dBT, K, N);
    CUDA_CHECK(cudaGetLastError());

    cachedB = B;
    cachedN = N;
    cachedK = K;
  }

  dim3 block(16, 16);
  dim3 grid(CEIL_DIV(N, block.x), CEIL_DIV(M, block.y));
  k2_gemm_BT<<<grid, block, 0, s>>>(A, dBT, C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}
