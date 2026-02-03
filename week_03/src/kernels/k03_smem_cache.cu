#include "common.cuh"
#include "kernels.cuh"
#include <cuda_runtime.h>

template<int TILE>
__global__ void k3_smem(const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  float acc = 0.0f;
  for (int t = 0; t < CEIL_DIV(K, TILE); ++t) {
    int a_col = t * TILE + threadIdx.x;
    int b_row = t * TILE + threadIdx.y;

    As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < TILE; ++kk) {
      acc += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < M && col < N) C[row * N + col] = acc;
}

void launch_kernel_3(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
  constexpr int TILE = 32;
  dim3 block(TILE, TILE);
  dim3 grid(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
  k3_smem<TILE><<<grid, block, 0, s>>>(A, B, C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}
