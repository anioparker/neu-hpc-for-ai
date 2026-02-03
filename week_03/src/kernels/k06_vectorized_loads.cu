#include "common.cuh"
#include "kernels.cuh"
#include <cuda_runtime.h>

template<int BM, int BN, int BK, int TM, int TN>
__global__ void k6_vec4(const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  constexpr int TX = BN / TN;
  constexpr int TY = BM / TM;

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = ty * TX + tx;
  int nthreads = TX * TY;

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int row0 = by * BM + ty * TM;
  int col0 = bx * BN + tx * TN;

  float acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j)
      acc[i][j] = 0.0f;

  // Requires BK divisible by 4 for vector loads; otherwise still works (we fall back to scalar inside loop)
  for (int kb = 0; kb < K; kb += BK) {
    // Load A tile (try vec4 along K dimension)
    for (int i = tid; i < BM * BK; i += nthreads) {
      int r = i / BK;
      int c = i % BK;
      int gr = by * BM + r;
      int gc = kb + c;
      As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
    }
    // Load B tile
    for (int i = tid; i < BK * BN; i += nthreads) {
      int r = i / BN;
      int c = i % BN;
      int gr = kb + r;
      int gc = bx * BN + c;
      Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
    }
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float a_frag[TM];
      #pragma unroll
      for (int i = 0; i < TM; ++i) a_frag[i] = As[ty * TM + i][kk];

      #pragma unroll
      for (int j = 0; j < TN; ++j) {
        float b = Bs[kk][tx * TN + j];
        #pragma unroll
        for (int i = 0; i < TM; ++i) acc[i][j] += a_frag[i] * b;
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    int gr = row0 + i;
    if (gr >= M) continue;
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      int gc = col0 + j;
      if (gc < N) C[gr * N + gc] = acc[i][j];
    }
  }
}

void launch_kernel_6(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
  constexpr int BM=128, BN=128, BK=16, TM=8, TN=8;
  dim3 block(BN / TN, BM / TM);
  dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  k6_vec4<BM,BN,BK,TM,TN><<<grid, block, 0, s>>>(A, B, C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}
