#include "common.cuh"
#include "kernels.cuh"
#include <cuda_runtime.h>

template<int BM, int BN, int BK, int TM>
__global__ void k4_1d_blocktiling(const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  // We use 1D threads; each thread computes TM outputs in M for one column in N.
  // Threads per block = BN * (BM/TM)
  constexpr int ROW_GROUPS = BM / TM;
  constexpr int THREADS = BN * ROW_GROUPS;

  int tid = threadIdx.x;
  if (tid >= THREADS) return;

  int col_in_tile = tid % BN;        // 0..BN-1
  int row_group = tid / BN;          // 0..ROW_GROUPS-1
  int row0_in_tile = row_group * TM; // starting row inside BM tile

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int global_col = bx * BN + col_in_tile;
  int global_row0 = by * BM + row0_in_tile;

  float acc[TM];
  #pragma unroll
  for (int i = 0; i < TM; ++i) acc[i] = 0.0f;

  for (int kb = 0; kb < K; kb += BK) {
    // Cooperative load A tile (BM x BK)
    for (int i = tid; i < BM * BK; i += THREADS) {
      int r = i / BK;
      int c = i % BK;
      int gr = by * BM + r;
      int gc = kb + c;
      As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
    }

    // Cooperative load B tile (BK x BN)
    for (int i = tid; i < BK * BN; i += THREADS) {
      int r = i / BN;
      int c = i % BN;
      int gr = kb + r;
      int gc = bx * BN + c;
      Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
    }

    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float b = Bs[kk][col_in_tile];
      #pragma unroll
      for (int i = 0; i < TM; ++i) {
        acc[i] += As[row0_in_tile + i][kk] * b;
      }
    }

    __syncthreads();
  }

  // Store
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    int gr = global_row0 + i;
    if (gr < M && global_col < N) {
      C[gr * N + global_col] = acc[i];
    }
  }
}

void launch_kernel_4(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
  // Safe/correct config with <=1024 threads per block
  constexpr int BM = 128;
  constexpr int BN = 64;
  constexpr int BK = 16;
  constexpr int TM = 8;

  constexpr int THREADS = BN * (BM / TM); // 64 * 16 = 1024
  dim3 block(THREADS);
  dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

  k4_1d_blocktiling<BM, BN, BK, TM><<<grid, block, 0, s>>>(A, B, C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}
