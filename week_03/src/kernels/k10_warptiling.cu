#include "common.cuh"
#include "kernels.cuh"
#include <cuda_runtime.h>

template<int BM, int BN, int BK>
__global__ void k10_warptiling(const float* A, const float* B, float* C, int M, int N, int K) {
  // Block tile: BM x BN, K tile: BK
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  // 8 warps per block -> 256 threads
  int tid = threadIdx.x;
  int warp = tid >> 5;       // 0..7
  int lane = tid & 31;       // 0..31

  // Warp mapping: 2 warps in M, 4 warps in N => 8 warps total
  // Each warp computes a 64 x 32 tile (64 rows, 32 cols)
  int warp_m = warp >> 2;    // 0..1
  int warp_n = warp & 3;     // 0..3

  // Within warp: 8x4 thread grid
  int trow = lane & 7;       // 0..7
  int tcol = lane >> 3;      // 0..3

  // Each thread computes an 8x8 micro-tile
  constexpr int MT = 8;
  constexpr int NT = 8;

  int block_row = blockIdx.y * BM;
  int block_col = blockIdx.x * BN;

  int warp_row0 = block_row + warp_m * 64;
  int warp_col0 = block_col + warp_n * 32;

  int row0 = warp_row0 + trow * MT;   // 0..56 inside warp-tile
  int col0 = warp_col0 + tcol * NT;   // 0..24 inside warp-tile

  float acc[MT][NT];
  #pragma unroll
  for (int i = 0; i < MT; ++i)
    #pragma unroll
    for (int j = 0; j < NT; ++j)
      acc[i][j] = 0.0f;

  // Cooperative load uses all threads
  for (int kb = 0; kb < K; kb += BK) {
    // Load A: BM*BK floats
    for (int idx = tid; idx < BM * BK; idx += 256) {
      int r = idx / BK;
      int c = idx % BK;
      int gr = block_row + r;
      int gc = kb + c;
      As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
    }
    // Load B: BK*BN floats
    for (int idx = tid; idx < BK * BN; idx += 256) {
      int r = idx / BN;
      int c = idx % BN;
      int gr = kb + r;
      int gc = block_col + c;
      Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
    }

    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float a_frag[MT];
      float b_frag[NT];

      #pragma unroll
      for (int i = 0; i < MT; ++i) {
        int r = row0 + i;
        a_frag[i] = (r < M) ? As[r - block_row][kk] : 0.0f;
      }
      #pragma unroll
      for (int j = 0; j < NT; ++j) {
        int c = col0 + j;
        b_frag[j] = (c < N) ? Bs[kk][c - block_col] : 0.0f;
      }

      #pragma unroll
      for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j)
          acc[i][j] += a_frag[i] * b_frag[j];
    }

    __syncthreads();
  }

  // Store microtile
  #pragma unroll
  for (int i = 0; i < MT; ++i) {
    int gr = row0 + i;
    if (gr < M) {
      #pragma unroll
      for (int j = 0; j < NT; ++j) {
        int gc = col0 + j;
        if (gc < N) {
          C[gr * N + gc] = acc[i][j];
        }
      }
    }
  }
}

void launch_kernel_10(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 16;
  dim3 block(256);
  dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  k10_warptiling<BM, BN, BK><<<grid, block, 0, s>>>(A, B, C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}
