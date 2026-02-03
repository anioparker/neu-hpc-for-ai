#include "common.cuh"
#include "kernels.cuh"
#include <cuda_runtime.h>

template<int BM, int BN, int BK, int TM, int TN>
__global__ void k5_2d_regtiling(const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  int bx = blockIdx.x;
  int by = blockIdx.y;

  // 2D threadblock covering BN columns and BM rows via per-thread tiles TMxTN
  constexpr int THREADS_Y = BM / TM;
  constexpr int THREADS_X = BN / TN;

  int tx = threadIdx.x; // 0..THREADS_X-1
  int ty = threadIdx.y; // 0..THREADS_Y-1

  int row0 = by * BM + ty * TM;
  int col0 = bx * BN + tx * TN;

  float acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j)
      acc[i][j] = 0.0f;

  // cooperative load mapping (flattened thread id)
  int tid = ty * THREADS_X + tx;
  int nthreads = THREADS_X * THREADS_Y;

  for (int kb = 0; kb < K; kb += BK) {
    // Load As
    for (int i = tid; i < BM * BK; i += nthreads) {
      int r = i / BK;
      int c = i % BK;
      int gr = by * BM + r;
      int gc = kb + c;
      As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
    }
    // Load Bs
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
        for (int i = 0; i < TM; ++i) {
          acc[i][j] += a_frag[i] * b;
        }
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

void launch_kernel_5(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
  // Strong default starter config
  constexpr int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8;
  dim3 block(BN / TN, BM / TM); // x=16, y=16 => 256 threads
  dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  k5_2d_regtiling<BM, BN, BK, TM, TN><<<grid, block, 0, s>>>(A, B, C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}

// Expose variants for autotune (kernel 9)
template<int BM, int BN, int BK, int TM, int TN>
void launch_k5_variant(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
  dim3 block(BN / TN, BM / TM);
  dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  k5_2d_regtiling<BM, BN, BK, TM, TN><<<grid, block, 0, s>>>(A, B, C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}

// simple explicit instantiations used by kernel9 autotune
extern "C" void k5_launch_128_128_16_8_8(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
  launch_k5_variant<128,128,16,8,8>(A,B,C,M,N,K,s);
}
extern "C" void k5_launch_128_64_16_8_8(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
  launch_k5_variant<128,64,16,8,8>(A,B,C,M,N,K,s);
}
extern "C" void k5_launch_64_128_16_8_8(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
  launch_k5_variant<64,128,16,8,8>(A,B,C,M,N,K,s);
}
extern "C" void k5_launch_64_64_16_8_8(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
  launch_k5_variant<64,64,16,8,8>(A,B,C,M,N,K,s);
}
