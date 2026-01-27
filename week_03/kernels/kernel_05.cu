// kernel_05_2d_blocktiling.cu
// nvcc -O3 -std=c++17 -arch=sm_90 -lineinfo \
  -DBM=128 -DBN=128 -DBK=8 -DTM=8 -DTN=8 \
  kernel_05.cu -o 2d_blocktiling

// ./2d_blocktiling 4096 4096 4096 50


// 2D Blocktiling SGEMM (row-major).
// Each block computes a BM x BN tile of C.
// Each thread computes a TM x TN micro-tile (in registers).
//
// Cooperative SMEM loads use strideA/strideB loops like in your snippet.
//
// GFLOPs/s uses FLOPs ~= 2*M*N*K.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>

#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 128
#endif
#ifndef BK
#define BK 16
#endif
#ifndef TM
#define TM 8
#endif
#ifndef TN
#define TN 8
#endif

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define CHECK_CUDA(call) do {                                   \
  cudaError_t e = (call);                                       \
  if (e != cudaSuccess) {                                       \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
            __FILE__, __LINE__, cudaGetErrorString(e));         \
    std::exit(1);                                               \
  }                                                            \
} while (0)

struct GpuTimer {
  cudaEvent_t start, stop;
  GpuTimer() { CHECK_CUDA(cudaEventCreate(&start)); CHECK_CUDA(cudaEventCreate(&stop)); }
  ~GpuTimer(){ cudaEventDestroy(start); cudaEventDestroy(stop); }
  void tic() { CHECK_CUDA(cudaEventRecord(start)); }
  float toc() {
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

static inline void fill_random(float* p, size_t n, uint64_t seed=123) {
  uint64_t x = seed;
  auto rng = [&]() {
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    return x * 2685821657736338717ULL;
  };
  for (size_t i=0;i<n;++i) {
    uint64_t r = rng();
    float u = (float)((double)((r >> 11) & 0x1FFFFFULL) / (double)0x1FFFFFULL);
    p[i] = 2.f*u - 1.f;
  }
}

static inline double gflops_sgemm(int M, int N, int K, double ms_avg) {
  double flops = 2.0 * (double)M * (double)N * (double)K;
  return flops / (ms_avg * 1e6);
}

// 2D blocktiling kernel (micro-tiles TM x TN per thread).
__global__ void sgemm_2d_blocktiling(int M, int N, int K,
                                    float alpha,
                                    const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float beta,
                                    float* __restrict__ C) {
  static_assert(BM % TM == 0, "BM must be divisible by TM");
  static_assert(BN % TN == 0, "BN must be divisible by TN");

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Which C-tile this block computes
  const int cRow = (int)blockIdx.y;
  const int cCol = (int)blockIdx.x;

  // Thread coordinates in the block's (BM/TM) x (BN/TN) thread grid
  const int threadRow = (int)threadIdx.y; // 0..(BM/TM)-1
  const int threadCol = (int)threadIdx.x; // 0..(BN/TN)-1

  // Advance base pointers to start of this block tile (row-major)
  const float* A_ptr = A + (size_t)cRow * BM * (size_t)K;              // row=cRow*BM, col=0
  const float* B_ptr = B + (size_t)cCol * BN;                          // row=0, col=cCol*BN
  float*       C_ptr = C + (size_t)cRow * BM * (size_t)N + (size_t)cCol * BN; // row=cRow*BM, col=cCol*BN

  // ---- Cooperative loading mapping (match your snippet structure) ----
  // We interpret threads as a 2D grid:
  //   blockDim.x = BN/TN
  //   blockDim.y = BM/TM
  // Total threads: (BM/TM) * (BN/TN)  (must be <= 1024)

  // For A loads: each thread loads one element at (innerRowA, innerColA) and repeats with loadOffset
  const uint tid = (uint)(threadRow * blockDim.x + threadCol);

  // A tile (BM x BK) has BM*BK elements
  // strideA: how many rows (of A-tile) we advance per iteration for this thread
  const uint numThreads = (uint)(blockDim.x * blockDim.y);
  if (numThreads % BK != 0 || numThreads % BN != 0) {
  printf("Bad params: numThreads=%u must be divisible by BK=%u and BN=%u\n",
         numThreads, (uint)BK, (uint)BN);
  return;
    }

  const uint strideA = numThreads / BK; // how many A-rows covered per "round"
  const uint strideB = numThreads / BN; // how many B-rows covered per "round"

  // Inner coordinates for the first element this thread loads
  // (same conceptual names as your snippet)
  const uint innerRowA = tid / BK;   // 0..(BM-1) ideally
  const uint innerColA = tid % BK;   // 0..(BK-1)

  const uint innerRowB = tid / BN;   // 0..(BK-1) ideally
  const uint innerColB = tid % BN;   // 0..(BN-1)

  // Thread-local accumulators
  float threadResults[TM * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  // Outer loop over K tiles
  for (uint bkIdx = 0; bkIdx < (uint)K; bkIdx += BK) {

    // ---- Populate SMEM caches (strided loops exactly like your snippet) ----
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      uint r = innerRowA + loadOffset;
      uint c = innerColA;
      int gRow = cRow * BM + (int)r;
      int gCol = (int)bkIdx + (int)c;
      if (r < BM && gRow < M && gCol < K) {
        As[r * BK + c] = A_ptr[(size_t)r * (size_t)K + (size_t)c];
      } else if (r < BM) {
        As[r * BK + c] = 0.0f;
      }
    }

    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      uint r = innerRowB + loadOffset;
      uint c = innerColB;
      int gRow = (int)bkIdx + (int)r;
      int gCol = cCol * BN + (int)c;
      if (r < BK && gRow < K && gCol < N) {
        Bs[r * BN + c] = B_ptr[(size_t)r * (size_t)N + (size_t)c];
      } else if (r < BK) {
        Bs[r * BN + c] = 0.0f;
      }
    }

    __syncthreads();

    // advance blocktile (exactly as your snippet)
    A_ptr += BK;               // move BK columns to right
    B_ptr += (size_t)BK * (size_t)N; // move BK rows down

    // ---- Compute: dotIdx outer loop, reg cache, outer product ----
    #pragma unroll
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {

      // Load As rows needed for this thread into regM
      #pragma unroll
      for (uint i = 0; i < TM; ++i) {
        uint aRow = (uint)threadRow * TM + i; // 0..BM-1
        regM[i] = As[aRow * BK + dotIdx];
      }

      // Load Bs cols needed for this thread into regN
      #pragma unroll
      for (uint i = 0; i < TN; ++i) {
        uint bCol = (uint)threadCol * TN + i; // 0..BN-1
        regN[i] = Bs[dotIdx * BN + bCol];
      }

      // Outer product accumulate into threadResults
      #pragma unroll
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        #pragma unroll
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] =
              fmaf(regM[resIdxM], regN[resIdxN],
                   threadResults[resIdxM * TN + resIdxN]);
        }
      }
    }

    __syncthreads();
  }

  // ---- Write back TM x TN results ----
  const int baseRow = cRow * BM + threadRow * TM;
  const int baseCol = cCol * BN + threadCol * TN;

  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    int r = baseRow + i;
    if (r < M) {
      #pragma unroll
      for (int j = 0; j < TN; ++j) {
        int c = baseCol + j;
        if (c < N) {
          float* out = &C_ptr[(size_t)(threadRow * TM + i) * (size_t)N + (size_t)(threadCol * TN + j)];
          *out = alpha * threadResults[i * TN + j] + beta * (*out);
        }
      }
    }
  }
}

static void launch(int M, int N, int K,
                   float alpha, const float* dA, const float* dB,
                   float beta, float* dC) {
  dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

  const int threadsX = BN / TN;
  const int threadsY = BM / TM;
  const int threads  = threadsX * threadsY;
  if (threads > 1024) {
    fprintf(stderr, "ERROR: threads per block = %d > 1024. Adjust BM/BN/TM/TN.\n", threads);
    std::exit(1);
  }
  dim3 block(threadsX, threadsY);

  sgemm_2d_blocktiling<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
  CHECK_CUDA(cudaGetLastError());
}

int main(int argc, char** argv) {
  int M=4096, N=4096, K=4096;
  int reps=50, warmup=5;

  // Usage: ./bin [M N K] [reps]
  if (argc >= 4) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }
  if (argc >= 5) { reps = std::atoi(argv[4]); }

  printf("kernel_05_2d_blocktiling | BM=%d BN=%d BK=%d TM=%d TN=%d | M=%d N=%d K=%d reps=%d\n",
         BM, BN, BK, TM, TN, M, N, K, reps);

  const size_t sizeA = (size_t)M * (size_t)K;
  const size_t sizeB = (size_t)K * (size_t)N;
  const size_t sizeC = (size_t)M * (size_t)N;

  float* hA = (float*)std::malloc(sizeA*sizeof(float));
  float* hB = (float*)std::malloc(sizeB*sizeof(float));
  float* hC = (float*)std::malloc(sizeC*sizeof(float));
  if (!hA || !hB || !hC) { fprintf(stderr, "host malloc failed\n"); return 1; }

  fill_random(hA, sizeA, 1);
  fill_random(hB, sizeB, 2);
  fill_random(hC, sizeC, 3);

  float *dA=nullptr, *dB=nullptr, *dC=nullptr;
  CHECK_CUDA(cudaMalloc(&dA, sizeA*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dB, sizeB*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dC, sizeC*sizeof(float)));

  CHECK_CUDA(cudaMemcpy(dA, hA, sizeA*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB, sizeB*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dC, hC, sizeC*sizeof(float), cudaMemcpyHostToDevice));

  const float alpha = 1.0f;
  const float beta  = 0.0f;

  // Warmup
  for (int i=0;i<warmup;++i) {
    launch(M, N, K, alpha, dA, dB, beta, dC);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  // Timed
  GpuTimer t;
  t.tic();
  for (int i=0;i<reps;++i) {
    launch(M, N, K, alpha, dA, dB, beta, dC);
  }
  float ms_total = t.toc();
  CHECK_CUDA(cudaDeviceSynchronize());

  const double ms_avg = ms_total / (double)reps;
  const double gflops = gflops_sgemm(M, N, K, ms_avg);

  printf("avg time: %.4f ms  |  throughput: %.2f GFLOP/s\n", ms_avg, gflops);

  CHECK_CUDA(cudaFree(dA));
  CHECK_CUDA(cudaFree(dB));
  CHECK_CUDA(cudaFree(dC));
  std::free(hA); std::free(hB); std::free(hC);
  return 0;
}
