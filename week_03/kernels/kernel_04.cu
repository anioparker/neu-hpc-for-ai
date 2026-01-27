// kernel_04.cu
// nvcc -O3 -std=c++17 -arch=sm_90 \
  -DBM=64 -DBN=64 -DBK=8 -DTM=8 \
  kernel_04.cu -o 1d_blocktiling

// ./1d_blocktiling 4096 4096 4096 50


// 1D Blocktiling SGEMM (row-major): each thread computes TM rows for one column.
// Thread-local accumulation in registers: threadResults[TM].
//
// Output tile per block: BM x BN
// K tile depth: BK
//
// FLOPs: ~ 2*M*N*K  => GFLOPs/s = (2*M*N*K)/(ms_avg*1e6)

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

// 1D blocktiling kernel.
// Mapping:
// - Block computes C tile: rows [cRow*BM .. cRow*BM+BM), cols [cCol*BN .. cCol*BN+BN)
// - Each thread has (threadRow, threadCol) where:
//     threadCol = threadIdx.x % BN
//     threadRow = threadIdx.x / BN
// - Each thread computes TM rows for one column => TM outputs.
//   So number of threads per block must satisfy: (BM/TM) * BN threads.
//   Example: BM=128, BN=128, TM=8 => (128/8)*128 = 2048 threads (too many!)
// So choose realistic defaults (below) or compile-time adjust.
// Recommended typical: BM=128, BN=64, TM=8 => (16*64)=1024 threads (still max).
// Better: BM=128, BN=32, TM=8 => 16*32=512 threads.
//
// IMPORTANT: Ensure (BM % TM)==0 and threads <= 1024.

__global__ void sgemm_1d_blocktiling(int M, int N, int K,
                                    float alpha,
                                    const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float beta,
                                    float* __restrict__ C) {
  static_assert(BM % TM == 0, "BM must be divisible by TM");

  // Shared memory caches
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const int cRow = (int)blockIdx.y; // tile row index
  const int cCol = (int)blockIdx.x; // tile col index

  // Thread mapping
  const int tid = (int)threadIdx.x;

  const int threadCol = tid % BN;        // 0..BN-1
  const int threadRow = tid / BN;        // 0..(BM/TM)-1  (conceptually)
  const int rowBase   = threadRow * TM;  // starting row within BM for this thread

  // Early exit if this thread maps outside the tile grid (happens if blockDim is padded)
  if (threadRow >= (BM / TM)) return;

  // Advance pointers to start of this C tile (row-major)
  const float* A_ptr = A + (size_t)cRow * BM * (size_t)K;              // row=cRow*BM, col=0
  const float* B_ptr = B + (size_t)cCol * BN;                          // row=0, col=cCol*BN
  float*       C_ptr = C + (size_t)cRow * BM * (size_t)N + (size_t)cCol * BN; // row=cRow*BM, col=cCol*BN

  // Allocate thread-local cache for results in registers
  float threadResults[TM];
  #pragma unroll
  for (int i=0;i<TM;++i) threadResults[i] = 0.0f;

  // Compute helpers for loading SMEM:
  // We want to load As (BM x BK) and Bs (BK x BN) using all threads.
  // Use linear index over threads to cover each SMEM array.
  const int numThreads = (int)blockDim.x;

  // Outer loop over K tiles
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {

    // ---- Populate SMEM caches (same as before) ----
    // Load As: BM*BK elements
    for (int idx = tid; idx < BM * BK; idx += numThreads) {
      int r = idx / BK;      // 0..BM-1
      int c = idx % BK;      // 0..BK-1
      int globalRow = cRow * BM + r;
      int globalCol = bkIdx + c;
      if (globalRow < M && globalCol < K) {
        As[idx] = A_ptr[(size_t)r * (size_t)K + (size_t)c];
      } else {
        As[idx] = 0.0f;
      }
    }

    // Load Bs: BK*BN elements
    for (int idx = tid; idx < BK * BN; idx += numThreads) {
      int r = idx / BN;      // 0..BK-1
      int c = idx % BN;      // 0..BN-1
      int globalRow = bkIdx + r;
      int globalCol = cCol * BN + c;
      if (globalRow < K && globalCol < N) {
        Bs[idx] = B_ptr[(size_t)r * (size_t)N + (size_t)c];
      } else {
        Bs[idx] = 0.0f;
      }
    }

    __syncthreads();

    // advance blocktile for outer loop (exactly like your snippet)
    A_ptr += BK;
    B_ptr += (size_t)BK * (size_t)N;

    // ---- Calculate per-thread results ----
    #pragma unroll
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // reuse Bs entry by caching in a tmp var
      float Btmp = Bs[dotIdx * BN + threadCol];

      #pragma unroll
      for (int resIdx = 0; resIdx < TM; ++resIdx) {
        // As is stored as [BM x BK], row-major in SMEM
        float Aval = As[(rowBase + resIdx) * BK + dotIdx];
        threadResults[resIdx] = fmaf(Aval, Btmp, threadResults[resIdx]);
      }
    }

    __syncthreads();
  }

  // ---- Store results ----
  const int outCol = cCol * BN + threadCol;
  if (outCol < N) {
    #pragma unroll
    for (int resIdx = 0; resIdx < TM; ++resIdx) {
      const int outRow = cRow * BM + rowBase + resIdx;
      if (outRow < M) {
        float* c = &C_ptr[(size_t)(rowBase + resIdx) * (size_t)N + (size_t)threadCol];
        *c = alpha * threadResults[resIdx] + beta * (*c);
      }
    }
  }
}

static void launch(int M, int N, int K,
                   float alpha, const float* dA, const float* dB,
                   float beta, float* dC) {
  dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

  // threads = (BM/TM) * BN
  const int threads = (BM / TM) * BN;
  if (threads > 1024) {
    fprintf(stderr,
            "ERROR: threads per block = %d > 1024. "
            "Adjust BM/BN/TM.\n", threads);
    std::exit(1);
  }
  dim3 block(threads);

  sgemm_1d_blocktiling<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
  CHECK_CUDA(cudaGetLastError());
}

int main(int argc, char** argv) {
  int M=4096, N=4096, K=4096;
  int reps=50, warmup=5;

  // Usage: ./bin [M N K] [reps]
  if (argc >= 4) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }
  if (argc >= 5) { reps = std::atoi(argv[4]); }

  printf("kernel_04_1d_blocktiling | BM=%d BN=%d BK=%d TM=%d | M=%d N=%d K=%d reps=%d\n",
         BM, BN, BK, TM, M, N, K, reps);

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
