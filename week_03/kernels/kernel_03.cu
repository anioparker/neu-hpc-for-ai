// kernel_03_smem_caching.cu
// nvcc -O3 -arch=native -lineinfo kernel_03.cu -o smem_caching
// ./smem_caching 4096 4096 4096 50



// Shared-memory (SMEM) caching / tiling SGEMM (row-major).
// Matches the structure you pasted: advance A/B/C pointers per (cRow,cCol),
// loop bkIdx over K in steps of BLOCKSIZE, load As/Bs to SMEM, dot, store.
// Prints GFLOPs/s using FLOPs ~= 2*M*N*K.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>

#ifndef BLOCKSIZE
#define BLOCKSIZE 32
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
  // SGEMM FLOPs ~ 2*M*N*K (mul+add per inner K)
  // GFLOPs/s = (2*M*N*K) / (ms_avg * 1e6)
  double flops = 2.0 * (double)M * (double)N * (double)K;
  return flops / (ms_avg * 1e6);
}

__global__ void sgemm_smem_cache(int M, int N, int K,
                                 float alpha,
                                 const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float beta,
                                 float* __restrict__ C) {
  // Shared memory tiles: BLOCKSIZE x BLOCKSIZE
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // Block index in C tiles
  const int cRow = (int)blockIdx.y;
  const int cCol = (int)blockIdx.x;

  // Thread coordinates inside tile
  const int threadRow = (int)threadIdx.y;
  const int threadCol = (int)threadIdx.x;

  // Guard: if thread maps outside the actual matrix edges, we can still run,
  // but must bounds-check loads/stores. We'll do bounds checks per access.
  // (This keeps the code close to your snippet but safe for non-multiples.)

  // advance pointers to the starting positions (exactly like your snippet)
  const float* A_ptr = A + (size_t)cRow * BLOCKSIZE * (size_t)K;                    // row=cRow, col=0
  const float* B_ptr = B + (size_t)cCol * BLOCKSIZE;                               // row=0, col=cCol
  float*       C_ptr = C + (size_t)cRow * BLOCKSIZE * (size_t)N + (size_t)cCol * BLOCKSIZE; // row=cRow, col=cCol

  float tmp = 0.0f;

  // Outer loop over K dimension in BLOCKSIZE chunks
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // Global coordinates for the elements this thread will load
    const int aRow = cRow * BLOCKSIZE + threadRow;
    const int aCol = bkIdx + threadCol;

    const int bRow = bkIdx + threadRow;
    const int bCol = cCol * BLOCKSIZE + threadCol;

    // Load A and B tiles into shared memory (coalesced on threadCol)
    if (aRow < M && aCol < K) {
      As[threadRow * BLOCKSIZE + threadCol] = A_ptr[(size_t)threadRow * (size_t)K + threadCol];
    } else {
      As[threadRow * BLOCKSIZE + threadCol] = 0.0f;
    }

    if (bRow < K && bCol < N) {
      Bs[threadRow * BLOCKSIZE + threadCol] = B_ptr[(size_t)threadRow * (size_t)N + threadCol];
    } else {
      Bs[threadRow * BLOCKSIZE + threadCol] = 0.0f;
    }

    __syncthreads();

    // advance pointers onto next chunk (exactly like your snippet)
    A_ptr += BLOCKSIZE;         // move along A columns
    B_ptr += (size_t)BLOCKSIZE * (size_t)N; // move along B rows

    // dot product for this tile
    #pragma unroll
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp = fmaf(As[threadRow * BLOCKSIZE + dotIdx],
                 Bs[dotIdx * BLOCKSIZE + threadCol],
                 tmp);
    }

    __syncthreads();
  }

  // Write C
  const int cRowAbs = cRow * BLOCKSIZE + threadRow;
  const int cColAbs = cCol * BLOCKSIZE + threadCol;
  if (cRowAbs < M && cColAbs < N) {
    const int idx = threadRow * N + threadCol; // relative within this tile pointer
    C_ptr[idx] = alpha * tmp + beta * C_ptr[idx];
  }
}

int main(int argc, char** argv) {
  int M = 4096, N = 4096, K = 4096;
  int reps = 50, warmup = 5;

  // Usage: ./a.out [M N K] [reps]
  if (argc >= 4) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }
  if (argc >= 5) { reps = std::atoi(argv[4]); }

  printf("kernel_03_smem_caching | BLOCKSIZE=%d | M=%d N=%d K=%d reps=%d\n",
         BLOCKSIZE, M, N, K, reps);

  const size_t sizeA = (size_t)M * (size_t)K;
  const size_t sizeB = (size_t)K * (size_t)N;
  const size_t sizeC = (size_t)M * (size_t)N;

  float *hA = (float*)std::malloc(sizeA * sizeof(float));
  float *hB = (float*)std::malloc(sizeB * sizeof(float));
  float *hC = (float*)std::malloc(sizeC * sizeof(float));
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

  // One block computes one C tile
  dim3 gridDim(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE));
  dim3 blockDim(BLOCKSIZE, BLOCKSIZE);

  // Warmup
  for (int i=0;i<warmup;++i) {
    sgemm_smem_cache<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // Timed
  GpuTimer t;
  t.tic();
  for (int i=0;i<reps;++i) {
    sgemm_smem_cache<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
  }
  CHECK_CUDA(cudaGetLastError());
  float ms_total = t.toc();
  CHECK_CUDA(cudaDeviceSynchronize());

  double ms_avg = ms_total / (double)reps;
  double gflops = gflops_sgemm(M, N, K, ms_avg);

  printf("avg time: %.4f ms  |  throughput: %.2f GFLOP/s\n", ms_avg, gflops);

  CHECK_CUDA(cudaFree(dA));
  CHECK_CUDA(cudaFree(dB));
  CHECK_CUDA(cudaFree(dC));
  std::free(hA); std::free(hB); std::free(hC);
  return 0;
}
