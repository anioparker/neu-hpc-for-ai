// kernel_02_gmem_coalescing.cu
// Build: nvcc -O3 -arch=native -lineinfo kernel_02.cu -o gmem_coalescing
// Run: ./gmem_coalescing 4096 4096 4096 50

// GMEM coalescing via 1D block -> 2D tile mapping.
// One block computes a BLOCKSIZE x BLOCKSIZE tile of C (row-major).

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

// threadId = threadIdx.x + blockDim.x*(threadIdx.y + blockDim.y*threadIdx.z)
// Here we use a 1D block, so threadId == threadIdx.x.
//
// Mapping (exactly as you wrote):
//   x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE)
//   y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE)
__global__ void sgemm_coalescing(int M, int N, int K,
                                float alpha,
                                const float* __restrict__ A,
                                const float* __restrict__ B,
                                float beta,
                                float* __restrict__ C) {
  const int tid = (int)threadIdx.x;  // 0 .. BLOCKSIZE*BLOCKSIZE-1

  const int x = (int)blockIdx.x * BLOCKSIZE + (tid / BLOCKSIZE); // row in C
  const int y = (int)blockIdx.y * BLOCKSIZE + (tid % BLOCKSIZE); // col in C

  if (x < M && y < N) {
    float tmp = 0.0f;
    // A is row-major (M x K), B is row-major (K x N), C is row-major (M x N)
    for (int i = 0; i < K; ++i) {
      tmp = fmaf(A[x * K + i], B[i * N + y], tmp);
    }
    const int cidx = x * N + y;
    C[cidx] = alpha * tmp + beta * C[cidx];
  }
}

static inline double gflops_sgemm(int M, int N, int K, double ms_avg) {
  // SGEMM FLOPs ~ 2*M*N*K (mul+add per inner K)
  // GFLOPs/s = flops / time_s / 1e9
  // time_s = ms/1e3 => GFLOPs/s = (2*M*N*K) / (ms*1e6)
  double flops = 2.0 * (double)M * (double)N * (double)K;
  return flops / (ms_avg * 1e6);
}

int main(int argc, char** argv) {
  int M = 4096, N = 4096, K = 4096;
  int reps = 50, warmup = 5;

  // Usage: ./a.out [M N K] [reps]
  if (argc >= 4) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }
  if (argc >= 5) { reps = std::atoi(argv[4]); }

  printf("kernel_02_gmem_coalescing | BLOCKSIZE=%d | M=%d N=%d K=%d reps=%d\n",
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

  // gridDim stays the same (as your snippet, just note we map x->M, y->N):
  dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
  // make blockDim 1D, but keep same number of threads:
  dim3 blockDim(BLOCKSIZE * BLOCKSIZE);

  // Warmup
  for (int i=0;i<warmup;++i) {
    sgemm_coalescing<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // Timed
  GpuTimer t;
  t.tic();
  for (int i=0;i<reps;++i) {
    sgemm_coalescing<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
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
