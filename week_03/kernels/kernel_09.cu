// kernel_09.cu

// Build (H100 / Hopper):
//   nvcc -O3 -std=c++17 -arch=sm_90 -lineinfo kernel_09.cu -o autotuning
//
// Run (recommended: multiples of BM/BN/BK and multiples of 4 for float4):
//   ./autotuning 4096 4096 4096 50
//
// Notes:
// Kernel 9: Autotuning (vectorized float4 GMEM/SMEM + transposed A in SMEM + warptiling)
// - This autotuner benchmarks a small set of (BM,BN,BK,TM,TN) candidates and prints the best.
// - This code intentionally assumes “clean” sizes for performance benchmarking:
// M % BM == 0, N % BN == 0, K % BK == 0, and N,K multiples of 4.
//   If you want fully general sizes, you need boundary guards in loads/stores.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
static constexpr int K9_NUM_THREADS = 256;

#define CHK(call) do {                                      \
    cudaError_t err = (call);                               \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA error %s:%d: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(1);                                       \
    }                                                       \
} while (0)

static void fill(float* p, size_t n) {
    for (size_t i = 0; i < n; ++i) p[i] = (float)rand() / RAND_MAX - 0.5f;
}

static double gemm_flops(int M, int N, int K) {
    // GEMM convention: 2*M*N*K FLOPs
    return 2.0 * (double)M * (double)N * (double)K;
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(K9_NUM_THREADS)
sgemmAutotuned(int M, int N, int K, float alpha,
               const float* __restrict__ A,
               const float* __restrict__ B,
               float beta,
               float* __restrict__ C) {

  // Basic constraints for this kernel shape (compile-time):
  static_assert((BK % 4) == 0, "BK must be multiple of 4 for float4 loads");
  static_assert((BN % 4) == 0, "BN must be multiple of 4 for float4 loads/stores");
  static_assert((TN % 4) == 0, "TN must be multiple of 4 for float4 stores");

  const uint cRow = (uint)blockIdx.y;
  const uint cCol = (uint)blockIdx.x;

  // size of warptile
  constexpr int WM = TM * 16;
  constexpr int WN = TN * 16;
  // iterations of warptile inside block tile
  constexpr int WMITER = CEIL_DIV(BM, WM);
  constexpr int WNITER = CEIL_DIV(BN, WN);

  // For safety: require exact tiling (no partial warptiles inside a block tile)
  static_assert((BM % WM) == 0, "BM must be divisible by (TM*16) for this implementation");
  static_assert((BN % WN) == 0, "BN must be divisible by (TN*16) for this implementation");

  // Placement of the thread in the warptile
  // Threadblock is 1D: 256 threads
  const int threadCol = (int)threadIdx.x % (WN / TN);
  const int threadRow = (int)threadIdx.x / (WN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BK * BM];   // NOTE: A is stored TRANSPOSED: As[dotIdx * BM + row]
  __shared__ float Bs[BK * BN];

  // Move blocktile pointers to beginning of A's row and B's column and C tile
  A += (size_t)cRow * BM * (size_t)K;
  B += (size_t)cCol * BN;
  C += (size_t)cRow * BM * (size_t)N + (size_t)cCol * BN;

  // calculating the indices that this thread will load into SMEM
  // We'll load 128-bit / 32-bit = 4 elements per thread at each step
  const uint innerRowA = (uint)threadIdx.x / (BK / 4);
  const uint innerColA = (uint)threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (K9_NUM_THREADS * 4) / BK;   // = 1024/BK rows per "sweep"

  const uint innerRowB = (uint)threadIdx.x / (BN / 4);
  const uint innerColB = (uint)threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = K9_NUM_THREADS / (BN / 4);   // = 256 / (BN/4) rows per "sweep"

  // allocate thread-local cache for results in registerfile
  float threadResults[WMITER * WNITER * TM * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  // outer-most loop over block tiles along K
  for (uint bkIdx = 0; bkIdx < (uint)K; bkIdx += BK) {

    // populate the SMEM caches (vectorized float4 GMEM loads)
    // A: transpose while storing into As (so later loads are contiguous from As)
    for (uint offset = 0; offset + rowStrideA <= (uint)BM; offset += rowStrideA) {
      // global A index: (innerRowA+offset, innerColA*4..+3)
      float4 tmp = *reinterpret_cast<const float4*>(
          &A[(size_t)(innerRowA + offset) * (size_t)K + (size_t)innerColA * 4]);

      // store transposed: As[(dot)*BM + row]
      As[(innerColA * 4 + 0) * BM + (innerRowA + offset)] = tmp.x;
      As[(innerColA * 4 + 1) * BM + (innerRowA + offset)] = tmp.y;
      As[(innerColA * 4 + 2) * BM + (innerRowA + offset)] = tmp.z;
      As[(innerColA * 4 + 3) * BM + (innerRowA + offset)] = tmp.w;
    }

    // B: direct store, vectorized float4
    for (uint offset = 0; offset + rowStrideB <= (uint)BK; offset += rowStrideB) {
      *reinterpret_cast<float4*>(
          &Bs[(innerRowB + offset) * BN + innerColB * 4]) =
          *reinterpret_cast<const float4*>(
              &B[(size_t)(innerRowB + offset) * (size_t)N + (size_t)innerColB * 4]);
    }

    __syncthreads();

    // compute block tile in warptiles
    for (uint wmIdx = 0; wmIdx < (uint)WMITER; ++wmIdx) {
      for (uint wnIdx = 0; wnIdx < (uint)WNITER; ++wnIdx) {
        for (uint dotIdx = 0; dotIdx < (uint)BK; ++dotIdx) {
          // load A (from transposed As) into regM
          #pragma unroll
          for (uint i = 0; i < (uint)TM; ++i) {
            regM[i] = As[dotIdx * BM + (wmIdx * WM) + (uint)(threadRow * TM) + i];
          }
          // load B into regN
          #pragma unroll
          for (uint i = 0; i < (uint)TN; ++i) {
            regN[i] = Bs[dotIdx * BN + (wnIdx * WN) + (uint)(threadCol * TN) + i];
          }

          // outer-product accumulate into threadResults
          #pragma unroll
          for (uint resIdxM = 0; resIdxM < (uint)TM; ++resIdxM) {
            #pragma unroll
            for (uint resIdxN = 0; resIdxN < (uint)TN; ++resIdxN) {
              // Layout: [wm block of rows][wn block of cols]
              const uint idx =
                  (wmIdx * TM + resIdxM) * (WNITER * TN) + wnIdx * TN + resIdxN;
              threadResults[idx] = fmaf(regM[resIdxM], regN[resIdxN], threadResults[idx]);
            }
          }
        }
      }
    }

    __syncthreads();

    // advance blocktile
    A += BK;           // move BK columns to right
    B += (size_t)BK * (size_t)N;  // move BK rows down
  }

  // write out results (vectorized float4 stores)
  for (uint wmIdx = 0; wmIdx < (uint)WMITER; ++wmIdx) {
    for (uint wnIdx = 0; wnIdx < (uint)WNITER; ++wnIdx) {
      float* C_interim = C + (size_t)wmIdx * WM * (size_t)N + (size_t)wnIdx * WN;

      for (uint resIdxM = 0; resIdxM < (uint)TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < (uint)TN; resIdxN += 4) {
          // Load C vector into registers
          float4 tmp = *reinterpret_cast<float4*>(
              &C_interim[(size_t)(threadRow * TM + resIdxM) * (size_t)N +
                         (size_t)(threadCol * TN + resIdxN)]);

          // perform GEMM update in registers
          const uint base =
              (wmIdx * TM + resIdxM) * (WNITER * TN) + wnIdx * TN + resIdxN;

          tmp.x = alpha * threadResults[base + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[base + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[base + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[base + 3] + beta * tmp.w;

          // store back
          *reinterpret_cast<float4*>(
              &C_interim[(size_t)(threadRow * TM + resIdxM) * (size_t)N +
                         (size_t)(threadCol * TN + resIdxN)]) = tmp;
        }
      }
    }
  }
}

// ---------------- Autotuning harness ----------------

struct Cand {
  int BM, BN, BK, TM, TN;
  const char* name;
};

static constexpr Cand CANDS[] = {
  // Keep this set small at first; expand after you have a stable baseline.
  {128, 128,  8, 8, 8, "128x128x8  TM8 TN8"},
  {128, 128, 16, 8, 8, "128x128x16 TM8 TN8"},
  {128,  64, 16, 8, 4, "128x64x16  TM8 TN4"},
  { 64, 128, 16, 4, 8, "64x128x16  TM4 TN8"},
};

template<int BM, int BN, int BK, int TM, int TN>
static void launch_one(int M, int N, int K,
                       float alpha,
                       const float* dA,
                       const float* dB,
                       float beta,
                       float* dC) {
  dim3 block(K9_NUM_THREADS, 1, 1);
  dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);
  sgemmAutotuned<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
}

static void launch_by_id(int id, int M, int N, int K,
                         float alpha, const float* dA, const float* dB,
                         float beta, float* dC) {
  switch (id) {
    case 0: return launch_one<128,128, 8,8,8>(M,N,K,alpha,dA,dB,beta,dC);
    case 1: return launch_one<128,128,16,8,8>(M,N,K,alpha,dA,dB,beta,dC);
    case 2: return launch_one<128, 64,16,8,4>(M,N,K,alpha,dA,dB,beta,dC);
    case 3: return launch_one< 64,128,16,4,8>(M,N,K,alpha,dA,dB,beta,dC);
    default: return;
  }
}

static void die_bad_shape(const char* msg, int M, int N, int K) {
  fprintf(stderr, "Bad shape: %s (M=%d N=%d K=%d)\n", msg, M, N, K);
  std::exit(1);
}

int main(int argc, char** argv) {
  int M = 4096, N = 4096, K = 4096, reps = 50;
  if (argc >= 4) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }
  if (argc >= 5) reps = std::atoi(argv[4]);

  // Hard requirements for this benchmark kernel (no edge guards)
  if ((N % 4) != 0 || (K % 4) != 0) die_bad_shape("N and K must be multiples of 4 for float4", M,N,K);

  const float alpha = 1.0f;
  const float beta  = 0.0f;

  size_t bytesA = (size_t)M * (size_t)K * sizeof(float);
  size_t bytesB = (size_t)K * (size_t)N * sizeof(float);
  size_t bytesC = (size_t)M * (size_t)N * sizeof(float);

  float* hA = (float*)std::malloc(bytesA);
  float* hB = (float*)std::malloc(bytesB);
  float* hC = (float*)std::malloc(bytesC);
  if (!hA || !hB || !hC) { fprintf(stderr, "Host malloc failed\n"); return 1; }

  fill(hA, (size_t)M * (size_t)K);
  fill(hB, (size_t)K * (size_t)N);
  fill(hC, (size_t)M * (size_t)N);

  float *dA=nullptr, *dB=nullptr, *dC=nullptr;
  CHK(cudaMalloc(&dA, bytesA));
  CHK(cudaMalloc(&dB, bytesB));
  CHK(cudaMalloc(&dC, bytesC));
  CHK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(dC, hC, bytesC, cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CHK(cudaEventCreate(&start));
  CHK(cudaEventCreate(&stop));

  int best_id = -1;
  float best_ms = 1e30f;

  printf("Kernel 9 autotune | M=%d N=%d K=%d reps=%d | candidates=%zu\n",
         M, N, K, reps, sizeof(CANDS)/sizeof(CANDS[0]));

  for (int id = 0; id < (int)(sizeof(CANDS)/sizeof(CANDS[0])); ++id) {
    const auto& c = CANDS[id];

    // Since kernel has no boundary guards, ensure exact tiling for each candidate.
    if ((M % c.BM) != 0 || (N % c.BN) != 0 || (K % c.BK) != 0) {
      printf("cand %d: %s  (skipped: require M%%BM==0, N%%BN==0, K%%BK==0)\n", id, c.name);
      continue;
    }

    // warmup
    launch_by_id(id, M, N, K, alpha, dA, dB, beta, dC);
    CHK(cudaGetLastError());
    CHK(cudaDeviceSynchronize());

    CHK(cudaEventRecord(start));
    for (int r = 0; r < reps; ++r) {
      launch_by_id(id, M, N, K, alpha, dA, dB, beta, dC);
    }
    CHK(cudaGetLastError());
    CHK(cudaEventRecord(stop));
    CHK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHK(cudaEventElapsedTime(&ms, start, stop));
    float ms_avg = ms / reps;

    double gflops = gemm_flops(M, N, K) / (ms_avg * 1e6);
    printf("cand %d: %-22s  %.3f ms  %.2f GFLOP/s\n", id, c.name, ms_avg, gflops);

    if (ms_avg < best_ms) { best_ms = ms_avg; best_id = id; }
  }

  if (best_id < 0) {
    fprintf(stderr, "No candidate matched the shape constraints. Try M=N=K=4096.\n");
  } else {
    printf("\nBEST: cand %d: %s  %.3f ms\n", best_id, CANDS[best_id].name, best_ms);
  }

  CHK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));

  CHK(cudaFree(dA));
  CHK(cudaFree(dB));
  CHK(cudaFree(dC));
  std::free(hA);
  std::free(hB);
  std::free(hC);
  CHK(cudaEventDestroy(start));
  CHK(cudaEventDestroy(stop));
  return 0;
}
