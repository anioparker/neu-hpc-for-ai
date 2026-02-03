#include "common.cuh"
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>

// Variant symbols from k05_blocktiling_2d.cu
extern "C" void k5_launch_128_128_16_8_8(const float*, const float*, float*, int, int, int, cudaStream_t);
extern "C" void k5_launch_128_64_16_8_8(const float*, const float*, float*, int, int, int, cudaStream_t);
extern "C" void k5_launch_64_128_16_8_8(const float*, const float*, float*, int, int, int, cudaStream_t);
extern "C" void k5_launch_64_64_16_8_8(const float*, const float*, float*, int, int, int, cudaStream_t);

struct Cand {
  const char* name;
  void (*fn)(const float*, const float*, float*, int, int, int, cudaStream_t);
};

static float time_candidate(const Cand& c, const float* A, const float* B, float* C, int M, int N, int K, int reps, int warmup, cudaStream_t s) {
  for (int i = 0; i < warmup; ++i) c.fn(A, B, C, M, N, K, s);
  CUDA_CHECK(cudaStreamSynchronize(s));
  GpuTimer t;
  t.tic(s);
  for (int i = 0; i < reps; ++i) c.fn(A, B, C, M, N, K, s);
  float ms = t.toc(s);
  return ms / reps;
}

void launch_kernel_9_autotune(const float* A, const float* B, float* C, int M, int N, int K, int reps, int warmup, cudaStream_t s) {
  Cand cands[] = {
    {"k5_128x128x16_tm8_tn8", &k5_launch_128_128_16_8_8},
    {"k5_128x64x16_tm8_tn8",  &k5_launch_128_64_16_8_8},
    {"k5_64x128x16_tm8_tn8",  &k5_launch_64_128_16_8_8},
    {"k5_64x64x16_tm8_tn8",   &k5_launch_64_64_16_8_8},
  };

  // quick-tune reps: small number so tuning overhead is low
  int tune_reps = 10;
  int tune_warm = warmup > 0 ? 2 : 0;

  float best_ms = 1e30f;
  int best_i = 0;

  for (int i = 0; i < (int)(sizeof(cands)/sizeof(cands[0])); ++i) {
    float ms = time_candidate(cands[i], A, B, C, M, N, K, tune_reps, tune_warm, s);
    if (ms < best_ms) { best_ms = ms; best_i = i; }
  }

  std::printf("Autotune picked: %s (%.6f ms)\n", cands[best_i].name, best_ms);

  // Run the winner for the requested reps (no extra warmup unless caller wants)
  for (int i = 0; i < reps; ++i) cands[best_i].fn(A, B, C, M, N, K, s);
  CUDA_CHECK(cudaGetLastError());
}
