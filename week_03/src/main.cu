#include "common.cuh"
#include "runner.cuh"
#include "kernels.cuh"
#include "metrics.cuh"

#include <cstdio>
#include <vector>
#include <random>

// From cublas_fp32.cu
float run_cublas_fp32(const float* A, const float* B, float* C, int M, int N, int K, int reps, int warmup, cudaStream_t stream);
// From verify.cu
float max_abs_diff(const float* dA, const float* dB, int n, cudaStream_t stream);

static void fill_random(std::vector<float>& v) {
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : v) x = dist(rng);
}

static void zero_out(float* d, size_t bytes, cudaStream_t s) {
  CUDA_CHECK(cudaMemsetAsync(d, 0, bytes, s));
}

static void run_custom_kernel(int kid, const float* A, const float* B, float* C, int M, int N, int K, int reps, int warmup, cudaStream_t s) {
  // Warmup
  for (int i = 0; i < warmup; ++i) {
    switch (kid) {
      case 1: launch_kernel_1(A, B, C, M, N, K, s); break;
      case 2: launch_kernel_2(A, B, C, M, N, K, s); break;
      case 3: launch_kernel_3(A, B, C, M, N, K, s); break;
      case 4: launch_kernel_4(A, B, C, M, N, K, s); break;
      case 5: launch_kernel_5(A, B, C, M, N, K, s); break;
      case 6: launch_kernel_6(A, B, C, M, N, K, s); break;
      case 7: launch_kernel_7(A, B, C, M, N, K, s); break;
      case 8: launch_kernel_8(A, B, C, M, N, K, s); break;
      case 10: launch_kernel_10(A, B, C, M, N, K, s); break;
      case 11: launch_kernel_11(A, B, C, M, N, K, s); break;
      default: break;
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(s));

  // Timed
  GpuTimer t;
  t.tic(s);
  if (kid == 9) {
    // Autotune does its own timing internally and then launches the best config for reps.
    launch_kernel_9_autotune(A, B, C, M, N, K, reps, 0, s);
  } else {
    for (int i = 0; i < reps; ++i) {
      switch (kid) {
        case 1: launch_kernel_1(A, B, C, M, N, K, s); break;
        case 2: launch_kernel_2(A, B, C, M, N, K, s); break;
        case 3: launch_kernel_3(A, B, C, M, N, K, s); break;
        case 4: launch_kernel_4(A, B, C, M, N, K, s); break;
        case 5: launch_kernel_5(A, B, C, M, N, K, s); break;
        case 6: launch_kernel_6(A, B, C, M, N, K, s); break;
        case 7: launch_kernel_7(A, B, C, M, N, K, s); break;
        case 8: launch_kernel_8(A, B, C, M, N, K, s); break;
        case 10: launch_kernel_10(A, B, C, M, N, K, s); break;
        case 11: launch_kernel_11(A, B, C, M, N, K, s); break;
        default: break;
      }
    }
  }
  float ms_total = t.toc(s);
  std::printf("Kernel time (avg): %.6f ms\n", ms_total / reps);
}

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  printDeviceInfo();

  int M = args.m, N = args.n, K = args.k;
  size_t bytesA = (size_t)M * K * sizeof(float);
  size_t bytesB = (size_t)K * N * sizeof(float);
  size_t bytesC = (size_t)M * N * sizeof(float);

  std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
  fill_random(hA);
  fill_random(hB);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  float *dA=nullptr, *dB=nullptr, *dC=nullptr, *dCref=nullptr;
  CUDA_CHECK(cudaMalloc(&dA, bytesA));
  CUDA_CHECK(cudaMalloc(&dB, bytesB));
  CUDA_CHECK(cudaMalloc(&dC, bytesC));
  CUDA_CHECK(cudaMalloc(&dCref, bytesC));

  CUDA_CHECK(cudaMemcpyAsync(dA, hA.data(), bytesA, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(dB, hB.data(), bytesB, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Always build a cuBLAS reference (for correctness + relative perf)
  zero_out(dCref, bytesC, stream);
  float cublas_ms = run_cublas_fp32(dA, dB, dCref, M, N, K, args.reps, args.warmup, stream);

  double flops = flops_gemm(M, N, K);
  double cublas_gflops = gflops_per_s(flops, cublas_ms);

  std::printf("M=%d N=%d K=%d reps=%d\n", M, N, K, args.reps);
  std::printf("cuBLAS FP32 avg time: %.6f ms\n", cublas_ms);
  std::printf("cuBLAS throughput: %.2f GFLOP/s\n", cublas_gflops);

  // If user requested cuBLAS only
  if (args.kernel == 0) {
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dCref));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
  }

  // Run custom kernel
  zero_out(dC, bytesC, stream);
  run_custom_kernel(args.kernel, dA, dB, dC, M, N, K, args.reps, args.warmup, stream);

  // Compute perf
  // (Kernel prints its own avg time; we recompute by re-timing quickly to record a number.)
  // For simplicity: do one more timed loop
  zero_out(dC, bytesC, stream);
  GpuTimer t;
  t.tic(stream);
  if (args.kernel == 9) {
    launch_kernel_9_autotune(dA, dB, dC, M, N, K, args.reps, args.warmup, stream);
  } else {
    for (int i = 0; i < args.reps; ++i) {
      switch (args.kernel) {
        case 1: launch_kernel_1(dA, dB, dC, M, N, K, stream); break;
        case 2: launch_kernel_2(dA, dB, dC, M, N, K, stream); break;
        case 3: launch_kernel_3(dA, dB, dC, M, N, K, stream); break;
        case 4: launch_kernel_4(dA, dB, dC, M, N, K, stream); break;
        case 5: launch_kernel_5(dA, dB, dC, M, N, K, stream); break;
        case 6: launch_kernel_6(dA, dB, dC, M, N, K, stream); break;
        case 7: launch_kernel_7(dA, dB, dC, M, N, K, stream); break;
        case 8: launch_kernel_8(dA, dB, dC, M, N, K, stream); break;
        case 10: launch_kernel_10(dA, dB, dC, M, N, K, stream); break;
        case 11: launch_kernel_11(dA, dB, dC, M, N, K, stream); break;
      }
    }
  }
  float ms_total = t.toc(stream);
  float ms_avg = ms_total / args.reps;

  double gflops = gflops_per_s(flops, ms_avg);
  std::printf("Throughput: %.2f GFLOP/s\n", gflops);
  std::printf("Relative to cuBLAS: %.4f\n", gflops / cublas_gflops);

  if (args.verify) {
    float err = max_abs_diff(dC, dCref, M * N, stream);
    std::printf("Max abs error vs cuBLAS: %.6e\n", err);
  }

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaFree(dCref));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
