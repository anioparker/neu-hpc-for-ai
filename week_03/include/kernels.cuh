#pragma once
#include <cuda_runtime.h>

struct KernelResult {
  float ms = 0.0f;
  float max_abs_err = 0.0f;
  bool pass = true;
};

void launch_kernel_1(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s);
void launch_kernel_2(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s);
void launch_kernel_3(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s);
void launch_kernel_4(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s);
void launch_kernel_5(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s);
void launch_kernel_6(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s);

void launch_kernel_7(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s);
void launch_kernel_8(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s);

void launch_kernel_10(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s);
void launch_kernel_11(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s);

// Autotune (kernel 9)
void launch_kernel_9_autotune(const float* A, const float* B, float* C, int M, int N, int K, int reps, int warmup, cudaStream_t s);
