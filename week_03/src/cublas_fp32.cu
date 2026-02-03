#include "common.cuh"
#include <cublas_v2.h>
#include <cstdio>

static void cublasCheck(cublasStatus_t st, const char* file, int line) {
  if (st != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "[cuBLAS ERROR] %s:%d status=%d\n", file, line, (int)st);
    std::exit(EXIT_FAILURE);
  }
}
#define CUBLAS_CHECK(x) cublasCheck((x), __FILE__, __LINE__)

float run_cublas_fp32(const float* A, const float* B, float* C, int M, int N, int K, int reps, int warmup, cudaStream_t stream) {
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetStream(handle, stream));

  // Keep default math mode (do NOT enable TF32 tensor-op math).
  // This matches "FP32 baseline" intent.
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

  const float alpha = 1.0f, beta = 0.0f;

  // Our matrices are row-major. cuBLAS assumes column-major.
  // Trick: compute C^T = B^T * A^T in column-major, which corresponds to row-major C = A * B.
  // Dimensions:
  // A: MxK row-major => A^T is KxM column-major with leading dim K
  // B: KxN row-major => B^T is NxK column-major with leading dim N
  // C: MxN row-major => C^T is NxM column-major with leading dim N
  //
  // So call: (NxK) * (KxM) = (NxM)
  int lda = N; // for B^T
  int ldb = K; // for A^T
  int ldc = N; // for C^T

  // Warmup
  for (int i = 0; i < warmup; ++i) {
    CUBLAS_CHECK(cublasSgemm(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K,
      &alpha,
      B, lda,
      A, ldb,
      &beta,
      C, ldc
    ));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  GpuTimer t;
  t.tic(stream);
  for (int i = 0; i < reps; ++i) {
    CUBLAS_CHECK(cublasSgemm(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K,
      &alpha,
      B, lda,
      A, ldb,
      &beta,
      C, ldc
    ));
  }
  float ms = t.toc(stream);

  CUBLAS_CHECK(cublasDestroy(handle));
  return ms / reps;
}
