#include "kernels.cuh"

// Placeholder: real bank-conflict fixes would change shared layout.
// For now, just alias to kernel 6 so the pipeline compiles and sweeps.
void launch_kernel_7(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
  launch_kernel_6(A, B, C, M, N, K, s);
}
