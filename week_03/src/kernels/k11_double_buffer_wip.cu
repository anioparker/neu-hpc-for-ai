#include "kernels.cuh"

// WIP placeholder. Replace with real double-buffered pipeline later.
void launch_kernel_11(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
  launch_kernel_6(A, B, C, M, N, K, s);
}
