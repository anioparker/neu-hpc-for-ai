#include "kernels.cuh"

// Placeholder alias; replace with offset-padding shared memory version later.
void launch_kernel_8(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
  launch_kernel_6(A, B, C, M, N, K, s);
}


