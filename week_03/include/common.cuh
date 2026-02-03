#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

inline void cudaCheck(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", file, line, cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }
}
#define CUDA_CHECK(val) cudaCheck((val), __FILE__, __LINE__)

struct GpuTimer {
  cudaEvent_t start{}, stop{};
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }
  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  void tic(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(start, s)); }
  float toc(cudaStream_t s = 0) {
    CUDA_CHECK(cudaEventRecord(stop, s));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

inline void printDeviceInfo() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::printf("Device %d: %s\n", dev, prop.name);
  std::printf("  SMs: %d\n", prop.multiProcessorCount);
  std::printf("  CC: %d.%d\n", prop.major, prop.minor);
  std::printf("  Global mem: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  std::printf("  Shared per block: %zu bytes\n", prop.sharedMemPerBlock);
  std::printf("  Max threads/block: %d\n", prop.maxThreadsPerBlock);
}
