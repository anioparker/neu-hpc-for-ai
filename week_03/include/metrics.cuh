#pragma once
#include <cstdint>

inline double flops_gemm(int m, int n, int k) {
  // Same spirit as the blog: 2*m*n*k + m*n (for the output update)
  return 2.0 * (double)m * (double)n * (double)k + 1.0 * (double)m * (double)n;
}

inline double bytes_min_gemm(int m, int n, int k) {
  // Read A (m*k), read B (k*n), read C (m*n), write C (m*n)
  // float32 => 4 bytes
  double elems = (double)m * k + (double)k * n + (double)m * n + (double)m * n;
  return elems * 4.0;
}

inline double gflops_per_s(double flops, double ms) {
  return (flops / 1.0e9) / (ms / 1.0e3);
}
