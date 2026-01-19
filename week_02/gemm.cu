// gemm.cu
// D = alpha * (A @ B) + beta * C
// Row-major storage for all matrices.
//
// Build:
//   nvcc -O3 -std=c++17 gemm.cu -o gemm
//
// Run:
//   ./gemm

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                   __FILE__, __LINE__, cudaGetErrorString(err));              \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

// ✅ FIX: make idx2 usable from BOTH host (CPU) and device (GPU kernel)
__host__ __device__ static inline size_t idx2(size_t r, size_t c, size_t stride) {
  return r * stride + c;
}

// ----------------------------- CPU reference -----------------------------
void gemm_cpu(const float* A, const float* B, const float* C,
              float* D, int m, int k, int n,
              float alpha, float beta) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int t = 0; t < k; ++t) {
        acc += A[idx2(i, t, k)] * B[idx2(t, j, n)];
      }
      D[idx2(i, j, n)] = alpha * acc + beta * C[idx2(i, j, n)];
    }
  }
}

// ----------------------------- CUDA kernel -----------------------------
// Tiled GEMM: each thread computes one D(i,j)
template<int TILE>
__global__ void gemm_tiled(const float* __restrict__ A,
                           const float* __restrict__ B,
                           const float* __restrict__ C,
                           float* __restrict__ D,
                           int m, int k, int n,
                           float alpha, float beta) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y; // i in [0, m)
  int col = blockIdx.x * TILE + threadIdx.x; // j in [0, n)

  float acc = 0.0f;

  // Loop over tiles along K dimension
  for (int t0 = 0; t0 < k; t0 += TILE) {
    // Load tile of A: (row, t0 + tx)
    int a_col = t0 + threadIdx.x;
    if (row < m && a_col < k) {
      As[threadIdx.y][threadIdx.x] = A[idx2(row, a_col, k)];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.0f;
    }

    // Load tile of B: (t0 + ty, col)
    int b_row = t0 + threadIdx.y;
    if (b_row < k && col < n) {
      Bs[threadIdx.y][threadIdx.x] = B[idx2(b_row, col, n)];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Compute partial dot product for this tile
    #pragma unroll
    for (int tt = 0; tt < TILE; ++tt) {
      acc += As[threadIdx.y][tt] * Bs[tt][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < m && col < n) {
    float cval = C ? C[idx2(row, col, n)] : 0.0f;
    D[idx2(row, col, n)] = alpha * acc + beta * cval;
  }
}

// ----------------------------- Test / main -----------------------------
bool allclose(const std::vector<float>& a,
              const std::vector<float>& b,
              float atol = 1e-4f, float rtol = 1e-4f) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    float diff = std::fabs(a[i] - b[i]);
    float tol = atol + rtol * std::fabs(b[i]);
    if (diff > tol) {
      std::cerr << "Mismatch at i=" << i
                << " got=" << a[i]
                << " expected=" << b[i]
                << " diff=" << diff
                << " tol=" << tol << "\n";
      return false;
    }
  }
  return true;
}

int main() {
  // Example sizes (change as you like)
  int m = 128;
  int k = 256;
  int n = 192;

  float alpha = 1.25f;
  float beta  = -0.5f;

  size_t Asz = (size_t)m * k;
  size_t Bsz = (size_t)k * n;
  size_t Csz = (size_t)m * n;

  std::vector<float> hA(Asz), hB(Bsz), hC(Csz), hD(Csz), hDref(Csz);

  // Fill inputs (deterministic so runs are repeatable)
  for (size_t i = 0; i < Asz; ++i) hA[i] = (float)((i % 13) - 6) * 0.1f;
  for (size_t i = 0; i < Bsz; ++i) hB[i] = (float)((i % 17) - 8) * 0.07f;
  for (size_t i = 0; i < Csz; ++i) hC[i] = (float)((i % 19) - 9) * 0.05f;

  // CPU reference
  gemm_cpu(hA.data(), hB.data(), hC.data(), hDref.data(), m, k, n, alpha, beta);

  // Device buffers
  float *dA=nullptr, *dB=nullptr, *dC=nullptr, *dD=nullptr;
  CUDA_CHECK(cudaMalloc(&dA, Asz * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, Bsz * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, Csz * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dD, Csz * sizeof(float)));

  // Copy inputs to GPU
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), Asz * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), Bsz * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dC, hC.data(), Csz * sizeof(float), cudaMemcpyHostToDevice));

  // Launch tiled GEMM
  constexpr int TILE = 16;
  dim3 block(TILE, TILE, 1);
  dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE, 1);

  gemm_tiled<TILE><<<grid, block>>>(dA, dB, dC, dD, m, k, n, alpha, beta);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy output back to CPU
  CUDA_CHECK(cudaMemcpy(hD.data(), dD, Csz * sizeof(float), cudaMemcpyDeviceToHost));

  // Validate
  bool ok = allclose(hD, hDref);
  std::cout << (ok ? "PASS" : "FAIL") << "\n";

  // Cleanup
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaFree(dD));

  return ok ? 0 : 1;
}
