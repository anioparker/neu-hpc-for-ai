// gemm_inplace_transpose.cu
// C <- alpha * op(A) * op(B) + beta * C
// Optional transpose of A and/or B (row-major).
//
// Build:
//   nvcc -O3 -std=c++17 gemm_inplace_transpose.cu -o gemm_1
//
// Run:
//   ./gemm_1

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>  // ✅ for std::copy

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                   __FILE__, __LINE__, cudaGetErrorString(err));              \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

__host__ __device__ static inline size_t idx2(size_t r, size_t c, size_t stride) {
  return r * stride + c;
}

// ----------------------------- CPU reference (in-place) -----------------------------
// Same storage conventions as GPU.
// op(A): m x k, op(B): k x n, C: m x n
static void gemm_cpu_inplace(float alpha,
                             const float* A, bool transA,
                             const float* B, bool transB,
                             float beta,
                             float* C,
                             int m, int n, int k) {
  // Snapshot old C so we can update C in-place safely
  std::vector<float> C_old((size_t)m * (size_t)n);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      C_old[idx2((size_t)i, (size_t)j, (size_t)n)] =
          C[idx2((size_t)i, (size_t)j, (size_t)n)];

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int t = 0; t < k; ++t) {
        float a = transA ? A[idx2((size_t)t, (size_t)i, (size_t)m)]  // A stored (k x m)
                         : A[idx2((size_t)i, (size_t)t, (size_t)k)]; // A stored (m x k)

        float b = transB ? B[idx2((size_t)j, (size_t)t, (size_t)k)]  // B stored (n x k)
                         : B[idx2((size_t)t, (size_t)j, (size_t)n)]; // B stored (k x n)

        acc += a * b;
      }
      C[idx2((size_t)i, (size_t)j, (size_t)n)] =
          alpha * acc + beta * C_old[idx2((size_t)i, (size_t)j, (size_t)n)];
    }
  }
}

static bool allclose(const std::vector<float>& a,
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

// ----------------------------- Device access helpers -----------------------------
template<bool TA>
__device__ __forceinline__ float loadA(const float* __restrict__ A,
                                       int i, int t,
                                       int m, int k) {
  if constexpr (!TA) {
    return A[(size_t)i * (size_t)k + (size_t)t]; // A stored (m x k)
  } else {
    return A[(size_t)t * (size_t)m + (size_t)i]; // A stored (k x m), op(A)=A^T
  }
}

template<bool TB>
__device__ __forceinline__ float loadB(const float* __restrict__ B,
                                       int t, int j,
                                       int n, int k) {
  if constexpr (!TB) {
    return B[(size_t)t * (size_t)n + (size_t)j]; // B stored (k x n)
  } else {
    return B[(size_t)j * (size_t)k + (size_t)t]; // B stored (n x k), op(B)=B^T
  }
}

// ----------------------------- CUDA kernel (in-place C) -----------------------------
template<int TILE, bool TA, bool TB>
__global__ void gemm_tiled_inplace(float alpha,
                                  const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float beta,
                                  float* __restrict__ C,
                                  int m, int n, int k) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  float acc = 0.0f;

  for (int t0 = 0; t0 < k; t0 += TILE) {
    int tA = t0 + threadIdx.x;
    int tB = t0 + threadIdx.y;

    if (row < m && tA < k) As[threadIdx.y][threadIdx.x] = loadA<TA>(A, row, tA, m, k);
    else                  As[threadIdx.y][threadIdx.x] = 0.0f;

    if (tB < k && col < n) Bs[threadIdx.y][threadIdx.x] = loadB<TB>(B, tB, col, n, k);
    else                   Bs[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    #pragma unroll
    for (int tt = 0; tt < TILE; ++tt) {
      acc += As[threadIdx.y][tt] * Bs[tt][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < m && col < n) {
    size_t off = (size_t)row * (size_t)n + (size_t)col;
    float cold = C[off];
    C[off] = alpha * acc + beta * cold;
  }
}

// Host launcher: choose specialization
static void gemm(float alpha,
                 const float* dA, bool transposeA,
                 const float* dB, bool transposeB,
                 float beta,
                 float* dC,
                 int m, int n, int k) {
  constexpr int TILE = 16;
  dim3 block(TILE, TILE, 1);
  dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE, 1);

  if (!transposeA && !transposeB) {
    gemm_tiled_inplace<TILE, false, false><<<grid, block>>>(alpha, dA, dB, beta, dC, m, n, k);
  } else if (transposeA && !transposeB) {
    gemm_tiled_inplace<TILE, true,  false><<<grid, block>>>(alpha, dA, dB, beta, dC, m, n, k);
  } else if (!transposeA && transposeB) {
    gemm_tiled_inplace<TILE, false, true ><<<grid, block>>>(alpha, dA, dB, beta, dC, m, n, k);
  } else {
    gemm_tiled_inplace<TILE, true,  true ><<<grid, block>>>(alpha, dA, dB, beta, dC, m, n, k);
  }

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

// ----------------------------- Demo / test -----------------------------
int main() {
  int m = 64, k = 96, n = 80;
  float alpha = 1.25f;
  float beta  = -0.5f;

  struct Case { bool TA; bool TB; const char* name; };
  Case cases[] = {
    {false, false, "C <- alpha*A*B + beta*C"},
    {true,  false, "C <- alpha*A^T*B + beta*C"},
    {false, true,  "C <- alpha*A*B^T + beta*C"},
    {true,  true,  "C <- alpha*A^T*B^T + beta*C"},
  };

  for (auto cs : cases) {
    bool TA = cs.TA;
    bool TB = cs.TB;

    int A_rows = TA ? k : m;
    int A_cols = TA ? m : k;

    int B_rows = TB ? n : k;
    int B_cols = TB ? k : n;

    size_t Asz = (size_t)A_rows * (size_t)A_cols;
    size_t Bsz = (size_t)B_rows * (size_t)B_cols;
    size_t Csz = (size_t)m * (size_t)n;

    std::vector<float> hA(Asz), hB(Bsz), hC(Csz), hC_gpu(Csz), hC_ref(Csz);

    for (size_t i = 0; i < Asz; ++i) hA[i] = (float)((i % 23) - 11) * 0.03f;
    for (size_t i = 0; i < Bsz; ++i) hB[i] = (float)((i % 29) - 14) * 0.02f;
    for (size_t i = 0; i < Csz; ++i) hC[i] = (float)((i % 31) - 15) * 0.01f;

    // ✅ Avoid vector assignment warning: copy into hC_ref explicitly
    std::copy(hC.begin(), hC.end(), hC_ref.begin());
    gemm_cpu_inplace(alpha, hA.data(), TA, hB.data(), TB, beta, hC_ref.data(), m, n, k);

    float *dA=nullptr, *dB=nullptr, *dC=nullptr;
    CUDA_CHECK(cudaMalloc(&dA, Asz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, Bsz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, Csz * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), Asz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), Bsz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC.data(), Csz * sizeof(float), cudaMemcpyHostToDevice));

    gemm(alpha, dA, TA, dB, TB, beta, dC, m, n, k);

    CUDA_CHECK(cudaMemcpy(hC_gpu.data(), dC, Csz * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = allclose(hC_gpu, hC_ref);
    std::cout << (ok ? "PASS: " : "FAIL: ") << cs.name << "\n";

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    if (!ok) return 1;
  }

  return 0;
}
