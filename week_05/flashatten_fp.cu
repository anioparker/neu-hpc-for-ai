// nvcc -std=c++17 -O2 -I./cutlass/include flashatten_fp.cu -o fa_cute

// ./fa_cute 128

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>   // CuTe: Layout/Tensor/local_tile

using namespace cute;

// ------------------------- utility -------------------------

static void ck(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(e));
    std::exit(1);
  }
}

static inline float randf() {
  return float(rand()) / float(RAND_MAX) - 0.5f;
}

// -------------------- CPU reference (Alg 1) --------------------
// Q,K,V: [N,D], O: [N,D], L: [N]
template<int Br, int Bc, int D>
void flashattn_alg1_cpu(const float* Q, const float* K, const float* V,
                        float* O, float* L, int N)
{
  auto idxQD = [](int i, int k){ return i*D + k; };

  for (int i0 = 0; i0 < N; i0 += Br) {
    int Br_eff = std::min(Br, N - i0);

    // per-row state
    std::vector<float> m(Br_eff, -INFINITY);
    std::vector<float> l(Br_eff, 0.0f);
    std::vector<float> o(Br_eff * D, 0.0f);

    for (int j0 = 0; j0 < N; j0 += Bc) {
      int Bc_eff = std::min(Bc, N - j0);

      for (int r = 0; r < Br_eff; ++r) {
        // compute scores s[c] = dot(Q[i0+r], K[j0+c])
        float rowmax = -INFINITY;
        std::vector<float> s(Bc_eff);

        for (int c = 0; c < Bc_eff; ++c) {
          float acc = 0.0f;
          for (int k = 0; k < D; ++k) {
            acc += Q[idxQD(i0+r,k)] * K[idxQD(j0+c,k)];
          }
          s[c] = acc;
          rowmax = std::max(rowmax, acc);
        }

        float m_new = std::max(m[r], rowmax);
        float alpha = std::exp(m[r] - m_new);

        float l_new = l[r] * alpha;
        for (int c = 0; c < Bc_eff; ++c) {
          l_new += std::exp(s[c] - m_new);
        }

        // O update: O = alpha*O + sum_c exp(s[c]-m_new)*V[c]
        for (int k = 0; k < D; ++k) {
          float out = o[r*D + k] * alpha;
          float sum = 0.0f;
          for (int c = 0; c < Bc_eff; ++c) {
            sum += std::exp(s[c] - m_new) * V[idxQD(j0+c,k)];
          }
          o[r*D + k] = out + sum;
        }

        m[r] = m_new;
        l[r] = l_new;
      }
    }

    // finalize
    for (int r = 0; r < Br_eff; ++r) {
      float inv_l = 1.0f / l[r];
      for (int k = 0; k < D; ++k) {
        O[idxQD(i0+r,k)] = o[r*D + k] * inv_l;
      }
      L[i0+r] = m[r] + std::log(l[r]);
    }
  }
}

// ---------------------- CUDA kernel (Alg 1) ----------------------
// One block handles one Qi tile of Br rows.
// threadIdx.x in [0,Br) handles one row r.
template<int Br, int Bc, int D>
__global__ void flashattn_alg1_cute_kernel(const float* __restrict__ Qp,
                                          const float* __restrict__ Kp,
                                          const float* __restrict__ Vp,
                                          float* __restrict__ Op,
                                          float* __restrict__ Lp,
                                          int N)
{
  // CuTe tensor views (row-major)
  auto Q = make_tensor(make_gmem_ptr(Qp),
                       make_layout(make_shape(N, Int<D>{}),
                                   make_stride(Int<D>{}, Int<1>{})));
  auto K = make_tensor(make_gmem_ptr(Kp),
                       make_layout(make_shape(N, Int<D>{}),
                                   make_stride(Int<D>{}, Int<1>{})));
  auto V = make_tensor(make_gmem_ptr(Vp),
                       make_layout(make_shape(N, Int<D>{}),
                                   make_stride(Int<D>{}, Int<1>{})));
  auto O = make_tensor(make_gmem_ptr(Op),
                       make_layout(make_shape(N, Int<D>{}),
                                   make_stride(Int<D>{}, Int<1>{})));
  auto L = make_tensor(make_gmem_ptr(Lp),
                       make_layout(make_shape(N), make_stride(Int<1>{})));

  int tile_i = int(blockIdx.x);         // which Qi block
  int i0 = tile_i * Br;
  int r  = int(threadIdx.x);            // row within tile

  if (r >= Br) return;                  // requires blockDim.x >= Br
  if (i0 + r >= N) return;              // tail guard

  // Shared memory tiles: Q[Br,D], K[Bc,D], V[Bc,D]
  extern __shared__ float smem[];
  float* sQ = smem;
  float* sK = sQ + Br * D;
  float* sV = sK + Bc * D;

  // CuTe shared tensors (row-major)
  auto tQ = make_tensor(make_smem_ptr(sQ),
                        make_layout(make_shape(Int<Br>{}, Int<D>{}),
                                    make_stride(Int<D>{}, Int<1>{})));
  auto tK = make_tensor(make_smem_ptr(sK),
                        make_layout(make_shape(Int<Bc>{}, Int<D>{}),
                                    make_stride(Int<D>{}, Int<1>{})));
  auto tV = make_tensor(make_smem_ptr(sV),
                        make_layout(make_shape(Int<Bc>{}, Int<D>{}),
                                    make_stride(Int<D>{}, Int<1>{})));

  // Tile views in global memory using CuTe local_tile
  // Q tile at (tile_i, 0) in tiling (N by Br, D by D)
  auto Qi = local_tile(Q, make_shape(Int<Br>{}, Int<D>{}), make_coord(tile_i, 0));
  auto Oi = local_tile(O, make_shape(Int<Br>{}, Int<D>{}), make_coord(tile_i, 0));

  // Load Qi into shared (each thread loads its row)
  #pragma unroll
  for (int k = 0; k < D; ++k) {
    tQ(r,k) = Qi(r,k);
  }
  __syncthreads();

  // Per-row accumulators (registers)
  float m = -INFINITY;
  float l = 0.0f;
  float o[D];
  #pragma unroll
  for (int k = 0; k < D; ++k) o[k] = 0.0f;

  int Tc = (N + Bc - 1) / Bc;

  for (int tile_j = 0; tile_j < Tc; ++tile_j) {
    int j0 = tile_j * Bc;

    // Load K,V tile into shared: all threads cooperate over [Bc,D]
    for (int linear = int(threadIdx.x); linear < Bc * D; linear += int(blockDim.x)) {
      int c = linear / D;
      int k = linear - c * D;
      int gj = j0 + c;
      tK(c,k) = (gj < N) ? K(gj,k) : 0.0f;
      tV(c,k) = (gj < N) ? V(gj,k) : 0.0f;
    }
    __syncthreads();

    // Compute scores s[c] for this row
    float s[Bc];
    float rowmax = -INFINITY;

    #pragma unroll
    for (int c = 0; c < Bc; ++c) {
      float acc = 0.0f;
      #pragma unroll
      for (int k = 0; k < D; ++k) {
        acc += tQ(r,k) * tK(c,k);
      }
      s[c] = acc;
      rowmax = fmaxf(rowmax, acc);
    }

    // Online softmax update
    float m_new = fmaxf(m, rowmax);
    float alpha = expf(m - m_new);

    float l_new = l * alpha;
    #pragma unroll
    for (int c = 0; c < Bc; ++c) {
      int gj = j0 + c;
      if (gj < N) l_new += expf(s[c] - m_new);
    }

    // O update
    #pragma unroll
    for (int k = 0; k < D; ++k) {
      float out = o[k] * alpha;
      float sum = 0.0f;
      #pragma unroll
      for (int c = 0; c < Bc; ++c) {
        int gj = j0 + c;
        if (gj < N) sum += expf(s[c] - m_new) * tV(c,k);
      }
      o[k] = out + sum;
    }

    m = m_new;
    l = l_new;

    __syncthreads();
  }

  // Normalize and write back
  float inv_l = 1.0f / l;
  #pragma unroll
  for (int k = 0; k < D; ++k) {
    Oi(r,k) = o[k] * inv_l;
  }
  L(i0 + r) = m + logf(l);
}

// ------------------------------- main -------------------------------

template<int Br, int Bc, int D>
int run(int N)
{
  printf("Running FlashAttention Alg1 (CuTe) N=%d D=%d Br=%d Bc=%d\n", N, D, Br, Bc);

  size_t bytesQD = size_t(N) * D * sizeof(float);
  size_t bytesL  = size_t(N) * sizeof(float);

  std::vector<float> hQ(N*D), hK(N*D), hV(N*D);
  std::vector<float> hO_cpu(N*D, 0), hL_cpu(N, 0);
  std::vector<float> hO_gpu(N*D, 0), hL_gpu(N, 0);

  for (int i = 0; i < N*D; ++i) {
    hQ[i] = randf();
    hK[i] = randf();
    hV[i] = randf();
  }

  float *dQ, *dK, *dV, *dO, *dL;
  ck(cudaMalloc(&dQ, bytesQD), "malloc dQ");
  ck(cudaMalloc(&dK, bytesQD), "malloc dK");
  ck(cudaMalloc(&dV, bytesQD), "malloc dV");
  ck(cudaMalloc(&dO, bytesQD), "malloc dO");
  ck(cudaMalloc(&dL, bytesL ), "malloc dL");

  ck(cudaMemcpy(dQ, hQ.data(), bytesQD, cudaMemcpyHostToDevice), "H2D Q");
  ck(cudaMemcpy(dK, hK.data(), bytesQD, cudaMemcpyHostToDevice), "H2D K");
  ck(cudaMemcpy(dV, hV.data(), bytesQD, cudaMemcpyHostToDevice), "H2D V");

  // launch
  int Tr = (N + Br - 1) / Br;
  dim3 grid(Tr);
  dim3 block(256); // must be >= Br; 256 is safe for Br<=128

  size_t smem_bytes = (Br*D + Bc*D + Bc*D) * sizeof(float);

  flashattn_alg1_cute_kernel<Br,Bc,D><<<grid, block, smem_bytes>>>(
    dQ, dK, dV, dO, dL, N
  );
  ck(cudaGetLastError(), "kernel launch");
  ck(cudaDeviceSynchronize(), "sync");

  ck(cudaMemcpy(hO_gpu.data(), dO, bytesQD, cudaMemcpyDeviceToHost), "D2H O");
  ck(cudaMemcpy(hL_gpu.data(), dL, bytesL,  cudaMemcpyDeviceToHost), "D2H L");

  // CPU ref
  flashattn_alg1_cpu<Br,Bc,D>(hQ.data(), hK.data(), hV.data(),
                             hO_cpu.data(), hL_cpu.data(), N);

  // compare
  float max_abs_O = 0.f, max_rel_O = 0.f;
  for (int i = 0; i < N*D; ++i) {
    float a = hO_cpu[i], b = hO_gpu[i];
    float abs = fabsf(a - b);
    float rel = abs / (fabsf(a) + 1e-6f);
    max_abs_O = std::max(max_abs_O, abs);
    max_rel_O = std::max(max_rel_O, rel);
  }

  float max_abs_L = 0.f, max_rel_L = 0.f;
  for (int i = 0; i < N; ++i) {
    float a = hL_cpu[i], b = hL_gpu[i];
    float abs = fabsf(a - b);
    float rel = abs / (fabsf(a) + 1e-6f);
    max_abs_L = std::max(max_abs_L, abs);
    max_rel_L = std::max(max_rel_L, rel);
  }

  printf("O: max_abs=%.3e  max_rel=%.3e\n", max_abs_O, max_rel_O);
  printf("L: max_abs=%.3e  max_rel=%.3e\n", max_abs_L, max_rel_L);

  cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dL);
  return 0;
}

int main(int argc, char** argv)
{
  // Defaults
  int N = (argc > 1) ? std::atoi(argv[1]) : 128;

  // Choose a small D for compile-time template; change as needed
  // optional: make multiple instantiations.
  return run</*Br=*/64, /*Bc=*/64, /*D=*/64>(N);
}
