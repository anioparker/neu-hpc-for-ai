// flash_attention_cuda.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) do {                           \
  cudaError_t err = (x);                             \
  if (err != cudaSuccess) {                          \
    printf("CUDA error %s:%d: %s\n",                 \
           __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1);                                         \
  }                                                  \
} while(0)
#endif

// Choose simple tile sizes (can change later)
#ifndef BR
#define BR 64
#endif
#ifndef BC
#define BC 64
#endif

// Utility
__device__ __forceinline__ float neg_inf() { return -INFINITY; }

// Kernel: one block processes one Qi block (i)
__global__ void flash_attention_fwd_alg1(
    const float* __restrict__ Q, // (N,d)
    const float* __restrict__ K, // (N,d)
    const float* __restrict__ V, // (N,d)
    float* __restrict__ O,       // (N,d) output
    float* __restrict__ L,       // (N) logsumexp output
    int N, int d)
{
  // Which Qi block?
  int i_blk = blockIdx.x;
  int row0 = i_blk * BR;
  int Br_eff = min(BR, N - row0);
  if (Br_eff <= 0) return;

  // Shared memory: Qi, Kj, Vj, and S/P tiles
  extern __shared__ float smem[];
  float* sQi = smem;                               // BR*d
  float* sKj = sQi + BR * d;                       // BC*d
  float* sVj = sKj + BC * d;                       // BC*d
  float* sS  = sVj + BC * d;                       // BR*BC
  float* sP  = sS  + BR * BC;                      // BR*BC
  // Running stats in shared for simplicity:
  float* sMi = sP  + BR * BC;                      // BR
  float* sLi = sMi + BR;                           // BR
  float* sAlpha = sLi + BR;                        // BR  (exp(m_prev - m_new))

  // (Alg 1 line 4) Load Qi to shared
  // Parallel copy: threads cover elements
  for (int idx = threadIdx.x; idx < Br_eff * d; idx += blockDim.x) {
    int r = idx / d;
    int c = idx % d;
    sQi[r * d + c] = Q[(row0 + r) * d + c];
  }

  // (Alg 1 line 5) init Oi=0, li=0, mi=-inf
  // We'll write O directly to global at the end, but keep partial O in global? better keep a shared O tile:
  // For simplicity, store Oi in global scratch? We'll use shared temp for Oi to be correct.
  // Allocate Oi in registers is hard; use global O buffer region for Oi in this kernel:
  // We'll use shared Oi in the same smem by reusing sS after each j? safer: add shared Oi.
  // -> easiest: put Oi in global O and update in-place. But must initialize here.
  for (int idx = threadIdx.x; idx < Br_eff * d; idx += blockDim.x) {
    int r = idx / d;
    int c = idx % d;
    O[(row0 + r) * d + c] = 0.0f;
  }
  for (int r = threadIdx.x; r < Br_eff; r += blockDim.x) {
    sLi[r] = 0.0f;
    sMi[r] = neg_inf();
  }
  __syncthreads();

  int Tc = (N + BC - 1) / BC;

  // (Alg 1 line 6) loop j blocks
  for (int j_blk = 0; j_blk < Tc; ++j_blk) {
    int col0 = j_blk * BC;
    int bc = min(BC, N - col0);

    // (Alg 1 line 7) Load Kj, Vj into shared (pad by leaving rest unused)
    for (int idx = threadIdx.x; idx < bc * d; idx += blockDim.x) {
      int r = idx / d;   // 0..bc-1
      int c = idx % d;
      sKj[r * d + c] = K[(col0 + r) * d + c];
      sVj[r * d + c] = V[(col0 + r) * d + c];
    }
    __syncthreads();

    // (Alg 1 line 8) S = Qi * Kj^T   (Br_eff x bc)
    // Parallelize over (r,c)
    for (int idx = threadIdx.x; idx < Br_eff * BC; idx += blockDim.x) {
      int r = idx / BC;
      int c = idx % BC;
      if (c >= bc) {
        sS[r * BC + c] = neg_inf(); // mask padded keys
      } else {
        float sum = 0.0f;
        for (int k = 0; k < d; ++k) {
          sum += sQi[r * d + k] * sKj[c * d + k];
        }
        sS[r * BC + c] = sum;
      }
    }
    __syncthreads();

    // (Alg 1 line 9) row-wise online softmax update:
    // For simplicity: one thread handles each row r
    for (int r = threadIdx.x; r < Br_eff; r += blockDim.x) {
      float m_prev = sMi[r];
      // rowmax over BC (masked cols are -inf)
      float rowmax = neg_inf();
      for (int c = 0; c < BC; ++c) {
        float v = sS[r * BC + c];
        rowmax = (v > rowmax) ? v : rowmax;
      }
      float m_new = (m_prev > rowmax) ? m_prev : rowmax;

      // Ptilde = exp(S - m_new), and rowsum
      float rowsum = 0.0f;
      for (int c = 0; c < BC; ++c) {
        float s_rc = sS[r * BC + c];
        float p = (isinf(s_rc) && s_rc < 0) ? 0.0f : expf(s_rc - m_new);
        sP[r * BC + c] = p;
        rowsum += p;
      }

      float l_prev = sLi[r];
      float l_new = expf(m_prev - m_new) * l_prev + rowsum;

      sAlpha[r] = expf(m_prev - m_new); // used in line 10
      sMi[r] = m_new;
      sLi[r] = l_new;
    }
    __syncthreads();

    // (Alg 1 line 10) O = diag(alpha)*O + Ptilde*Vj
    // Parallelize over O elements (r,outc)
    for (int idx = threadIdx.x; idx < Br_eff * d; idx += blockDim.x) {
      int r = idx / d;
      int outc = idx % d;

      float o = O[(row0 + r) * d + outc];
      o *= sAlpha[r];

      float add = 0.0f;
      for (int c = 0; c < bc; ++c) {
        add += sP[r * BC + c] * sVj[c * d + outc];
      }
      O[(row0 + r) * d + outc] = o + add;
    }
    __syncthreads();
  }

  // (Alg 1 line 12) O /= l (row-wise)
  for (int idx = threadIdx.x; idx < Br_eff * d; idx += blockDim.x) {
    int r = idx / d;
    int c = idx % d;
    float denom = sLi[r];
    O[(row0 + r) * d + c] /= denom;
  }

  // (Alg 1 line 13) L = m + log(l)
  for (int r = threadIdx.x; r < Br_eff; r += blockDim.x) {
    L[row0 + r] = sMi[r] + logf(sLi[r]);
  }
}

// Host helper to launch
void launch_flash(int N, int d,
                  const float* Q, const float* K, const float* V,
                  float* O, float* L)
{
  int Tr = (N + BR - 1) / BR;

  // shared memory size (floats):
  // Qi: BR*d, Kj: BC*d, Vj: BC*d, S: BR*BC, P: BR*BC, Mi: BR, Li: BR, Alpha: BR
  size_t sh_floats = (size_t)BR*d + (size_t)BC*d + (size_t)BC*d
                   + (size_t)BR*BC + (size_t)BR*BC
                   + (size_t)BR + (size_t)BR + (size_t)BR;
  size_t sh_bytes = sh_floats * sizeof(float);

  int threads = 128; // simple
  flash_attention_fwd_alg1<<<Tr, threads, sh_bytes>>>(Q,K,V,O,L,N,d);
  CHECK_CUDA(cudaGetLastError());
}

int main() {
  // Tiny sanity test (same as yours)
  int N = 2, d = 4;

  float hQ[] = {1,0,1,0,  0,1,0,1};
  float hK[] = {1,0,1,0,  0,1,0,1};
  float hV[] = {10,20,30,40,  50,60,70,80};

  float hO[8] = {0};
  float hL[2] = {0};

  float *dQ, *dK, *dV, *dO, *dL;
  CHECK_CUDA(cudaMalloc(&dQ, N*d*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dK, N*d*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dV, N*d*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dO, N*d*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dL, N*sizeof(float)));

  CHECK_CUDA(cudaMemcpy(dQ, hQ, N*d*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dK, hK, N*d*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dV, hV, N*d*sizeof(float), cudaMemcpyHostToDevice));

  launch_flash(N,d,dQ,dK,dV,dO,dL);

  CHECK_CUDA(cudaMemcpy(hO, dO, N*d*sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(hL, dL, N*sizeof(float), cudaMemcpyDeviceToHost));

  printf("O:\n");
  for(int i=0;i<N;i++){
    for(int j=0;j<d;j++) printf("%.5f ", hO[i*d+j]);
    printf("\n");
  }
  printf("L:\n");
  for(int i=0;i<N;i++) printf("%.5f\n", hL[i]);

  cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dL);
  return 0;
}
