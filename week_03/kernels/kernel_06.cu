// kernel_06.cu
//  nvcc -O3 -std=c++17 -arch=sm_90 -lineinfo \
     -DBM=128 -DBN=128 -DBK=16 -DTM=8 -DTN=8 \
     kernel_06.cu -o vectorized_mem_access
//  ./vectorized_mem_access 4096 4096 4096 50
//
// Notes:
// - For float4 vectorization, you want BN%4==0 and BK%4==0, and N,K multiples of 4 for clean fast paths.
// - This is row-major: A[MxK], B[KxN], C[MxN].

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 128
#endif
#ifndef BK
#define BK 16
#endif
#ifndef TM
#define TM 8
#endif
#ifndef TN
#define TN 8
#endif

#define CEIL_DIV(a,b) (((a) + (b) - 1) / (b))

#define CHK(call) do {                                      \
    cudaError_t err = (call);                               \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA error %s:%d: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(1);                                       \
    }                                                       \
} while (0)

__device__ __forceinline__ float load_guarded(const float* p, int idx, int limit) {
    return (idx < limit) ? p[idx] : 0.0f;
}

__global__ void sgemm_kernel_06(int M, int N, int K,
                                float alpha, const float* __restrict__ A,
                                const float* __restrict__ B,
                                float beta, float* __restrict__ C)
{
    // 2D thread-tile: each thread computes TM x TN outputs
    const int threadCol = (int)threadIdx.x; // 0..(BN/TN -1)
    const int threadRow = (int)threadIdx.y; // 0..(BM/TM -1)

    const int blockCol = (int)blockIdx.x;   // along N
    const int blockRow = (int)blockIdx.y;   // along M

    const int rowBase = blockRow * BM + threadRow * TM;     // global row start for this thread-tile
    const int colBase = blockCol * BN + threadCol * TN;     // global col start for this thread-tile

    // Transposed A tile in SMEM: AsT[dotIdx][row] => AsT[dotIdx*BM + row]
    __shared__ float AsT[BK * BM];
    __shared__ float Bs[BK * BN];

    // Register tile accumulators
    float acc[TM * TN];
#pragma unroll
    for (int i = 0; i < TM * TN; ++i) acc[i] = 0.0f;

    // Register caches
    float regM[TM];
    float regN[TN];

    // thread id for cooperative loading
    const int tix = (int)threadIdx.x;
    const int tiy = (int)threadIdx.y;
    const int tid = tiy * (BN / TN) + tix;
    const int numThreads = (BN / TN) * (BM / TM);

    // Optional runtime sanity checks (only one thread prints)
    if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
        if ((BK % 4) != 0 || (BN % 4) != 0 || (TM % 4) != 0 || (TN % 4) != 0) {
            printf("Warning: BK, BN, TM, TN should be multiples of 4 for full vectorization.\n");
            printf("Current: BK=%d BN=%d TM=%d TN=%d\n", BK, BN, TM, TN);
        }
    }

    // Outer loop over K tiles
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {

        // Cooperative vectorized GMEM->SMEM loads
        // Total float4 loads:
        // - A tile: BM*BK floats => (BM*BK)/4 float4
        // - B tile: BK*BN floats => (BK*BN)/4 float4
        const int vecA = (BM * BK) / 4;
        const int vecB = (BK * BN) / 4;
        const int vecTotal = vecA + vecB;

        for (int v = tid; v < vecTotal; v += numThreads) {
            if (v < vecA) {
                // A tile float4 load at (row, col4*4 .. col4*4+3)
                const int row = v / (BK / 4);
                const int col4 = v % (BK / 4); // vector column index within BK
                const int gRow = blockRow * BM + row;
                const int gCol = bkIdx + col4 * 4;

                float4 tmp;
                if (gRow < M && (gCol + 3) < K) {
                    tmp = *reinterpret_cast<const float4*>(&A[gRow * K + gCol]);
                } else {
                    // guarded scalar load (edge tiles)
                    float t0 = (gRow < M) ? load_guarded(&A[gRow * K], gCol + 0, K) : 0.0f;
                    float t1 = (gRow < M) ? load_guarded(&A[gRow * K], gCol + 1, K) : 0.0f;
                    float t2 = (gRow < M) ? load_guarded(&A[gRow * K], gCol + 2, K) : 0.0f;
                    float t3 = (gRow < M) ? load_guarded(&A[gRow * K], gCol + 3, K) : 0.0f;
                    tmp = make_float4(t0, t1, t2, t3);
                }

                // Transpose while storing: AsT[(dotIdx)*BM + row]
                const int dot0 = col4 * 4 + 0;
                const int dot1 = col4 * 4 + 1;
                const int dot2 = col4 * 4 + 2;
                const int dot3 = col4 * 4 + 3;
                if (dot0 < BK) AsT[dot0 * BM + row] = tmp.x;
                if (dot1 < BK) AsT[dot1 * BM + row] = tmp.y;
                if (dot2 < BK) AsT[dot2 * BM + row] = tmp.z;
                if (dot3 < BK) AsT[dot3 * BM + row] = tmp.w;
            } else {
                // B tile float4 load at (row, col4*4 .. col4*4+3)
                const int t = v - vecA;
                const int row = t / (BN / 4);   // row within BK
                const int col4 = t % (BN / 4);  // vector column index within BN
                const int gRow = bkIdx + row;
                const int gCol = blockCol * BN + col4 * 4;

                float4 tmp;
                if (gRow < K && (gCol + 3) < N) {
                    tmp = *reinterpret_cast<const float4*>(&B[gRow * N + gCol]);
                } else {
                    // guarded scalar load (edge tiles)
                    float t0 = (gRow < K && (gCol + 0) < N) ? B[gRow * N + (gCol + 0)] : 0.0f;
                    float t1 = (gRow < K && (gCol + 1) < N) ? B[gRow * N + (gCol + 1)] : 0.0f;
                    float t2 = (gRow < K && (gCol + 2) < N) ? B[gRow * N + (gCol + 2)] : 0.0f;
                    float t3 = (gRow < K && (gCol + 3) < N) ? B[gRow * N + (gCol + 3)] : 0.0f;
                    tmp = make_float4(t0, t1, t2, t3);
                }

                // Store to Bs (not transposed)
                float4* dst = reinterpret_cast<float4*>(&Bs[row * BN + col4 * 4]);
                *dst = tmp;
            }
        }
        __syncthreads();

        // Compute
#pragma unroll
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Load AsT (transposed) into regM: contiguous in row -> good for vectorized SMEM loads
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int r = threadRow * TM + i; // row within BM
                regM[i] = AsT[dotIdx * BM + r];
            }

            // Load Bs into regN
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                regN[j] = Bs[dotIdx * BN + threadCol * TN + j];
            }

            // Outer product accumulate
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i * TN + j] = fmaf(regM[i], regN[j], acc[i * TN + j]);
                }
            }
        }
        __syncthreads();
    }

    // Vectorized GMEM store (float4) when possible; scalar fallback on edges
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = rowBase + i;
        if (r >= M) continue;

        // store TN columns (prefer float4 chunks)
        const int c0 = colBase;

        // TN is assumed multiple of 4 in the fast path
#pragma unroll
        for (int j4 = 0; j4 < TN; j4 += 4) {
            const int c = c0 + j4;
            if ((c + 3) < N) {
                float4 out;
                out.x = alpha * acc[i * TN + (j4 + 0)];
                out.y = alpha * acc[i * TN + (j4 + 1)];
                out.z = alpha * acc[i * TN + (j4 + 2)];
                out.w = alpha * acc[i * TN + (j4 + 3)];

                float4 old = *reinterpret_cast<float4*>(&C[r * N + c]);
                old.x = beta * old.x + out.x;
                old.y = beta * old.y + out.y;
                old.z = beta * old.z + out.z;
                old.w = beta * old.w + out.w;

                *reinterpret_cast<float4*>(&C[r * N + c]) = old;
            } else {
                // edge fallback
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int cc = c + j;
                    if (cc < N) {
                        float val = alpha * acc[i * TN + (j4 + j)] + beta * C[r * N + cc];
                        C[r * N + cc] = val;
                    }
                }
            }
        }
    }
}

static void fill(float* p, size_t n) {
    for (size_t i = 0; i < n; ++i) p[i] = (float)rand() / RAND_MAX - 0.5f;
}

static double gemm_flops(int M, int N, int K) {
    // GEMM convention: 2*M*N*K (+ optional small terms ignored for large sizes)
    return 2.0 * (double)M * (double)N * (double)K;
}

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096, reps = 50;
    if (argc >= 4) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }
    if (argc >= 5) reps = std::atoi(argv[4]);

    const float alpha = 1.0f, beta = 0.0f;

    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float *hA = (float*)std::malloc(bytesA);
    float *hB = (float*)std::malloc(bytesB);
    float *hC = (float*)std::malloc(bytesC);
    if (!hA || !hB || !hC) { fprintf(stderr, "Host malloc failed\n"); return 1; }

    fill(hA, (size_t)M * K);
    fill(hB, (size_t)K * N);
    fill(hC, (size_t)M * N);

    float *dA, *dB, *dC;
    CHK(cudaMalloc(&dA, bytesA));
    CHK(cudaMalloc(&dB, bytesB));
    CHK(cudaMalloc(&dC, bytesC));
    CHK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dC, hC, bytesC, cudaMemcpyHostToDevice));

    dim3 block(BN / TN, BM / TM, 1);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);

    // warmup
    sgemm_kernel_06<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    CHK(cudaGetLastError());
    CHK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHK(cudaEventCreate(&start));
    CHK(cudaEventCreate(&stop));

    CHK(cudaEventRecord(start));
    for (int r = 0; r < reps; ++r) {
        sgemm_kernel_06<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    }
    CHK(cudaGetLastError());
    CHK(cudaEventRecord(stop));
    CHK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHK(cudaEventElapsedTime(&ms, start, stop));
    float ms_per = ms / reps;

    double flops = gemm_flops(M, N, K);
    double gflops = flops / (ms_per * 1e6);

    printf("kernel_06 | BM=%d BN=%d BK=%d TM=%d TN=%d | M=%d N=%d K=%d reps=%d\n",
           BM, BN, BK, TM, TN, M, N, K, reps);
    printf("Launch: grid=(%u,%u,%u) block=(%u,%u,%u)\n",
           grid.x, grid.y, grid.z, block.x, block.y, block.z);
    printf("Time: %.3f ms (avg)\n", ms_per);
    printf("Throughput: %.2f GFLOP/s\n", gflops);

    CHK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));

    CHK(cudaFree(dA)); CHK(cudaFree(dB)); CHK(cudaFree(dC));
    std::free(hA); std::free(hB); std::free(hC);
    CHK(cudaEventDestroy(start)); CHK(cudaEventDestroy(stop));
    return 0;
}
