// kernel_10.cu
// Kernel 10: Warptiling (between block-tiling and thread-tiling)
// - Vectorized float4 GMEM->SMEM loads
// - Transpose A while storing to SMEM (AsT) for contiguous loads
// - Warptiling: each warp computes a WMxWN tile; each thread computes TMxTN
//
// Build (H100 / Hopper):
//   nvcc -O3 -std=c++17 -arch=sm_90 -lineinfo kernel_10.cu -o kernel_10
//
// Run:
//   ./kernel_10 4096 4096 4096 50
//
// Requirements for this “benchmark” version (no edge guards):
//   M % BM == 0, N % BN == 0, K % BK == 0, and N,K multiple of 4.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CEIL_DIV(a,b) (((a) + (b) - 1) / (b))

#define CHK(call) do {                                      \
    cudaError_t err = (call);                               \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA error %s:%d: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(1);                                       \
    }                                                       \
} while (0)

static void fill(float* p, size_t n) {
    for (size_t i = 0; i < n; ++i) p[i] = (float)rand() / RAND_MAX - 0.5f;
}

static double gemm_flops(int M, int N, int K) {
    return 2.0 * (double)M * (double)N * (double)K;
}

// Tunables (kept as constexpr for simplicity)
static constexpr int NUM_THREADS = 256;      // 8 warps
static constexpr int BM = 128;
static constexpr int BN = 128;
static constexpr int BK = 16;

// Warps arranged as a 2D grid inside the block
static constexpr int WARPS_M = 2;
static constexpr int WARPS_N = 4;
static_assert(WARPS_M * WARPS_N == (NUM_THREADS / 32), "Warp grid must match warps per block");

// Per-warp tile sizes
static constexpr int WM = BM / WARPS_M;      // 64
static constexpr int WN = BN / WARPS_N;      // 32

// Thread tile (per thread outputs)
static constexpr int TM = 4;
static constexpr int TN = 4;
static_assert((WN % TN) == 0, "WN must be divisible by TN");
static_assert((WM % TM) == 0, "WM must be divisible by TM");
static_assert((BN % 4) == 0 && (BK % 4) == 0 && (TN % 4) == 0, "float4 requires multiples of 4");

// Inside a warp, we shape lanes into (threadRowInWarp, threadColInWarp)
static constexpr int WARP_COLS = (WN / TN);  // 32/4 = 8
static constexpr int WARP_ROWS = 32 / WARP_COLS; // 32/8 = 4
static_assert(WARP_ROWS * WARP_COLS == 32, "Warp lane tiling must cover 32 threads");

// “Sub-iterations” inside a warp to cover WM rows (because warp has only WARP_ROWS groups)
static constexpr int WSUBM = WARP_ROWS * TM; // 4*4 = 16 rows per sub-iteration
static constexpr int WMITER = WM / WSUBM;    // 64/16 = 4
static constexpr int WNITER = 1;             // because WARP_COLS covers all WN columns already

__global__ void __launch_bounds__(NUM_THREADS)
sgemm_kernel_10(int M, int N, int K,
                float alpha, const float* __restrict__ A,
                const float* __restrict__ B,
                float beta, float* __restrict__ C) {

    const int tid    = (int)threadIdx.x;   // 0..255
    const int lane   = tid & 31;           // 0..31
    const int warpId = tid >> 5;           // 0..7

    const int warpRow = warpId / WARPS_N;  // 0..1
    const int warpCol = warpId % WARPS_N;  // 0..3

    const int threadColInWarp = lane % WARP_COLS; // 0..7
    const int threadRowInWarp = lane / WARP_COLS; // 0..3

    const int blockRow = (int)blockIdx.y;
    const int blockCol = (int)blockIdx.x;

    // Shared memory:
    // - AsT stores A tile transposed: AsT[dotIdx * BM + row]  (BK x BM)
    // - Bs stores B tile normal:      Bs[dotIdx * BN + col]  (BK x BN)
    __shared__ float AsT[BK * BM];
    __shared__ float Bs[BK * BN];

    // Move A/B/C pointers to this block tile
    const float* Abase = A + (size_t)blockRow * BM * (size_t)K;
    const float* Bbase = B + (size_t)blockCol * BN;
    float*       Cbase = C + (size_t)blockRow * BM * (size_t)N + (size_t)blockCol * BN;

    // Cooperative load mapping (float4 loads):
    // A tile has BM*BK floats => (BM*BK)/4 float4s
    // B tile has BK*BN floats => (BK*BN)/4 float4s
    const int innerRowA = tid / (BK / 4);
    const int innerColA = tid % (BK / 4);
    constexpr int rowStrideA = (NUM_THREADS * 4) / BK;  // 1024/BK -> 64 rows per sweep for BK=16

    const int innerRowB = tid / (BN / 4);
    const int innerColB = tid % (BN / 4);
    constexpr int rowStrideB = NUM_THREADS / (BN / 4);  // 256/(BN/4) -> 8 rows per sweep for BN=128

    // Registers:
    // - each thread accumulates (WMITER*TM) x TN = 16 x 4 = 64 results
    float acc[WMITER * TM * TN];
#pragma unroll
    for (int i = 0; i < WMITER * TM * TN; ++i) acc[i] = 0.0f;

    float regM[WMITER * TM];
    float regN[TN];

    // Outer loop over K tiles
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        const float* Acur = Abase + bkIdx;
        const float* Bcur = Bbase + (size_t)bkIdx * (size_t)N;

        // Load A (transpose into AsT)
        for (int off = 0; off + rowStrideA <= BM; off += rowStrideA) {
            const int row = innerRowA + off;      // 0..BM-1
            const int col = innerColA * 4;        // 0..BK-4 step 4
            float4 tmp = *reinterpret_cast<const float4*>(&Acur[(size_t)row * (size_t)K + (size_t)col]);

            // transpose store: AsT[(dot)*BM + row]
            AsT[(col + 0) * BM + row] = tmp.x;
            AsT[(col + 1) * BM + row] = tmp.y;
            AsT[(col + 2) * BM + row] = tmp.z;
            AsT[(col + 3) * BM + row] = tmp.w;
        }

        // Load B (store normal into Bs)
        for (int off = 0; off + rowStrideB <= BK; off += rowStrideB) {
            const int row = innerRowB + off;      // 0..BK-1
            const int col = innerColB * 4;        // 0..BN-4 step 4
            *reinterpret_cast<float4*>(&Bs[row * BN + col]) =
                *reinterpret_cast<const float4*>(&Bcur[(size_t)row * (size_t)N + (size_t)col]);
        }

        __syncthreads();

        // Warptiled compute
#pragma unroll
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {

            // Load this warp's A rows into regs (transposed AsT -> contiguous by row)
#pragma unroll
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    const int rowInBlock =
                        warpRow * WM
                        + wSubRowIdx * WSUBM
                        + threadRowInWarp * TM
                        + i;
                    regM[wSubRowIdx * TM + i] = AsT[dotIdx * BM + rowInBlock];
                }
            }

            // Load this warp's B cols into regs
            // Warp covers WN columns fully with (threadColInWarp * TN)
#pragma unroll
            for (int i = 0; i < TN; ++i) {
                const int colInBlock =
                    warpCol * WN
                    + threadColInWarp * TN
                    + i;
                regN[i] = Bs[dotIdx * BN + colInBlock];
            }

            // FMA accumulate
#pragma unroll
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
#pragma unroll
                for (int resM = 0; resM < TM; ++resM) {
                    const float aVal = regM[wSubRowIdx * TM + resM];
#pragma unroll
                    for (int resN = 0; resN < TN; ++resN) {
                        const int idx = (wSubRowIdx * TM + resM) * TN + resN;
                        acc[idx] = fmaf(aVal, regN[resN], acc[idx]);
                    }
                }
            }
        }

        __syncthreads();
    }

    // Write results back to C (vectorized float4 store because TN=4)
#pragma unroll
    for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
#pragma unroll
        for (int resM = 0; resM < TM; ++resM) {
            const int row =
                blockRow * BM
                + warpRow * WM
                + wSubRowIdx * WSUBM
                + threadRowInWarp * TM
                + resM;

            const int col =
                blockCol * BN
                + warpCol * WN
                + threadColInWarp * TN;

            float4 out;
            const int base = (wSubRowIdx * TM + resM) * TN;
            out.x = alpha * acc[base + 0];
            out.y = alpha * acc[base + 1];
            out.z = alpha * acc[base + 2];
            out.w = alpha * acc[base + 3];

            float4 old = *reinterpret_cast<float4*>(&Cbase[(size_t)(row - blockRow * BM) * (size_t)N + (size_t)(col - blockCol * BN)]);
            old.x = beta * old.x + out.x;
            old.y = beta * old.y + out.y;
            old.z = beta * old.z + out.z;
            old.w = beta * old.w + out.w;

            *reinterpret_cast<float4*>(&Cbase[(size_t)(row - blockRow * BM) * (size_t)N + (size_t)(col - blockCol * BN)]) = old;
        }
    }
}

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096, reps = 50;
    if (argc >= 4) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }
    if (argc >= 5) reps = std::atoi(argv[4]);

    if ((M % BM) || (N % BN) || (K % BK) || (N % 4) || (K % 4)) {
        fprintf(stderr, "Require M%%BM==0, N%%BN==0, K%%BK==0 and N,K multiples of 4.\n");
        fprintf(stderr, "Got M=%d N=%d K=%d\n", M, N, K);
        return 1;
    }

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    size_t bytesA = (size_t)M * (size_t)K * sizeof(float);
    size_t bytesB = (size_t)K * (size_t)N * sizeof(float);
    size_t bytesC = (size_t)M * (size_t)N * sizeof(float);

    float* hA = (float*)std::malloc(bytesA);
    float* hB = (float*)std::malloc(bytesB);
    float* hC = (float*)std::malloc(bytesC);
    if (!hA || !hB || !hC) { fprintf(stderr, "Host malloc failed\n"); return 1; }

    fill(hA, (size_t)M * (size_t)K);
    fill(hB, (size_t)K * (size_t)N);
    fill(hC, (size_t)M * (size_t)N);

    float *dA=nullptr, *dB=nullptr, *dC=nullptr;
    CHK(cudaMalloc(&dA, bytesA));
    CHK(cudaMalloc(&dB, bytesB));
    CHK(cudaMalloc(&dC, bytesC));
    CHK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dC, hC, bytesC, cudaMemcpyHostToDevice));

    dim3 block(NUM_THREADS, 1, 1);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);

    // warmup
    sgemm_kernel_10<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    CHK(cudaGetLastError());
    CHK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHK(cudaEventCreate(&start));
    CHK(cudaEventCreate(&stop));

    CHK(cudaEventRecord(start));
    for (int r = 0; r < reps; ++r) {
        sgemm_kernel_10<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    }
    CHK(cudaGetLastError());
    CHK(cudaEventRecord(stop));
    CHK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHK(cudaEventElapsedTime(&ms, start, stop));
    float ms_avg = ms / reps;

    double gflops = gemm_flops(M, N, K) / (ms_avg * 1e6);

    printf("kernel_10 warptiling | BM=%d BN=%d BK=%d | WARPS_M=%d WARPS_N=%d | WM=%d WN=%d | TM=%d TN=%d\n",
           BM, BN, BK, WARPS_M, WARPS_N, WM, WN, TM, TN);
    printf("Launch: grid=(%u,%u,%u) block=(%u,%u,%u)\n",
           grid.x, grid.y, grid.z, block.x, block.y, block.z);
    printf("Time: %.3f ms (avg)\n", ms_avg);
    printf("Throughput: %.2f GFLOP/s\n", gflops);

    CHK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));

    CHK(cudaFree(dA)); CHK(cudaFree(dB)); CHK(cudaFree(dC));
    std::free(hA); std::free(hB); std::free(hC);
    CHK(cudaEventDestroy(start)); CHK(cudaEventDestroy(stop));
    return 0;
}
