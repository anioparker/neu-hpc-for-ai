// sgemm_naive_flops.cu
// Build: nvcc -O3 -std=c++17 kernel_01.cu -o sgemm_naive_flops
// Run:./sgemm_naive_flops

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

// C = alpha*A*B + beta*C
// A: MxK, B: KxN, C: MxN (row-major)
__global__ void sgemm_naive(int M, int N, int K,
                            float alpha, const float* A,
                            const float* B, float beta, float* C) {
    // compute position in C that this thread is responsible for
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; // row in C
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; // col in C

    // if condition is necessary when M or N aren't multiples of 32.
    if (x < (unsigned)M && y < (unsigned)N) {
        float tmp = 0.0f;
        // dot product of row x of A and col y of B
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        // C = alpha*(A@B) + beta*C
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

// Naive FLOP count for C = alpha*A*B + beta*C
// Per C[x,y]:
//   dot: K mul + K add (GEMM convention) => 2K FLOPs
//   alpha*tmp (1 mul) + beta*C (1 mul) + final add (1 add) => +3 FLOPs
// Total: M*N*(2K+3) = 2MNK + 3MN
static double gemm_flops(int M, int N, int K) {
    return (double)M * (double)N * (2.0 * (double)K + 3.0);
}

static void fill(float* p, size_t n) {
    for (size_t i = 0; i < n; ++i) p[i] = (float)rand() / RAND_MAX - 0.5f;
}

int main(int argc, char** argv) {
    int M = 1024, N = 1024, K = 1024;
    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }

    float alpha = 1.25f;
    float beta  = 0.75f;

    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float *hA = (float*)std::malloc(bytesA);
    float *hB = (float*)std::malloc(bytesB);
    float *hC = (float*)std::malloc(bytesC);
    if (!hA || !hB || !hC) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

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

    // Launch: blocks map C with 32x32 threads
    dim3 blockDim(32, 32, 1);
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);

    // Time kernel with events
    cudaEvent_t start, stop;
    CHK(cudaEventCreate(&start));
    CHK(cudaEventCreate(&stop));

    // Warmup
    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
    CHK(cudaGetLastError());
    CHK(cudaDeviceSynchronize());

    CHK(cudaEventRecord(start));
    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
    CHK(cudaGetLastError());
    CHK(cudaEventRecord(stop));
    CHK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHK(cudaEventElapsedTime(&ms, start, stop));

    double flops = gemm_flops(M, N, K);
    double gflops = flops / (ms * 1e6); // FLOPs / (ms*1e6) = GFLOP/s

    printf("Launch config: grid=(%u,%u,%u) block=(%u,%u,%u)\n",
           gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
    printf("M=%d N=%d K=%d\n", M, N, K);
    printf("FLOPs (naive): %.0f  = M*N*(2K+3) = 2*M*N*K + 3*M*N\n", flops);
    printf("Kernel time: %.3f ms\n", ms);
    printf("Throughput: %.2f GFLOP/s\n", gflops);

    // Cleanup
    CHK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));
    CHK(cudaFree(dA));
    CHK(cudaFree(dB));
    CHK(cudaFree(dC));
    std::free(hA);
    std::free(hB);
    std::free(hC);
    CHK(cudaEventDestroy(start));
    CHK(cudaEventDestroy(stop));

    return 0;
}
