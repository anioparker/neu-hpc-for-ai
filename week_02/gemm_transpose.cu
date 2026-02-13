// nvcc -O2 -std=c++17 gemm_transpose.cu -o gemm_transpose && ./gemm_transpose

#include <cstdio>
#include <cuda_runtime.h>
#include <stdbool.h>

__device__ __forceinline__ float loadA(const float* A, int m, int k,
                                       int row, int t, bool transposeA) {
    return transposeA ? A[t * m + row] : A[row * k + t];
}

__device__ __forceinline__ float loadB(const float* B, int n, int k,
                                       int t, int col, bool transposeB) {
    return transposeB ? B[col * k + t] : B[t * n + col];
}

__global__ void gemm_kernel(int m, int n, int k,
                            float alpha, const float* A, bool transposeA,
                            const float* B, bool transposeB,
                            float beta, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;

    float acc = 0.0f;
    for (int t = 0; t < k; ++t) {
        acc += loadA(A, m, k, row, t, transposeA) *
               loadB(B, n, k, t, col, transposeB);
    }

    int idx = row * n + col;
    C[idx] = alpha * acc + beta * C[idx];
}

static inline int div_up(int a, int b) { return (a + b - 1) / b; }

void gemm(int m, int n, int k,
          float alpha, float *A, bool transposeA,
          float *B, bool transposeB,
          float beta, float *C,
          cudaStream_t stream = 0) {
    dim3 block(16, 16);
    dim3 grid(div_up(n, (int)block.x), div_up(m, (int)block.y));
    gemm_kernel<<<grid, block, 0, stream>>>(m, n, k, alpha, A, transposeA, B, transposeB, beta, C);
}

static void print_matrix(const char* name, const float* M, int R, int C) {
    printf("%s (%dx%d):\n", name, R, C);
    for (int r = 0; r < R; ++r) {
        printf("  ");
        for (int c = 0; c < C; ++c) printf("%8.3f ", M[r * C + c]);
        printf("\n");
    }
}

int main() {
    // Example: C <- alpha * A*B + beta*C
    // A: [m x k] = [2 x 3], B: [k x n] = [3 x 2], C: [m x n] = [2 x 2]
    int m = 2, k = 3, n = 2;
    float alpha = 1.0f, beta = 1.0f;
    bool transposeA = false, transposeB = false;

    float hA[2 * 3] = {
        1, 2, 3,
        4, 5, 6
    };
    float hB[3 * 2] = {
        7,  8,
        9, 10,
        11,12
    };
    float hC[2 * 2] = {
        1, 1,
        1, 1
    };

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, sizeof(hA));
    cudaMalloc(&dB, sizeof(hB));
    cudaMalloc(&dC, sizeof(hC));

    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, sizeof(hC), cudaMemcpyHostToDevice);

    gemm(m, n, k, alpha, dA, transposeA, dB, transposeB, beta, dC);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, sizeof(hC), cudaMemcpyDeviceToHost);

    print_matrix("A", hA, m, k);
    print_matrix("B", hB, k, n);
    print_matrix("C (updated)", hC, m, n);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
