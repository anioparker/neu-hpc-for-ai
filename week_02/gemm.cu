// nvcc -O2 -std=c++17 gemm.cu -o gemm && ./gemm


#include <cstdio>
#include <cuda_runtime.h>

__global__ void gemm(
    int m, int n, int k,
    float alpha, float *A, float *B,
    float beta,  float *C, float *D)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 0..m-1
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 0..n-1
    if (row >= m || col >= n) return;

    float acc = 0.0f;
    for (int t = 0; t < k; ++t) {
        float a = A[row * k + t];
        float b = B[t * n + col];
        acc += a * b;
    }

    float c = (C != nullptr) ? C[row * n + col] : 0.0f;
    D[row * n + col] = alpha * acc + beta * c;
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
    // Example: m=2, k=3, n=2
    int m = 2, k = 3, n = 2;
    float alpha = 1.0f, beta = 1.0f;

    // Host matrices (row-major)
    float hA[2 * 3] = { 1, 2, 3,
                        4, 5, 6 };          // [m x k]
    float hB[3 * 2] = { 7,  8,
                        9, 10,
                       11, 12 };            // [k x n]
    float hC[2 * 2] = { 1, 1,
                        1, 1 };              // [m x n]
    float hD[2 * 2] = { 0, 0, 0, 0 };         // [m x n]

    // Device buffers
    float *dA=nullptr, *dB=nullptr, *dC=nullptr, *dD=nullptr;
    cudaMalloc(&dA, sizeof(hA));
    cudaMalloc(&dB, sizeof(hB));
    cudaMalloc(&dC, sizeof(hC));
    cudaMalloc(&dD, sizeof(hD));

    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, sizeof(hC), cudaMemcpyHostToDevice);

    // Launch: 16x16 threads per block
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x,
              (m + block.y - 1) / block.y);

    gemm<<<grid, block>>>(m, n, k, alpha, dA, dB, beta, dC, dD);
    cudaDeviceSynchronize();

    cudaMemcpy(hD, dD, sizeof(hD), cudaMemcpyDeviceToHost);

    print_matrix("A", hA, m, k);
    print_matrix("B", hB, k, n);
    print_matrix("C", hC, m, n);
    print_matrix("D = alpha*A*B + beta*C", hD, m, n);

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
    return 0;
}
