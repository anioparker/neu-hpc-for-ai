// matmul_single_thread.c
// Build: gcc -std=c11 -O3 -march=native -Wall -Wextra matmul_single_thread.c -o matmul -lm
// Run:   ./matmul

#include <stdio.h>
#include <stddef.h>

static inline size_t idx(size_t r, size_t c, size_t stride) {
    return r * stride + c;
}

/* Pretty-print a row-major matrix */
static void print_matrix(const char *name, const double *X, size_t rows, size_t cols) {
    printf("%s (%zux%zu):\n", name, rows, cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            printf("%10.4f ", X[idx(i, j, cols)]);
        }
        printf("\n");
    }
    printf("\n");
}

// C[N×K] = A[N×M] * B[M×K]
void matmul_ijk(const double *A, const double *B, double *C,
                size_t N, size_t M, size_t K)
{
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < K; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < M; k++) {
                sum += A[idx(i, k, M)] * B[idx(k, j, K)];
            }
            C[idx(i, j, K)] = sum;
        }
    }
}

int main(void) {
    /* Example 1: A = 1x1, B = 1x1 */
    {
        size_t N = 1, M = 1, K = 1;
        double A[1] = { 2.0 };
        double B[1] = { 3.5 };
        double C[1];

        matmul_ijk(A, B, C, N, M, K);

        print_matrix("A", A, N, M);
        print_matrix("B", B, M, K);
        print_matrix("C = A*B", C, N, K);
    }

    /* Example 2: A = 1x1, B = 1x5 */
    {
        size_t N = 1, M = 1, K = 5;
        double A[1] = { -2.0 };
        double B[5] = { 1.0, -1.5, 0.0, 4.0, 2.25 };
        double C[5];

        matmul_ijk(A, B, C, N, M, K);

        print_matrix("A", A, N, M);
        print_matrix("B", B, M, K);
        print_matrix("C = A*B", C, N, K);
    }

    return 0;
}
