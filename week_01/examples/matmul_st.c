// matmul_single_thread.c
// Build: gcc -std=c11 -O3 -march=native -Wall -Wextra matmul_single_thread.c -o matmul -lm
// Run:   ./matmul

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>

static inline size_t idx(size_t r, size_t c, size_t stride) {
    return r * stride + c;
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

// Reference implementation (different loop order) to cross-check.
// We zero C and accumulate into it: i-k-j
static void matmul_ref_ikj(const double *A, const double *B, double *C,
                           size_t N, size_t M, size_t K)
{
    if (N == 0 || K == 0) return;
    memset(C, 0, N * K * sizeof(double));

    for (size_t i = 0; i < N; i++) {
        for (size_t k = 0; k < M; k++) {
            double a = A[idx(i, k, M)];
            const double *b_row = &B[idx(k, 0, K)];
            double *c_row = &C[idx(i, 0, K)];
            for (size_t j = 0; j < K; j++) {
                c_row[j] += a * b_row[j];
            }
        }
    }
}

// Deterministic filler so tests are reproducible
static void fill_matrix(double *X, size_t rows, size_t cols, unsigned seed) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            // Small values to avoid overflow, but not all integers
            X[idx(i, j, cols)] = (double)((seed + 1u) * 131u + (unsigned)i * 17u + (unsigned)j * 31u) / 1000.0;
        }
    }
}

static int nearly_equal(double a, double b) {
    double diff = fabs(a - b);
    double scale = fmax(1.0, fmax(fabs(a), fabs(b)));
    return diff <= 1e-9 * scale;
}

static int run_one_test(size_t N, size_t M, size_t K) {
    size_t a_elems = N * M;
    size_t b_elems = M * K;
    size_t c_elems = N * K;

    // malloc(0) is allowed but can return NULL; handle sizes carefully.
    double *A = (a_elems ? (double*)malloc(a_elems * sizeof(double)) : NULL);
    double *B = (b_elems ? (double*)malloc(b_elems * sizeof(double)) : NULL);
    double *C = (c_elems ? (double*)malloc(c_elems * sizeof(double)) : NULL);
    double *R = (c_elems ? (double*)malloc(c_elems * sizeof(double)) : NULL);

    if ((a_elems && !A) || (b_elems && !B) || (c_elems && (!C || !R))) {
        fprintf(stderr, "Allocation failed for N=%zu M=%zu K=%zu\n", N, M, K);
        free(A); free(B); free(C); free(R);
        return 0;
    }

    if (A) fill_matrix(A, N, M, 1u);
    if (B) fill_matrix(B, M, K, 2u);

    // If N==0 or K==0, there is no C to compute; treat as pass.
    if (N == 0 || K == 0) {
        free(A); free(B); free(C); free(R);
        return 1;
    }

    // Initialize C to something nonzero to detect "forgot to write" bugs.
    for (size_t i = 0; i < c_elems; i++) C[i] = 123.456;

    // For M==0, correct output is all zeros (empty sum).
    // matmul_ijk naturally sets sum=0 and writes it for each (i,j), so it's fine.
    matmul_ijk(A, B, C, N, M, K);
    matmul_ref_ikj(A, B, R, N, M, K);

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < K; j++) {
            double x = C[idx(i, j, K)];
            double y = R[idx(i, j, K)];
            if (!nearly_equal(x, y)) {
                fprintf(stderr, "FAIL N=%zu M=%zu K=%zu at (%zu,%zu): got=%g ref=%g\n",
                        N, M, K, i, j, x, y);
                free(A); free(B); free(C); free(R);
                return 0;
            }
        }
    }

    free(A); free(B); free(C); free(R);
    return 1;
}

int main(void) {
    // A set of sizes that hits many corner cases and variety:
    // - includes 0, 1 (corner cases)
    // - small primes (3,5,7)
    // - a few larger-ish sizes
    size_t sizes[] = {0, 1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 31, 32};
    size_t n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    size_t total = 0, passed = 0;

    // Exhaustive combinations over these sizes for (N,M,K)
    for (size_t a = 0; a < n_sizes; a++) {
        for (size_t b = 0; b < n_sizes; b++) {
            for (size_t c = 0; c < n_sizes; c++) {
                size_t N = sizes[a], M = sizes[b], K = sizes[c];
                total++;
                if (run_one_test(N, M, K)) passed++;
                else {
                    fprintf(stderr, "Stopped on first failure.\n");
                    printf("Passed %zu/%zu tests\n", passed, total);
                    return 1;
                }
            }
        }
    }

    printf("All tests passed: %zu/%zu\n", passed, total);
    return 0;
}

