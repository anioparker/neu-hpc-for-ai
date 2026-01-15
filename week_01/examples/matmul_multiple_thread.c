// matmul_multiple_thread.c
// Multi-threaded matrix multiplication (pthreads) + built-in correctness tests.
//
// Build:
//   gcc -std=c11 -O3 -march=native -Wall -Wextra matmul_multiple_thread.c -o matmul -pthread -lm
//
// Run:
//   ./matmul

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <pthread.h>
#include <unistd.h>   // sysconf

static inline size_t idx(size_t r, size_t c, size_t stride) {
    return r * stride + c;
}

/* -------------------- Pretty print -------------------- */
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

/* -------------------- Reference (single-thread, different loop order) --------------------
   This is only used for testing correctness. */
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

/* -------------------- Multi-threaded matmul (pthreads): split by output rows -------------------- */
typedef struct {
    const double *A;
    const double *B;
    double *C;
    size_t N, M, K;
    size_t i_start; // inclusive
    size_t i_end;   // exclusive
} WorkerArgs;

static void *worker_rows_ijk(void *arg) {
    WorkerArgs *w = (WorkerArgs*)arg;

    for (size_t i = w->i_start; i < w->i_end; i++) {
        for (size_t j = 0; j < w->K; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < w->M; k++) {
                sum += w->A[idx(i, k, w->M)] * w->B[idx(k, j, w->K)];
            }
            w->C[idx(i, j, w->K)] = sum;
        }
    }
    return NULL;
}

// C[N×K] = A[N×M] * B[M×K]
void matmul_pthreads(const double *A, const double *B, double *C,
                     size_t N, size_t M, size_t K,
                     size_t num_threads)
{
    if (N == 0 || K == 0) return;

    if (num_threads == 0) {
        long cores = sysconf(_SC_NPROCESSORS_ONLN);
        num_threads = (cores > 0) ? (size_t)cores : 1;
    }
    if (num_threads == 0) num_threads = 1;
    if (num_threads > N) num_threads = N; // more threads than rows is pointless

    pthread_t *threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    WorkerArgs *args   = (WorkerArgs*)malloc(num_threads * sizeof(WorkerArgs));
    if (!threads || !args) {
        fprintf(stderr, "Thread allocation failed.\n");
        free(threads); free(args);
        // fallback: do it in current thread
        WorkerArgs w = (WorkerArgs){A, B, C, N, M, K, 0, N};
        worker_rows_ijk(&w);
        return;
    }

    size_t base = N / num_threads;
    size_t rem  = N % num_threads;

    size_t i = 0;
    size_t created = 0;

    for (size_t t = 0; t < num_threads; t++) {
        size_t take = base + (t < rem ? 1 : 0);
        args[t] = (WorkerArgs){
            .A=A, .B=B, .C=C, .N=N, .M=M, .K=K,
            .i_start=i,
            .i_end=i+take
        };
        i += take;

        int rc = pthread_create(&threads[t], NULL, worker_rows_ijk, &args[t]);
        if (rc != 0) {
            fprintf(stderr, "pthread_create failed (rc=%d). Falling back.\n", rc);
            // join already created
            for (size_t j = 0; j < created; j++) pthread_join(threads[j], NULL);
            // compute all rows single-threaded
            WorkerArgs w = (WorkerArgs){A, B, C, N, M, K, 0, N};
            worker_rows_ijk(&w);
            free(threads); free(args);
            return;
        }
        created++;
    }

    for (size_t t = 0; t < created; t++) {
        pthread_join(threads[t], NULL);
    }

    free(threads);
    free(args);
}

/* -------------------- Test harness -------------------- */
static void fill_matrix(double *X, size_t rows, size_t cols, unsigned seed) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            // deterministic small-ish values
            X[idx(i, j, cols)] = (double)((seed + 1u) * 131u + (unsigned)i * 17u + (unsigned)j * 31u) / 1000.0;
        }
    }
}

static int nearly_equal(double a, double b) {
    double diff  = fabs(a - b);
    double scale = fmax(1.0, fmax(fabs(a), fabs(b)));
    return diff <= 1e-9 * scale;
}

static int run_one_test(size_t N, size_t M, size_t K, size_t threads) {
    size_t a_elems = N * M;
    size_t b_elems = M * K;
    size_t c_elems = N * K;

    double *A   = a_elems ? (double*)malloc(a_elems * sizeof(double)) : NULL;
    double *B   = b_elems ? (double*)malloc(b_elems * sizeof(double)) : NULL;
    double *C   = c_elems ? (double*)malloc(c_elems * sizeof(double)) : NULL;
    double *Ref = c_elems ? (double*)malloc(c_elems * sizeof(double)) : NULL;

    if ((a_elems && !A) || (b_elems && !B) || (c_elems && (!C || !Ref))) {
        fprintf(stderr, "Allocation failed for N=%zu M=%zu K=%zu\n", N, M, K);
        free(A); free(B); free(C); free(Ref);
        return 0;
    }

    if (A) fill_matrix(A, N, M, 1u);
    if (B) fill_matrix(B, M, K, 2u);

    if (N == 0 || K == 0) { // nothing to compare
        free(A); free(B); free(C); free(Ref);
        return 1;
    }

    // sentinel init to catch "forgot to write" bugs
    for (size_t i = 0; i < c_elems; i++) C[i] = 123.456;

    matmul_pthreads(A, B, C, N, M, K, threads);
    matmul_ref_ikj(A, B, Ref, N, M, K);

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < K; j++) {
            double x = C[idx(i, j, K)];
            double y = Ref[idx(i, j, K)];
            if (!nearly_equal(x, y)) {
                fprintf(stderr,
                        "FAIL threads=%zu N=%zu M=%zu K=%zu at (%zu,%zu): got=%g ref=%g\n",
                        threads, N, M, K, i, j, x, y);
                free(A); free(B); free(C); free(Ref);
                return 0;
            }
        }
    }

    free(A); free(B); free(C); free(Ref);
    return 1;
}

/* -------------------- Demo cases (print actual results) -------------------- */
static void demo_case_2x1_times_1x3(void) {
    // A: 2x1, B: 1x3 -> C: 2x3
    const size_t N = 2, M = 1, K = 3;
    double A[2] = {
        2.0,
       -1.0
    };
    double B[3] = {
        1.5, 0.0, -2.0
    };
    double C[2 * 3];

    matmul_pthreads(A, B, C, N, M, K, 0); // 0 = auto threads

    printf("=== Demo: A (2x1) * B (1x3) ===\n");
    print_matrix("A", A, N, M);
    print_matrix("B", B, M, K);
    print_matrix("C = A*B", C, N, K);
}

static void demo_case_2x2_times_2x2(void) {
    // A: 2x2, B: 2x2 -> C: 2x2
    const size_t N = 2, M = 2, K = 2;
    double A[4] = {
        1.0, 2.0,
        3.0, 4.0
    };
    double B[4] = {
        5.0, 6.0,
        7.0, 8.0
    };
    double C[4];

    matmul_pthreads(A, B, C, N, M, K, 0); // 0 = auto threads

    printf("=== Demo: A (2x2) * B (2x2) ===\n");
    print_matrix("A", A, N, M);
    print_matrix("B", B, M, K);
    print_matrix("C = A*B", C, N, K);
}

int main(void) {
    // wide set of sizes including corner cases
    size_t sizes[] = {0, 1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 31, 32};
    size_t n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    // test multiple thread counts (0 means "auto")
    size_t thread_counts[] = {1, 2, 3, 4, 8, 0};
    size_t n_threads = sizeof(thread_counts) / sizeof(thread_counts[0]);

    // Run tests silently; print only if a failure occurs.
    for (size_t ti = 0; ti < n_threads; ti++) {
        size_t T = thread_counts[ti];
        for (size_t a = 0; a < n_sizes; a++) {
            for (size_t b = 0; b < n_sizes; b++) {
                for (size_t c = 0; c < n_sizes; c++) {
                    size_t N = sizes[a], M = sizes[b], K = sizes[c];
                    if (!run_one_test(N, M, K, T)) {
                        fprintf(stderr, "Stopped on first failure.\n");
                        return 1;
                    }
                }
            }
        }
    }

    // Print requested demo results
    demo_case_2x1_times_1x3();
    demo_case_2x2_times_2x2();

    return 0;
}
