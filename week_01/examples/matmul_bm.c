// matmul_benchmark.c
// Benchmark pthreads matrix multiplication speedup.
//
// Build:
//   gcc -std=c11 -O3 -march=native -Wall -Wextra matmul_benchmark.c -o bench -pthread -lm
//
// Run (defaults N=M=K=2048, repeats=3):
//   ./bench
//
// Run custom (e.g., 4096^3, repeats=5):
//   ./bench 4096 4096 4096 5

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

static inline size_t idx(size_t r, size_t c, size_t stride) {
    return r * stride + c;
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
    if (num_threads > N) num_threads = N;

    pthread_t *threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    WorkerArgs *args   = (WorkerArgs*)malloc(num_threads * sizeof(WorkerArgs));
    if (!threads || !args) {
        free(threads); free(args);
        WorkerArgs w = {A, B, C, N, M, K, 0, N};
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
            for (size_t j = 0; j < created; j++) pthread_join(threads[j], NULL);
            WorkerArgs w = {A, B, C, N, M, K, 0, N};
            worker_rows_ijk(&w);
            free(threads); free(args);
            return;
        }
        created++;
    }

    for (size_t t = 0; t < created; t++) pthread_join(threads[t], NULL);

    free(threads);
    free(args);
}

/* -------------------- Benchmark utilities -------------------- */

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// static void *xmalloc_aligned(size_t bytes, size_t alignment) {
//     void *p = NULL;
//     // posix_memalign requires alignment to be power of 2 and multiple of sizeof(void*)
//     if (posix_memalign(&p, alignment, bytes) != 0) return NULL;
//     return p;
// }

// deterministic fill
static void fill_matrix(double *X, size_t rows, size_t cols, unsigned seed) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            X[idx(i, j, cols)] = (double)((seed + 1u) * 131u + (unsigned)i * 17u + (unsigned)j * 31u) / 1000.0;
        }
    }
}

static double checksum_matrix(const double *C, size_t rows, size_t cols) {
    // Summing all entries prevents the compiler from "optimizing away" work.
    // Cost is tiny vs O(N*M*K) for large matmul.
    double s = 0.0;
    for (size_t i = 0; i < rows * cols; i++) s += C[i];
    return s;
}

static double best_time_for_threads(const double *A, const double *B, double *C,
                                    size_t N, size_t M, size_t K,
                                    size_t threads, int repeats,
                                    volatile double *out_checksum)
{
    double best = 1e300;

    for (int r = 0; r < repeats; r++) {
        double t0 = now_seconds();
        matmul_pthreads(A, B, C, N, M, K, threads);
        double t1 = now_seconds();

        // Touch results (prevents dead-code elimination)
        double cs = checksum_matrix(C, N, K);
        *out_checksum += cs;

        double dt = t1 - t0;
        if (dt < best) best = dt; // best-of-N repeats helps reduce noise
    }
    return best;
}

int main(int argc, char **argv) {
    size_t N = 2048, M = 2048, K = 2048;
    int repeats = 3;

    if (argc == 5) {
        N = (size_t)strtoull(argv[1], NULL, 10);
        M = (size_t)strtoull(argv[2], NULL, 10);
        K = (size_t)strtoull(argv[3], NULL, 10);
        repeats = (int)strtol(argv[4], NULL, 10);
        if (repeats < 1) repeats = 1;
    } else if (argc != 1) {
        fprintf(stderr, "Usage: %s [N M K repeats]\n", argv[0]);
        return 1;
    }

    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (cores < 1) cores = 1;
    printf("CPU cores reported: %ld\n", cores);
    printf("Matrix sizes: A=%zux%zu, B=%zux%zu, C=%zux%zu, repeats=%d\n", N, M, M, K, N, K, repeats);

    size_t a_elems = N * M;
    size_t b_elems = M * K;
    size_t c_elems = N * K;

    // size_t align = 64; // cacheline-friendly
    // double *A = (double*)xmalloc_aligned(a_elems * sizeof(double), align);
    // double *B = (double*)xmalloc_aligned(b_elems * sizeof(double), align);
    // double *C = (double*)xmalloc_aligned(c_elems * sizeof(double), align);

    double *A = a_elems ? (double*)malloc(a_elems * sizeof(double)) : NULL;
    double *B = b_elems ? (double*)malloc(b_elems * sizeof(double)) : NULL;
    double *C = c_elems ? (double*)malloc(c_elems * sizeof(double)) : NULL;


    if ((a_elems && !A) || (b_elems && !B) || (c_elems && !C)) {
        fprintf(stderr, "Allocation failed (try smaller N/M/K).\n");
        free(A); free(B); free(C);
        return 2;
    }

    fill_matrix(A, N, M, 1u);
    fill_matrix(B, M, K, 2u);

    // Warm-up (helps stabilize CPU frequency / caches)
    volatile double sink = 0.0;
    matmul_pthreads(A, B, C, N, M, K, 1);
    sink += checksum_matrix(C, N, K);

    size_t thread_list[] = {1, 4, 16, 32, 64, 128};
    size_t nt = sizeof(thread_list) / sizeof(thread_list[0]);

    double t1 = 0.0;

    printf("\n%-10s %-12s %-12s %-12s\n", "threads", "time(s)", "speedup", "GFLOP/s");
    for (size_t i = 0; i < nt; i++) {
        size_t T = thread_list[i];

        double dt = best_time_for_threads(A, B, C, N, M, K, T, repeats, &sink);

        if (T == 1) t1 = dt;

        double speedup = (t1 > 0.0) ? (t1 / dt) : 0.0;

        long double flops = 2.0L * (long double)N * (long double)M * (long double)K;
        long double gflops = (dt > 0.0) ? (flops / ( (long double)dt * 1e9L )) : 0.0L;

        printf("%-10zu %-12.6f %-12.3f %-12.2Lf\n", T, dt, speedup, gflops);
    }

    // Print checksum sink to prevent “unused” warnings and keep work “observable”
    printf("\n(ignore) checksum accumulator: %.5f\n", (double)sink);

    free(A); free(B); free(C);
    return 0;
}
