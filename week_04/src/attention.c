
#include <math.h>
#include <string.h>

#include "assertc.h"
#include "trace.h"
#include "float_eq.h"
#include "mat_mul.h"
#include "transpose.h"
#include "softmax.h"
#include "mat_print.h"


void dot_mul(float alpha, float * A, float * B, size_t N) {
    for (size_t i = 0; i < N; i++) {
        B[i] = alpha * A[i];
    }
}

void attention(float *Q, float *K, float *V, float *O, size_t N, size_t d) {
    float* Kx = (float*) malloc(N * d * sizeof(float));
    transpose(K, Kx, N, d);

    mat_print("Kx", Kx, d, N);

    float* QKx = (float*) malloc(N * N * sizeof(float));
    mat_mul(Q, Kx, QKx, N, d, N);

    mat_print("QKx", QKx, N, N);

    free(Kx);

    float* S = (float*) malloc(N * N * sizeof(float));
    dot_mul(1.0f / sqrtf((float)d), QKx, S, N * N);

    mat_print("S", S, N, N);

    free(QKx);

    float* P = (float*) malloc(N * N * sizeof(float));
    softmax(S, P, N, N);

    mat_print("P", P, N, N);


    free(S);

    mat_mul(P, V, O, N, N, d);

    mat_print("O", O, N, d);

    free(P);
}

int main(void) {
    signal(SIGABRT, handler);

    size_t N = 2;
    size_t d = 4;

    float Qdata[] = {
        1, 0, 1, 0,
        0, 1, 0, 1
    };
    float Kdata[] = {
        1, 0, 1, 0,
        0, 1, 0, 1
    };
    float Vdata[] = {
        10, 20, 30, 40,
        50, 60, 70, 80
    };

    float Odata[] = {
        20.75766f, 30.75766f, 40.75766f, 50.75766f,
        39.24234f, 49.24235f, 59.24235f, 69.24235f,
    };

    size_t size = N * d * sizeof(float);

    float *h_Q = (float*) malloc(size); assertc(h_Q != 0);
    float *h_K = (float*) malloc(size); assertc(h_K != 0);
    float *h_V = (float*) malloc(size); assertc(h_V != 0);
    float *h_O = (float*) malloc(size); assertc(h_O != 0);

    memcpy(h_Q, Qdata, size);
    memcpy(h_K, Kdata, size);
    memcpy(h_V, Vdata, size);

    attention(h_Q, h_K, h_V, h_O, N, d);

    check_float_array_eq(h_O, Odata, N * d);

    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);

   return 0;
}
