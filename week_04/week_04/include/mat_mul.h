#pragma once

#include <stdlib.h>
#include "mat_util.h"

void mat_mul(float *A, float *B, float *C, size_t M, size_t K, size_t N) {

    for (size_t x = 0; x < M; x++) {
        for (size_t y = 0; y < N; y++) {
            float sum = 0;
            for (size_t k = 0; k < K; k++) {
                size_t AIdx = at(x, k, M, K);
                size_t BIdx = at(k, y, K, N);

                sum += A[AIdx] * B[BIdx];
            }

            size_t CIdx = at(x, y, M, N);
            C[CIdx] = sum;
        }
    }

}
