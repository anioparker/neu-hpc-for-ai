#pragma once

#include "assertc.h"
#include <math.h>
#include <stdlib.h>
#include "mat_util.h"

void softmax(float *A, float *B, size_t M, size_t N) {

    for (size_t r = 0; r < M; r++) {
        float max = 0;
        for (size_t c = 0; c < N; c++) {
            size_t AIdx = at(r, c, M, N);
            if (A[c] > max) {
                max = A[AIdx];
            }
        }

        float sum = 0;
        for (size_t c = 0; c < N; c++) {
            size_t AIdx = at(r, c, M, N);
            sum += expf(A[AIdx] - max);
        }

        for (size_t c = 0; c < N; c++) {
            size_t AIdx = at(r, c, M, N);
            size_t BIdx = at(r, c, M, N);

            B[BIdx] = expf(A[AIdx] - max) / sum;
        }
    }
}

