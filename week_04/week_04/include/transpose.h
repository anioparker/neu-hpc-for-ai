#pragma once

#include "assertc.h"
#include <stdlib.h>
#include "mat_util.h"

void transpose(float *A, float *B, size_t M, size_t N) {

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            size_t AreadIdx = at(i, j, M, N);
            size_t BwriteIdx = at(j, i, N, M);
            B[BwriteIdx] = A[AreadIdx];
        }
    }
}
