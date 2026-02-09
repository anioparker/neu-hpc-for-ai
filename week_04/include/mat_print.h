
#pragma once

#include <stdlib.h>
#include "assertc.h"


void mat_print(char* name, float *A, size_t M, size_t N) {
    printf("%s:\n", name);
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            printf("%f ", A[i * N + j]);
        }
        printf("\n");
    }
}
