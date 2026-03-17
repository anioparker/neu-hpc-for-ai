#include "tensor.h"

#include <stdio.h>
#include <stdlib.h>

float *tensor_alloc(size_t n) {
    float *ptr = (float *)calloc(n, sizeof(float));
    if (!ptr) {
        fprintf(stderr, "tensor_alloc failed for %zu floats\n", n);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void tensor_free(float *ptr) {
    free(ptr);
}

void tensor_fill(float *x, size_t n, float value) {
    for (size_t i = 0; i < n; ++i) {
        x[i] = value;
    }
}

void tensor_copy(float *dst, const float *src, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = src[i];
    }
}

void tensor_print_2d(const float *x, int rows, int cols, const char *name) {
    printf("%s = [\n", name);
    for (int r = 0; r < rows; ++r) {
        printf("  [");
        for (int c = 0; c < cols; ++c) {
            printf("% .6f", x[r * cols + c]);
            if (c + 1 < cols) {
                printf(", ");
            }
        }
        printf("]\n");
    }
    printf("]\n");
}
