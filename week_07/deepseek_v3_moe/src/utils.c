#include "utils.h"

#include <math.h>

float sigmoid(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

float silu(float x) {
    return x * sigmoid(x);
}

float nearly_equal(float a, float b, float eps) {
    float diff = fabsf(a - b);
    return diff <= eps ? 1.0f : 0.0f;
}

void zero_vector(float *x, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] = 0.0f;
    }
}

void vec_add_scaled(float *dst, const float *src, float scale, int n) {
    for (int i = 0; i < n; ++i) {
        dst[i] += scale * src[i];
    }
}

void matvec(const float *x, const float *w, float *y, int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < in_dim; ++i) {
            sum += x[i] * w[i * out_dim + j];
        }
        y[j] = sum;
    }
}

void hadamard(float *out, const float *a, const float *b, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}
