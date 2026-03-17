#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

float silu(float x);
float sigmoid(float x);
float nearly_equal(float a, float b, float eps);
void zero_vector(float *x, int n);
void vec_add_scaled(float *dst, const float *src, float scale, int n);
void matvec(const float *x, const float *w, float *y, int in_dim, int out_dim);
void hadamard(float *out, const float *a, const float *b, int n);

#endif
