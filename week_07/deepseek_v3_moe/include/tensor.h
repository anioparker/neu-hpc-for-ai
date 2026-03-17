#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

float *tensor_alloc(size_t n);
void tensor_free(float *ptr);
void tensor_fill(float *x, size_t n, float value);
void tensor_copy(float *dst, const float *src, size_t n);
void tensor_print_2d(const float *x, int rows, int cols, const char *name);

#endif
