#pragma once
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <cstdio>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x, dt) TORCH_CHECK((x).scalar_type() == (dt), #x " has wrong dtype")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

static inline void cuda_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s : %s\n", msg, cudaGetErrorString(e));
        std::abort();
    }
}