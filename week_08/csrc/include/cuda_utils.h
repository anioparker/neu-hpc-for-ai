#pragma once
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_DTYPE_LONG(x) TORCH_CHECK((x).scalar_type() == at::kLong, #x " must be int64")
#define CHECK_INPUT_FLOAT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_DTYPE_FLOAT(x)
#define CHECK_INPUT_LONG(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_DTYPE_LONG(x)

inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error at " << file << ":" << line
            << " code=" << static_cast<int>(err)
            << " \"" << cudaGetErrorString(err) << "\"";
        throw std::runtime_error(oss.str());
    }
}
#define CUDA_CHECK(err) cuda_check((err), __FILE__, __LINE__)