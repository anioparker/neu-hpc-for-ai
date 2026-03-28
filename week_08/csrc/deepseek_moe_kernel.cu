#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <cmath>
#include "include/deepseek_moe.h"
#include "include/cuda_utils.h"

namespace {

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// One block handles one token row.
__global__ void local_expert_ffn_kernel(
    const float* __restrict__ x,          // [M, H]
    const int64_t* __restrict__ expert_id,// [M]
    const float* __restrict__ gate_w,     // [E, H, I]
    const float* __restrict__ up_w,       // [E, H, I]
    const float* __restrict__ down_w,     // [E, I, H]
    float* __restrict__ y,                // [M, H]
    int M,
    int E,
    int H,
    int I
) {
    int row = blockIdx.x;
    if (row >= M) return;

    int e = static_cast<int>(expert_id[row]);
    if (e < 0 || e >= E) return;

    extern __shared__ float smem[];
    float* gate_buf = smem;       // [I]
    float* up_buf   = smem + I;   // [I]

    // Compute gate and up projections
    for (int j = threadIdx.x; j < I; j += blockDim.x) {
        float g = 0.0f;
        float u = 0.0f;
        for (int h = 0; h < H; ++h) {
            float xv = x[row * H + h];
            g += xv * gate_w[(e * H + h) * I + j];
            u += xv * up_w[(e * H + h) * I + j];
        }
        gate_buf[j] = g;
        up_buf[j] = u;
    }
    __syncthreads();

    // Down projection after SiLU(gate) * up
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        float out = 0.0f;
        for (int j = 0; j < I; ++j) {
            float z = silu(gate_buf[j]) * up_buf[j];
            out += z * down_w[(e * I + j) * H + h];
        }
        y[row * H + h] = out;
    }
}

} // namespace

torch::Tensor local_expert_forward_cuda(
    torch::Tensor x,
    torch::Tensor expert_id,
    torch::Tensor gate_w,
    torch::Tensor up_w,
    torch::Tensor down_w
) {
    CHECK_INPUT_FLOAT(x);
    CHECK_INPUT_LONG(expert_id);
    CHECK_INPUT_FLOAT(gate_w);
    CHECK_INPUT_FLOAT(up_w);
    CHECK_INPUT_FLOAT(down_w);

    TORCH_CHECK(x.dim() == 2, "x must be [M, H]");
    TORCH_CHECK(expert_id.dim() == 1, "expert_id must be [M]");
    TORCH_CHECK(gate_w.dim() == 3, "gate_w must be [E_local, H, I]");
    TORCH_CHECK(up_w.dim() == 3, "up_w must be [E_local, H, I]");
    TORCH_CHECK(down_w.dim() == 3, "down_w must be [E_local, I, H]");

    const int64_t M = x.size(0);
    const int64_t H = x.size(1);
    const int64_t E = gate_w.size(0);
    const int64_t I = gate_w.size(2);

    TORCH_CHECK(expert_id.size(0) == M, "expert_id size mismatch");
    TORCH_CHECK(gate_w.size(1) == H, "gate_w H mismatch");
    TORCH_CHECK(up_w.size(0) == E && up_w.size(1) == H && up_w.size(2) == I, "up_w shape mismatch");
    TORCH_CHECK(down_w.size(0) == E && down_w.size(1) == I && down_w.size(2) == H, "down_w shape mismatch");

    auto y = torch::zeros({M, H}, x.options());

    const int threads = 256;
    const int blocks = static_cast<int>(M);
    const size_t shmem = static_cast<size_t>(2 * I * sizeof(float));

    local_expert_ffn_kernel<<<blocks, threads, shmem, at::cuda::getDefaultCUDAStream()>>>(
        x.data_ptr<float>(),
        expert_id.data_ptr<int64_t>(),
        gate_w.data_ptr<float>(),
        up_w.data_ptr<float>(),
        down_w.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<int>(M),
        static_cast<int>(E),
        static_cast<int>(H),
        static_cast<int>(I)
    );
    CUDA_CHECK(cudaGetLastError());

    return y;
}