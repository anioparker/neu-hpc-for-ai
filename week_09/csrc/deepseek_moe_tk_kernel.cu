#include "common.cuh"

#include <mma.h>

using namespace nvcuda;

// ------------------------------
// Kernel configuration
// ------------------------------
constexpr int TILE_M = 32;   // tokens tile
constexpr int TILE_N = 32;   // output/hidden tile
constexpr int TILE_K = 32;   // reduction tile for tensor core MMA

// one block handles one (token_tile, expert_route, hidden_tile)
constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS = WARPS_PER_BLOCK * 32;

template <typename T>
__device__ inline float to_float(T x) { return static_cast<float>(x); }

template <>
__device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ inline __nv_bfloat16 to_bf16(float x) {
    return __float2bfloat16(x);
}

// ------------------------------
// Simple activation
// ------------------------------
__device__ inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

// ------------------------------
// Stage 1: expert up-projection
//
// x:       [T, H]
// w1:      [E, H, I]
// hidden:  [T, Kroute, I] logically, but we stream it
// ------------------------------
__global__ void moe_up_kernel(
    const __nv_bfloat16* __restrict__ x,        // [T, H]
    const int64_t* __restrict__ topk_idx,       // [T, K]
    const float* __restrict__ topk_w,           // [T, K]
    const __nv_bfloat16* __restrict__ w1,       // [E, H, I]
    float* __restrict__ hidden_buf,             // [T, K, I] fp32
    int T,
    int H,
    int I,
    int E,
    int Ksel)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = T * Ksel * I;
    if (idx >= total) return;

    const int i = idx % I;
    const int tmp = idx / I;
    const int kslot = tmp % Ksel;
    const int t = tmp / Ksel;

    const int expert = static_cast<int>(topk_idx[t * Ksel + kslot]);
    if (expert < 0 || expert >= E) {
        hidden_buf[idx] = 0.0f;
        return;
    }

    float acc = 0.0f;
    for (int h = 0; h < H; ++h) {
        const float xv = __bfloat162float(x[t * H + h]);
        const float wv = __bfloat162float(w1[(expert * H + h) * I + i]);
        acc += xv * wv;
    }

    hidden_buf[idx] = silu(acc);
}

// ------------------------------
// Stage 2: down projection and merge
// hidden_buf: [T, K, I] fp32
// w2:         [E, I, H] bf16
// out:        [T, H] bf16
// ------------------------------
__global__ void moe_down_kernel(
    const float* __restrict__ hidden_buf,       // [T, K, I]
    const int64_t* __restrict__ topk_idx,       // [T, K]
    const float* __restrict__ topk_w,           // [T, K]
    const __nv_bfloat16* __restrict__ w2,       // [E, I, H]
    __nv_bfloat16* __restrict__ out,            // [T, H]
    int T,
    int I,
    int H,
    int E,
    int Ksel)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || h >= H) return;

    float y = 0.0f;

    for (int k = 0; k < Ksel; ++k) {
        int e = static_cast<int>(topk_idx[t * Ksel + k]);
        float gate = topk_w[t * Ksel + k];

        float partial = 0.0f;
        for (int i = 0; i < I; ++i) {
            float hv = hidden_buf[(t * Ksel + k) * I + i];
            float wv = __bfloat162float(w2[(e * I + i) * H + h]);
            partial += hv * wv;
        }
        y += gate * partial;
    }

    out[t * H + h] = to_bf16(y);
}

torch::Tensor deepseek_moe_tk_forward_cuda(
    torch::Tensor x,
    torch::Tensor topk_idx,
    torch::Tensor topk_w,
    torch::Tensor w1,
    torch::Tensor w2)
{
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(w1.scalar_type() == torch::kBFloat16, "w1 must be bf16");
    TORCH_CHECK(w2.scalar_type() == torch::kBFloat16, "w2 must be bf16");
    TORCH_CHECK(topk_idx.scalar_type() == torch::kInt64 || topk_idx.scalar_type() == torch::kInt32,
                "topk_idx must be int32/int64");
    TORCH_CHECK(topk_w.scalar_type() == torch::kFloat || topk_w.scalar_type() == torch::kBFloat16,
                "topk_w must be fp32 or bf16");

    const int T = x.size(0);
    const int H = x.size(1);
    const int Ksel = topk_idx.size(1);
    const int E = w1.size(0);
    const int I = w1.size(2);

    auto hidden_buf = torch::zeros({T, Ksel, I}, x.options().dtype(torch::kFloat));
    auto out = torch::zeros({T, H}, x.options());

    torch::Tensor topk_idx_i64 = topk_idx;
    if (topk_idx.scalar_type() == torch::kInt32) {
        topk_idx_i64 = topk_idx.to(torch::kInt64);
    }

    torch::Tensor topk_w_f32 = topk_w.scalar_type() == torch::kFloat
        ? topk_w
        : topk_w.to(torch::kFloat);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    const int up_threads = 256;
    dim3 grid_up(ceil_div(T * Ksel * I, up_threads));
    dim3 block_up(up_threads);

    moe_up_kernel<<<grid_up, block_up, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        topk_idx_i64.data_ptr<int64_t>(),
        topk_w_f32.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(w1.data_ptr<at::BFloat16>()),
        hidden_buf.data_ptr<float>(),
        T, H, I, E, Ksel
    );

    dim3 block_down(256);
    dim3 grid_down(T, ceil_div(H, 256));

    moe_down_kernel<<<grid_down, block_down, 0, stream>>>(
        hidden_buf.data_ptr<float>(),
        topk_idx_i64.data_ptr<int64_t>(),
        topk_w_f32.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(w2.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        T, I, H, E, Ksel
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
