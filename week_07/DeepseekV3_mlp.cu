// nvcc -O2 -std=c++17 DeepseekV3_mlp.cu -o DeepseekV3_mlp
// ./DeepseekV3_mlp
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <iomanip>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " -> " << cudaGetErrorString(err) << std::endl;       \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

struct Config {
    int hidden_size = 2;
    int intermediate_size = 4;
};

// -----------------------------
// Device helpers
// -----------------------------
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// y = x * W^T
// x: [tokens, in_features]
// w: [out_features, in_features]
// y: [tokens, out_features]
__global__ void linear_forward_kernel(
    const float* x,
    const float* w,
    float* y,
    int tokens,
    int in_features,
    int out_features
) {
    int token_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int out_idx   = blockIdx.x * blockDim.x + threadIdx.x;

    if (token_idx < tokens && out_idx < out_features) {
        float acc = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            float xv = x[token_idx * in_features + k];
            float wv = w[out_idx * in_features + k];
            acc += xv * wv;
        }
        y[token_idx * out_features + out_idx] = acc;
    }
}

// z = silu(gate) * up
__global__ void silu_mul_kernel(
    const float* gate,
    const float* up,
    float* z,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        z[idx] = silu(gate[idx]) * up[idx];
    }
}

// -----------------------------
// Host-side MLP wrapper
// -----------------------------
class DeepseekV3MLP {
public:
    DeepseekV3MLP(const Config& cfg,
                  const std::vector<float>& gate_w,
                  const std::vector<float>& up_w,
                  const std::vector<float>& down_w)
        : config(cfg)
    {
        const int h = config.hidden_size;
        const int i = config.intermediate_size;

        assert((int)gate_w.size() == i * h);
        assert((int)up_w.size()   == i * h);
        assert((int)down_w.size() == h * i);

        CUDA_CHECK(cudaMalloc(&d_gate_w, gate_w.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_up_w,   up_w.size()   * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_down_w, down_w.size() * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_gate_w, gate_w.data(), gate_w.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_up_w,   up_w.data(),   up_w.size()   * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_down_w, down_w.data(), down_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    ~DeepseekV3MLP() {
        cudaFree(d_gate_w);
        cudaFree(d_up_w);
        cudaFree(d_down_w);
    }

    // x shape: [B, S, H]
    // returns y shape: [B, S, H]
    std::vector<float> forward(const std::vector<float>& x, int B, int S) const {
        const int H = config.hidden_size;
        const int I = config.intermediate_size;
        const int T = B * S;

        assert((int)x.size() == T * H);

        float* d_x = nullptr;
        float* d_gate = nullptr;
        float* d_up = nullptr;
        float* d_hidden = nullptr;
        float* d_y = nullptr;

        CUDA_CHECK(cudaMalloc(&d_x,      T * H * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gate,   T * I * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_up,     T * I * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_hidden, T * I * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y,      T * H * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_x, x.data(), T * H * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block1(16, 16);
        dim3 grid_gate((I + block1.x - 1) / block1.x, (T + block1.y - 1) / block1.y);
        dim3 grid_down((H + block1.x - 1) / block1.x, (T + block1.y - 1) / block1.y);

        // gate_proj(x)
        linear_forward_kernel<<<grid_gate, block1>>>(d_x, d_gate_w, d_gate, T, H, I);
        CUDA_CHECK(cudaGetLastError());

        // up_proj(x)
        linear_forward_kernel<<<grid_gate, block1>>>(d_x, d_up_w, d_up, T, H, I);
        CUDA_CHECK(cudaGetLastError());

        // silu(gate) * up
        int numel = T * I;
        int threads = 256;
        int blocks = (numel + threads - 1) / threads;
        silu_mul_kernel<<<blocks, threads>>>(d_gate, d_up, d_hidden, numel);
        CUDA_CHECK(cudaGetLastError());

        // down_proj(hidden)
        linear_forward_kernel<<<grid_down, block1>>>(d_hidden, d_down_w, d_y, T, I, H);
        CUDA_CHECK(cudaGetLastError());

        std::vector<float> y(T * H);
        CUDA_CHECK(cudaMemcpy(y.data(), d_y, T * H * sizeof(float), cudaMemcpyDeviceToHost));

        cudaFree(d_x);
        cudaFree(d_gate);
        cudaFree(d_up);
        cudaFree(d_hidden);
        cudaFree(d_y);

        return y;
    }

private:
    Config config;
    float* d_gate_w = nullptr;
    float* d_up_w = nullptr;
    float* d_down_w = nullptr;
};

// -----------------------------
// Utility printing
// -----------------------------
void print_tensor_3d(const std::vector<float>& v, int B, int S, int H, const std::string& name) {
    std::cout << name << " shape = [" << B << ", " << S << ", " << H << "]\n";
    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
            std::cout << "  [" << b << ", " << s << "] : ";
            for (int h = 0; h < H; ++h) {
                int idx = (b * S + s) * H + h;
                std::cout << std::fixed << std::setprecision(6) << v[idx] << " ";
            }
            std::cout << "\n";
        }
    }
}

// -----------------------------
// Demo main
// -----------------------------
int main() {
    Config config;
    const int H = config.hidden_size;
    const int I = config.intermediate_size;

    // Example deterministic weights
    // gate_proj.weight shape [I, H]
    std::vector<float> gate_w = {
         0.10f, -0.20f,
         0.30f,  0.40f,
        -0.50f,  0.60f,
         0.70f, -0.80f
    };

    // up_proj.weight shape [I, H]
    std::vector<float> up_w = {
         0.20f,  0.10f,
        -0.30f,  0.50f,
         0.40f, -0.60f,
         0.70f,  0.80f
    };

    // down_proj.weight shape [H, I]
    std::vector<float> down_w = {
         0.11f, -0.12f,  0.13f, -0.14f,
         0.21f, -0.22f,  0.23f, -0.24f
    };

    DeepseekV3MLP mlp(config, gate_w, up_w, down_w);

    // Example input: shape [B, S, H] = [2, 3, 2]
    int B = 2;
    int S = 3;
    std::vector<float> x = {
         0.5f, -1.0f,
         1.5f,  0.3f,
        -0.7f,  0.2f,

         0.0f,  1.0f,
        -1.2f,  0.8f,
         0.9f, -0.4f
    };

    auto y = mlp.forward(x, B, S);

    print_tensor_3d(x, B, S, H, "input");
    print_tensor_3d(y, B, S, H, "output");

    return 0;
}