#include <torch/extension.h>
#include <vector>
#include "include/deepseek_moe.h"

torch::Tensor local_expert_forward_cuda(
    torch::Tensor x,
    torch::Tensor expert_id,
    torch::Tensor gate_w,
    torch::Tensor up_w,
    torch::Tensor down_w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "local_expert_forward",
        &local_expert_forward_cuda,
        "DeepSeek-style local expert forward (CUDA)"
    );
}