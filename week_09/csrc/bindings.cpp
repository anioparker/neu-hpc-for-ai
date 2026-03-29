#include "common.cuh"

torch::Tensor deepseek_moe_tk_forward_cuda(
    torch::Tensor x,
    torch::Tensor topk_idx,
    torch::Tensor topk_w,
    torch::Tensor w1,
    torch::Tensor w2);

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor topk_idx,
    torch::Tensor topk_w,
    torch::Tensor w1,
    torch::Tensor w2) {

    CHECK_INPUT(x);
    CHECK_INPUT(topk_idx);
    CHECK_INPUT(topk_w);
    CHECK_INPUT(w1);
    CHECK_INPUT(w2);

    TORCH_CHECK(x.dim() == 2, "x must be [T, H]");
    TORCH_CHECK(topk_idx.dim() == 2, "topk_idx must be [T, K]");
    TORCH_CHECK(topk_w.dim() == 2, "topk_w must be [T, K]");
    TORCH_CHECK(w1.dim() == 3, "w1 must be [E, H, I]");
    TORCH_CHECK(w2.dim() == 3, "w2 must be [E, I, H]");

    TORCH_CHECK(x.size(0) == topk_idx.size(0), "x/topk_idx T mismatch");
    TORCH_CHECK(x.size(0) == topk_w.size(0), "x/topk_w T mismatch");
    TORCH_CHECK(w1.size(0) == w2.size(0), "expert count mismatch");
    TORCH_CHECK(w1.size(2) == w2.size(1), "intermediate size mismatch");
    TORCH_CHECK(w1.size(1) == x.size(1), "hidden size mismatch");
    TORCH_CHECK(w2.size(2) == x.size(1), "hidden size mismatch");

    return deepseek_moe_tk_forward_cuda(x, topk_idx, topk_w, w1, w2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "DeepSeek MoE ThunderKittens forward (CUDA)");
}