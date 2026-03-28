#pragma once
#include <torch/extension.h>

// x:        [M, H]
// expert_id:[M]          local expert ids in [0, E_local)
// gate_w:   [E_local, H, I]
// up_w:     [E_local, H, I]
// down_w:   [E_local, I, H]
torch::Tensor local_expert_forward_cuda(
    torch::Tensor x,
    torch::Tensor expert_id,
    torch::Tensor gate_w,
    torch::Tensor up_w,
    torch::Tensor down_w);