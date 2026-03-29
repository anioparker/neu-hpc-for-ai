import torch
import deepseek_moe_tk_ext


def moe_ffn_forward(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_w: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """
    x:       [T, H]           bf16
    topk_idx:[T, K]           int32/int64
    topk_w:  [T, K]           fp32/bf16
    w1:      [E, H, I]        bf16
    w2:      [E, I, H]        bf16

    returns: [T, H]           bf16
    """
    assert x.is_cuda
    assert w1.is_cuda and w2.is_cuda
    assert topk_idx.is_cuda and topk_w.is_cuda
    return deepseek_moe_tk_ext.forward(x, topk_idx, topk_w, w1, w2)