import torch
import torch.nn.functional as F


def deepseek_expert_ffn(x, gate_w, up_w, down_w):
    # x: [M, H]
    # gate_w/up_w: [H, I]
    # down_w: [I, H]
    g = x @ gate_w
    u = x @ up_w
    z = F.silu(g) * u
    y = z @ down_w
    return y


def reference_moe_forward(
    x,              # [B, T, H]
    router_w,       # [H, E]
    gate_w,         # [E, H, I]
    up_w,           # [E, H, I]
    down_w,         # [E, I, H]
    top_k=2,
):
    b, t, h = x.shape
    n = b * t
    x_flat = x.reshape(n, h)

    logits = x_flat @ router_w
    topv, topi = torch.topk(logits, k=top_k, dim=-1)
    probs = torch.softmax(topv, dim=-1)

    out = torch.zeros_like(x_flat)
    for token in range(n):
        for j in range(top_k):
            e = topi[token, j].item()
            w = probs[token, j].item()
            y = deepseek_expert_ffn(
                x_flat[token:token + 1],
                gate_w[e],
                up_w[e],
                down_w[e],
            )
            out[token] += w * y[0]

    return out.reshape(b, t, h)