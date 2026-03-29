import torch
import torch.nn.functional as F
from deepseek_moe_tk import DeepseekMoETwitterKittens


def reference_forward(x, router_w, w1, w2, top_k):
    T, H = x.shape
    E = w1.shape[0]

    logits = x.float() @ router_w.t().float()
    probs = F.softmax(logits, dim=-1)
    topk_w, topk_idx = torch.topk(probs, k=top_k, dim=-1)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    y = torch.zeros_like(x)
    for t in range(T):
        acc = torch.zeros(H, device=x.device, dtype=torch.float32)
        for k in range(top_k):
            e = topk_idx[t, k].item()
            gate = topk_w[t, k].item()
            hidden = F.silu(x[t].float() @ w1[e].float())
            out = hidden @ w2[e].float()
            acc += gate * out
        y[t] = acc.to(x.dtype)
    return y


@torch.no_grad()
def test_forward_close():
    if not torch.cuda.is_available():
        return

    torch.manual_seed(0)
    model = DeepseekMoETwitterKittens(
        hidden_size=128,
        intermediate_size=256,
        num_experts=4,
        top_k=2,
        dtype=torch.bfloat16,
        device="cuda",
    ).eval()

    x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)

    y = model(x)
    y_ref = reference_forward(
        x,
        model.router.weight,
        model.w1,
        model.w2,
        top_k=model.top_k,
    )

    max_abs = (y.float() - y_ref.float()).abs().max().item()
    print("max_abs:", max_abs)
    assert max_abs < 3e-1