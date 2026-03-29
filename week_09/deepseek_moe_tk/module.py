import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import moe_ffn_forward


class DeepseekMoETwitterKittens(nn.Module):
    """
    DeepSeek-style MoE FFN block:
      router(x) -> top-k experts
      expert_i: SiLU(x @ W1_i) @ W2_i
      weighted sum over top-k
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int = 2,
        bias: bool = False,
        dtype=torch.bfloat16,
        device="cuda",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(hidden_size, num_experts, bias=bias, device=device, dtype=dtype)

        # expert weights
        self.w1 = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size, device=device, dtype=dtype)
        )
        self.w2 = nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_size, device=device, dtype=dtype)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.router.weight)
        if self.router.bias is not None:
            nn.init.zeros_(self.router.bias)
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, H] or [T, H]
        """
        orig_shape = x.shape
        if x.dim() == 3:
            b, s, h = x.shape
            x2 = x.reshape(b * s, h)
        else:
            x2 = x

        # Router in PyTorch for simplicity/stability.
        logits = self.router(x2).float()                  # [T, E]
        probs = F.softmax(logits, dim=-1)                 # [T, E]
        topk_w, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)

        # Normalize top-k weights so they sum to 1 over selected experts.
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        y = moe_ffn_forward(
            x2.contiguous(),
            topk_idx.contiguous(),
            topk_w.contiguous(),
            self.w1.contiguous(),
            self.w2.contiguous(),
        )

        if x.dim() == 3:
            y = y.reshape(b, s, h)
        return y
