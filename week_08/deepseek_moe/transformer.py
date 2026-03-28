import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import DeepseekMoE


@dataclass
class DeepseekMoEConfig:
    vocab_size: int = 50257
    max_seq_len: int = 1024
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 6
    dense_intermediate_size: int = 2048

    # DeepSeekMoE-specific
    num_experts: int = 8
    top_k: int = 2
    moe_intermediate_size: int = 512
    num_shared_experts: int = 1

    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.weight


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, h = x.shape
        nh, hd = self.num_heads, self.head_dim

        q = self.q_proj(x).view(b, t, nh, hd).transpose(1, 2)  # [B, nh, T, hd]
        k = self.k_proj(x).view(b, t, nh, hd).transpose(1, 2)
        v = self.v_proj(x).view(b, t, nh, hd).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)  # [B, nh, T, T]

        causal = torch.triu(
            torch.ones(t, t, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal, float("-inf"))

        if attn_mask is not None:
            # attn_mask expected [B, T], 1=keep, 0=mask
            expanded = attn_mask[:, None, None, :].to(torch.bool)
            scores = scores.masked_fill(~expanded, float("-inf"))

        probs = F.softmax(scores, dim=-1)
        probs = F.dropout(probs, p=self.dropout, training=self.training)

        y = torch.matmul(probs, v)  # [B, nh, T, hd]
        y = y.transpose(1, 2).contiguous().view(b, t, h)
        y = self.o_proj(y)
        return y


class DenseFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate_proj(x)
        u = self.up_proj(x)
        z = F.silu(g) * u
        return self.down_proj(z)


class SharedExperts(nn.Module):
    """
    Shared experts are always enabled for every token.
    We implement them as a sum of FFNs, matching the DeepSeekMoE idea
    of always-on shared experts.
    """
    def __init__(self, hidden_size: int, intermediate_size: int, num_shared_experts: int):
        super().__init__()
        self.experts = nn.ModuleList(
            [DenseFFN(hidden_size, intermediate_size) for _ in range(num_shared_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.experts) == 0:
            return torch.zeros_like(x)
        out = 0.0
        for expert in self.experts:
            out = out + expert(x)
        return out


class DeepseekMoEBlock(nn.Module):
    """
    DeepSeekMoE block:
      output = shared_experts(x) + routed_moe(x)

    Fine-grained expert segmentation is represented by using many small experts
    with smaller moe_intermediate_size rather than fewer large dense FFNs.
    """
    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        top_k: int,
        num_shared_experts: int = 1,
    ):
        super().__init__()
        self.shared = SharedExperts(
            hidden_size=hidden_size,
            intermediate_size=moe_intermediate_size,
            num_shared_experts=num_shared_experts,
        )
        self.routed = DeepseekMoE(
            hidden_size=hidden_size,
            intermediate_size=moe_intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared(x) + self.routed(x)


class DeepseekMoETransformerLayer(nn.Module):
    def __init__(self, config: DeepseekMoEConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.ffn_norm = RMSNorm(config.hidden_size)

        self.attn = MultiHeadSelfAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

        self.moe = DeepseekMoEBlock(
            hidden_size=config.hidden_size,
            moe_intermediate_size=config.moe_intermediate_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
            num_shared_experts=config.num_shared_experts,
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attn_mask=attn_mask)
        x = x + self.moe(self.ffn_norm(x))
        return x


class DeepseekMoETransformer(nn.Module):
    def __init__(self, config: DeepseekMoEConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DeepseekMoETransformerLayer(config) for _ in range(config.num_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(x, attn_mask=attention_mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if labels is None:
            return {"logits": logits}

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return {"loss": loss, "logits": logits}