import torch
import torch.nn as nn
import torch.distributed as dist
import deepseek_moe_cuda

from .distributed_utils import get_rank, get_world_size


class DeepseekMoE(nn.Module):
    """
    Data-parallel + expert-parallel MoE forward.

    Global expert layout:
        experts are sharded evenly across ranks.

    Communication:
        torch.distributed backend='nccl'
        all_to_all_single is used for token dispatch and return.
    """

    def __init__(self, hidden_size, intermediate_size, num_experts, top_k=2):
        super().__init__()
        assert num_experts > 0
        assert top_k >= 1
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k

        world_size = get_world_size()
        assert num_experts % world_size == 0, "num_experts must divide world_size"
        self.local_num_experts = num_experts // world_size

        # Router replicated on all ranks
        self.router = nn.Parameter(torch.randn(hidden_size, num_experts) * 0.02)

        # Local experts only
        self.gate_w = nn.Parameter(
            torch.randn(self.local_num_experts, hidden_size, intermediate_size) * 0.02
        )
        self.up_w = nn.Parameter(
            torch.randn(self.local_num_experts, hidden_size, intermediate_size) * 0.02
        )
        self.down_w = nn.Parameter(
            torch.randn(self.local_num_experts, intermediate_size, hidden_size) * 0.02
        )

    def forward(self, x):
        """
        x: [B, T, H] local token shard on each rank
        returns: [B, T, H]
        """
        assert x.is_cuda
        rank = get_rank()
        world_size = get_world_size()
        experts_per_rank = self.local_num_experts

        b, t, h = x.shape
        n_tokens = b * t
        x_flat = x.reshape(n_tokens, h).contiguous()

        # Router is replicated on every rank
        logits = x_flat @ self.router                        # [N, E]
        topv, topi = torch.topk(logits, k=self.top_k, dim=-1)
        topw = torch.softmax(topv, dim=-1)                  # [N, K]

        # Expand tokens by top-k
        k = self.top_k
        expanded = x_flat.repeat_interleave(k, dim=0).contiguous()          # [N*K, H]
        expanded_gid = topi.reshape(-1).contiguous()                        # [N*K]
        expanded_w = topw.reshape(-1).contiguous()                          # [N*K]
        expanded_tok = torch.arange(n_tokens, device=x.device).repeat_interleave(k)

        dst_rank = torch.div(expanded_gid, experts_per_rank, rounding_mode="floor")
        local_eid = (expanded_gid % experts_per_rank).contiguous()

        # Sort by destination rank for all_to_all_single
        perm = torch.argsort(dst_rank)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)

        send_x = expanded[perm].contiguous()
        send_e = local_eid[perm].contiguous()
        send_w = expanded_w[perm].contiguous()
        send_tok = expanded_tok[perm].contiguous()
        send_dst = dst_rank[perm].contiguous()

        send_counts = torch.bincount(send_dst, minlength=world_size).to(torch.int64)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)

        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()

        recv_total = int(recv_counts.sum().item())
        recv_x = torch.empty((recv_total, h), device=x.device, dtype=x.dtype)
        recv_e = torch.empty((recv_total,), device=x.device, dtype=torch.long)

        dist.all_to_all_single(recv_x, send_x, output_split_sizes=recv_splits, input_split_sizes=send_splits)
        dist.all_to_all_single(recv_e, send_e, output_split_sizes=recv_splits, input_split_sizes=send_splits)

        # Local expert compute on this rank's expert shard
        recv_y = deepseek_moe_cuda.local_expert_forward(
            recv_x.contiguous(),
            recv_e.contiguous(),
            self.gate_w.contiguous(),
            self.up_w.contiguous(),
            self.down_w.contiguous(),
        )

        # Return outputs back to source ranks
        ret_total = int(send_counts.sum().item())
        ret_y = torch.empty((ret_total, h), device=x.device, dtype=x.dtype)

        dist.all_to_all_single(
            ret_y,
            recv_y,
            output_split_sizes=send_splits,
            input_split_sizes=recv_splits,
        )

        # Undo the destination sort
        local_out_expanded = ret_y[inv_perm].contiguous()   # [N*K, H]
        local_w_expanded = send_w[inv_perm].contiguous()    # [N*K]
        local_tok_expanded = send_tok[inv_perm].contiguous()

        weighted = local_out_expanded * local_w_expanded.unsqueeze(-1)
        out = torch.zeros((n_tokens, h), device=x.device, dtype=x.dtype)
        out.index_add_(0, local_tok_expanded, weighted)

        return out.view(b, t, h)