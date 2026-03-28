# import os
# import time
# from dataclasses import dataclass

# import torch
# import torch.distributed as dist
# import torch.nn as nn

# from .distributed_utils import init_distributed, get_rank
# from .transformer import DeepseekMoEConfig, DeepseekMoETransformer, DenseFFN, RMSNorm, MultiHeadSelfAttention


# class DenseTransformerLayer(nn.Module):
#     def __init__(self, config: DeepseekMoEConfig):
#         super().__init__()
#         self.attn_norm = RMSNorm(config.hidden_size)
#         self.ffn_norm = RMSNorm(config.hidden_size)
#         self.attn = MultiHeadSelfAttention(config.hidden_size, config.num_heads, config.dropout)
#         self.ffn = DenseFFN(config.hidden_size, config.dense_intermediate_size)

#     def forward(self, x, attn_mask=None):
#         x = x + self.attn(self.attn_norm(x), attn_mask=attn_mask)
#         x = x + self.ffn(self.ffn_norm(x))
#         return x


# class DenseTransformer(nn.Module):
#     def __init__(self, config: DeepseekMoEConfig):
#         super().__init__()
#         self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
#         self.layers = nn.ModuleList([DenseTransformerLayer(config) for _ in range(config.num_layers)])
#         self.norm = RMSNorm(config.hidden_size)
#         self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#     def forward(self, input_ids):
#         x = self.embed(input_ids)
#         for layer in self.layers:
#             x = layer(x)
#         x = self.norm(x)
#         return self.head(x)


# @dataclass
# class BenchmarkConfig:
#     batch_size: int = 8
#     seq_len: int = 512
#     warmup_steps: int = 10
#     measure_steps: int = 30


# @torch.no_grad()
# def benchmark_model(model: nn.Module, input_ids: torch.Tensor, warmup_steps: int, measure_steps: int):
#     for _ in range(warmup_steps):
#         _ = model(input_ids)
#     torch.cuda.synchronize()

#     start = time.time()
#     for _ in range(measure_steps):
#         _ = model(input_ids)
#     torch.cuda.synchronize()
#     elapsed = time.time() - start

#     toks = input_ids.numel() * measure_steps
#     return toks / elapsed


# def benchmark_main():
#     init_distributed()

#     rank = get_rank()
#     local_rank = int(os.environ["LOCAL_RANK"])
#     device = torch.device(f"cuda:{local_rank}")

#     cfg = DeepseekMoEConfig(
#         vocab_size=32768,
#         max_seq_len=512,
#         hidden_size=512,
#         num_heads=8,
#         num_layers=6,
#         dense_intermediate_size=2048,
#         num_experts=8,
#         top_k=2,
#         moe_intermediate_size=512,
#         num_shared_experts=1,
#         dropout=0.0,
#     )
#     bench = BenchmarkConfig()

#     input_ids = torch.randint(
#         0,
#         cfg.vocab_size,
#         (bench.batch_size, bench.seq_len),
#         device=device,
#         dtype=torch.long,
#     )

#     dense = DenseTransformer(cfg).to(device).eval()
#     moe = DeepseekMoETransformer(cfg).to(device).eval()

#     dense_tps = benchmark_model(dense, input_ids, bench.warmup_steps, bench.measure_steps)
#     moe_tps = benchmark_model(moe, input_ids, bench.warmup_steps, bench.measure_steps)

#     dense_tensor = torch.tensor([dense_tps], device=device)
#     moe_tensor = torch.tensor([moe_tps], device=device)

#     dist.all_reduce(dense_tensor, op=dist.ReduceOp.SUM)
#     dist.all_reduce(moe_tensor, op=dist.ReduceOp.SUM)

#     if rank == 0:
#         dense_global = dense_tensor.item()
#         moe_global = moe_tensor.item()
#         speedup = moe_global / max(dense_global, 1e-6)
#         print(f"[benchmark] dense_tokens_per_sec={dense_global:.2f}")
#         print(f"[benchmark] moe_tokens_per_sec={moe_global:.2f}")
#         print(f"[benchmark] moe_vs_dense_speedup={speedup:.4f}")