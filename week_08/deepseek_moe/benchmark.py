import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .dataset import SyntheticCausalLMDataset, SyntheticDatasetConfig, collate_causal_lm
from .distributed_utils import init_distributed, get_rank, get_world_size
from .module import DeepseekMoE
from .transformer import DeepseekMoEConfig, DeepseekMoETransformer


@dataclass
class BenchConfig:
    batch_size: int = 8
    seq_len: int = 512
    total_tokens: int = 10_000_000
    warmup_steps: int = 10
    measure_steps: int = 50
    hidden_size: int = 512
    intermediate_size: int = 512
    num_experts: int = 8
    top_k: int = 2


@torch.no_grad()
def bench_operator_only(
    moe: DeepseekMoE,
    loader: DataLoader,
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
):
    moe.eval()
    times = []
    total_tokens = 0
    total_batches = 0

    it = iter(loader)

    for step in range(warmup_steps + measure_steps):
        batch = next(it)
        input_ids = batch["input_ids"].to(device, non_blocking=True)

        # synthetic hidden states matching transformer hidden size
        x = torch.randn(
            input_ids.size(0),
            input_ids.size(1),
            moe.hidden_size,
            device=device,
            dtype=torch.float32,
        )

        torch.cuda.synchronize()
        t0 = time.time()
        _ = moe(x)
        torch.cuda.synchronize()
        dt = time.time() - t0

        if step >= warmup_steps:
            times.append(dt)
            total_tokens += x.size(0) * x.size(1)
            total_batches += x.size(0)

    avg_time = sum(times) / len(times)
    return {
        "avg_step_time_s": avg_time,
        "tokens_per_sec": total_tokens / sum(times),
        "samples_per_sec": total_batches / sum(times),
    }


@torch.no_grad()
def bench_transformer(
    model: DeepseekMoETransformer,
    loader: DataLoader,
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
):
    model.eval()
    times = []
    total_tokens = 0
    total_batches = 0

    it = iter(loader)

    for step in range(warmup_steps + measure_steps):
        batch = next(it)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        torch.cuda.synchronize()
        t0 = time.time()
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
        torch.cuda.synchronize()
        dt = time.time() - t0

        if step >= warmup_steps:
            times.append(dt)
            total_tokens += input_ids.numel()
            total_batches += input_ids.size(0)

    avg_time = sum(times) / len(times)
    return {
        "avg_step_time_s": avg_time,
        "tokens_per_sec": total_tokens / sum(times),
        "samples_per_sec": total_batches / sum(times),
    }


def reduce_metric_dict(metrics: dict, device: torch.device):
    out = {}
    world_size = get_world_size()
    for k, v in metrics.items():
        t = torch.tensor([float(v)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        if "time" in k:
            out[k] = t.item() / world_size
        else:
            out[k] = t.item()
    return out


def main():
    init_distributed()
    rank = get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(1234 + rank)
    torch.cuda.set_device(device)

    cfg = BenchConfig()

    dataset = SyntheticCausalLMDataset(
        SyntheticDatasetConfig(
            vocab_size=32768,
            seq_len=cfg.seq_len,
            total_tokens=max(
                cfg.total_tokens,
                (cfg.warmup_steps + cfg.measure_steps + 8) * cfg.batch_size * cfg.seq_len,
            ),
            seed=1234 + rank,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate_causal_lm,
        num_workers=0,
        pin_memory=True,
    )

    moe = DeepseekMoE(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
    ).to(device)

    transformer_cfg = DeepseekMoEConfig(
        vocab_size=32768,
        max_seq_len=cfg.seq_len,
        hidden_size=cfg.hidden_size,
        num_heads=8,
        num_layers=6,
        dense_intermediate_size=2048,
        num_experts=cfg.num_experts,
        moe_intermediate_size=cfg.intermediate_size,
        top_k=cfg.top_k,
        num_shared_experts=1,
        dropout=0.0,
    )
    transformer = DeepseekMoETransformer(transformer_cfg).to(device)

    op_metrics = bench_operator_only(
        moe=moe,
        loader=loader,
        device=device,
        warmup_steps=cfg.warmup_steps,
        measure_steps=cfg.measure_steps,
    )
    tr_metrics = bench_transformer(
        model=transformer,
        loader=loader,
        device=device,
        warmup_steps=cfg.warmup_steps,
        measure_steps=cfg.measure_steps,
    )

    op_metrics = reduce_metric_dict(op_metrics, device)
    tr_metrics = reduce_metric_dict(tr_metrics, device)

    if rank == 0:
        ratio = tr_metrics["tokens_per_sec"] / max(op_metrics["tokens_per_sec"], 1e-9)
        print("=== Synthetic benchmark: Operator vs Transformer ===")
        print(f"world_size: {get_world_size()}")
        print(f"batch_size_per_rank: {cfg.batch_size}")
        print(f"seq_len: {cfg.seq_len}")
        print(f"hidden_size: {cfg.hidden_size}")
        print(f"num_experts: {cfg.num_experts}")
        print(f"top_k: {cfg.top_k}")
        print()
        print("[operator_only]")
        print(f"avg_step_time_s: {op_metrics['avg_step_time_s']:.6f}")
        print(f"tokens_per_sec:  {op_metrics['tokens_per_sec']:.2f}")
        print(f"samples_per_sec: {op_metrics['samples_per_sec']:.2f}")
        print()
        print("[transformer]")
        print(f"avg_step_time_s: {tr_metrics['avg_step_time_s']:.6f}")
        print(f"tokens_per_sec:  {tr_metrics['tokens_per_sec']:.2f}")
        print(f"samples_per_sec: {tr_metrics['samples_per_sec']:.2f}")
        print()
        print("[comparison]")
        print(f"transformer_over_operator_tokens_ratio: {ratio:.4f}")


if __name__ == "__main__":
    main()
