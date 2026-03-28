import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .dataset import SyntheticDatasetConfig, SyntheticCausalLMDataset, collate_causal_lm
from .distributed_utils import init_distributed, get_rank, get_world_size
from .transformer import DeepseekMoEConfig, DeepseekMoETransformer


@dataclass
class TrainConfig:
    batch_size: int = 4
    seq_len: int = 512
    total_tokens: int = 20_000_000
    lr: float = 3e-4
    num_steps: int = 50
    grad_accum_steps: int = 1
    log_every: int = 5


def train_main():
    init_distributed()

    rank = get_rank()
    world_size = get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(1234 + rank)
    torch.cuda.set_device(device)

    model_cfg = DeepseekMoEConfig(
        vocab_size=32768,
        max_seq_len=512,
        hidden_size=512,
        num_heads=8,
        num_layers=6,
        dense_intermediate_size=2048,
        num_experts=8,
        top_k=2,
        moe_intermediate_size=512,
        num_shared_experts=1,
    )

    train_cfg = TrainConfig()

    model = DeepseekMoETransformer(model_cfg).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

    dataset = SyntheticCausalLMDataset(
        SyntheticDatasetConfig(
            vocab_size=model_cfg.vocab_size,
            seq_len=train_cfg.seq_len,
            total_tokens=train_cfg.total_tokens,
            seed=1234 + rank,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        collate_fn=collate_causal_lm,
        num_workers=0,
        pin_memory=True,
    )

    model.train()

    total_tokens_local = 0
    start_time = time.time()

    for step, batch in enumerate(loader):
        if step >= train_cfg.num_steps:
            break

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out["loss"] / train_cfg.grad_accum_steps
        loss.backward()

        if (step + 1) % train_cfg.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_tokens_local += input_ids.numel()

        if (step + 1) % train_cfg.log_every == 0:
            elapsed = time.time() - start_time
            local_toks_per_sec = total_tokens_local / max(elapsed, 1e-6)

            toks_tensor = torch.tensor([local_toks_per_sec], device=device)
            dist.all_reduce(toks_tensor, op=dist.ReduceOp.SUM)
            global_toks_per_sec = toks_tensor.item()

            if rank == 0:
                print(
                    f"[train] step={step+1} "
                    f"loss={loss.item() * train_cfg.grad_accum_steps:.6f} "
                    f"global_tokens_per_sec={global_toks_per_sec:.2f}"
                )

    dist.barrier()
    if rank == 0:
        print("Training run finished.")