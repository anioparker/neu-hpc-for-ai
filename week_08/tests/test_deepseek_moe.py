import os
import torch
import torch.distributed as dist

from deepseek_moe.distributed_utils import init_distributed, get_rank, get_world_size
from deepseek_moe.module import DeepseekMoE
from deepseek_moe.reference import reference_moe_forward


def build_global_weights_from_local(module):
    """
    Gather local expert weights from all ranks to rank 0 for reference check.
    """
    rank = get_rank()
    world_size = get_world_size()

    gather_gate = [torch.empty_like(module.gate_w) for _ in range(world_size)] if rank == 0 else None
    gather_up = [torch.empty_like(module.up_w) for _ in range(world_size)] if rank == 0 else None
    gather_down = [torch.empty_like(module.down_w) for _ in range(world_size)] if rank == 0 else None

    dist.gather(module.gate_w, gather_gate, dst=0)
    dist.gather(module.up_w, gather_up, dst=0)
    dist.gather(module.down_w, gather_down, dst=0)

    if rank == 0:
        gate = torch.cat(gather_gate, dim=0)
        up = torch.cat(gather_up, dim=0)
        down = torch.cat(gather_down, dim=0)
        return gate, up, down
    return None, None, None


def main():
    init_distributed()
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    hidden = 16
    inter = 32
    num_experts = 4
    top_k = 2

    assert num_experts % world_size == 0

    moe = DeepseekMoE(hidden, inter, num_experts, top_k=top_k).to(device)
    moe.eval()

    # Make all routers identical across ranks
    dist.broadcast(moe.router.data, src=0)

    b_local = 2
    t = 3
    x_local = torch.randn(b_local, t, hidden, device=device)

    # Forward distributed
    y_local = moe(x_local)

    # Gather inputs to rank 0 for reference
    gather_x = [torch.empty_like(x_local) for _ in range(world_size)] if rank == 0 else None
    dist.gather(x_local, gather_x, dst=0)

    gate, up, down = build_global_weights_from_local(moe)

    if rank == 0:
        x_global = torch.cat(gather_x, dim=0)  # [B_global, T, H]
        y_ref = reference_moe_forward(
            x_global,
            moe.router.data,
            gate,
            up,
            down,
            top_k=top_k,
        )

        chunks = list(y_ref.chunk(world_size, dim=0))
    else:
        chunks = None

    y_ref_local = torch.empty_like(y_local)
    dist.scatter(y_ref_local, scatter_list=chunks, src=0)

    max_abs = (y_local - y_ref_local).abs().max().item()
    print(f"[rank {rank}] max_abs={max_abs:.6e}")

    torch.testing.assert_close(y_local, y_ref_local, atol=1e-4, rtol=1e-4)

    if rank == 0:
        print("PASS: distributed MoE matches reference")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()