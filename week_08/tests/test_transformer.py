import torch

from deepseek_moe.transformer import DeepseekMoEConfig, DeepseekMoETransformer


def test_transformer_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = DeepseekMoEConfig(
        vocab_size=1000,
        max_seq_len=32,
        hidden_size=64,
        num_heads=4,
        num_layers=2,
        dense_intermediate_size=128,
        num_experts=2,
        top_k=1,
        moe_intermediate_size=32,
        num_shared_experts=1,
    )
    model = DeepseekMoETransformer(cfg).to(device)

    input_ids = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
    labels = input_ids.clone()

    out = model(input_ids=input_ids, labels=labels)
    assert "loss" in out
    assert "logits" in out
    assert out["logits"].shape == (2, 16, cfg.vocab_size)