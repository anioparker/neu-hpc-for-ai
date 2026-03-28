from dataclasses import dataclass
from typing import Dict, Iterator

import torch
from torch.utils.data import IterableDataset


@dataclass
class SyntheticDatasetConfig:
    vocab_size: int = 50257
    seq_len: int = 512
    total_tokens: int = 10_000_000
    seed: int = 1234


class SyntheticCausalLMDataset(IterableDataset):
    """
    Large synthetic dataset for stress-testing throughput on Modal.
    This is useful before plugging in a real corpus.
    """
    def __init__(self, config: SyntheticDatasetConfig):
        super().__init__()
        self.config = config

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        g = torch.Generator()
        g.manual_seed(self.config.seed)

        seq_len = self.config.seq_len
        total_steps = self.config.total_tokens // seq_len

        for _ in range(total_steps):
            input_ids = torch.randint(
                low=0,
                high=self.config.vocab_size,
                size=(seq_len,),
                generator=g,
                dtype=torch.long,
            )
            labels = input_ids.clone()
            yield {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": torch.ones(seq_len, dtype=torch.long),
            }


def collate_causal_lm(batch):
    input_ids = torch.stack([x["input_ids"] for x in batch], dim=0)
    labels = torch.stack([x["labels"] for x in batch], dim=0)
    attention_mask = torch.stack([x["attention_mask"] for x in batch], dim=0)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }