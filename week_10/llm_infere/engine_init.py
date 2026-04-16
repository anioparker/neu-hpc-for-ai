import torch
from typing import List, Dict
from .model import Model, Tokenizer
from .block_manager import BlockManager
from .sequence import Sequence, SequenceStatus
from .sampling import SamplingParams, sample_token
from .config import Config

class InferenceEngine:
    def __init__(self, config: Config):
        self.config = config
        self.model = Model(config.model_path)
        self.tokenizer = Tokenizer(config.model_path)
        self.block_manager = BlockManager(
            num_blocks=config.max_seq_len // config.block_size,
            block_size=config.block_size,
            num_layers=self.model.num_layers,
            num_heads=self.model.num_heads,
            head_dim=self.model.head_dim,
            device=config.device,
            dtype=getattr(torch, config.dtype),
        )
        self.active_seqs: Dict[int, Sequence] = {}  # seq_id → Sequence

    def add_sequence(self, input_ids: List[int], sampling_params: SamplingParams) -> int:
        """Add a new sequence for generation."""
        seq_id = len(self.active_seqs)
        seq = Sequence(seq_id, input_ids, sampling_params)
        self.active_seqs[seq_id] = seq
        self.block_manager.allocate(seq_id, len(input_ids))
        return seq_id