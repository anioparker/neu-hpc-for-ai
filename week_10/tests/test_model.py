
from ..llm_infere.config import Config
from ..llm_infere.engine_init import InferenceEngine
from dataclasses import dataclass

@dataclass
config = Config(model_path="Qwen/Qwen3-0.6B")
engine = InferenceEngine(config)

# Add a prompt
prompt = "Hello, how are you?"
input_ids = engine.tokenizer.encode(prompt)
seq_id = engine.add_sequence(input_ids, SamplingParams(temperature=0.7))

# Run generation
for _ in range(100):
    engine.step()
    if seq_id not in engine.active_seqs:
        break

output = engine.tokenizer.decode(engine.active_seqs[seq_id].output_ids)
print(output)