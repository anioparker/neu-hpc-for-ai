# llm_infere

A minimal LLM inference engine. Model: Qwen3-0.6B.

## Run tests

```bash
pytest tests/test_kv_cache.py -v          # local, no GPU
modal run modal_run.py::test              # all tests on GPU
```

## Build nano-sglang Core

**Part 1: KV Cache** — `kv_cache.py`
- Store and retrieve key/value tensors across layers

**Part 2: Engine** — `engine.py`
- wire prefill + decode loop, stop at EOS or max_tokens

**Part 3: Scheduler** — `scheduler.py`
- run all decodes in one GPU call
- manage request lifecycle: waiting → running → finished

**Part 4: Benchmark**
- Measure throughput (tokens/sec) vs. number of concurrent requests
- Compare batched scheduler vs. generating one request at a time

## Reference

- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [nano-vllm walkthrough](https://neutree.ai/blog/nano-vllm-part-1)
- [nano-sglang](https://github.com/lixiaohua-neu/nano-sglang)