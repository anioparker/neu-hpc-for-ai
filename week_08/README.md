# DeepSeekMoE Multi-GPU CUDA + NCCL on Modal

This repository implements a **multi-GPU DeepSeek-style Mixture-of-Experts (MoE) forward operator** using:

- **CUDA** for local expert FFN compute
- **NCCL** via `torch.distributed` for token dispatch and return
- **expert parallelism** by sharding experts across GPUs
- **data parallelism** by sharding tokens across GPUs
- **Modal** for single-node multi-GPU execution


## 1. Overview

A DeepSeek-style MoE layer replaces the dense FFN in a transformer block with a set of experts and a router.

For each token:

1. the **router** computes expert scores
2. the **top-k experts** are selected
3. the token is sent to those experts
4. expert outputs are computed
5. outputs are weighted by router probabilities
6. results are summed back to produce the final token output

This repo focuses on the **distributed MoE operator** itself.

---

## 2. What is implemented

### Implemented

- replicated router on every GPU
- expert sharding across GPUs
- token dispatch with `all_to_all_single`
- local CUDA expert FFN forward
- return path with `all_to_all_single`
- weighted combine for top-k routing
- reference PyTorch implementation for correctness testing
- Modal app for multi-GPU execution

### Not yet implemented

- backward kernel
- autograd support for CUDA extension
- grouped GEMM / fused kernels
- BF16 / FP16 kernels
- capacity factor and token dropping
- auxiliary load balancing loss
- tensor parallelism inside experts
- training script

This is a **forward-pass distributed operator baseline**.

---

## 3. Repository structure

```text
deepseek_moe_modal/
├── modal_app.py
├── setup.py
├── requirements.txt
├── scripts/
│   ├── build_extension.sh
│   ├── run_local.sh
│   └── run_modal.sh
├── deepseek_moe/
│   ├── __init__.py
│   ├── module.py
│   ├── reference.py
│   └── distributed_utils.py
├── csrc/
│   ├── include/
│   │   ├── deepseek_moe.h
│   │   └── cuda_utils.h
│   ├── bindings.cpp
│   └── deepseek_moe_kernel.cu
└── tests/
    └── test_deepseek_moe.py
```

## 4. Core idea

Let:

- `H` = hidden size
- `I` = intermediate size
- `E` = total number of experts
- `K` = top-k experts selected per token
- `P` = number of GPUs

Each rank owns:

- all router weights
- only `E / P` local experts

Each rank starts with its own shard of tokens.

### Forward flow

1. flatten local tokens
2. compute router logits
3. take top-k experts
4. expand tokens into `K` routed copies
5. map each global expert id to destination rank
6. all-to-all tokens to expert owners
7. run local expert CUDA forward
8. all-to-all outputs back
9. combine weighted outputs for each original token

## 5. DeepSeek-style expert formula

For a token input `x`:

```text
gate = x @ W_gate
up   = x @ W_up
z    = SiLU(gate) * up
y    = z @ W_down


So each expert computes:
```bash
Expert(x) = down( SiLU(gate(x)) * up(x) )
```

The final MoE output is:
```bash
MoE(x) = sum_{k in topK} p_k * Expert_k(x)
```

where p_k are the normalized top-k router weights.

## 6. Run

```bash
modal run modal_app.py
```

## 7. Multi-GPU CUDA + NCCL DeepseekMoE Operator

### Strengths

- Scales routed FFN compute across multiple GPUs
- Activates only top-k experts per token
- Uses lower active compute than a dense FFN at a similar total parameter count
- Expert parallelism helps significantly when the number of experts is large

### Weaknesses

- Incurs **all-to-all communication cost**
- Has **token permutation / unpermutation overhead**
- **Router imbalance** can reduce GPU utilization
- A naive CUDA kernel is still **not fully optimized**

---

## 8. Full DeepseekMoE Transformer

### Strengths

- Provides much larger total model capacity for similar per-token active FLOPs
- Scales better in large-dataset, large-model-capacity regimes
- Shared experts can help stabilize common patterns
- Fine-grained expert segmentation increases routing flexibility

### Weaknesses

- Attention remains **dense**
- MoE benefits are largest when the **FFN dominates compute**
- Distributed efficiency depends heavily on:
  - routing locality
  - all-to-all communication cost

---

## 9. Practical Benchmark Setup

The most useful benchmark setup is:

- **Dense Transformer baseline**
- **DeepseekMoE Transformer using your distributed operator**

### Compare the following metrics

- **tokens/sec**
- **max memory**
- **step time**
- **scaling from 1 GPU → 2 GPU → 4 GPU**

---

## 10. Benchmark

`deepseek_moe/benchmark.py` compares two distributed 4-GPU workloads:

- `operator_only`: the standalone `DeepseekMoE` CUDA operator with NCCL token exchange
- `transformer`: the full `DeepseekMoETransformer`, which includes attention, embeddings, and MoE blocks

It uses a large synthetic token dataset, runs warmup and timed steps, and reports:

- `avg_step_time_s`
- `tokens_per_sec`
- `samples_per_sec`
- `transformer_over_operator_tokens_ratio`

Run it on Modal with:

```bash
modal run modal_app.py::main --mode benchmark
```


## Summary

In practice, you should expect the distributed DeepseekMoE operator to outperform a dense FFN-style baseline when:

- expert count is sufficiently large,
- routing is reasonably balanced,
- and communication overhead is controlled.

However, end-to-end gains in a full transformer depend not only on expert execution, but also on:

- dense attention cost,
- routing quality,
- communication efficiency,
- and overall kernel optimization.