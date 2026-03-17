# Implement DeepSeekV3 MoE Operator 

This project implements the **Mixture-of-Experts (MoE) operator** used in **DeepSeekV3** using **pure C**, with:

- **no parallelism**
- **no CUDA**
- **no OpenMP**
- **no BLAS**
- **no external ML frameworks**

The goal is to build a minimal, readable, and fully CPU-based reference implementation of the MoE forward pass.

---

## Objective

Implement the MoE operator for DeepSeekV3 in plain C as a correctness-first baseline.

This project focuses on:

- understanding the MoE execution flow
- reproducing the routing and expert computation logic
- keeping the code simple and debuggable
- avoiding hardware-specific optimizations

This is **not** a high-performance implementation. It is a **reference implementation** for learning, testing, and verification.

---

## What the MoE Operator Does

At a high level, the DeepSeekV3 MoE layer works like this:

1. Take an input tensor of token hidden states
2. Use a **gating/router** mechanism to score experts for each token
3. Select the **top-k experts** per token
4. Send each token to its selected experts
5. Run each selected expert MLP
6. Weight each expert output by its routing weight
7. Sum the weighted outputs back into the final token representation

In formula form, for token hidden state `x`:

```text
router_logits = x * W_gate
topk_experts, topk_weights = TopK(Softmax(router_logits), k)

output = sum_{i in topk_experts} topk_weights[i] * Expert_i(x)

## Project structure

```text
deepseek_v3_moe_c/
├── README.md
├── Makefile
├── include/
│   ├── tensor.h
│   ├── moe.h
│   ├── router.h
│   ├── expert.h
│   └── utils.h
├── src/
│   ├── main.c
│   ├── tensor.c
│   ├── moe.c
│   ├── router.c
│   ├── expert.c
│   └── utils.c
├── tests/
│   ├── test_router.c
│   ├── test_topk.c
│   ├── test_expert.c
│   └── test_moe.c
└── data/
    └── sample_weights.txt
```

## Build

Build the demo binary:

```bash
make
```

Run it:

```bash
./deepseek_v3_moe
```

Build and run tests:

```bash
make test
./test_router
./test_topk
./test_expert
./test_moe
```

## Notes

This is a simple reference implementation, not an optimized production kernel.

- routing is done token by token
- experts are executed serially
- tensors are raw contiguous `float*` buffers
- all math uses straightforward loops

## Expert definition

Each expert uses a gated MLP:

```text
u = x @ up_proj
v = x @ gate_proj
a = silu(v)
z = u * a
y = z @ down_proj
```

## MoE forward

For each token:

1. compute router logits
2. softmax over experts
3. select top-k experts
4. run each selected expert
5. combine outputs with routing weights

## Sample usage

The demo in `src/main.c` constructs a tiny MoE with:

- 2 tokens
- hidden dimension 4
- 3 experts
- top-k = 2
- intermediate dimension 6

and prints the resulting output tensor.
