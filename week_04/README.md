# 04_week FlashAttention-2 — Algorithm 1 (CPU C + CUDA)

This repo answers two assignment questions using **Algorithm 1 (FlashAttention-2 forward pass, Sec. 3.1)**:
1) **Unparallelized C (CPU)** implementation for correctness.  
2) **Parallelized CUDA (GPU)** implementation for correctness (no low-level optimization yet).  

Paper: https://arxiv.org/abs/2307.08691

---

## Q1) Unparallelized C: Algorithm 1 (CPU reference)

### File
- `flash_attention_alg1.c` — CPU reference that matches Algorithm 1 logic:
  - blocks over `i` (query blocks) and `j` (key/value blocks)
  - maintains online softmax state per row: `m` (running max), `l` (running sum)
  - outputs:
    - `O` (attention output)
    - `L` (row-wise logsumexp) where `L = m + log(l)`

### Build
From the folder containing `flash_attention_alg1.c` (your `04_week/`):
```bash
gcc -O2 -std=gnu11 -Wall -Wextra -I../include flash_attention_alg1.c -lm -o flash_attention_alg1

## Q2) Parallelized CUDA: Algorithm 1 (correctness-first)

### File (suggested)
`flash_attention_cuda.cu`

### Expected mapping (simple + correct)
- **One threadblock per `Qi` block** (one `i`)
- Inside the block, **loop over all `j` blocks** of `K,V`
- Keep `Oi`, `mi`, `li` in **registers/shared memory**
- Parallelize:
  - computing score tile `Sij = Qi * Kj^T`
  - `rowmax` reduction
  - `rowsum` reduction + `Ptilde`
  - `Oi` update

### Build
```bash
nvcc -O2 -std=c++17 -arch=sm_89 flash_attention_cuda.cu -o flash_attention_cuda
