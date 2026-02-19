# FlashAttention Algorithm 1 with CuTe

CuTe is a collection of C++ CUDA template abstractions for defining and operating on hierarchically multidimensional layouts of threads and data. This repo contains a CUDA implementation of **FlashAttention-2 forward pass algorithm** using **CuTe** for tensor/layout/tile views.

## Files
- `flashatten_fp.cu` — CUDA + CPU reference implementation (single head, FP32, row-major `[N, D]`).

## Requirements
- CUDA toolkit with `nvcc`
- CUTLASS repo available locally (CuTe headers live in `cutlass/include/cute/...`)

## Test
- sequence length N = 1, 7, 63, 64, 65, 127, 128, 129, 256;

## CuTe 

In the GPU kernel:

- `make_layout` / `make_tensor` create global-memory tensor views for `Q, K, V, O, L`
- `local_tile` creates block tiles `Qi`, `Oi` (so you can write `Qi(r,k)` instead of pointer math)
- `make_smem_ptr` + `make_tensor` create shared-memory tiles `tQ, tK, tV`

---

## Correctness Check

The program runs:

1. GPU FlashAttention Algorithm 1 kernel  
2. CPU reference Algorithm 1  
3. Reports max absolute / relative errors for `O` and `L` (log-sum-exp)


