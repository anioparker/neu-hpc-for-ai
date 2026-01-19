# CUDA GEMM (No cuBLAS/cuDNN)

This repo implements a CUDA **GEMM** kernel (generalized matrix multiplication) without using high-level CUDA libraries.

## What it does
Computes **in-place**:
\[
C \leftarrow \alpha \cdot \mathrm{op}(A)\mathrm{op}(B) + \beta \cdot C
\]
where `op(A)` and `op(B)` are either the matrix itself or its transpose.

Supported forms:
- `C <- alpha * A  * B  + beta * C`
- `C <- alpha * A^T* B  + beta * C`
- `C <- alpha * A  * B^T+ beta * C`
- `C <- alpha * A^T* B^T+ beta * C`

## Files
- `gemm.cu` — baseline GEMM (produces `D = alpha*(A@B) + beta*C`).
- `gemm_inplace_transpose.cu` — extended GEMM: optional transpose of `A`/`B`, updates `C` in-place, includes a small CPU reference + correctness tests.

## Requirements
- NVIDIA GPU + CUDA toolkit (provides `nvcc`)

## Build & Run

### 1) Baseline GEMM
```bash
nvcc -O3 -std=c++17 gemm.cu -o gemm
./gemm
