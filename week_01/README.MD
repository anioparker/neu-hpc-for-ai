# Week_01 — Matrix Multiplication in C (Single-thread + pthreads)

This README documents how I completed each requirement in the assignment and where to find/run the code.

---

## 1) Modal setup

### ✅ Create a modal.com account using GitHub
- Action: Signed up / logged in to **modal.com** using the **“Continue with GitHub”** option.

### ✅ Join the course workspace
- Action: Opened the instructor-provided invite link and joined the **course workspace** in Modal.

### ✅ Read docs + run examples
I followed and ran the examples from these official docs pages:
- Hello World: https://modal.com/docs/examples/hello_world  
- Images: https://modal.com/docs/guide/images  
- GPU: https://modal.com/docs/guide/gpu  
- CUDA: https://modal.com/docs/guide/cuda  
- Resources: https://modal.com/docs/guide/resources  

### ✅ Avoid unnecessary runs / monitor usage (important: limited shared funds)
Best practices I used to avoid extra charges:
- **Prefer local execution** when debugging:
  - Use `f.local(...)` instead of `f.remote(...)` until the code is correct.
- **Avoid repeated remote runs** while iterating:
  - Debug logic locally first, then do a single remote run to verify.
- **Check the Modal dashboard** (Apps / Runs / Logs / Usage) after running jobs:
  - Confirm that no old jobs are still running.
- **Be careful with `.remote()` loops**:
  - A loop that calls `.remote()` many times can quickly create many paid tasks.
- **Stop/interrupt runs** if you started something accidentally:
  - Cancel from the Modal UI (or stop the process that launched it).
- **Use smaller inputs** for correctness checks before scaling up.

---

## 2) Implement matrix multiplication (single-threaded)

### ✅ Single-threaded C implementation
File: **`matmul_single_thread.c`**  
What it contains:
- A single-threaded matrix multiplication function (row-major):
  - `matmul_ijk(A, B, C, N, M, K)`
- A `main()` test harness that checks **many matrix dimension combinations**, including corner cases.

Build:
```bash
gcc -std=c11 -O3 -march=native -Wall -Wextra matmul_single_thread.c -o matmul_st -lm