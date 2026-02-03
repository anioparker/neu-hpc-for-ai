#reproduce the worklog’s calculation section for a GEMM of size 4092×4092×4092.
import argparse
from math import prod

def flops_gemm(m, n, k):
    return 2*m*n*k + m*n

def bytes_min_gemm(m, n, k):
    # Read A (m*k), read B (k*n), read C (m*n), write C (m*n), FP32 => 4B
    elems = (m*k + k*n + m*n + m*n)
    return elems * 4

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    args = ap.parse_args()

    m, n, k = args.m, args.n, args.k
    fl = flops_gemm(m, n, k)
    by = bytes_min_gemm(m, n, k)

    print(f"m={m} n={n} k={k}")
    print(f"FLOPs: {fl} ({fl/1e9:.3f} GFLOPs)")
    print(f"Min bytes: {by} ({by/1e6:.3f} MB, {by/1024/1024:.3f} MiB)")
    print(f"Arithmetic intensity (upper bound): {fl/by:.3f} FLOPs/byte")

if __name__ == "__main__":
    main()
