import argparse
import csv
import re
import subprocess
from pathlib import Path
from datetime import datetime

KERNELS = [
    (0, "cublas_fp32"),
    (1, "k01_naive"),
    (2, "k02_transposeB_cached"),
    (3, "k03_smem_tile32"),
    (4, "k04_1d_blocktiling_fixed"),
    (5, "k05_2d_regtiling"),
    (6, "k06_vec_loads"),
    (9, "k09_autotune"),
    (10, "k10_warptiling_placeholder"),
]

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "raw"
OUTDIR.mkdir(parents=True, exist_ok=True)

def run_one(binpath, kid, m, n, k, reps):
    cmd = [binpath, "--kernel", str(kid),
           "--m", str(m), "--n", str(n), "--k", str(k),
           "--reps", str(reps)]
    out = subprocess.check_output(cmd, text=True)

    def grab(pat, default=None, cast=float):
        mo = re.search(pat, out)
        return cast(mo.group(1)) if mo else default

    cublas_ms = grab(r"cuBLAS FP32 avg time:\s*([0-9.]+)", None)
    cublas_gflops = grab(r"cuBLAS throughput:\s*([0-9.]+)", None)
    k_ms = grab(r"Kernel time \(avg\):\s*([0-9.]+)", None)
    k_gflops = grab(r"Throughput:\s*([0-9.]+)", None)
    rel = grab(r"Relative to cuBLAS:\s*([0-9.]+)", None)
    err = grab(r"Max abs error vs cuBLAS:\s*([0-9.eE+-]+)", None)

    ms = cublas_ms if kid == 0 else k_ms
    gflops = cublas_gflops if kid == 0 else k_gflops
    rel_to = 1.0 if kid == 0 else rel
    max_err = 0.0 if kid == 0 else (err if err is not None else -1.0)

    return {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "kernel_id": kid,
        "name": dict(KERNELS)[kid],
        "m": m, "n": n, "k": k,
        "reps": reps,
        "ms": ms,
        "gflops": gflops,
        "rel_to_cublas": rel_to,
        "max_abs_err": max_err,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()

    rows = []
    for kid, name in KERNELS:
        print(f"=== running {kid} {name} ===")
        rows.append(run_one(args.bin, kid, args.m, args.n, args.k, args.reps))

    out_csv = OUTDIR / f"h100_{args.m}.csv"
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nWrote: {out_csv}\n")
    for r in rows:
        print(
            f'{r["kernel_id"]:>2} {r["name"]:<28} '
            f'ms={r["ms"]:.6f}  gflops={r["gflops"]:.2f}  '
            f'rel={r["rel_to_cublas"]:.4f}  err={r["max_abs_err"]:.3e}'
        )

if __name__ == "__main__":
    main()
