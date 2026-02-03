# Optional plotting script (kept minimal).
# You can extend this to matplotlib charts if you want.
import pandas as pd
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    print(df[["kernel_id","name","ms","gflops","rel_to_cublas"]].sort_values("kernel_id").to_string(index=False))

if __name__ == "__main__":
    main()
