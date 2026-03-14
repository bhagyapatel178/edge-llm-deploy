import argparse
import json
import os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/runs.csv")
    ap.add_argument("--device", default="auto", choices=["auto", "MTL0", "BLAS"])
    ap.add_argument("--n-prompt", type=int, default=512)
    ap.add_argument("--n-gen", type=int, default=128)
    ap.add_argument("--max-rss-mb", type=float, default=None)
    ap.add_argument("--max-ttft-s", type=float, default=None)
    ap.add_argument("--min-tg-tok-s", type=float, default=None)
    ap.add_argument("--out", default="results/recommendation.json")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # numeric cleanup
    
    for c in ["n_prompt", "n_gen", "pp_tok_s", "tg_tok_s", "ttft_proxy_s", "max_rss_mb", "n_gpu_layers"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

if __name__ == "__main__":
    main()