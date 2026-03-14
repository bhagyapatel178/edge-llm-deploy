import argparse
import json
import os
import pandas as pd


def pick_device(df: pd.DataFrame, device: str) -> str:
    # primarily mac focused at the moment
    if device != "auto":
        return device
    # prefer GPU device if present
    if "MTL0" in set(df["devices"]):
        return "MTL0"
    return "BLAS"

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

    # choose device and workload
    device = pick_device(df, args.device)
    df = df[(df["devices"] == device) & (df["n_prompt"] == args.n_prompt) & (df["n_gen"] == args.n_gen)].copy()

    if df.empty:
        raise SystemExit(f"No runs found for device={device}, n_prompt={args.n_prompt}, n_gen={args.n_gen}")
    
    # apply constraints (only if provided)
    if args.max_rss_mb is not None:
        df = df[df["max_rss_mb"] <= args.max_rss_mb]
    if args.max_ttft_s is not None:
        df = df[df["ttft_proxy_s"] <= args.max_ttft_s]
    if args.min_tg_tok_s is not None:
        df = df[df["tg_tok_s"] >= args.min_tg_tok_s]

    if df.empty:
        raise SystemExit("No configurations satisfy the given constraints.")


if __name__ == "__main__":
    main()