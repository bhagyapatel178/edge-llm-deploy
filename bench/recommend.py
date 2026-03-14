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
    
    # score: maximize tg, minimize ttft + rss
    # Still need to tune weights.
    df["score"] = (
        1.0 * df["tg_tok_s"]
        - 0.3 * df["ttft_proxy_s"]
        - 0.05 * df["max_rss_mb"]
    )
    best = df.sort_values("score", ascending=False).iloc[0].to_dict()


    rec = {
        "device": device,
        "workload": {"n_prompt": args.n_prompt, "n_gen": args.n_gen},
        "recommended": {
            "model_filename": best["model_filename"],
            "quant": best["quant"],
            "n_threads": int(best["n_threads"]),
            "n_gpu_layers": int(best["n_gpu_layers"]) if not pd.isna(best["n_gpu_layers"]) else None,
            "devices": best["devices"],
        },
        "expected": {
            "pp_tok_s": float(best["pp_tok_s"]),
            "tg_tok_s": float(best["tg_tok_s"]),
            "ttft_proxy_s": float(best["ttft_proxy_s"]),
            "max_rss_mb": float(best["max_rss_mb"]),
        },
        "note": "TTFT proxy computed as (n_prompt/pp_tok_s) + (1/tg_tok_s). Peak RSS from /usr/bin/time -l.",
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)

    print("Recommended configuration:")
    print(json.dumps(rec, indent=2))

if __name__ == "__main__":
    main()