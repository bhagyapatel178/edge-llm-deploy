import argparse
import json
import os
import pandas as pd

QUANT_RANK = {"q8_0": 3, "q6_k": 2, "q4_k_m": 1, "q3_k_m": 0}

def pick_device(df: pd.DataFrame, device: str) -> str:
    if device != "auto":
        return device
    # Prefer Metal if present, else CPU
    return "MTL0" if "MTL0" in set(df["devices"]) else "BLAS"

def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # numeric cleanup
    for c in ["n_prompt", "n_gen", "pp_tok_s", "tg_tok_s", "ttft_proxy_s", "max_rss_mb", "n_gpu_layers", "n_threads"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["quant_rank"] = df["quant"].map(QUANT_RANK).fillna(-1).astype(int)
    df["rss_gb"] = df["max_rss_mb"] / 1024.0

    return df


def apply_constraints(df: pd.DataFrame,
                      max_rss_mb: float | None,
                      min_headroom_mb: float | None,
                      max_ttft_s: float | None,
                      min_tg_tok_s: float | None) -> pd.DataFrame:    
    # RAM constraint (with optional headroom)
    if max_rss_mb is not None:
        if min_headroom_mb is not None:
            df = df[df["max_rss_mb"] <= (max_rss_mb - min_headroom_mb)]
        else:
            df = df[df["max_rss_mb"] <= max_rss_mb]

    # Latency / throughput constraints
    if max_ttft_s is not None:
        df = df[df["ttft_proxy_s"] <= max_ttft_s]
    if min_tg_tok_s is not None:
        df = df[df["tg_tok_s"] >= min_tg_tok_s]

    return df

def print_top(df: pd.DataFrame, objective: str, k: int = 5):
    cols = ["quant", "quant_rank", "devices", "n_prompt", "n_gen", "tg_tok_s", "ttft_proxy_s", "max_rss_mb", "pp_tok_s", "n_gpu_layers"]
    view = df[cols].copy()

    if objective == "balanced":
        view["score"] = 1.0 * view["tg_tok_s"] - 0.3 * view["ttft_proxy_s"] - 1.0 * (view["max_rss_mb"] / 1024.0)
        view = view.sort_values("score", ascending=False)
    elif objective == "speed_first":
        view = view.sort_values(["tg_tok_s", "ttft_proxy_s", "quant_rank"], ascending=[False, True, False])
    elif objective == "max_context":
        view = view.sort_values(["n_prompt", "quant_rank", "tg_tok_s"], ascending=[False, False, False])
    else:  # quality_first
        view = view.sort_values(["quant_rank", "tg_tok_s", "ttft_proxy_s"], ascending=[False, False, True])

    print("\nTop candidates (after filtering):")
    print(view.head(k).to_string(index=False))

def choose_best(df: pd.DataFrame, objective: str) -> pd.Series:
    if objective == "quality_first":
        # Highest precision first, then faster decode, then lower TTFT, then lower RSS
        return df.sort_values(
            ["quant_rank", "tg_tok_s", "ttft_proxy_s", "max_rss_mb"],
            ascending=[False, False, True, True]
        ).iloc[0]

    if objective == "speed_first":
        # Fastest decode first, then lower TTFT, then higher precision, then lower RSS
        return df.sort_values(
            ["tg_tok_s", "ttft_proxy_s", "quant_rank", "max_rss_mb"],
            ascending=[False, True, False, True]
        ).iloc[0]

    if objective == "balanced":
        # Weighted score (RSS in GB so units are sane)
        df = df.copy()
        df["score"] = (
            1.0 * df["tg_tok_s"]
            - 0.3 * df["ttft_proxy_s"]
            - 1.0 * df["rss_gb"]
        )
        return df.sort_values(["score", "quant_rank"], ascending=[False, False]).iloc[0]

    # max_context
    # Choose largest prompt budget that passes constraints;
    # tie-break by higher precision and then speed/latency.
    return df.sort_values(
        ["n_prompt", "quant_rank", "tg_tok_s", "ttft_proxy_s", "max_rss_mb"],
        ascending=[False, False, False, True, True]
    ).iloc[0]

def main():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/runs.csv")
    ap.add_argument("--device", default="auto", choices=["auto", "MTL0", "BLAS"])
    ap.add_argument(
        "--objective",
        default="quality_first",
        choices=["quality_first", "speed_first", "balanced", "max_context"],
        help="Recommendation objective after filtering by constraints."
    )
    # Workload selection
    ap.add_argument("--n-gen", type=int, default=128)
    ap.add_argument("--n-prompt", type=int, default=512,
                    help="Used for objectives except max_context. For max_context this acts as a minimum prompt length filter.")
    ap.add_argument("--min-n-prompt", type=int, default=None,
                    help="Minimum prompt tokens required (useful for max_context).")
    # Constraints (optional)
    ap.add_argument("--max-rss-mb", type=float, default=None)
    ap.add_argument("--min-headroom-mb", type=float, default=None,
                    help="Optional headroom margin under the RAM ceiling (e.g., 500–1000MB for edge realism).")
    ap.add_argument("--max-ttft-s", type=float, default=None)
    ap.add_argument("--min-tg-tok-s", type=float, default=None)

    ap.add_argument("--out", default="results/recommendation.json")
    ap.add_argument("--topk", type=int, default=5)

    args = ap.parse_args()

    df = load_and_clean(args.csv)
    device = pick_device(df, args.device)

    # Filter to device + generation length
    df = df[(df["devices"] == device) & (df["n_gen"] == args.n_gen)].copy()
    if df.empty:
        raise SystemExit(f"No runs found for device={device}, n_gen={args.n_gen}")

    # For non-max_context objectives, also filter to a specific prompt length
    if args.objective != "max_context":
        df = df[df["n_prompt"] == args.n_prompt].copy()
        if df.empty:
            raise SystemExit(f"No runs found for device={device}, n_prompt={args.n_prompt}, n_gen={args.n_gen}")
    else:
        # max_context: optionally require at least some prompt length
        min_prompt = args.min_n_prompt if args.min_n_prompt is not None else args.n_prompt
        df = df[df["n_prompt"] >= min_prompt].copy()
        if df.empty:
            raise SystemExit(f"No runs found for device={device} with n_prompt >= {min_prompt} and n_gen={args.n_gen}")

    # Apply constraints
    df = apply_constraints(df, args.max_rss_mb, args.min_headroom_mb, args.max_ttft_s, args.min_tg_tok_s)
    if df.empty:
        raise SystemExit("No configurations satisfy the given constraints.")

    print_top(df, args.objective, k=args.topk)

    best = choose_best(df, args.objective).to_dict()

    rec = {
        "device": device,
        "objective": args.objective,
        "workload": {"n_prompt": int(best["n_prompt"]), "n_gen": int(best["n_gen"])},
        "constraints": {
            "max_rss_mb": args.max_rss_mb,
            "min_headroom_mb": args.min_headroom_mb,
            "max_ttft_s": args.max_ttft_s,
            "min_tg_tok_s": args.min_tg_tok_s,
            "min_n_prompt": args.min_n_prompt,
        },
        "recommended": {
            "model_filename": best["model_filename"],
            "quant": best["quant"],
            "n_threads": int(best["n_threads"]) if not pd.isna(best["n_threads"]) else None,
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

    print("\nRecommended configuration:")
    print(json.dumps(rec, indent=2))
    

if __name__ == "__main__":
    main()
    