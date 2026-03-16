import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "results/runs.csv"
OUT_DIR = "results/plots"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # Keep only this model/quant for now (avoids clutter)
    df = df[df["quant"] == "q4_k_m"].copy()

    df["n_prompt"] = pd.to_numeric(df["n_prompt"], errors="coerce")
    df["max_rss_mb"] = pd.to_numeric(df["max_rss_mb"], errors="coerce")
    df = df.dropna(subset=["n_prompt", "max_rss_mb"])

    plt.figure()
    for dev in sorted(df["devices"].dropna().unique()):
        sub = df[df["devices"] == dev].sort_values("n_prompt")
        plt.plot(sub["n_prompt"], sub["max_rss_mb"], marker="o", label=dev)

    plt.xlabel("Prompt tokens (proxy for context/KV usage)")
    plt.ylabel("Peak RSS (MB)")
    plt.title("Peak RSS vs Prompt Token Budget (Q4_K_M)")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "rss_vs_prompt_tokens.png")
    plt.savefig(out, dpi=200)
    print("Saved:", out)


if __name__ == "__main__":
    main()