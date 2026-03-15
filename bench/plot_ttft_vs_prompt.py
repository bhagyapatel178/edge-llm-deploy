import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "results/runs.csv"
OUT_DIR = "results/plots"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    # numeric
    for c in ["n_prompt", "n_gen", "ttft_proxy_s", "tg_tok_s", "max_rss_mb", "n_gpu_layers"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # focus on the context/prompt sweep runs
    df = df[df["n_gen"] == 128].copy()

    # only the quants we have prompt sweeps for
    df = df[df["quant"].isin(["q4_k_m", "q6_k"])].copy()

    # Avoid mixing “MTL0 but ngl=0” into Metal curves
    df = df[
        (df["devices"] == "BLAS") |
        ((df["devices"] == "MTL0") & (df["n_gpu_layers"] > 0))
    ].copy()

    plt.figure()
    for (dev, quant), sub in df.groupby(["devices", "quant"]):
        sub = sub.sort_values("n_prompt")
        plt.plot(sub["n_prompt"], sub["ttft_proxy_s"], marker="o", label=f"{dev}-{quant}")

    plt.xlabel("Prompt tokens")
    plt.ylabel("TTFT proxy (s)")
    plt.title("TTFT proxy vs prompt tokens (n_gen=128)")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "ttft_vs_prompt_tokens_q4_q6.png")
    plt.savefig(out, dpi=200)
    print("Saved:", out)

if __name__ == "__main__":
    main()
