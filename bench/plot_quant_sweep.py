# bench/plot_quant_sweep.py
import os
import pandas as pd
import matplotlib.pyplot as plt

PLOTS_DIR = "results/plots"
CSV_PATH = "results/runs.csv"

# Order quants in a sensible "more compressed -> less compressed" direction
QUANT_ORDER = ["q3_k_m", "q4_k_m", "q6_k", "q8_0"]

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # Ensure numeric columns are numeric
    for col in ["n_prompt", "n_gen", "pp_tok_s", "tg_tok_s", "ttft_proxy_s", "max_rss_mb"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter to the quant sweep runs only (avoid your q4 p=3968 rows)
    df = df[(df["n_prompt"] == 512) & (df["n_gen"] == 128)].copy()

    # Keep only the quants we care about (and ensure consistent ordering)
    df = df[df["quant"].isin(QUANT_ORDER)].copy()
    df["quant"] = pd.Categorical(df["quant"], categories=QUANT_ORDER, ordered=True)

    # In case you have duplicates later (re-runs), average them
    agg = (
        df.groupby(["devices", "quant"], as_index=False)
          .agg({
              "tg_tok_s": "mean",
              "pp_tok_s": "mean",
              "ttft_proxy_s": "mean",
              "max_rss_mb": "mean",
          })
          .sort_values(["devices", "quant"])
    )

    # Helper to plot a metric vs quant with one line per device
    def plot_metric(metric: str, ylabel: str, title: str, out_name: str):
        plt.figure()

        for dev in sorted(agg["devices"].unique()):
            sub = agg[agg["devices"] == dev].sort_values("quant")
            plt.plot(sub["quant"].astype(str), sub[metric], marker="o", label=dev)

        plt.xlabel("Quantization")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(PLOTS_DIR, out_name)
        plt.savefig(out_path, dpi=200)
        print(f"Saved: {out_path}")

    plot_metric(
        metric="tg_tok_s",
        ylabel="Decode throughput (tokens/sec)",
        title="Decode throughput vs quant (p=512, n=128)",
        out_name="tg_tok_s_vs_quant.png",
    )

    plot_metric(
        metric="max_rss_mb",
        ylabel="Peak RSS (MB)",
        title="Peak RSS vs quant (p=512, n=128)",
        out_name="max_rss_mb_vs_quant.png",
    )

    plot_metric(
        metric="ttft_proxy_s",
        ylabel="TTFT proxy (s)",
        title="TTFT proxy vs quant (p=512, n=128)",
        out_name="ttft_proxy_s_vs_quant.png",
    )

if __name__ == "__main__":
    main()