import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/runs.csv")

# Keep only this model/quant for now (avoids clutter)
df = df[df["quant"] == "q4_k_m"].copy()

df["n_prompt"] = df["n_prompt"].astype(int)
df["max_rss_mb"] = df["max_rss_mb"].astype(float)

plt.figure()
for dev in sorted(df["devices"].unique()):
    sub = df[df["devices"] == dev].sort_values("n_prompt")
    plt.plot(sub["n_prompt"], sub["max_rss_mb"], marker="o", label=dev)

plt.xlabel("Prompt tokens (proxy for context/KV usage)")
plt.ylabel("Peak RSS (MB)")
plt.title("Peak RSS vs Prompt Token Budget (Q4_K_M)")
plt.legend()
plt.tight_layout()
plt.savefig("results/plots/rss_vs_prompt_tokens.png", dpi=200)
print("Saved: results/plots/rss_vs_prompt_tokens.png")
