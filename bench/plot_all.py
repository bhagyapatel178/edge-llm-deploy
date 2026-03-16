from __future__ import annotations

import subprocess
import sys


def main() -> None:
    scripts = [
        "bench/plot_quant_sweep.py",
        "bench/plot_ttft_vs_prompt.py",
        "bench/plot_rss_vs_prompt.py",
    ]

    for script in scripts:
        print(f"[plot] running {script}")
        subprocess.run([sys.executable, script], check=True)

    print("[plot] all plots generated")


if __name__ == "__main__":
    main()