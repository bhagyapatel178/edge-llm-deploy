from __future__ import annotations

from pathlib import Path
import json
import math
from typing import Any

import gguf


LOW_GGUF = "models/qwen2.5-3b/qwen2.5-3b-instruct-q4_k_m.gguf"
HIGH_GGUF = "models/qwen2.5-3b/qwen2.5-3b-instruct-q8_0.gguf"
SALIENCY_JSON = "artifacts/mixer/saliency/qwen25_3b_v2_saliency.json"
OUTPUT_JSON = "artifacts/mixer/recipes/qwen25_3b_v2_budget_match.json"

# Match the extra storage cost of your best v1 recipe:
# output + last2 q8 = 2.09 GB
# vanilla q4        = 1.96 GB
TARGET_EXTRA_GB = 2.09 - 1.96

# Use 1 MB bins for knapsack
BIN_BYTES = 1024 * 1024

def load_tensor_map(path: str) -> dict[str, Any]:
    reader = gguf.GGUFReader(path)
    return {tensor.name: tensor for tensor in reader.tensors}


def is_candidate_tensor(name: str, low_tensor: Any, high_tensor: Any) -> bool:
    """
    Restrict to tensors where promotion actually matters:
    - tensor exists in both
    - promotion increases bytes
    - tensor is at least 2D
    - skip biases/norms that are already same type or tiny
    """
    low_shape = tuple(int(x) for x in low_tensor.shape.tolist())
    high_shape = tuple(int(x) for x in high_tensor.shape.tolist())

    if low_shape != high_shape:
        return False

    if len(low_shape) < 2:
        return False

    delta_bytes = int(high_tensor.n_bytes) - int(low_tensor.n_bytes)
    if delta_bytes <= 0:
        return False

    return True

def main() -> None:
    low_map = load_tensor_map(LOW_GGUF)
    high_map = load_tensor_map(HIGH_GGUF)

    saliency = json.loads(Path(SALIENCY_JSON).read_text(encoding="utf-8"))

    target_extra_bytes = int(round(TARGET_EXTRA_GB * (1024 ** 3)))
    budget_bins = target_extra_bytes // BIN_BYTES

    candidates: list[dict[str, Any]] = []

    for name, scores in saliency.items():
        if name not in low_map or name not in high_map:
            continue

        low_tensor = low_map[name]
        high_tensor = high_map[name]

        if not is_candidate_tensor(name, low_tensor, high_tensor):
            continue

        delta_bytes = int(high_tensor.n_bytes) - int(low_tensor.n_bytes)
        cost_bins = max(1, math.ceil(delta_bytes / BIN_BYTES))

        # Base v2 score: mean_abs only
        saliency_score = float(scores["mean_abs"])

        candidates.append({
            "name": name,
            "score": saliency_score,
            "mean_abs": float(scores["mean_abs"]),
            "rms": float(scores["rms"]),
            "count": int(scores["count"]),
            "delta_bytes": delta_bytes,
            "cost_bins": cost_bins,
            "low_type": getattr(low_tensor.tensor_type, "name", str(low_tensor.tensor_type)),
            "high_type": getattr(high_tensor.tensor_type, "name", str(high_tensor.tensor_type)),
            "shape": [int(x) for x in low_tensor.shape.tolist()],
        })

    if not candidates:
        raise RuntimeError("No candidate tensors found. Check saliency JSON and GGUF compatibility.")
    
    # 0-1 knapsack DP
    n = len(candidates)
    dp = [0.0] * (budget_bins + 1)
    keep = [[False] * (budget_bins + 1) for _ in range(n)]

    for i, item in enumerate(candidates):
        w = item["cost_bins"]
        v = item["score"]

        for b in range(budget_bins, w - 1, -1):
            alt = dp[b - w] + v
            if alt > dp[b]:
                dp[b] = alt
                keep[i][b] = True

    # Recover chosen set
    b = max(range(budget_bins + 1), key=lambda x: dp[x])
    selected: list[dict[str, Any]] = []

    for i in range(n - 1, -1, -1):
        if keep[i][b]:
            item = candidates[i]
            selected.append(item)
            b -= item["cost_bins"]

    selected.reverse()

    used_bytes = sum(item["delta_bytes"] for item in selected)
    used_bins = sum(item["cost_bins"] for item in selected)
    total_score = sum(item["score"] for item in selected)

    # Also compute greedy baseline for comparison/debugging
    greedy_candidates = sorted(
        candidates,
        key=lambda x: x["score"] / x["delta_bytes"],
        reverse=True,
    )

    greedy_selected = []
    greedy_used = 0
    for item in greedy_candidates:
        if greedy_used + item["delta_bytes"] <= target_extra_bytes:
            greedy_selected.append(item)
            greedy_used += item["delta_bytes"]

    out = {
        "policy": "saliency_knapsack_budget_match_v2",
        "low": LOW_GGUF,
        "high": HIGH_GGUF,
        "saliency_json": SALIENCY_JSON,
        "score_definition": "mean_abs",
        "budget_bytes": target_extra_bytes,
        "budget_bins": budget_bins,
        "bin_bytes": BIN_BYTES,
        "used_bytes": used_bytes,
        "used_bins": used_bins,
        "selected_count": len(selected),
        "total_score": total_score,
        "selected_tensors": selected,
        "greedy_baseline": {
            "policy": "score_per_byte_greedy",
            "selected_count": len(greedy_selected),
            "used_bytes": greedy_used,
            "total_score": sum(item["score"] for item in greedy_selected),
            "selected_tensor_names": [item["name"] for item in greedy_selected],
        },
    }

    out_path = Path(OUTPUT_JSON)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Wrote recipe: {out_path}")
    print(f"Candidates: {len(candidates)}")
    print(f"Budget: {target_extra_bytes / (1024**2):.1f} MB")
    print(f"Used:   {used_bytes / (1024**2):.1f} MB")
    print(f"Selected tensors: {len(selected)}")
    print(f"Total score: {total_score:.6f}")

if __name__ == "__main__":
    main()