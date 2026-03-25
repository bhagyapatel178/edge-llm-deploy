from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import gguf


DEFAULT_LOW_GGUF = "models/qwen2.5-3b/qwen2.5-3b-instruct-q4_k_m.gguf"
DEFAULT_HIGH_GGUF = "models/qwen2.5-3b/qwen2.5-3b-instruct-q8_0.gguf"
DEFAULT_SALIENCY_JSON = "artifacts/mixer/saliency/qwen25_3b_v2_saliency.json"
DEFAULT_OUTPUT_JSON = "artifacts/mixer/recipes/qwen25_3b_v2_budget_match.json"

# 0.5 MiB bins
BIN_BYTES = 512 * 1024

# Keep the same best-working v2.1 choice
DEFAULT_FORCED_NAMES = {"output.weight"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a budgeted tensor-promotion recipe from saliency scores."
    )
    parser.add_argument(
        "--low-gguf",
        default=DEFAULT_LOW_GGUF,
        help="Low-precision GGUF path.",
    )
    parser.add_argument(
        "--high-gguf",
        default=DEFAULT_HIGH_GGUF,
        help="High-precision GGUF path.",
    )
    parser.add_argument(
        "--saliency-json",
        default=DEFAULT_SALIENCY_JSON,
        help="Saliency JSON path.",
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT_JSON,
        help="Output recipe JSON path.",
    )
    parser.add_argument(
        "--target-extra-mib",
        type=float,
        required=True,
        help="Target extra budget over the low model, in MiB.",
    )
    parser.add_argument(
        "--force-name",
        action="append",
        default=None,
        help="Tensor name to always include. Repeatable. Default: output.weight",
    )
    parser.add_argument(
        "--policy-name",
        default="saliency_knapsack_budget_sweep_v2_1",
        help="Policy label to write into the output JSON.",
    )
    return parser.parse_args()


def load_tensor_map(path: str) -> dict[str, Any]:
    reader = gguf.GGUFReader(path)
    return {tensor.name: tensor for tensor in reader.tensors}


def is_candidate_tensor(low_tensor: Any, high_tensor: Any) -> bool:
    """
    Restrict to tensors where promotion actually matters:
    - tensor exists in both
    - promotion increases bytes
    - tensor is at least 2D
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
    args = parse_args()

    low_gguf = args.low_gguf
    high_gguf = args.high_gguf
    saliency_json = args.saliency_json
    output_json = args.output_json
    forced_names = set(args.force_name) if args.force_name else set(DEFAULT_FORCED_NAMES)

    low_map = load_tensor_map(low_gguf)
    high_map = load_tensor_map(high_gguf)

    saliency = json.loads(Path(saliency_json).read_text(encoding="utf-8"))

    target_extra_bytes = int(round(args.target_extra_mib * (1024 ** 2)))

    candidates: list[dict[str, Any]] = []

    for name, scores in saliency.items():
        if name not in low_map or name not in high_map:
            continue

        low_tensor = low_map[name]
        high_tensor = high_map[name]

        if not is_candidate_tensor(low_tensor, high_tensor):
            continue

        weight_rms = scores.get("weight_rms")
        if weight_rms is None:
            continue

        rms = scores.get("rms")
        mean_abs = scores.get("mean_abs")
        count = scores.get("count")

        if rms is None or mean_abs is None or count is None:
            continue

        delta_bytes = int(high_tensor.n_bytes) - int(low_tensor.n_bytes)
        cost_bins = max(1, math.ceil(delta_bytes / BIN_BYTES))

        # v2.1 score: rms activation * rms weight
        saliency_score = float(rms) * float(weight_rms)

        candidates.append({
            "name": name,
            "score": saliency_score,
            "mean_abs": float(mean_abs),
            "rms": float(rms),
            "weight_rms": float(weight_rms),
            "count": int(count),
            "numel": scores.get("numel"),
            "delta_bytes": delta_bytes,
            "cost_bins": cost_bins,
            "low_type": getattr(low_tensor.tensor_type, "name", str(low_tensor.tensor_type)),
            "high_type": getattr(high_tensor.tensor_type, "name", str(high_tensor.tensor_type)),
            "shape": [int(x) for x in low_tensor.shape.tolist()],
        })

    if not candidates:
        raise RuntimeError("No candidate tensors found. Check saliency JSON and GGUF compatibility.")

    forced = [c for c in candidates if c["name"] in forced_names]
    remaining = [c for c in candidates if c["name"] not in forced_names]

    forced_bytes = sum(c["delta_bytes"] for c in forced)
    forced_score = sum(c["score"] for c in forced)

    remaining_budget_bytes = target_extra_bytes - forced_bytes
    if remaining_budget_bytes < 0:
        forced_mib = forced_bytes / (1024 ** 2)
        target_mib = target_extra_bytes / (1024 ** 2)
        raise RuntimeError(
            f"Forced tensors exceed target budget: "
            f"forced={forced_mib:.2f} MiB, target={target_mib:.2f} MiB. "
            f"Increase --target-extra-mib or remove forced tensors."
        )

    budget_bins = remaining_budget_bytes // BIN_BYTES

    # 0-1 knapsack over remaining tensors
    n = len(remaining)
    dp = [0.0] * (budget_bins + 1)
    keep = [[False] * (budget_bins + 1) for _ in range(n)]

    for i, item in enumerate(remaining):
        w = item["cost_bins"]
        v = item["score"]

        for b in range(budget_bins, w - 1, -1):
            alt = dp[b - w] + v
            if alt > dp[b]:
                dp[b] = alt
                keep[i][b] = True

    # Recover chosen set
    b = max(range(budget_bins + 1), key=lambda x: dp[x])
    selected_remaining: list[dict[str, Any]] = []

    for i in range(n - 1, -1, -1):
        if keep[i][b]:
            item = remaining[i]
            selected_remaining.append(item)
            b -= item["cost_bins"]

    selected_remaining.reverse()

    # Greedy fill leftover budget from unselected tensors
    selected_names = {item["name"] for item in selected_remaining}
    used_remaining_bytes = sum(item["delta_bytes"] for item in selected_remaining)

    leftover_candidates = sorted(
        [c for c in remaining if c["name"] not in selected_names],
        key=lambda x: x["score"] / x["delta_bytes"],
        reverse=True,
    )

    for item in leftover_candidates:
        if used_remaining_bytes + item["delta_bytes"] <= remaining_budget_bytes:
            selected_remaining.append(item)
            selected_names.add(item["name"])
            used_remaining_bytes += item["delta_bytes"]

    selected = forced + selected_remaining
    used_bytes = forced_bytes + used_remaining_bytes
    used_bins = sum(item["cost_bins"] for item in selected_remaining) + sum(item["cost_bins"] for item in forced)
    total_score = forced_score + sum(item["score"] for item in selected_remaining)

    # Greedy baseline for comparison/debugging
    greedy_remaining = sorted(
        remaining,
        key=lambda x: x["score"] / x["delta_bytes"],
        reverse=True,
    )

    greedy_selected_remaining = []
    greedy_used_remaining = 0
    for item in greedy_remaining:
        if greedy_used_remaining + item["delta_bytes"] <= remaining_budget_bytes:
            greedy_selected_remaining.append(item)
            greedy_used_remaining += item["delta_bytes"]

    greedy_selected = forced + greedy_selected_remaining
    greedy_used = forced_bytes + greedy_used_remaining
    greedy_score = forced_score + sum(item["score"] for item in greedy_selected_remaining)

    out = {
        "policy": args.policy_name,
        "low": low_gguf,
        "high": high_gguf,
        "saliency_json": saliency_json,
        "score_definition": "rms_activation_times_rms_weight",
        "forced_names": sorted(forced_names),
        "target_extra_mib": args.target_extra_mib,
        "target_budget_bytes": target_extra_bytes,
        "forced_bytes": forced_bytes,
        "remaining_budget_bytes": remaining_budget_bytes,
        "budget_bins": budget_bins,
        "bin_bytes": BIN_BYTES,
        "used_bytes": used_bytes,
        "used_mib": used_bytes / (1024 ** 2),
        "selected_count": len(selected),
        "used_bins": used_bins,
        "total_score": total_score,
        "selected_tensors": selected,
        "selected_tensor_names": [item["name"] for item in selected],
        "greedy_baseline": {
            "policy": "forced_plus_score_per_byte_greedy",
            "selected_count": len(greedy_selected),
            "used_bytes": greedy_used,
            "used_mib": greedy_used / (1024 ** 2),
            "total_score": greedy_score,
            "selected_tensor_names": [item["name"] for item in greedy_selected],
        },
    }

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Wrote recipe: {out_path}")
    print(f"Candidates:       {len(candidates)}")
    print(f"Forced tensors:   {len(forced)}")
    print(f"Target budget:    {target_extra_bytes / (1024**2):.2f} MiB")
    print(f"Forced bytes:     {forced_bytes / (1024**2):.2f} MiB")
    print(f"Used bytes:       {used_bytes / (1024**2):.2f} MiB")
    print(f"Selected tensors: {len(selected)}")
    print(f"Total score:      {total_score:.6f}")


if __name__ == "__main__":
    main()