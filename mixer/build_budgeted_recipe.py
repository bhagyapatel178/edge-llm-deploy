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
