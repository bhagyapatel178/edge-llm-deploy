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