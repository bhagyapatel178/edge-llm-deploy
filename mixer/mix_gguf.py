from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

try:
    import gguf
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The 'gguf' package is required. Install with: pip install gguf"
    ) from exc


LOGGER = logging.getLogger("mixer.mix_gguf")
GGUF_INTERNAL_KEYS = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
BLOCK_RE = re.compile(r"^blk\.(\d+)\.")


@dataclass
class MixerConfig:
    low: Path
    high: Path
    output: Path
    policy: str
    last_n: int = 0
    include_regex: list[str] | None = None
    exclude_regex: list[str] | None = None
    recipe_json: Path | None = None
    dry_run: bool = False


@dataclass
class TensorRecord:
    name: str
    tensor_type: Any
    shape: tuple[int, ...]
    n_bytes: int
    data: Any


class MixerError(RuntimeError):
    pass