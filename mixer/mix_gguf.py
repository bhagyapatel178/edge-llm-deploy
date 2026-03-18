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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a mixed-precision GGUF by copying selected tensors from a higher-precision source."
    )
    parser.add_argument("--config", type=Path, help="Optional YAML config file.")
    parser.add_argument("--low", type=Path, help="Low-precision GGUF source, e.g. Q4_K_M.")
    parser.add_argument("--high", type=Path, help="Higher-precision GGUF source, e.g. Q8_0 or F16.")
    parser.add_argument("--output", type=Path, help="Output mixed GGUF path.")
    parser.add_argument(
        "--policy",
        choices=["last_n_blocks", "output_plus_last_n_blocks", "regex_list"],
        help="Tensor selection policy.",
    )
    parser.add_argument("--last-n", type=int, default=0, help="Number of final transformer blocks to protect.")
    parser.add_argument(
        "--include-regex",
        action="append",
        default=None,
        help="Regex for tensors to take from the high-precision source. Repeatable.",
    )
    parser.add_argument(
        "--exclude-regex",
        action="append",
        default=None,
        help="Regex for tensors to force back to the low-precision source. Repeatable.",
    )
    parser.add_argument("--recipe-json", type=Path, help="Optional provenance JSON output.")
    parser.add_argument("--dry-run", action="store_true", help="Print the selection only; do not write a GGUF.")
    parser.add_argument("--verbose", action="store_true", help="Enable info logging.")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> MixerConfig:
    config_data: dict[str, Any] = {}
    if args.config:
        with args.config.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
            if not isinstance(raw, dict):
                raise MixerError("Config file must contain a YAML mapping.")
            config_data = raw

    def pick(name: str, default: Any = None) -> Any:
        cli_value = getattr(args, name)
        if cli_value is not None:
            return cli_value
        return config_data.get(name, default)

    low_raw = pick("low")
    high_raw = pick("high")
    output_raw = pick("output")

    if not low_raw:
        raise MixerError("Missing required 'low' path.")
    if not high_raw:
        raise MixerError("Missing required 'high' path.")
    if not output_raw:
        raise MixerError("Missing required 'output' path.")

    low = Path(low_raw)
    high = Path(high_raw)
    output = Path(output_raw)

    policy = pick("policy")
    if not policy:
        raise MixerError("A policy is required.")

    return MixerConfig(
        low=low,
        high=high,
        output=output,
        policy=policy,
        last_n=int(pick("last_n", 0) or 0),
        include_regex=list(pick("include_regex") or []),
        exclude_regex=list(pick("exclude_regex") or []),
        recipe_json=Path(pick("recipe_json")) if pick("recipe_json") else None,
        dry_run=bool(args.dry_run or config_data.get("dry_run", False)),
    )


def reader_arch(reader: Any) -> str:
    field = reader.get_field("general.architecture")
    if field is None:
        raise MixerError("Could not find general.architecture in GGUF metadata.")
    return str(field.contents())


def reader_alignment(reader: Any) -> int:
    field = reader.get_field("general.alignment")
    if field is None:
        return 32
    return int(field.contents())


def tensors_by_name(reader: Any) -> dict[str, TensorRecord]:
    return {
        tensor.name: TensorRecord(
            name=tensor.name,
            tensor_type=tensor.tensor_type,
            shape=tuple(int(x) for x in tensor.shape.tolist()),
            n_bytes=int(tensor.n_bytes),
            data=tensor.data,
        )
        for tensor in reader.tensors
    }


def block_index(name: str) -> int | None:
    match = BLOCK_RE.match(name)
    if match:
        return int(match.group(1))
    return None


def compile_many(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    return [re.compile(p) for p in patterns]


def select_tensors(
    names: list[str],
    policy: str,
    last_n: int,
    include_regex: list[str],
    exclude_regex: list[str],
) -> set[str]:
    selected: set[str] = set()
    block_ids = sorted({bid for bid in (block_index(name) for name in names) if bid is not None})
    protected_blocks: set[int] = set(block_ids[-last_n:]) if last_n > 0 else set()

    if policy == "last_n_blocks":
        selected = {name for name in names if (bid := block_index(name)) is not None and bid in protected_blocks}
    elif policy == "output_plus_last_n_blocks":
        selected = {
            name
            for name in names
            if ((bid := block_index(name)) is not None and bid in protected_blocks)
            or name == "output.weight"
        }
    elif policy == "regex_list":
        selected = set()
    else:
        raise MixerError(f"Unsupported policy: {policy}")

    include_res = compile_many(include_regex)
    exclude_res = compile_many(exclude_regex)

    for name in names:
        if any(rx.search(name) for rx in include_res):
            selected.add(name)
    for name in list(selected):
        if any(rx.search(name) for rx in exclude_res):
            selected.discard(name)

    return selected