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
GGUF_INTERNAL_KEYS = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count","general.architecture"}
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
    selected_recipe: Path | None = None
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
        choices=["last_n_blocks", "output_plus_last_n_blocks", "regex_list", "explicit_list"],
        help="Tensor selection policy.",
    )
    parser.add_argument("--last-n", type=int, default=None, help="Number of final transformer blocks to protect.")
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
    parser.add_argument(
        "--selected-recipe",
        type=Path,
        help="Optional recipe JSON containing explicitly selected tensor names.",
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

    selected_recipe_raw = pick("selected_recipe")
    selected_recipe = Path(selected_recipe_raw) if selected_recipe_raw else None

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
        selected_recipe=selected_recipe,
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

    if policy in {"last_n_blocks", "output_plus_last_n_blocks"} and last_n <= 0:
        raise MixerError(f"Policy {policy!r} requires last_n > 0")

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


def validate_compatible(low_reader: Any, high_reader: Any) -> tuple[dict[str, TensorRecord], dict[str, TensorRecord]]:
    low_arch = reader_arch(low_reader)
    high_arch = reader_arch(high_reader)
    if low_arch != high_arch:
        raise MixerError(f"Architecture mismatch: low={low_arch!r}, high={high_arch!r}")

    low_tensors = tensors_by_name(low_reader)
    high_tensors = tensors_by_name(high_reader)

    low_names = set(low_tensors)
    high_names = set(high_tensors)
    if low_names != high_names:
        missing_from_high = sorted(low_names - high_names)
        missing_from_low = sorted(high_names - low_names)
        raise MixerError(
            "Tensor name mismatch between low/high GGUFs. "
            f"Missing from high: {missing_from_high[:5]} | Missing from low: {missing_from_low[:5]}"
        )

    for name in low_tensors:
        lt = low_tensors[name]
        ht = high_tensors[name]
        if lt.shape != ht.shape:
            raise MixerError(f"Shape mismatch for tensor {name}: low={lt.shape}, high={ht.shape}")

    return low_tensors, high_tensors


def copy_metadata(reader: Any, writer: Any) -> None:
    for key, field in reader.fields.items():
        if key in GGUF_INTERNAL_KEYS or key == "general.alignment":
            continue
        value = field.contents()
        if not field.types:
            continue
        main_type = field.types[0]
        sub_type = field.types[-1] if main_type == gguf.GGUFValueType.ARRAY and len(field.types) > 1 else None
        writer.add_key_value(key, value, main_type, sub_type)


def write_recipe_json(
    path: Path,
    config: MixerConfig,
    selected: set[str],
    low_reader: Any,
    high_reader: Any,
    low_tensors: dict[str, TensorRecord],
    high_tensors: dict[str, TensorRecord],
) -> None:
    protected = []
    for name in sorted(selected):
        ht = high_tensors[name]
        lt = low_tensors[name]
        protected.append(
            {
                "name": name,
                "shape": list(ht.shape),
                "high_type": str(ht.tensor_type.name),
                "low_type": str(lt.tensor_type.name),
                "n_bytes_high": ht.n_bytes,
                "n_bytes_low": lt.n_bytes,
            }
        )

    recipe = {
        "low": str(config.low),
        "high": str(config.high),
        "output": str(config.output),
        "policy": config.policy,
        "last_n": config.last_n,
        "include_regex": config.include_regex, 
        "exclude_regex": config.exclude_regex,
        "architecture": reader_arch(low_reader),
        "protected_tensor_count": len(protected),
        "protected_tensors": protected,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    config = load_config(args)

    if not config.low.exists():
        raise MixerError(f"Low-precision GGUF not found: {config.low}")
    if not config.high.exists():
        raise MixerError(f"High-precision GGUF not found: {config.high}")

    low_reader = gguf.GGUFReader(config.low)
    high_reader = gguf.GGUFReader(config.high)
    low_tensors, high_tensors = validate_compatible(low_reader, high_reader)

    ordered_names = [tensor.name for tensor in low_reader.tensors]
    selected = select_tensors(
        ordered_names,
        policy=config.policy,
        last_n=config.last_n,
        include_regex=config.include_regex or [],
        exclude_regex=config.exclude_regex or [],
    )

    LOGGER.info("Selected %d protected tensors.", len(selected))
    for name in sorted(selected)[:20]:
        LOGGER.info("protect -> %s", name)
    if len(selected) > 20:
        LOGGER.info("... plus %d more", len(selected) - 20)

    if config.recipe_json:
        write_recipe_json(config.recipe_json, config, selected, low_reader, high_reader, low_tensors, high_tensors)

    if config.dry_run:
        print(json.dumps({
            "selected_count": len(selected),
            "selected": sorted(selected),
        }, indent=2))
        return

    config.output.parent.mkdir(parents=True, exist_ok=True)

    arch = reader_arch(low_reader)
    alignment = reader_alignment(low_reader)
    writer = gguf.GGUFWriter(path=config.output, arch=arch, use_temp_file=False)
    writer.add_custom_alignment(alignment)
    copy_metadata(low_reader, writer)

    protected_count = 0
    for name in ordered_names:
        source = high_tensors[name] if name in selected else low_tensors[name]
        if name in selected:
            protected_count += 1
        writer.add_tensor(
            name=name,
            tensor=source.data,
            raw_shape=tuple(int(x) for x in source.data.shape),
            raw_dtype=source.tensor_type,
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    summary = {
        "output": str(config.output),
        "architecture": arch,
        "protected_tensor_count": protected_count,
        "total_tensor_count": len(ordered_names),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
