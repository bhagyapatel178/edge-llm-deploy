from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

try:
    import gguf
except ImportError as exc:  # pragma: no cover
    raise SystemExit("The 'gguf' package is required. Install with: pip install gguf") from exc

BLOCK_RE = re.compile(r"^blk\.(\d+)\.")


def block_index(name: str) -> int | None:
    m = BLOCK_RE.match(name)
    return int(m.group(1)) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump GGUF tensor inventory to CSV.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reader = gguf.GGUFReader(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["name", "block", "tensor_type", "shape", "n_bytes"],
        )
        writer.writeheader()
        for tensor in reader.tensors:
            writer.writerow(
                {
                    "name": tensor.name,
                    "block": block_index(tensor.name),
                    "tensor_type": tensor.tensor_type.name,
                    "shape": list(int(x) for x in tensor.shape.tolist()),
                    "n_bytes": int(tensor.n_bytes),
                }
            )


if __name__ == "__main__":
    main()
