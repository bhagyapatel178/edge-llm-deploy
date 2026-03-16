from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable


def ensure_models(
    repo_id: str,
    filenames: Iterable[str],
    model_dir: str | Path,
    prefer_cli: bool = True,
) -> list[Path]:
    """
    Ensure the requested GGUF files exist locally.

    First tries the HF CLI (`hf download ... --local-dir ...`) to mirror the
    workflow you're already using. Falls back to huggingface_hub if needed.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    filenames = list(filenames)
    wanted_paths = [model_dir / name for name in filenames]
    missing = [name for name, path in zip(filenames, wanted_paths) if not path.exists()]

    if not missing:
        return wanted_paths

    hf_cli = shutil.which("hf") or shutil.which("huggingface-cli")

    if prefer_cli and hf_cli:
        for filename in missing:
            cmd = [
                hf_cli,
                "download",
                repo_id,
                filename,
                "--local-dir",
                str(model_dir),
            ]
            print("Downloading:", " ".join(cmd))
            subprocess.run(cmd, check=True)
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "Could not find `hf` CLI and `huggingface_hub` is not installed. "
                "Install one of them to enable downloads."
            ) from exc

        print(f"Downloading from {repo_id} into {model_dir} via huggingface_hub...")
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=filenames,
            local_dir=str(model_dir),
        )

    for path in wanted_paths:
        if not path.exists():
            raise FileNotFoundError(f"Expected downloaded file not found: {path}")

    return wanted_paths