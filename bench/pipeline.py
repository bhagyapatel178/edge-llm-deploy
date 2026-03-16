from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .download_models import ensure_models
from .ingest_jsonl_time import main as ingest_one
from .recommend import apply_constraints, choose_best, load_and_clean, pick_device
from .run_grid import build_run_specs, run_benchmark


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def existing_ingested_pairs(csv_path: str | Path) -> set[tuple[str, str]]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return set()

    pairs: set[tuple[str, str]] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.add((row.get("jsonl_path", ""), row.get("time_path", "")))
    return pairs


def remove_ingested_pair(
    csv_path: str | Path,
    jsonl_path: str | Path,
    time_path: str | Path,
) -> None:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)

    if "jsonl_path" not in df.columns or "time_path" not in df.columns:
        return

    jsonl_str = str(jsonl_path)
    time_str = str(time_path)

    before = len(df)
    df = df[~((df["jsonl_path"] == jsonl_str) & (df["time_path"] == time_str))]
    after = len(df)

    if after != before:
        df.to_csv(csv_path, index=False)
        print(f"[ingest] removed existing row for {Path(jsonl_str).name}")