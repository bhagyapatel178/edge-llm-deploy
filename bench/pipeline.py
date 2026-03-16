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


def sort_candidates(df: pd.DataFrame, objective: str) -> pd.DataFrame:
    df = df.copy()

    if objective == "quality_first":
        return df.sort_values(
            ["quant_rank", "tg_tok_s", "ttft_proxy_s", "max_rss_mb"],
            ascending=[False, False, True, True],
        )

    if objective == "speed_first":
        return df.sort_values(
            ["tg_tok_s", "ttft_proxy_s", "quant_rank", "max_rss_mb"],
            ascending=[False, True, False, True],
        )

    if objective == "balanced":
        df["score"] = (
            1.0 * df["tg_tok_s"]
            - 0.3 * df["ttft_proxy_s"]
            - 1.0 * df["rss_gb"]
        )
        return df.sort_values(["score", "quant_rank"], ascending=[False, False])

    # max_context
    return df.sort_values(
        ["n_prompt", "quant_rank", "tg_tok_s", "ttft_proxy_s", "max_rss_mb"],
        ascending=[False, False, False, True, True],
    )


def build_filtered_df(
    runs_csv: str | Path,
    constraints: dict[str, Any],
    objective: str,
) -> tuple[pd.DataFrame, str]:
    df = load_and_clean(str(runs_csv))

    device = pick_device(df, constraints.get("device", "auto"))

    n_gen = int(constraints.get("n_gen", 128))
    df = df[(df["devices"] == device) & (df["n_gen"] == n_gen)].copy()
    if df.empty:
        raise SystemExit(f"No runs found for device={device}, n_gen={n_gen}")

    if objective != "max_context":
        n_prompt = int(constraints.get("n_prompt", 512))
        df = df[df["n_prompt"] == n_prompt].copy()
        if df.empty:
            raise SystemExit(
                f"No runs found for device={device}, n_prompt={n_prompt}, n_gen={n_gen}"
            )
    else:
        min_n_prompt = constraints.get("min_n_prompt", constraints.get("n_prompt", 512))
        min_n_prompt = int(min_n_prompt)
        df = df[df["n_prompt"] >= min_n_prompt].copy()
        if df.empty:
            raise SystemExit(
                f"No runs found for device={device} with n_prompt >= {min_n_prompt} and n_gen={n_gen}"
            )

    df = apply_constraints(
        df,
        constraints.get("max_rss_mb"),
        constraints.get("min_headroom_mb"),
        constraints.get("max_ttft_s"),
        constraints.get("min_tg_tok_s"),
    )
    if df.empty:
        raise SystemExit("No configurations satisfy the given constraints.")

    return df, device


def write_recommendation(
    runs_csv: str | Path,
    constraints: dict[str, Any],
    objective: str,
    out_path: str | Path,
    topk: int = 5,
) -> dict[str, Any]:
    filtered_df, device = build_filtered_df(runs_csv, constraints, objective)
    ranked = sort_candidates(filtered_df, objective)
    best = choose_best(filtered_df, objective).to_dict()

    top_candidates = []
    for _, row in ranked.head(topk).iterrows():
        top_candidates.append(
            {
                "model_filename": row["model_filename"],
                "quant": row["quant"],
                "devices": row["devices"],
                "n_prompt": int(row["n_prompt"]),
                "n_gen": int(row["n_gen"]),
                "tg_tok_s": float(row["tg_tok_s"]),
                "ttft_proxy_s": float(row["ttft_proxy_s"]),
                "max_rss_mb": float(row["max_rss_mb"]),
            }
        )

    rec = {
        "device": device,
        "objective": objective,
        "workload": {
            "n_prompt": int(best["n_prompt"]),
            "n_gen": int(best["n_gen"]),
        },
        "constraints": {
            "max_rss_mb": constraints.get("max_rss_mb"),
            "min_headroom_mb": constraints.get("min_headroom_mb"),
            "max_ttft_s": constraints.get("max_ttft_s"),
            "min_tg_tok_s": constraints.get("min_tg_tok_s"),
            "min_n_prompt": constraints.get("min_n_prompt"),
        },
        "recommended": {
            "model_filename": best["model_filename"],
            "quant": best["quant"],
            "n_threads": int(best["n_threads"]) if not pd.isna(best["n_threads"]) else None,
            "n_gpu_layers": int(best["n_gpu_layers"]) if not pd.isna(best["n_gpu_layers"]) else None,
            "devices": best["devices"],
        },
        "expected": {
            "pp_tok_s": float(best["pp_tok_s"]),
            "tg_tok_s": float(best["tg_tok_s"]),
            "ttft_proxy_s": float(best["ttft_proxy_s"]),
            "max_rss_mb": float(best["max_rss_mb"]),
        },
        "top_candidates": top_candidates,
        "note": "TTFT proxy computed as (n_prompt/pp_tok_s) + (1/tg_tok_s). Peak RSS from /usr/bin/time.",
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)

    print(f"[recommend] wrote {out_path}")
    print(json.dumps(rec, indent=2))
    return rec


def cleanup_models(
    run_config: dict[str, Any],
    runs_csv: str | Path,
    constraints: dict[str, Any],
    objective: str,
    keep_top: int = 2,
) -> None:
    filtered_df, _device = build_filtered_df(runs_csv, constraints, objective)
    ranked = sort_candidates(filtered_df, objective)

    keep_paths: list[str] = []
    for model_filename in ranked["model_filename"].tolist():
        if model_filename not in keep_paths:
            keep_paths.append(model_filename)
        if len(keep_paths) >= keep_top:
            break

    keep_names = {Path(p).name for p in keep_paths}
    model_dir = Path(run_config["model_dir"])

    for quant_filename in run_config["quants"]:
        path = model_dir / quant_filename
        if path.exists() and path.name not in keep_names:
            print(f"[cleanup] deleting {path}")
            path.unlink()
