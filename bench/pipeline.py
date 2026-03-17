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

from collections import defaultdict
from datetime import datetime, timezone

def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def grouped_specs_by_model_device(specs):
    groups = defaultdict(list)
    for spec in specs:
        groups[(str(spec.model_path), spec.device)].append(spec)

    for _, group in groups.items():
        group.sort(key=lambda s: s.n_prompt)
        yield group


def load_result_row(
    runs_csv: str | Path,
    jsonl_path: str | Path,
    time_path: str | Path,
) -> dict[str, Any] | None:
    csv_path = Path(runs_csv)
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if "jsonl_path" not in df.columns or "time_path" not in df.columns:
        return None

    sub = df[
        (df["jsonl_path"] == str(jsonl_path)) &
        (df["time_path"] == str(time_path))
    ]
    if sub.empty:
        return None

    row = sub.iloc[-1].to_dict()
    for col in ["pp_tok_s", "tg_tok_s", "ttft_proxy_s", "max_rss_mb", "n_prompt", "n_gen"]:
        if col in row:
            row[col] = pd.to_numeric(row[col], errors="coerce")
    return row


SKIP_FIELDS = [
    "timestamp_utc",
    "reason",
    "model_filename",
    "quant",
    "devices",
    "n_prompt",
    "n_gen",
    "jsonl_path",
    "time_path",
    "predicted_next_ttft_s",
    "current_ttft_s",
    "current_max_rss_mb",
    "elapsed_s",
]


def upsert_skip_record(skip_csv: str | Path, record: dict[str, Any]) -> None:
    skip_csv = Path(skip_csv)
    skip_csv.parent.mkdir(parents=True, exist_ok=True)

    if skip_csv.exists():
        df = pd.read_csv(skip_csv)
        if "jsonl_path" in df.columns and "time_path" in df.columns:
            df = df[
                ~(
                    (df["jsonl_path"] == record["jsonl_path"]) &
                    (df["time_path"] == record["time_path"])
                )
            ]
    else:
        df = pd.DataFrame(columns=SKIP_FIELDS)

    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(skip_csv, index=False)


def effective_rss_limit_mb(constraints: dict[str, Any]) -> float | None:
    max_rss_mb = constraints.get("max_rss_mb")
    min_headroom_mb = constraints.get("min_headroom_mb")

    if max_rss_mb is None:
        return None
    if min_headroom_mb is not None:
        return float(max_rss_mb) - float(min_headroom_mb)
    return float(max_rss_mb)


def prune_decision(
    current_row: dict[str, Any],
    next_prompt: int,
    objective: str,
    constraints: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    prune_cfg = constraints.get("prune", {})
    if not prune_cfg.get("enabled", False):
        return None, {}

    rss_limit = effective_rss_limit_mb(constraints)
    current_rss = float(current_row["max_rss_mb"])
    current_ttft = float(current_row["ttft_proxy_s"])

    if prune_cfg.get("stop_on_memory", True) and rss_limit is not None:
        if current_rss > rss_limit:
            return "memory_budget_exceeded", {
                "current_max_rss_mb": current_rss,
                "current_ttft_s": current_ttft,
            }

    apply_to = set(prune_cfg.get("apply_to_objectives", ["balanced", "speed_first"]))
    if objective in apply_to and prune_cfg.get("stop_on_predicted_ttft", True):
        max_ttft_s = constraints.get("max_ttft_s")
        if max_ttft_s is not None:
            pp_tok_s = float(current_row["pp_tok_s"])
            tg_tok_s = float(current_row["tg_tok_s"])
            predicted_next_ttft = (next_prompt / pp_tok_s) + (1.0 / tg_tok_s)

            multiplier = float(prune_cfg.get("ttft_multiplier", 1.0))
            if predicted_next_ttft > float(max_ttft_s) * multiplier:
                return "predicted_ttft_exceeds_threshold", {
                    "predicted_next_ttft_s": predicted_next_ttft,
                    "current_max_rss_mb": current_rss,
                    "current_ttft_s": current_ttft,
                }

    return None, {}


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--constraints", required=True)
    ap.add_argument(
        "--objective",
        default="balanced",
        choices=["quality_first", "speed_first", "balanced", "max_context"],
    )
    ap.add_argument("--cleanup", action="store_true")
    ap.add_argument("--keep-top", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    run_cfg = load_yaml(args.config)
    constraint_cfg = load_yaml(args.constraints)

    runs_csv = Path(run_cfg.get("runs_csv", "results/runs.csv"))
    plots_dir = Path(run_cfg.get("plots_dir", "results/plots"))
    recommendation_out = Path(
        run_cfg.get("recommendation_out", "results/recommendation.json")
    )

    runs_csv.parent.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    recommendation_out.parent.mkdir(parents=True, exist_ok=True)

    ensure_models(
        repo_id=run_cfg["repo_id"],
        filenames=run_cfg["quants"],
        model_dir=run_cfg["model_dir"],
        prefer_cli=bool(run_cfg.get("prefer_hf_cli", True)),
    )

    specs = build_run_specs(run_cfg)
    ingested_pairs = existing_ingested_pairs(runs_csv)

    skip_csv = Path(run_cfg.get("skipped_runs_csv", "results/skipped_runs.csv"))
    prune_cfg = constraint_cfg.get("prune", {})
    timeout_s = prune_cfg.get("timeout_s")

    for group in grouped_specs_by_model_device(specs):
        stop_reason = None
        stop_details: dict[str, Any] = {}

        for i, spec in enumerate(group):
            if stop_reason is not None:
                upsert_skip_record(
                    skip_csv,
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "reason": stop_reason,
                        "model_filename": str(spec.model_path),
                        "quant": spec.quant,
                        "devices": spec.device,
                        "n_prompt": spec.n_prompt,
                        "n_gen": spec.n_gen,
                        "jsonl_path": str(spec.jsonl_path),
                        "time_path": str(spec.time_path),
                        "predicted_next_ttft_s": stop_details.get("predicted_next_ttft_s"),
                        "current_ttft_s": stop_details.get("current_ttft_s"),
                        "current_max_rss_mb": stop_details.get("current_max_rss_mb"),
                        "elapsed_s": stop_details.get("elapsed_s"),
                    },
                )
                print(f"[prune] skipped {spec.jsonl_path.stem}: {stop_reason}")
                continue

            result = run_benchmark(
                spec,
                llama_bench_bin=run_cfg.get("llama_bench_bin", "llama-bench"),
                time_bin=run_cfg.get("time_bin", "/usr/bin/time"),
                time_args=run_cfg.get("time_args", ["-l"]),
                force=args.force,
                timeout_s=timeout_s,
            )

            pair = (str(spec.jsonl_path), str(spec.time_path))

            if result["status"] == "timeout":
                stop_reason = "timeout"
                stop_details = {"elapsed_s": result["elapsed_s"]}
                upsert_skip_record(
                    skip_csv,
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "reason": "timeout",
                        "model_filename": str(spec.model_path),
                        "quant": spec.quant,
                        "devices": spec.device,
                        "n_prompt": spec.n_prompt,
                        "n_gen": spec.n_gen,
                        "jsonl_path": str(spec.jsonl_path),
                        "time_path": str(spec.time_path),
                        "predicted_next_ttft_s": None,
                        "current_ttft_s": None,
                        "current_max_rss_mb": None,
                        "elapsed_s": result["elapsed_s"],
                    },
                )
                continue

            if spec.jsonl_path.exists() and spec.time_path.exists():
                if args.force and pair in ingested_pairs:
                    remove_ingested_pair(runs_csv, spec.jsonl_path, spec.time_path)
                    ingested_pairs.remove(pair)

                if pair not in ingested_pairs:
                    ingest_one(str(spec.jsonl_path), str(spec.time_path), str(runs_csv))
                    ingested_pairs.add(pair)
                else:
                    print(f"[ingest] already indexed {spec.jsonl_path.name}")

            row = load_result_row(runs_csv, spec.jsonl_path, spec.time_path)
            if row is None:
                continue

            if i + 1 < len(group):
                next_prompt = group[i + 1].n_prompt
                reason, details = prune_decision(
                    current_row=row,
                    next_prompt=next_prompt,
                    objective=args.objective,
                    constraints=constraint_cfg,
                )
                if reason is not None:
                    stop_reason = reason
                    stop_details = details
                    print(
                        f"[prune] stopping larger prompts for "
                        f"{spec.model_path.name} on {spec.device}: {reason}"
                    )

    subprocess.run([sys.executable, "-m", "bench.plot_all"], check=True)

    write_recommendation(
        runs_csv=runs_csv,
        constraints=constraint_cfg,
        objective=args.objective,
        out_path=recommendation_out,
        topk=int(constraint_cfg.get("topk", 5)),
    )

    if args.cleanup:
        cleanup_models(
            run_config=run_cfg,
            runs_csv=runs_csv,
            constraints=constraint_cfg,
            objective=args.objective,
            keep_top=args.keep_top,
        )


if __name__ == "__main__":
    main()