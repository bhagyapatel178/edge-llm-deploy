import csv
import json 
import re
import sys 
from datetime import datetime
from pathlib import Path

RSS_RE = re.compile(r"^\s*(\d+)\s+maximum resident set size\s*$")

FIELDS = [
    "timestamp_utc",
    "test_time_utc",
    "model_type",
    "model_filename",
    "quant",
    "devices",
    "backends",
    "n_threads",
    "n_gpu_layers",
    "n_prompt",
    "n_gen",
    "pp_tok_s",
    "tg_tok_s",
    "ttft_proxy_s",
    "max_rss_mb",
    "build_commit",
    "build_number",
    "jsonl_path",
    "time_path",
]

def parse_time_max_rss_bytes(time_text:str) -> int:
    for line in time_text.splitlines():
        m = RSS_RE.match(line)
        if m:
            return int(m.group(1))
    raise ValueError("Could not find 'maximum resident set size' in time output")

def infer_quant(model_filename:str) -> str:
    s = model_filename.lower()
    for q in ["q8_0", "q6_k", "q4_k_m", "q3_k_m", "f16", "fp16"]:
        if q in s:
            return q
    return "unknown"

def read_jsonl(path:Path):
    rows = [] 
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No JSONL rows in {path}")
    return rows

def main(jsonl_path: str, time_path:str, out_csv:str = "results/runs.csv"):
    jsonl_path = Path(jsonl_path)
    time_path = Path(time_path)

    rows = read_jsonl(jsonl_path)
    time_text = time_path.read_text(encoding="utf-8", errors="ignore")
    max_rss_bytes = parse_time_max_rss_bytes(time_text)
    max_rss_mb = max_rss_bytes/ 1024 / 1024

    # find pp (Prompt Processing) and tg (Token Generation) entries
    pp = next((r for r in rows if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0), None)
    tg = next((r for r in rows if r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0), None)
    if pp is None or tg is None:
        raise ValueError("Could not find both pp (n_prompt>0) and tg (n_gen>0) entries in JSONL.")
    
    pp_tps = float(pp["avg_ts"])
    tg_tps = float(tg["avg_ts"])
    n_prompt = int(pp["n_prompt"])
    n_gen = int(tg["n_gen"])

    ttft_proxy = (n_prompt / pp_tps) + (1.0 / tg_tps)

    record = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "test_time_utc": tg.get("test_time", pp.get("test_time", "")),
        "model_type": pp.get("model_type", ""),
        "model_filename": pp.get("model_filename", ""),
        "quant": infer_quant(pp.get("model_filename", "")),
        "devices": pp.get("devices", ""),
        "backends": pp.get("backends", ""),
        "n_threads": pp.get("n_threads", ""),
        "n_gpu_layers": pp.get("n_gpu_layers", ""),
        "n_prompt": n_prompt,
        "n_gen": n_gen,
        "pp_tok_s": f"{pp_tps:.2f}",
        "tg_tok_s": f"{tg_tps:.2f}",
        "ttft_proxy_s": f"{ttft_proxy:.3f}",
        "max_rss_mb": f"{max_rss_mb:.1f}",
        "build_commit": pp.get("build_commit", ""),
        "build_number": pp.get("build_number", ""),
        "jsonl_path": str(jsonl_path),
        "time_path": str(time_path),
    }

    out_csv = Path(out_csv)
    write_header = not out_csv.exists()
    out_csv.parent.mkdir(parents=True, exist_ok = True)

    with out_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(record)
    
    print(f"Appended 1 run to {out_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python bench/ingest_jsonl_time.py <run.jsonl> <run.times>")
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2])
