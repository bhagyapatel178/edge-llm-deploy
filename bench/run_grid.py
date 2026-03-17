from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


QUANT_PATTERNS = [
    "q8_0",
    "q6_k",
    "q5_k_m",
    "q5_k_s",
    "q4_k_m",
    "q4_k_s",
    "q3_k_m",
    "q3_k_s",
    "f16",
    "fp16",
]


@dataclass(frozen=True)
class RunSpec:
    model_path: Path
    quant: str
    device: str
    ngl: int
    n_prompt: int
    n_gen: int
    threads: int
    repetitions: int
    jsonl_path: Path
    time_path: Path


def infer_quant_tag(filename: str) -> str:
    s = filename.lower()

    if "q3_k_m" in s or "q3_k_s" in s:
        return "q3"
    if "q4_k_m" in s or "q4_k_s" in s:
        return "q4"
    if "q5_k_m" in s or "q5_k_s" in s:
        return "q5"
    if "q6_k" in s:
        return "q6"
    if "q8_0" in s:
        return "q8"
    if "f16" in s or "fp16" in s:
        return "f16"

    return Path(filename).stem.lower()


def log_stem(quant: str, device: str, n_prompt: int, n_gen: int) -> str:
    return f"{quant}_{device.lower()}_p{n_prompt}_n{n_gen}"


def build_run_specs(config: dict[str, Any]) -> list[RunSpec]:
    model_dir = Path(config["model_dir"])
    raw_logs_dir = Path(config["raw_logs_dir"])
    raw_logs_dir.mkdir(parents=True, exist_ok=True)

    quants = config["quants"]
    prompt_tokens = config["prompt_tokens"]
    n_gen = int(config.get("n_gen", 128))
    threads = int(config.get("threads", 8))
    repetitions = int(config.get("repetitions", 1))
    devices = config["devices"]

    specs: list[RunSpec] = []
    for quant_filename in quants:
        model_path = model_dir / quant_filename
        quant = infer_quant_tag(quant_filename)

        for device_cfg in devices:
            device_name = str(device_cfg["name"])
            ngl = int(device_cfg["ngl"])

            for p in prompt_tokens:
                stem = log_stem(quant, device_name, int(p), n_gen)
                specs.append(
                    RunSpec(
                        model_path=model_path,
                        quant=quant,
                        device=device_name,
                        ngl=ngl,
                        n_prompt=int(p),
                        n_gen=n_gen,
                        threads=threads,
                        repetitions=repetitions,
                        jsonl_path=raw_logs_dir / f"{stem}.jsonl",
                        time_path=raw_logs_dir / f"{stem}.time",
                    )
                )

    return specs


def logs_exist_and_nonempty(jsonl_path: Path, time_path: Path) -> bool:
    return (
        jsonl_path.exists()
        and time_path.exists()
        and jsonl_path.stat().st_size > 0
        and time_path.stat().st_size > 0
    )


def _cleanup_partial_logs(jsonl_path: Path, time_path: Path) -> None:
    for path in [jsonl_path, time_path]:
        if path.exists() and path.stat().st_size == 0:
            path.unlink(missing_ok=True)


def run_benchmark(
    spec: RunSpec,
    llama_bench_bin: str = "llama-bench",
    time_bin: str = "/usr/bin/time",
    time_args: list[str] | None = None,
    force: bool = False,
    timeout_s: int | None = None,
) -> dict[str, Any]:
    """
    Returns:
      {"status": "ran" | "skipped_existing" | "timeout", "elapsed_s": float | None}
    """
    time_args = time_args or ["-l"]

    if not spec.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {spec.model_path}")

    if not force and logs_exist_and_nonempty(spec.jsonl_path, spec.time_path):
        print(f"[skip] logs already exist for {spec.jsonl_path.stem}")
        return {"status": "skipped_existing", "elapsed_s": None}

    spec.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    spec.time_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        time_bin,
        *time_args,
        llama_bench_bin,
        "-m", str(spec.model_path),
        "-p", str(spec.n_prompt),
        "-n", str(spec.n_gen),
        "-t", str(spec.threads),
        "-ngl", str(spec.ngl),
        "-r", str(spec.repetitions),
        "-dev", spec.device,
        "-o", "jsonl",
    ]

    print("[run]", " ".join(cmd))

    start = time.monotonic()
    with spec.jsonl_path.open("w", encoding="utf-8") as stdout_f, spec.time_path.open("w", encoding="utf-8") as stderr_f:
        proc = subprocess.Popen(
            cmd,
            stdout=stdout_f,
            stderr=stderr_f,
            start_new_session=True,
        )

        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()

            elapsed = time.monotonic() - start
            _cleanup_partial_logs(spec.jsonl_path, spec.time_path)
            print(f"[timeout] {spec.jsonl_path.stem} after {elapsed:.1f}s")
            return {"status": "timeout", "elapsed_s": elapsed}

        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

    elapsed = time.monotonic() - start
    return {"status": "ran", "elapsed_s": elapsed}