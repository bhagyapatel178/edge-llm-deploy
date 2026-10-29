# AI in a Box — Exploring Edge LLM Deployment, Benchmarking, and Mixed-Precision Compression

## Project theme
This final-year project investigates practical ways to run large language models (LLMs) locally under edge constraints such as limited memory, latency targets, and hardware variability, and to improve the quality–efficiency trade-off of on-device inference without introducing a new runtime.

The project is split into two complementary parts:

- **Part 1: A portable benchmarking, orchestration, and recommendation toolkit** for GGUF-based local inference using `llama.cpp`. It evaluates multiple quantised model variants and runtime settings on a target device, logs structured benchmark results, generates plots, and recommends a deployment configuration under user-defined constraints such as memory budget, TTFT target, throughput floor, and context length requirement.

- **Part 2: A tensor-level mixed-precision GGUF framework** that constructs runnable mixed models by combining tensors from a lower-precision GGUF and a higher-precision GGUF of the same base model. The framework was evaluated primarily on `Qwen2.5-3B-Instruct`, progressing from simple structural heuristics to an activation-aware, budgeted tensor-selection strategy.

The project does **not** build a new inference runtime. Instead, it uses **`llama.cpp`** as the execution engine and contributes the evaluation, orchestration, recommendation, and mixed-precision model-construction layers around it.

## Reports

- 📄 **[Final Report (PDF)](./reports/final-report.pdf)**
- 📊 **[Presentation Slides (PDF)](./reports/presentation-slides.pdf)**

## Final project outcome
### Part 1
Part 1 was validated across:
- MacBook Pro (`MTL0` / Metal-accelerated)
- MacBook Pro (`BLAS` / CPU-only)
- Raspberry Pi 5 (`BLAS` / CPU-only)
- an additional model-family portability sanity test on `Llama-3.2-1B-Instruct-GGUF` using configuration changes only

Outputs include:
- structured raw benchmark logs
- unified CSV results
- plots of throughput, TTFT proxy, memory usage, and context scaling
- scenario-specific deployment recommendations

### Part 2
Part 2 produced runnable mixed-precision GGUF models compatible with the same `llama.cpp` evaluation stack. The strongest results showed that selective tensor promotion can recover a large fraction of the quality lost under low-bit quantisation without paying the full storage cost of moving to a uniformly higher-precision model.

## Deliverables
### Part 1: Edge LLM benchmarking + recommendation toolkit
**Inputs**
- base model family in GGUF form
- candidate quantised variants (e.g. `Q8_0`, `Q6_K`, `Q4_K_M`, `Q3_K_M`)
- hardware/runtime profile
- deployment constraints and workload description

**Outputs**
- recommended deployment configuration
- CSV benchmark results
- plots for throughput, TTFT proxy, peak RSS, and context scaling
- recommendation JSON

### Part 2: Mixed-precision GGUF framework
**Inputs**
- compatible low-precision and higher-precision GGUF files of the same base model
- tensor-selection policy (heuristic or activation-aware)
- optional additional storage budget

**Outputs**
- runnable mixed GGUF model
- recipe / selected tensor list
- perplexity and runtime evaluation against uniform-quant baselines

**Goal**
- Improve the edge trade-off between **quality / latency / memory / compatibility** compared to “vanilla” uniform quantisation (e.g., Q4 everywhere), using the same benchmarking harness from Part 1 for evaluation.

## Repo structure
- `bench/` — benchmark orchestration, ingestion, plotting, recommendation
- `mixer/` — GGUF mixing logic and mixed-precision tooling
- `configs/` — run configs, device profiles, recommendation constraints, mixer configs
- `models/` — GGUF models (gitignored)
- `results/` — raw logs, CSVs, plots, recommendations (gitignored)
- `prompts/` — prompts and evaluation text used by benchmarking / perplexity workflows
- `reports/` — final report and presentation slides (PDF)

## Setup (Mac)
### 1) Install llama.cpp runtime (Brew)
```bash
brew install llama.cpp
llama-bench --help
llama-cli --version

## Example commands
### Part 1
python -m bench.pipeline --config configs/run.yaml --constraints configs/edge_8gb.yaml --objective balanced

### Part 2
python -m mixer.mix_gguf --config configs/mixer/qwen25_3b_last2block_q8.yaml --verbose 