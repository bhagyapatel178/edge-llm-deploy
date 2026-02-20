# AI in a Box — Exploring Edge LLM Deployment, Benchmarking, and Mixed-Precision Compression

## Project theme
This final-year project explores practical ways to **run large language models (LLMs) locally under edge constraints** (limited memory, latency targets, and hardware variability), and to **optimise models for on-device inference** while preserving usability and performance.

The work is split into two complementary parts:

- **Part 1: A portable benchmarking + orchestration + recommendation toolkit** that evaluates multiple GGUF model variants and runtime settings on a target device (e.g., Mac CPU vs Metal; later Raspberry Pi 5) and recommends the best configuration for user-defined constraints (e.g., “must fit in 8GB RAM”, “TTFT < 2s”, “≥ 10 tok/s”, “context length = 2k”).
- **Part 2: A proposed mixed-precision GGUF compression framework** (a “GGUF mixer”), inspired by the motivation of approaches like AWQ (preserving quality by selectively keeping sensitive components at higher precision), but implemented at the level of **selecting layers/tensors** to keep higher precision while quantising the remainder for edge compatibility.

I do **not** build a new inference runtime. Instead, I use an existing runtime (**llama.cpp**) as the execution engine and contribute the **evaluation + optimisation frameworks** around it:
- candidate variant selection (download / produce GGUF variants using existing tools)
- controlled benchmark execution
- structured logging to CSV
- analysis + plots
- constraint-driven recommendation
- (Part 2) mixed-precision model composition and evaluation

## Deliverables
### Part 1 (core): Edge LLM benchmarking + recommendation toolkit
**Inputs**
- Base model (GGUF variants such as FP16 / Q8 / Q6 / Q4 / Q3)
- Hardware profile (Mac CPU-only vs Mac Metal; vs later Raspberry Pi 5 CPU)
- Constraints/targets (RAM ceiling, TTFT, tokens/sec, context length, etc.)

**Outputs**
- Recommended deployment configuration (model variant + runtime flags)
- CSV results + plots (TTFT proxy, tok/s, peak RSS, context scaling)
- Optional: selected GGUF variant (downloaded or produced via existing tools)

### Part 2 (extension): Mixed-precision GGUF “mixer” (compression framework v1)
Create a mixed-precision GGUF by combining tensors from two GGUFs of the same base model (e.g., FP16 + Q4) according to a policy that keeps selected components at higher precision while the rest remains quantised.

**Goal**
- Improve the edge trade-off between **quality / latency / memory / compatibility** compared to “vanilla” uniform quantisation (e.g., Q4 everywhere), using the same benchmarking harness from Part 1 for evaluation.

## Repo structure
- `bench/` — Python benchmark runner, parsers, plotting, recommendation logic
- `configs/` — hardware profiles, constraint sets, experiment grids
- `scripts/` — setup + one-command quickstart orchestration
- `models/` — GGUF models (ignored by git)
- `results/` — logs, CSV, plots (ignored by git)
- `prompts/` — benchmark prompts (short/long)

## Setup (Mac)
### 1) Install llama.cpp runtime (Brew)
```bash
brew install llama.cpp
llama-bench --help
llama-cli --version