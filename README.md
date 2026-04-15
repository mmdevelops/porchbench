# feral

Rigorous quality benchmarking for local LLMs. Measures what actually matters when choosing between models, quantization levels, and prompt strategies on your own hardware — with real statistics, not vibes.

Most local LLM benchmarks measure tokens/sec. Model cards report scores on standard academic benchmarks under ideal conditions. Neither tells you whether Qwen 14B Q4 or Qwen 8B Q8 is the better choice for *your* GPU and *your* workload. feral does.

## What it does

- **Benchmark** — run curated prompt suites against any Ollama model and capture structured results (quality + throughput + full metadata)
- **Evaluate** — score responses with an LLM judge using rubric-based criteria and debiasing controls
- **Compare** — side-by-side model comparison with paired statistics
- **Route** — discover whether adapting prompt strategies to model size beats always using the biggest model
- **Profile** — measure model load times, VRAM usage, swap costs, and co-residency on your hardware

## Quick start

**Prerequisites:** [Ollama](https://ollama.com) installed and running, at least one model pulled.

```bash
# Install
pip install -e .

# Pull a model if you haven't already
ollama pull qwen2.5:3b

# Run the coding benchmark
feral run --suite suites/coding-basics.yaml --model qwen2.5:3b
```

That's it. Results are written to `results/` as structured JSON, and a summary table prints to your terminal.

### Evaluate quality with an LLM judge

```bash
# Score results using a local Ollama model as judge
feral evaluate \
  --result results/<your-result-file>.json \
  --rubric rubrics/default.yaml \
  --evaluator gemma3:12b
```

### Compare models

```bash
# Run the same suite on multiple models
feral run --suite suites/coding-basics.yaml --model qwen2.5:3b --model qwen2.5:7b

# Compare results side-by-side
feral compare \
  --result results/<model-a-result>.json \
  --result results/<model-b-result>.json
```

### Discover routing opportunities

Does adapting your prompt strategy to model size actually help?

```bash
# Run every prompt x strategy x model combination
feral discover-routes \
  --suite suites/routing-discovery.yaml \
  --model qwen2.5:3b --model qwen2.5:7b

# Analyze the results
feral analyze-routes \
  --result results/<discovery-result-1>.json \
  --result results/<discovery-result-2>.json
```

### Profile your hardware

```bash
feral profile --model qwen2.5:3b --model qwen2.5:7b
```

Measures model load/unload times, VRAM footprint, and co-residency capacity — the data you need to decide whether model routing is worth the swap overhead on your system.

### Run everything overnight

Queue up a full benchmark run before bed or work. Feral auto-discovers suites, detects which ones need routing discovery vs standard runs, checks your GPU is working, and handles errors without stopping.

```bash
feral overnight -m gemma4:e4b -m qwen3:8b -m phi4:14b --repeats 3
```

Add `--profile` to measure model load times and VRAM first, or `--yes` to skip the confirmation prompt for fully unattended runs.

## Benchmark suites

| Suite | Prompts | What it tests |
|-------|---------|---------------|
| `coding-basics.yaml` | 32 | Implementation quality across 3 difficulty tiers — not just "does it compile" but design, idiom, edge-case handling |
| `cross-domain.yaml` | 22 | Science problems requiring both Python implementation and domain reasoning (security, biology, physics, math) |
| `routing-discovery.yaml` | 100+ | Prompt strategy x model scale interactions with 5 strategies (universal, brevity, direct, chain-of-thought, structured) |
| `tool-use.yaml` | — | Agent-style tasks with sandboxed code execution, scored by outcome state |

## What makes this different

**It's not another MMLU wrapper.** Standard benchmarks tell you "model X scores 85% on MMLU." That doesn't help you decide between two quantization levels on your 24GB GPU. feral treats the deployment context — quantization, VRAM budget, model swap time, prompt strategy — as first-class experimental variables.

**Statistical rigor for local eval.** Paired comparisons (question-level deltas, not independent point estimates), bootstrap confidence intervals, Cohen's d effect sizes, and contamination tagging. Repeat runs at temperature=0 to detect floating-point non-determinism across quantization levels.

**Reproducibility built in.** Every result captures model SHA, suite SHA, Ollama version, quantization level, KV cache type, and full generation parameters. Same inputs, same outputs, verifiable later.

See [METHODOLOGY.md](docs/reference/METHODOLOGY.md) for the full statistical framework and academic references.

## Project structure

```
suites/          Benchmark prompt suites (YAML)
rubrics/         Evaluation rubrics for LLM-as-judge scoring
examples/        Sample results you can feed into feral compare/evaluate
results/         Run outputs (JSON, gitignored)
scorecards/      Evaluation scorecards (JSON, gitignored)
src/feral/       Python package
  cli.py         CLI entry point (typer + rich)
  overnight.py   Unattended multi-suite orchestration
  runner.py      Async benchmark execution
  evaluator.py   LLM-as-judge scoring with debiasing
  routing.py     Routing discovery and analysis
  profiler.py    Hardware and model profiling
  statistics.py  Bootstrap CIs, paired comparisons, effect sizes
  schemas.py     Pydantic models for all I/O
```

## Documentation

- [METHODOLOGY.md](docs/reference/METHODOLOGY.md) — statistical framework, evaluation methods, debiasing, reproducibility standards
- [docs/skills/](docs/skills/) — Claude Code skill templates for using your subscription as an evaluation backend

## Requirements

- Python 3.11+
- Ollama running locally (or specify `--host` for a remote instance)
- For API-based evaluation: `pip install -e ".[api]"` (adds Anthropic SDK)

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
