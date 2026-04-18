# porchbench

[![PyPI version](https://img.shields.io/pypi/v/porchbench.svg)](https://pypi.org/project/porchbench/)
[![Python versions](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/porchbench/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/mmdevelops/porchbench/blob/main/LICENSE)

Rigorous quality benchmarking for local LLMs. Measures what actually matters when choosing between models, quantization levels, and prompt strategies on your own hardware — with real statistics, not vibes.

Most local LLM benchmarks measure tokens/sec. Model cards report scores on standard academic benchmarks under ideal conditions. Neither tells you whether Qwen 14B Q4 or Qwen 8B Q8 is the better choice for *your* GPU and *your* workload. porchbench does.

## What it does

- **Benchmark** — run curated prompt suites against any Ollama model and capture structured results (quality + throughput + full metadata)
- **Evaluate** — score responses with an LLM judge using rubric-based criteria anchored by worked calibration examples
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

# Run the coding benchmark — suites ship with the package, reference by name
porchbench run --suite coding-basics --model qwen2.5:3b
```

That's it. A summary table prints to your terminal, and a structured JSON is written to `results/`:

```
results/<timestamp>_<suite>_<model>.json
  e.g. results/2026-04-16T21-29-46_coding-basics_qwen2.5-3b.json
```

Every later step (`evaluate`, `compare`, `leaderboard`) reads these files. If you don't want to type the filename, pass no `--result` flag to any command and you'll get an interactive picker.

### Evaluate quality with an LLM judge

```bash
# Score results using a local Ollama model as judge (default: gemma4:e4b).
# --rubric is auto-resolved from the result file if omitted.
porchbench evaluate --result results/<your-result-file>.json
```

Scorecards are written to `scorecards/` as `<timestamp>_<run-id-prefix>.json`.

### Compare models

```bash
# Run the same suite on multiple models
porchbench run --suite coding-basics --model qwen2.5:3b --model qwen2.5:7b

# Compare results side-by-side
porchbench compare \
  --result results/<model-a-result>.json \
  --result results/<model-b-result>.json
```

### Rank models in a leaderboard

Once you have multiple scorecards in `scorecards/`, rank them on a single rubric:

```bash
# Auto-scans scorecards/ for comparable entries (same rubric)
porchbench leaderboard

# Or point at specific scorecards
porchbench leaderboard --scorecard scorecards/<a>.json --scorecard scorecards/<b>.json
```

Pass `--strict` to require the same evaluator model, not just the same rubric.

> The leaderboard ranks by weighted mean only — it does not run a significance test or attach CIs to the ranking. To judge whether a gap between two models is real, run `porchbench compare` on their underlying result files; that path produces a paired test with a bootstrap CI and a Cohen's dz effect size.

### Discover routing opportunities

Does adapting your prompt strategy to model size actually help?

```bash
# Run every prompt x strategy x model combination
porchbench routes discover \
  --suite routing-discovery \
  --model qwen2.5:3b --model qwen2.5:7b

# Analyze the results
porchbench routes analyze \
  --result results/<discovery-result-1>.json \
  --result results/<discovery-result-2>.json
```

### Profile your hardware

```bash
porchbench profile --model qwen2.5:3b --model qwen2.5:7b
```

Measures model load/unload times, VRAM footprint, and co-residency capacity — the data you need to decide whether model routing is worth the swap overhead on your system.

### Run everything overnight

Queue up a full benchmark run before bed or work. porchbench auto-discovers suites, detects which ones need routing discovery vs standard runs, checks your GPU is working, and handles errors without stopping.

```bash
porchbench overnight -m gemma4:e4b -m qwen3:8b -m phi4:14b --repeats 3
```

Add `--profile` to measure model load times and VRAM first, or `--yes` to skip the confirmation prompt for fully unattended runs.

## Benchmark suites

Suites ship bundled with the package and are referenced by name:

| Suite | Prompts | What it tests |
|-------|---------|---------------|
| `coding-basics` | 28 | Implementation quality across 3 difficulty tiers — not just "does it compile" but design, idiom, edge-case handling |
| `cross-domain` | 22 | Science problems requiring both Python implementation and domain reasoning (security, biology, physics, math) |
| `routing-discovery` | 92 | Prompt strategy x model scale interactions with 5 strategies (universal, brevity, direct, chain-of-thought, structured) |
| `tool-use` | 19 | Agent-style tasks with sandboxed code execution, scored by outcome state |

**Customizing or adding your own:** drop a YAML file in `./suites/` next to where you run `porchbench` and it automatically overrides the packaged copy with the same name (or adds a new one). Same pattern for `./rubrics/`. You can also pass an explicit path: `--suite ./my-suite.yaml`.

## What makes this different

**It's not another MMLU wrapper.** Standard benchmarks tell you "model X scores 85% on MMLU." That doesn't help you decide between two quantization levels on your 24GB GPU. porchbench treats the deployment context — quantization, VRAM budget, model swap time, prompt strategy — as first-class experimental variables.

**Statistical rigor for local eval.** Paired comparisons (question-level deltas, not independent point estimates), bootstrap confidence intervals, Cohen's dz effect sizes, calibration-anchored judge prompts, and contamination tagging. Repeat runs at temperature=0 to detect floating-point non-determinism across quantization levels.

**Reproducibility built in.** Every result captures model SHA, suite SHA, Ollama version, quantization level, KV cache type, and full generation parameters. Same inputs, same outputs, verifiable later.

See [METHODOLOGY.md](https://github.com/mmdevelops/porchbench/blob/main/docs/reference/METHODOLOGY.md) for the full statistical framework and academic references.

## Project structure

```
examples/           Sample results you can feed into porchbench compare/evaluate
results/            Run outputs (JSON, gitignored)
scorecards/         Evaluation scorecards (JSON, gitignored)
src/porchbench/          Python package
  cli.py            CLI entry point (typer + rich + beaupy pickers)
  interactive.py    Interactive model/suite/result pickers
  assets.py         Asset resolver: cwd overrides → packaged defaults
  backend.py        Inference backend abstraction (Ollama, OpenAI-compat)
  runner.py         Async benchmark execution
  suite.py          Suite + prompt loading and validation
  evaluator.py      LLM-as-judge scoring with debiasing
  routing.py        Routing discovery and analysis
  profiler.py       Hardware and model profiling
  compare.py        Side-by-side comparison rendering
  leaderboard.py    Cross-scorecard ranking
  overnight.py      Unattended multi-suite orchestration
  metrics.py        Throughput and latency metrics
  statistics.py     Bootstrap CIs, paired comparisons, effect sizes
  schemas.py        Pydantic models for all I/O
  sandbox/          Sandboxed code execution for tool-use suites
  harness/          Prompt rendering and response parsing
  data/             Bundled assets (shipped inside the wheel)
    suites/         Benchmark prompt suites (YAML)
    rubrics/        LLM-as-judge scoring rubrics
```

## Documentation

- [METHODOLOGY.md](https://github.com/mmdevelops/porchbench/blob/main/docs/reference/METHODOLOGY.md) — statistical framework, evaluation methods, debiasing, reproducibility standards
- [CHANGELOG.md](https://github.com/mmdevelops/porchbench/blob/main/CHANGELOG.md) — release notes and known limitations
- [docs/skills/](https://github.com/mmdevelops/porchbench/tree/main/docs/skills) — Claude Code skill templates for using your subscription as an evaluation backend

## Configuration

Defaults work out of the box. To override them persistently, copy `.env.example` to `.env` and edit:

| Variable | Purpose |
|----------|---------|
| `OLLAMA_HOST` | Ollama server URL (default `http://localhost:11434`) |
| `OLLAMA_KV_CACHE_TYPE` | KV cache quantization passed through to Ollama (e.g. `q8_0`) |
| `PORCHBENCH_BACKEND` | Inference backend for `porchbench run`: `ollama` (default) or `openai-compat` |
| `PORCHBENCH_BASE_URL` | OpenAI-compatible server URL (when `PORCHBENCH_BACKEND=openai-compat`) |
| `PORCHBENCH_API_KEY` | API key for the OpenAI-compatible server |
| `PORCHBENCH_EVAL_BACKEND` | Judge backend: `ollama` (default), `api`, or `claude-code` |
| `PORCHBENCH_EVAL_MODEL` | Judge model override (defaults differ per backend) |
| `ANTHROPIC_API_KEY` | Required only when `PORCHBENCH_EVAL_BACKEND=api` |
| `PORCHBENCH_SEED` | RNG seed for bootstrap CIs in `porchbench compare` (default `42`). Override to probe sensitivity. |

CLI flags always take precedence over env vars. See `porchbench <command> --help` for per-command overrides.

## Troubleshooting

**`Connection refused` or `cannot connect to Ollama`** — Ollama isn't running. Start it with `ollama serve` (or the Ollama desktop app) and retry. For a remote instance, set `OLLAMA_HOST=http://host:11434` or pass `--host`.

**`model 'X' not found`** — pull it first: `ollama pull X`. `porchbench` does not auto-pull; this keeps runs reproducible.

**`porchbench: command not found`** — the package installed but the entry point isn't on `PATH`. Re-run `pip install -e .` inside the project's venv, or invoke via `python -m porchbench.cli`.

**Interactive picker shows no options** — either Ollama has no pulled models (`ollama list` to check) or your `suites/` / `results/` directory is empty. You can always pass `--model` / `--suite` / `--result` explicitly.

**AMD / ROCm: kernel errors on gfx1201 (RDNA 4)** — a rocblas override usually fixes most models. Some quantized models (notably parts of the Qwen 3.5 family) hit a missing `SOLVE_TRI` kernel upstream; fall back to a different model family until ROCm ships the fix.

**Results file not where you expected** — output goes to `--output-dir` (default `results/`). Check the command's `--help` for the relevant flag.

## Requirements

- Python 3.11+
- Ollama running locally (or specify `--host` for a remote instance)
- For API-based evaluation: `pip install -e ".[api]"` (adds Anthropic SDK)

## License

Apache 2.0 — see [LICENSE](https://github.com/mmdevelops/porchbench/blob/main/LICENSE) for details.
