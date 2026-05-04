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
pip install porchbench

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
# Score results using a local Ollama model as judge.
# --rubric is auto-resolved from the result file if omitted.
porchbench evaluate --result results/<your-result-file>.json
```

The first time you run an Ollama-backed eval without `--evaluator`, porchbench prompts you to pick a judge from your locally-pulled models and offers to save the choice as `PORCHBENCH_EVAL_MODEL` in `./.env`. Subsequent runs skip the prompt. Override anytime with `--evaluator <name>` or by editing `.env`. Cloud backends (`--backend api`, `--backend claude-code`) use stable defaults (`claude-sonnet-4-6`, `sonnet`) without prompting.

Scorecards are written to `scorecards/` as `<timestamp>_<run-id-prefix>.json`.

To collapse run + score into one step, pass `--evaluate` to `porchbench run`. All inference completes first, then the judge model loads once and scores every result in a single post-phase batch — no swap thrashing between target model and judge. Compose with `--eval-backend ollama|api|claude-code` and `--eval-model <name>` to override defaults. With `--yes` (unattended), you must pass `--eval-model` or set `PORCHBENCH_EVAL_MODEL` — the picker won't fire.

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

A **strategy** is a system-message wrapper defined in the suite YAML (e.g. `cot` prepends "Think step by step", `direct` prepends "Respond with only the final answer"). The `routing-discovery` suite ships with five: `universal`, `brevity`, `direct`, `cot`, `structured`. The `--strategies` flag on `run` runs every prompt through every strategy so `analyze-routes` can find cases where a smaller model paired with the right strategy beats a larger model's default.

```bash
# Run every prompt x strategy combination against the selected models
porchbench run --strategies \
  --suite routing-discovery \
  --model qwen2.5:3b --model qwen2.5:7b

# Analyze the results
porchbench analyze-routes \
  --result results/<discovery-result-1>.json \
  --result results/<discovery-result-2>.json
```

> **Migrating from v0.0.x routes commands?** `routes discover` was consolidated into `run --strategies` (same matrix expansion, now on the unified benchmark command). `routes analyze` was promoted to top-level `analyze-routes`. Old invocations print a one-line breadcrumb pointing at the new commands.

### Profile your hardware

```bash
porchbench profile --model qwen2.5:3b --model qwen2.5:7b
```

Measures model load/unload times, VRAM footprint, and co-residency capacity — the data you need to decide whether model routing is worth the swap overhead on your system.

### Run everything overnight

`run` accepts multiple suites — repeat `-s` to queue a full benchmark before bed or work. porchbench runs every suite as a straight benchmark by default (one row per prompt), checks your GPU is working, and handles errors without stopping. Add `--strategies` to expand strategies-bearing suites into the prompt × strategy × model matrix instead of running them as a baseline.

```bash
porchbench run -s coding-basics -s tool-use -s cross-domain \
  -m gemma4:e4b -m qwen3:8b -m phi4:14b --repeats 3
```

Add `--profile` to measure model load times and VRAM first, or `--yes` for fully unattended mode (skips the eval-model picker fallback if no judge is configured). Add `--evaluate` to chain LLM-as-judge scoring as a single post-phase after all inference completes — pick the judge backend with `--eval-backend ollama|api|claude-code` (defaults to local ollama). Running eval as a post-phase keeps the judge model loaded once for the whole batch instead of swapping between target and judge on every run. With `--evaluate` you wake up to scorecards, not just raw results.

Add `--resume` to skip `<suite, model, repeat>` triples whose result JSON is already in `results/`. Only completed runs are restored — a session interrupted mid-run loses its in-progress prompts because results write at the end of each run. Use it after a crash, an OOM, or an explicit Ctrl-C to pick up where the queue left off without re-paying for everything that already finished.

> **Migrating from v0.0.x?** `porchbench overnight ...` consolidated into `porchbench run ...` (every flag is supported). `routes discover` consolidated into `run --strategies`. `routes analyze` was renamed to top-level `analyze-routes`. Old invocations print a one-line breadcrumb pointing at the new commands.

## Benchmark suites

Suites ship bundled with the package and are referenced by name:

| Suite | Prompts | What it tests |
|-------|---------|---------------|
| `coding-basics` | 28 | Implementation quality across 3 difficulty tiers — not just "does it compile" but design, idiom, edge-case handling |
| `cross-domain` | 22 | Science problems requiring both Python implementation and domain reasoning (security, biology, physics, math) |
| `routing-discovery` | 92 | Prompt strategy x model scale interactions with 5 strategies (universal, brevity, direct, chain-of-thought, structured) |
| `tool-use` | 19 | Agent-style tasks with sandboxed code execution, scored by outcome state. Ships 4 tool-planning strategies (universal, cot, direct, structured) |

**Which suites benefit from `run --strategies`?** Only suites that define a `strategies:` block in their YAML — currently `routing-discovery` and `tool-use`. The strategy set is chosen per-suite because different domains have different strategy × scale interactions worth measuring: CoT helps reasoning, a "numbered plan" preamble helps tool-use planning, neither would meaningfully differentiate factual recall. `coding-basics` and `cross-domain` intentionally stick to a single default prompt — they answer "how good is this model at X" rather than "which prompt strategy unlocks the smaller model." Without `--strategies`, every suite (including the strategies-bearing ones) runs as a straight benchmark.

**Customizing or adding your own:** drop a YAML file in `./suites/` next to where you run `porchbench` and it automatically overrides the packaged copy with the same name (or adds a new one). Same pattern for `./rubrics/`. You can also pass an explicit path: `--suite ./my-suite.yaml`.

## What makes this different

**It's not another MMLU wrapper.** Standard benchmarks tell you "model X scores 85% on MMLU." That doesn't help you decide between two quantization levels on your 24GB GPU. porchbench treats the deployment context — quantization, VRAM budget, model swap time, prompt strategy — as first-class experimental variables.

**Statistical rigor for local eval.** Paired comparisons (question-level deltas, not independent point estimates), bootstrap confidence intervals, Cohen's dz effect sizes, calibration-anchored judge prompts, and contamination tagging. Repeat runs at temperature=0 to detect floating-point non-determinism across quantization levels.

**Reproducibility built in.** Every result captures model SHA, suite SHA, Ollama version, quantization level, KV cache type, and full generation parameters. Same inputs, same outputs, verifiable later.

See [METHODOLOGY.md](https://github.com/mmdevelops/porchbench/blob/main/docs/reference/METHODOLOGY.md) for the full statistical framework and academic references.

## Project structure

```
examples/              Sample results you can feed into porchbench compare/evaluate
results/               Run outputs (JSON, gitignored)
scorecards/            Evaluation scorecards (JSON, gitignored)
src/porchbench/        Python package
  __main__.py          Entry point for `python -m porchbench`
  cli.py               CLI entry point (typer + rich + beaupy pickers)
  interactive.py       Interactive model/suite/result pickers
  assets.py            Asset resolver: cwd overrides → packaged defaults
  backend.py           Inference backend abstraction (Ollama, OpenAI-compat)
  runner.py            Async benchmark execution
  suite.py             Suite + prompt loading and validation
  evaluator.py         LLM-as-judge scoring with debiasing
  routing.py           Routing discovery and analysis
  profiler.py          Hardware and model profiling
  compare.py           Side-by-side comparison rendering
  leaderboard.py       Cross-scorecard ranking
  overnight.py         Unattended multi-suite orchestration
  doctor.py            Environment diagnostics — `porchbench doctor`
  metrics.py           Throughput and latency metrics
  statistics.py        Bootstrap CIs, paired comparisons, effect sizes
  schemas.py           Pydantic models for all I/O
  errors.py            Typed exception hierarchy
  tool_runner.py       Tool-use prompt execution driver
  sandbox/             Sandboxed code execution for tool-use suites
  harness/             Prompt rendering and response parsing
  data/                Bundled assets (shipped inside the wheel)
    suites/            Benchmark prompt suites (YAML)
    rubrics/           LLM-as-judge scoring rubrics
```

## Documentation

- [METHODOLOGY.md](https://github.com/mmdevelops/porchbench/blob/main/docs/reference/METHODOLOGY.md) — statistical framework, evaluation methods, debiasing, reproducibility standards
- [CHANGELOG.md](https://github.com/mmdevelops/porchbench/blob/main/CHANGELOG.md) — release notes and known limitations
- [docs/skills/](https://github.com/mmdevelops/porchbench/tree/main/docs/skills) — Claude Code skill templates for using your subscription as an evaluation backend

## Configuration

Defaults work out of the box. Override any of these via shell export or a project-local `.env` file next to where you run `porchbench`:

| Variable | Purpose |
|----------|---------|
| `OLLAMA_HOST` | Ollama server URL (default `http://localhost:11434`) |
| `OLLAMA_KV_CACHE_TYPE` | KV cache quantization passed through to Ollama (e.g. `q8_0`) |
| `PORCHBENCH_BACKEND` | Inference backend for `porchbench run`: `ollama` (default) or `openai-compat` |
| `PORCHBENCH_BASE_URL` | OpenAI-compatible server URL (when `PORCHBENCH_BACKEND=openai-compat`) |
| `PORCHBENCH_API_KEY` | API key for the OpenAI-compatible server |
| `PORCHBENCH_EVAL_BACKEND` | Judge backend: `ollama` (default), `api`, or `claude-code` |
| `PORCHBENCH_EVAL_MODEL` | Judge model override. Cloud backends have stable defaults; for ollama, the first `--evaluate` prompts you to pick and persists your choice here in `.env`. |
| `ANTHROPIC_API_KEY` | Required only when `PORCHBENCH_EVAL_BACKEND=api` |
| `PORCHBENCH_SEED` | RNG seed for bootstrap CIs in `porchbench compare` (default `42`). Override to probe sensitivity. |

CLI flags always take precedence over env vars. See `porchbench <command> --help` for per-command overrides.

## Platform support

porchbench runs anywhere Ollama runs. GPU detection and VRAM polling try `nvidia-smi` first, then `rocm-smi`, with a graceful fallback via Ollama's `/api/ps`. No vendor-specific code path gates core functionality.

**Validation status:**

- **AMD ROCm (tested on RDNA 4, gfx1201):** actively developed and exercised here — see the ROCm note in Troubleshooting for a known gfx1201 workaround.
- **NVIDIA (CUDA):** implemented and expected to work, but not validated by the author, who does not currently have an NVIDIA rig. Bug reports from NVIDIA users are especially welcome.
- **Apple Silicon / CPU-only:** inherits Ollama's Metal/CPU backends; untested.

**Verifying your setup:** run `porchbench doctor` to check Python, Ollama, GPU detection, VRAM sampler selection, and package install state. When filing a bug, attach `porchbench doctor --json` output for fast triage.

**Shell completion:** run `porchbench --install-completion` to install tab completion for bash, zsh, fish, or PowerShell.

## Troubleshooting

**`Connection refused` or `cannot connect to Ollama`** — Ollama isn't running. Start it with `ollama serve` (or the Ollama desktop app) and retry. For a remote instance, set `OLLAMA_HOST=http://host:11434` or pass `--host`.

**`model 'X' not found`** — pull it first: `ollama pull X`. `porchbench` does not auto-pull; this keeps runs reproducible.

**`porchbench: command not found`** — the package installed but the entry point isn't on `PATH`. Re-run `pip install porchbench` inside the active venv (or `pip install -e .` from a checkout), or invoke via `python -m porchbench`.

**Commands return silently with no output (Windows PowerShell)** — open a new PowerShell window, re-activate your venv, and re-run. Rich-CLI tools that mix interactive pickers and live progress bars can occasionally leave PowerShell's terminal in a half-restored state after an aborted picker or a quirky exit; subsequent commands write output that the terminal swallows. A fresh shell resets the state. Same class of issue affects `uv`, `gh`, and other rich-CLI tools on Windows; not specific to porchbench.

**Interactive picker shows no options** — your `results/` directory is empty (for `evaluate` / `compare` / `leaderboard`) or you have no `suites/` directory and the packaged suites failed to resolve. You can always pass `--model` / `--suite` / `--result` explicitly. (For the model picker, an empty server now exits with an `ollama pull` hint rather than showing a blank picker.)

**AMD / ROCm: kernel errors on gfx1201 (RDNA 4)** — a rocblas override usually fixes most models. Some quantized models (notably parts of the Qwen 3.5 family) hit a missing `SOLVE_TRI` kernel upstream; fall back to a different model family until ROCm ships the fix.

**Reasoning-mode models run slow by default** — Qwen 3, DeepSeek-R1, and other models with a thinking mode emit `<think>...</think>` reasoning tokens before every answer unless explicitly disabled. Porchbench strips thinking tags before judging, but still pays the full generation cost per prompt. The fastest way to disable for a single run is the `--set` override:

```bash
porchbench run --suite coding-basics --model qwen3:8b --set think=false
```

`--set KEY=VALUE` is repeatable and layers over the suite's `defaults.options` (`--set think=false --set num_ctx=8192` is fine). Values parse as YAML so booleans, ints, and `null` round-trip to the right Python types. The override is captured in each `PromptResult.options_used` so the run JSON reflects what actually executed.

For a more permanent change, set `think: false` in your suite's `defaults.options` block instead:

```yaml
defaults:
  options:
    temperature: 0
    num_predict: 2048
    think: false   # disable reasoning tokens for thinking-capable models
```

Applies only to the Ollama backend; OpenAI-compatible servers ignore the field.

Not every thinking-model family honors `think: false`. Qwen 3 and DeepSeek-R1 respect the flag; some other families (e.g. LFM2.5-thinking) ignore it and continue emitting `<think>` blocks regardless. If you set `think: false` and still see prompts scored zero with the "truncated before answer emitted" diagnostic, the model's family likely doesn't honor the flag — raise `num_predict` to give it room to close the thinking trace, or pick a non-thinking variant if benchmark wall-clock matters.

**Results file not where you expected** — output goes to `--output-dir` (default `results/`). Check the command's `--help` for the relevant flag.

## Methodology notes

**Determinism is best-effort on GPU.** porchbench sets `temperature: 0` and `seed: 42` by default, which gives bit-identical outputs across repeats *in the steady state*. However, GPU floating-point reductions do not guarantee the same order across cold- vs warm-kernel states. On ROCm (and to a lesser extent CUDA), the first inference after a model (re)load can produce subtly different logits than subsequent inferences — enough to flip greedy token selection on near-tied probabilities. In practice this manifests as "repeat 1 differs slightly from repeats 2, 3, …" on a fresh model load, with later repeats converging.

This is not a porchbench bug; it's inherent to parallel GPU inference. Treat it as real measurement noise. `porchbench compare` uses paired bootstrap CIs (`seed=42` for the bootstrap RNG; override via `PORCHBENCH_SEED`), which handle cross-repeat variance honestly — wider intervals when the signal is noisier. If you need strict steady-state numbers (e.g., pure speed benchmarks), run a throwaway warmup pass before your measured repeats.

**Truncated / empty responses score zero, not skipped.** When a model exhausts `num_predict` before producing any user-facing content — common with reasoning-mode models emitting long `<think>` blocks — the evaluator records a zero score with a diagnostic rationale rather than silently excluding the prompt from aggregates. This keeps cross-model comparisons honest: a model that fails to answer 8 prompts shouldn't outscore a model that answers all 28. If you see prompts scored zero for "truncated before answer emitted", bump `num_predict` or set `think: false` in your suite's `defaults.options`.

## Requirements

- Python 3.11+
- Ollama running locally (or specify `--host` for a remote instance)
- For Anthropic-API-based LLM judge evaluation: `pip install "porchbench[api]"` (pulls in the Anthropic SDK; requires `PORCHBENCH_EVAL_BACKEND=api` + `ANTHROPIC_API_KEY`). Not needed for the default local-Ollama judge.

## License

Apache 2.0 — see [LICENSE](https://github.com/mmdevelops/porchbench/blob/main/LICENSE) for details.
