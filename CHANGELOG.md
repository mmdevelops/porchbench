# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Pre-1.0: expect breaking changes between minor versions.

## [0.1.0] - 2026-04-16

Initial public release.

### Added
- `run` — async Ollama benchmark runner with paired per-prompt results, `--repeats` for determinism verification, `--resume` for incremental runs, and structured JSON output under `results/`.
- `evaluate` — LLM-as-judge scoring with three backends: local Ollama (default), Anthropic API, and a Claude Code sandboxed backend. Rubric auto-resolution from suite metadata; calibration priming for few-shot scoring accuracy.
- `compare` — side-by-side comparison table grouping metrics with model columns.
- `leaderboard` — cross-scorecard ranking, with optional `--strict` mode to require a matching evaluator.
- `routes discover` / `routes analyze` — map prompt-strategy × model-scale interactions to find routing opportunities.
- `profile` — measure model load/unload time, VRAM footprint, and co-residency capacity (Ollama only).
- `overnight` — unattended multi-suite batch orchestration with optional `--evaluate` and `--profile` phases, auto-discovery of suites, and error-tolerant continuation.
- Interactive pickers (beaupy-backed) for models, suites, results, and scorecards — every command falls through to a picker when its primary argument is omitted.
- Four shipped suites: `coding-basics` (28 prompts), `cross-domain` (22), `routing-discovery` (92), `tool-use` (19).
- Tool-use benchmarking with sandboxed subprocess execution and outcome-state validators.
- Statistical tooling: bootstrap confidence intervals, paired per-question deltas (Wilcoxon signed-rank for n >= 6, paired t for smaller samples), Cohen's dz effect sizes, and contamination-aware aggregation. Paired-t p-values are gated to df >= 30 (below that, `p_value` and `significant` are `null` in the scorecard JSON and the CI plus effect size carry the inference — see METHODOLOGY.md for rationale).
- `.env` + `PORCHBENCH_*` environment variable configuration; CLI flags always win.
- `porchbench compare --seed` (env `PORCHBENCH_SEED`) exposes the bootstrap RNG seed (default `42`). Output is byte-identical across runs for a fixed seed; override to probe sensitivity of CI bounds and the Cohen's dz effect size.
- 260-test suite across backend, runner, evaluator, routing, sandbox, validators, schemas, statistics, and asset resolution.
- Benchmark suites and rubrics ship bundled with the package under `src/porchbench/data/`. Reference them by name (`-s coding-basics`, `--rubric default`) from any directory — no repo checkout required. Drop a YAML in `./suites/` or `./rubrics/` to override with a project-local copy.
- `RunMetadata.porchbench_version` records the installed package version on every new run for reproducibility.

### Known limitations
- `profile` is Ollama-only; OpenAI-compatible backends report stub values.
- AMD / ROCm on gfx1201 (RDNA 4) requires a rocblas override; some quantized Qwen 3.5 variants hit an unshipped `SOLVE_TRI` kernel upstream. See README troubleshooting.
