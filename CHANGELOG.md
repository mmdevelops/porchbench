# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Pre-1.0: expect breaking changes between minor versions.

## [Unreleased]

### Changed
- Packaging: the sdist now excludes `docs/assets/` (README-page imagery added after 0.1.0; installed package contents are unchanged).

## [0.1.0] - 2026-05-04

Initial public release.

### Added

#### Commands
- `run` — async benchmark runner for one or many suites against one or many models. Single-suite + single-model + single-repeat takes a fast inline path with per-prompt verbose progress and a `-p` prompt-id filter; larger plans flow through the orchestration pipeline (plan-table preview, optional `--profile` system phase, optional `--evaluate` post-phase batch with the judge model loaded once for all results, `--yes` for unattended runs). Includes `--repeats` for determinism verification, `--resume` for incremental recovery, and structured JSON output under `results/`.
- `evaluate` — LLM-as-judge scoring with three backends: local Ollama (default), Anthropic API, and a Claude Code sandboxed backend. Rubric auto-resolved from suite metadata; calibration priming for few-shot scoring accuracy.
- `compare` — side-by-side comparison table with paired per-question deltas. Same-model columns are disambiguated by run timestamp so duplicate selections stay distinguishable.
- `leaderboard` — cross-scorecard ranking, with `--strict` to require a matching evaluator and `--evaluator <label>` to pin a specific judge.
- `analyze-routes` — consumes `run --strategies` JSONs and produces a `RoutingAnalysis` (best route per problem, vs-default comparison, inverse-scaling detection, pattern grouping, headline verdict). Refuses on a single model with a clear "need ≥2 models" message.
- `profile` — measure model load/unload time, VRAM footprint, and co-residency capacity (Ollama only).
- `doctor` — environment diagnostics: Python, Ollama reachability, GPU detection, VRAM sampler selection, package install state. `--json` for machine-readable triage output.

#### Benchmark surface
- Four shipped suites: `coding-basics` (28 prompts), `cross-domain` (22), `routing-discovery` (92), `tool-use` (19). Bundled under `src/porchbench/data/`; reference by name from any directory or override with a project-local YAML in `./suites/`.
- `run --strategies` expands every prompt across all suite-defined strategies and every model — the matrix `analyze-routes` consumes.
- Tool-use benchmarking with sandboxed subprocess execution and outcome-state validators.
- `tool_use_metrics.tool_calls_via_text` counter on tool-use prompts: increments when an assistant turn returns no structured `tool_calls` but the message content parses as a tool-call shape. Recognised shapes: bare `{"name": str, "arguments": dict}`, list-of-objects, concatenated objects, fenced (```` ```json ````), Hermes/ChatML-style `<tool_call>{...}</tool_call>` (matched incidentally because the JSON inside parses cleanly), and Anthropic SDK / OpenAI-tool-call wrappers `{"function": {...}}` and `{"function_call": {...}}`. Surfaces the "model knows it should call a tool but emits the call as text" regression — distinct from "model can't tool-call." Detector uses `JSONDecoder.raw_decode` so concatenated objects (a common qwen2.5-coder emission) are caught. Shape set is mirrored with the agent-harness project per a cross-project detector pact.

#### Statistics and reproducibility
- Bootstrap confidence intervals, paired per-question deltas (Wilcoxon signed-rank for n ≥ 6, paired t for smaller samples), Cohen's dz effect sizes, and contamination-aware aggregation. Paired-t p-values are gated to df ≥ 30; below that, `p_value` and `significant` are `null` in the scorecard JSON and the CI plus effect size carry the inference. See METHODOLOGY.md for rationale.
- `porchbench compare --seed` (env `PORCHBENCH_SEED`, default `42`) exposes the bootstrap RNG seed. Output is byte-identical for a fixed seed; override to probe sensitivity.
- `RunMetadata.porchbench_version` records the installed package version on every run.
- Every result captures model SHA, suite SHA, Ollama version, quantization level, KV cache type, and full generation parameters.
- 553-test suite across backend, runner, evaluator, routing, sandbox, validators, schemas, statistics, asset resolution, and the public eval API surface.

#### Configuration and ergonomics
- Interactive pickers (beaupy-backed) for models, suites, results, and scorecards — every command falls through to a picker when its primary argument is omitted. Suite-first ordering in the `run` flow lets the model picker render Ollama capability badges (`[tools, vision, thinking]`) and tag missing-cap models so suite/model mismatches surface at selection time.
- Run options screen surfaces the resolved judge model on the Evaluate toggle, plus a sibling "Re-pick judge for this run" toggle that forces the picker for one invocation without overwriting `.env`. Re-pick implies `--evaluate`.
- `.env` and `PORCHBENCH_*` environment variables for all configuration; CLI flags always win.
- `--set KEY=VALUE` overrides for individual inference options (`--set think=false`, `--set num_ctx=16384`); values parse as YAML.

#### Public Python API
- Stable top-level library API: `from porchbench import RunResult, Scorecard, PromptScore, Suite, Rubric, RoutingAnalysis, SystemProfile` re-exports the Pydantic schemas for every serialized artifact the CLI produces.
- Public evaluator API for embedding porchbench scoring in other tools: `evaluate_single`, `evaluate_single_sync`, `batch_evaluate_results`, `batch_evaluate_results_sync`, `make_backend("ollama" | "api" | "claude-code", **kw)`, plus the three backend classes (`OllamaEvalBackend`, `AnthropicEvalBackend`, `ClaudeCodeEvalBackend`) and the `EvalBackend` Protocol.
- `SuiteReference.slug` property + module-level `slugify_suite_name(name)` helper for consumers that need to map suite display names to on-disk filename form.
- Package is typed (`py.typed` marker, `Typing :: Typed` classifier). Downstream type-checkers pick up the Pydantic models without a separate stubs package.

### Known limitations
- `profile` is Ollama-only; OpenAI-compatible backends report stub values.
- AMD / ROCm on gfx1201 (RDNA 4) requires a rocblas override; some quantized Qwen 3.5 variants hit an unshipped `SOLVE_TRI` kernel upstream. See README troubleshooting.
- `--resume` only restores work from runs that completed and wrote a result JSON. Mid-run interruption (Ctrl-C, OOM, session kill) loses all in-progress prompts because results write at the end of `run_suite`.
- The `tool_calls_via_text` detector covers JSON shapes (bare, list, concatenated, fenced), Hermes/ChatML `<tool_call>` wrappers, and `{"function": ...}` / `{"function_call": ...}` wrappers. Anthropic-style nested-XML emissions (`<function_calls><invoke name="...">...</invoke></function_calls>`) are not detected — this is the one shape neither porchbench nor agent-harness handles, tracked as a v0.2 coordination point.
