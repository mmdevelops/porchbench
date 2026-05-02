# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Pre-1.0: expect breaking changes between minor versions.

## [0.1.0] - 2026-04-30

Initial public release.

### Added
- `run` — async Ollama benchmark runner with paired per-prompt results, `--repeats` for determinism verification, `--resume` for incremental runs, and structured JSON output under `results/`.
- `evaluate` — LLM-as-judge scoring with three backends: local Ollama (default), Anthropic API, and a Claude Code sandboxed backend. Rubric auto-resolution from suite metadata; calibration priming for few-shot scoring accuracy.
- `compare` — side-by-side comparison table grouping metrics with model columns. Same-model columns disambiguated with `·HH:MM` (or short run-id on minute collisions) so duplicate selections stay distinguishable.
- `leaderboard` — cross-scorecard ranking, with optional `--strict` mode to require a matching evaluator and `--evaluator <label>` to pin a specific judge.
- `overnight --strategies` / `analyze-routes` — map prompt-strategy × model-scale interactions to find routing opportunities. The `--strategies` flag on `overnight` (and the matching options-screen toggle) expands every prompt across all suite-defined strategies and every model. `analyze-routes` consumes the resulting JSONs and produces a `RoutingAnalysis` (best route per problem, vs-default comparison, inverse-scaling detection, pattern grouping, headline verdict). Single-model `analyze-routes` refuses with a clear "need ≥2 models" message naming the supplied model. Picker filter on `analyze-routes` gates the result list to files that actually carry per-prompt strategy tags.
- `profile` — measure model load/unload time, VRAM footprint, and co-residency capacity (Ollama only).
- `overnight` — unattended multi-suite batch orchestration with optional `--evaluate` and `--profile` phases, auto-discovery of suites, and error-tolerant continuation.
- Interactive pickers (beaupy-backed) for models, suites, results, and scorecards — every command falls through to a picker when its primary argument is omitted.
- Four shipped suites: `coding-basics` (28 prompts), `cross-domain` (22), `routing-discovery` (92), `tool-use` (19).
- Tool-use benchmarking with sandboxed subprocess execution and outcome-state validators.
- `tool_use_metrics.tool_calls_via_text` counter on tool-use prompts: increments when an assistant turn returns no structured `tool_calls` but the message content parses as a tool-call shape (`{"name": str, "arguments": dict}`, single object, list-of-objects, concatenated, or fenced). Surfaces the "model knows it should call a tool but emits the call as text" regression — distinct from "model can't tool-call." Detector uses `JSONDecoder.raw_decode` so concatenated objects (a common qwen2.5-coder emission) are caught.
- Statistical tooling: bootstrap confidence intervals, paired per-question deltas (Wilcoxon signed-rank for n >= 6, paired t for smaller samples), Cohen's dz effect sizes, and contamination-aware aggregation. Paired-t p-values are gated to df >= 30 (below that, `p_value` and `significant` are `null` in the scorecard JSON and the CI plus effect size carry the inference — see METHODOLOGY.md for rationale).
- `.env` + `PORCHBENCH_*` environment variable configuration; CLI flags always win.
- `porchbench compare --seed` (env `PORCHBENCH_SEED`) exposes the bootstrap RNG seed (default `42`). Output is byte-identical across runs for a fixed seed; override to probe sensitivity of CI bounds and the Cohen's dz effect size.
- 502-test suite across backend, runner, evaluator, routing, sandbox, validators, schemas, statistics, and asset resolution.
- Benchmark suites and rubrics ship bundled with the package under `src/porchbench/data/`. Reference them by name (`-s coding-basics`, `--rubric default`) from any directory — no repo checkout required. Drop a YAML in `./suites/` or `./rubrics/` to override with a project-local copy.
- `RunMetadata.porchbench_version` records the installed package version on every new run for reproducibility.
- Stable top-level library API: `from porchbench import RunResult, Scorecard, Suite, Rubric, RoutingAnalysis, SystemProfile` re-exports the Pydantic schemas for every serialized artifact the CLI produces. Intended entry point for programmatic consumers of result and scorecard JSON.
- Package is typed (`py.typed` marker, `Typing :: Typed` classifier). Downstream type-checkers (mypy, pyright) pick up the Pydantic models directly without a separate stubs package.

### Changed
- `run --resume` now filters already-completed prompts before opening the Rich progress bar, so the bar is sized to the actual workload and the "Resuming: skipping N…" message no longer interleaves with live bar redraws. When every prompt is already done, the CLI short-circuits with `"Nothing to run for <model>…"` and skips both the bar and the result-file write.
- `run_suite` no longer writes an empty result JSON when zero prompts actually ran — defends `overnight --resume` from polluting `results/` on every no-op resume.
- `overnight` preflight now validates every target model name against the inference server before warmup; a typo or comma-separated `-m a,b` mistake hard-fails immediately with `"Model not found: <name>"` and an `ollama pull` hint instead of running the full plan with 100% errors under `--yes`.
- `CodeOutputValidator` failure reasons now report the final non-empty stderr line (typically the `ExceptionType: message` summary) instead of head-truncating the traceback. Per-prompt `[val-fail]` badges are usable again — previously they showed `Validation failed: Traceback (most recent call last):` and dropped the actual exception.
- `tool-use` suite `t3-pipeline` prompt clarified to specify `region` and `revenue` as the exact column names. Models had been emitting `total_revenue` / `Total Revenue` / `TotalRevenue` (the prompt's most prominent label from earlier steps), which the validator's exact-match column lookup couldn't handle — now both sides agree on a single name.
- Removed the static "First prompt can take several minutes…" cold-start hint from `overnight` output. The 60-second heartbeat already emits concrete `(Xm Ys elapsed)` lines during real cold compiles, which is more useful than a generic warning.
- Composite validator construction now reads `type` non-destructively from suite specs. The previous `dict.pop` mutated shared suite state and broke iteration 2+ during routing-discovery (`KeyError: 'type'` after the first strategy).
- `harness.ToolUseMetrics → ToolUseMetricsData` conversion centralized in a single `build_tool_use_metrics_data` helper used by both runner.py and routing.py. Adding a new metric to the tool-use path now needs one update site rather than two — the duplication had silently zeroed `tool_calls_via_text` on every routing-discovery JSON until caught by the agent-harness team during integration testing.
- Windows captured-output (pipes, file redirects, CI logs) now reconfigures `sys.stdout`/`sys.stderr` to UTF-8 with `errors="replace"` at CLI entry. Previously the leaderboard's score-distribution sparkline (U+2588 `█`) crashed mid-render under cp1252; captured runs now render the same as interactive terminal runs.
- `overnight` no longer auto-expands strategies when the picked suite has a `strategies:` block. Baseline (one row per prompt) is the new default; strategy matrix expansion is opt-in via the `--strategies` flag and a new options-screen toggle. Single-suite + `--strategies` against a non-strategy suite hard-fails with a clear message; multi-suite mixed selections warn per-non-strategy suite and fall back to baseline for those. Replaces the surprising prior behavior where `overnight tool-use` silently ran 19 × 4 = 76 cells when the user expected 19.
- Interactive picker order swapped to **suite-first** across `run` and `overnight`. Knowing the suite at model-pick time lets the model picker render Ollama capability badges (`[tools, vision, thinking]`) and tag missing-cap models with `· missing: tools` (sorted to the bottom) when the suite needs a capability the model lacks. Previously a tool-use run against e.g. `medgemma:4b` only surfaced the mismatch after the user had configured options/repeats; the picker now surfaces it at selection time. `check_tool_support_or_exit` is preserved as defense-in-depth for the CLI-args path. New backend method `list_available_models_with_capabilities()` parallelizes per-model `client.show()` calls; new helper `required_capabilities_for_suite()` is the single source of truth for the suite-needs-tools? check.

### Fixed
- `overnight` preflight now surfaces the resolved evaluator (`PASS Evaluator: ollama/<judge>`) before the VRAM cofit check. The cofit message had been referencing "target + eval" without ever naming which model the judge would be. Mirrors the `evaluate` command's existing `Evaluator: ollama/<judge>` preflight line.
- Interactive picker headers (`Select a suite:`, `Select model(s):`) now print a `[dim]Suite: <name>[/dim]` (or `Suites: ...`, `Models: ...`) confirmation line after the picker exits. Previously the prompt header would dangle without context once the next picker drew over it (most visible in `overnight`, where the suite-picker header stayed on screen with no record of what was picked while the model picker was active).
- `routes discover` no longer crashes when the inference server bounces between models. The per-model `get_model_info()` setup at `routing.py:106` was unwrapped — a transient `ConnectionError` (e.g. user restarts Ollama mid-run) propagated out of `asyncio.run()` and killed the whole command, losing every model scheduled after the disconnect. Now reuses the `get_model_info_safe` wrapper already proven in `run`: on connection failure it logs a warning, falls back to a stub `ModelInfo`, and the per-cell error-tolerant inference loop records each downstream chat failure as an errored cell before moving on to the next model. Cell-level resume to recover partial JSONs is tracked as a backlog item.

### Known limitations
- `profile` is Ollama-only; OpenAI-compatible backends report stub values.
- AMD / ROCm on gfx1201 (RDNA 4) requires a rocblas override; some quantized Qwen 3.5 variants hit an unshipped `SOLVE_TRI` kernel upstream. See README troubleshooting.
- `--resume` only restores work from runs that completed and wrote a result JSON. Mid-run interruption (Ctrl-C, OOM, session kill) loses all in-progress prompts because results write at the end of `run_suite`.
- The `tool_calls_via_text` detector matches `{"name": str, "arguments": dict}` shapes only. XML-style emissions (`<function_calls>...</function_calls>`) used by some Anthropic-trained model families are not detected. Tracked as a v0.2 coordination point with the agent-harness project.
