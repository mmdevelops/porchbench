# PRELIM: KV Cache Compression Benchmarking

**Status:** Prelim
**Date:** 2026-04-12
**Depends on:** DESIGN.md (core framework), METHODOLOGY.md (statistical rigor)

## Problem

Weight quantization (Q4_K_M, Q8, etc.) is already a first-class variable in
ollama-bench, but **KV cache compression** is not measured at all. KV cache is
the working memory that stores attention context during inference — it grows
linearly with context length and is often the bottleneck for long-context
workloads on consumer GPUs.

As of early 2026, the KV cache compression landscape has exploded:

1. **Ollama already supports** `f16`, `q8_0`, and `q4_0` KV cache types via
   `OLLAMA_KV_CACHE_TYPE`. These are untested and undocumented in benchmarking
   practice.
2. **TurboQuant** (Zandieh et al., ICLR 2026) achieves 6x compression at 3 bits
   with zero accuracy loss on standard benchmarks. llama.cpp implementations
   exist (`tq3`, `tq4`); an Ollama PR is open ([ollama/ollama#15090](https://github.com/ollama/ollama/pull/15090)).
3. **Community findings contradict the paper**: multiple implementations found
   MSE-only quantization (PolarQuant) outperforms the paper's MSE+QJL two-stage
   approach through softmax, because QJL's unbiased inner-product estimation
   becomes high-variance after exponentiation.
4. **No standardized benchmarking protocol** exists for comparing KV cache methods
   in the Ollama ecosystem. Users choose cache types without data.

### Evidence

- Ollama PR #15090 reports +6.5% gen speed at 128K context with `tq3` on RTX 3070 Ti
- vLLM TurboQuant plugin shows 3.76x KV cache compression on Molmo2-8B (1,639 MiB
  down to 435 MiB), but also discovered FP16 precision fails after ~11,385 tokens
- NVIDIA kvpress benchmarks 30+ methods but targets vLLM/HuggingFace, not Ollama
- No existing tool measures the accuracy/throughput/memory tradeoff across Ollama's
  supported cache types at varying context lengths

## Current State

The framework captures weight quantization level, throughput, TTFT, and timing
metrics per prompt. The profiler measures VRAM usage per model. But there is no
mechanism to systematically vary KV cache type, measure its effects, or compare
cache types while holding other variables constant.

| What | Where |
|------|-------|
| KV cache type field (new) | `schemas.py:SystemInfo.kv_cache_type` |
| Weight quantization tracking | `schemas.py:ModelDetails.quantization_level` |
| Throughput metrics | `schemas.py:PromptMetrics` (tokens/sec, TTFT) |
| VRAM profiling | `profiler.py:_profile_single_model()` |
| Metrics aggregation | `metrics.py:summarize_run()` |
| Run orchestration | `runner.py:run_suite()` |
| KV cache methodology | `METHODOLOGY.md` KV Cache Compression section |

## Design

### Core concept: KV cache sweep

A **cache sweep** runs the same suite + model combination across multiple KV cache
types, producing one run result per cache type. These results share identical
inputs (same prompts, same model weights, same quantization) and differ only in
the cache compression setting — isolating the effect of cache type on throughput,
memory, and accuracy.

### Component 1: Cache sweep orchestrator

A new CLI command that automates the set-restart-run cycle:

```
ollama-bench sweep-cache \
    --suite suites/long-context.yaml \
    --model qwen2.5:7b \
    --cache-type f16 --cache-type q8_0 --cache-type q4_0 \
    --context-lengths 4096,16384,65536,131072
```

For each `(cache_type, context_length)` combination:
1. Set `OLLAMA_KV_CACHE_TYPE` environment variable
2. Restart the Ollama server (or create a model variant via Modelfile)
3. Run the suite with `num_ctx` set to the target context length
4. Capture VRAM usage via `ollama.ps()` during inference
5. Record the run result with `system.kv_cache_type` populated

**Output:** a directory of run results, one per `(cache_type, context_length)` cell,
plus a sweep summary JSON that cross-references them.

### Component 2: Needle-in-a-Haystack (NIAH) task suite

The canonical accuracy test for KV cache compression. A fact is embedded at a
specific depth within a long context, and the model must retrieve it.

```yaml
# suites/niah.yaml
suite:
  name: "Needle-in-a-Haystack"
  version: "1.0"
  description: "KV cache compression accuracy at varying context lengths and depths"
  categories:
    - retrieval

defaults:
  options:
    temperature: 0
    seed: 42
    num_predict: 256
```

NIAH prompts are **generated programmatically** (not hand-authored) because the
combinatorial space is large: `depth_positions x context_lengths x needle_types`.

Parameters:
- **Depth positions:** 10%, 25%, 50%, 75%, 90% of context
- **Context lengths:** 4K, 8K, 16K, 32K, 64K, 128K (up to model/VRAM limit)
- **Needle types:**
  - Simple fact retrieval ("The secret code is 7429")
  - Multi-fact retrieval (2-3 facts placed at different depths)
  - Reasoning over retrieved fact ("What is the secret code plus 1000?")

Each prompt has `expected_answer` for automated correctness checking (exact match
or regex extraction).

A generator script (`scripts/generate_niah_suite.py`) produces the YAML suite from
configurable parameters. The haystack filler text is drawn from a fixed corpus
(e.g., Paul Graham essays, public domain text) with a content hash for reproducibility.

### Component 3: Memory profiling during inference

The current profiler (`profiler.py`) captures VRAM at model load time. Cache
compression benchmarking needs VRAM measurement **during inference at target
context length**, because that's when the KV cache is populated.

New profiling approach:
1. Start inference with a long prompt (context_length tokens of filler + question)
2. Sample `ollama.ps()` size_vram during the generation phase
3. Record peak VRAM as `peak_vram_bytes` in run results

This extends `PromptMetrics` with an optional `peak_vram_bytes` field, populated
only during cache sweep runs (to avoid the overhead of VRAM polling in normal runs).

### Component 4: Sweep analysis and comparison

A companion analysis command that reads sweep results and produces:

1. **Throughput vs. cache type** — tokens/sec (prefill and decode separately) for
   each cache type, at each context length
2. **Memory vs. cache type** — peak VRAM at each context length, compression ratio
   vs. f16 baseline
3. **Accuracy vs. cache type** — NIAH retrieval accuracy at each depth × context
   length, broken down by cache type
4. **Context capacity** — maximum achievable `num_ctx` before OOM for each cache type
5. **Degradation onset** — the context length at which accuracy begins to drop for
   each cache type (the "cliff")

Output: structured JSON + optional Rich table for terminal display. Visualization
is deferred (non-goal for now; downstream tools can consume the JSON).

## Config

| Field | Default | Validation | Purpose |
|-------|---------|------------|---------|
| `cache_types` | `["f16", "q8_0", "q4_0"]` | non-empty list of strings | Cache types to sweep |
| `context_lengths` | `[4096, 16384, 65536]` | non-empty list of positive ints | Context lengths to test |
| `niah_depths` | `[0.1, 0.25, 0.5, 0.75, 0.9]` | list of floats in (0, 1) | Needle placement depths |
| `niah_needle_count` | `3` | positive int | Number of distinct needles per depth |
| `vram_poll_interval_ms` | `100` | positive int | How often to sample VRAM during inference |
| `server_restart_delay_s` | `5` | non-negative | Wait after server restart before running |
| `haystack_corpus` | `"paulgraham"` | string | Filler text source identifier |

## Implementation Phases

### Phase 1: NIAH Suite Generator

**Goal:** Produce Needle-in-a-Haystack YAML suites programmatically for any
combination of context lengths and depths.

**Changes:**
- New script `scripts/generate_niah_suite.py` — takes context lengths, depths,
  needle count as CLI args; outputs a suite YAML
- New filler corpus file `data/haystack_paulgraham.txt` — source text for padding
- Uses existing `Suite` schema for validation

**Validation:**
- Generated YAML loads and validates through `suite.py`
- Prompts at each depth have correct expected_answer
- Context lengths in generated prompts match requested num_ctx
- Run a generated suite at 4K context with one model to verify end-to-end

### Phase 2: VRAM Profiling During Inference

**Goal:** Measure peak VRAM usage during active inference at a target context length.

**Changes:**
- Add `peak_vram_bytes: int | None = None` to `PromptMetrics` in `schemas.py`
- New function `measure_peak_vram()` in `profiler.py` that polls `ollama.ps()`
  in a background task during inference
- Integrate into `runner.py:run_prompt()` with an opt-in flag (avoid overhead
  in normal runs)

**Validation:**
- VRAM measurements increase with context length for the same model
- q4_0 cache shows lower peak VRAM than f16 at the same context length
- Measurements are within 5% of nvidia-smi readings (cross-check)

### Phase 3: Cache Sweep Orchestrator

**Goal:** Automate running the same suite across multiple cache types with
server restarts.

**Changes:**
- New CLI command `sweep-cache` in `cli.py`
- New module `src/ollama_bench/sweep.py` — orchestration logic:
  - Restart Ollama with target `OLLAMA_KV_CACHE_TYPE`
  - Wait for server readiness
  - Delegate to `runner.run_suite()` with VRAM profiling enabled
  - Collect results into a sweep result structure
- New schema `SweepResult` in `schemas.py` — references individual run result
  files plus sweep-level metadata (cache types tested, context lengths, etc.)

**Validation:**
- Sweep produces one run result per (cache_type, context_length) cell
- Each run result has correct `system.kv_cache_type` recorded
- Server is confirmed running with correct cache type before each cell
- Sweep completes end-to-end with at least 2 cache types and 2 context lengths

### Phase 4: Sweep Analysis

**Goal:** Compare results across cache types for throughput, memory, and accuracy.

**Changes:**
- New module `src/ollama_bench/sweep_analysis.py`
- New CLI command `analyze-sweep` in `cli.py`
- Reads sweep result + individual run results
- Computes compression ratios, throughput deltas, accuracy matrices
- Outputs structured JSON + Rich terminal table

**Validation:**
- Analysis correctly identifies the f16 baseline and computes ratios against it
- NIAH accuracy matrix shows per-depth, per-context-length, per-cache-type results
- Degradation onset is identified where accuracy drops below a threshold (default 95%)
- Results are consistent with known properties (q8_0 ~lossless, q4_0 shows some
  degradation at long contexts)

## Deferred (not blocking done)

- **TurboQuant integration** — waiting for Ollama merge of tq3/tq4 cache types
  (estimated 2-3 months). When available, add to default `cache_types` list.
  The sweep infrastructure will support them with zero code changes — they're
  just new values for `OLLAMA_KV_CACHE_TYPE`.
- **Multi-engine comparison** — benchmarking the same model via vLLM (where
  TurboQuant plugin is available today) alongside Ollama. Would require a
  second client backend. Deferred until there's a concrete need.
- **Perplexity measurement** — requires token-level log-probabilities, which
  Ollama does not currently expose. Would need a custom llama.cpp integration.
- **Visualization** — charts, heatmaps, NIAH depth-vs-length plots. Downstream
  tools (matplotlib, plotly) can consume the JSON output. No built-in viz for now.
- **Automated server management on remote hosts** — SSH-based restart for
  benchmarking against remote Ollama instances.
- **Model variant approach** — alternative to env var: create Ollama model variants
  with different cache types via Modelfile (`FROM model \n PARAMETER kv_cache_type q4_0`).
  This avoids server restarts but creates model copies. Worth exploring but not
  the primary mechanism.

## Files Involved

| File | Change |
|------|--------|
| `scripts/generate_niah_suite.py` | New — NIAH suite generator |
| `data/haystack_paulgraham.txt` | New — filler corpus for NIAH |
| `src/ollama_bench/schemas.py` | Add `peak_vram_bytes` to PromptMetrics, add `SweepResult` |
| `src/ollama_bench/profiler.py` | Add `measure_peak_vram()` background poller |
| `src/ollama_bench/runner.py` | Integrate VRAM profiling opt-in |
| `src/ollama_bench/sweep.py` | New — cache sweep orchestrator |
| `src/ollama_bench/sweep_analysis.py` | New — sweep result analysis |
| `src/ollama_bench/cli.py` | Add `sweep-cache` and `analyze-sweep` commands |

## Relationship to Other Systems

| Design | Relationship |
|--------|-------------|
| DESIGN.md (core framework) | Extends — uses existing runner, schemas, client |
| DESIGN-ROUTING.md (routing) | Independent — cache type could become a routing variable later |
| DESIGN-SANDBOX.md (tool-use) | Independent — no interaction |
| METHODOLOGY.md | Synergistic — KV cache section already added |
| HARDWARE-PLAN.md | Synergistic — cache compression directly affects VRAM capacity |

## Dependencies

- Ollama server with `OLLAMA_KV_CACHE_TYPE` support — exists today for f16/q8_0/q4_0
- Existing runner/client/profiler modules — implemented
- A model that supports long contexts (>32K) for meaningful NIAH testing

## Open Questions

1. **Server restart mechanism on Windows.** The sweep orchestrator needs to restart
   Ollama with a new env var. On Linux this is straightforward (`OLLAMA_KV_CACHE_TYPE=q4_0 ollama serve`).
   On Windows, the restart mechanism depends on whether Ollama runs as a service
   or a foreground process. Leaning toward: document both approaches, default to
   subprocess management for the foreground case.

2. **Modelfile variants vs. env var.** An alternative to server restarts is creating
   model variants with different cache types embedded in the Modelfile. This avoids
   restarts but duplicates the model entry in Ollama's model list. Leaning toward:
   env var as primary mechanism (simpler, no duplication), Modelfile variant as
   documented alternative for advanced users.

3. **NIAH filler corpus licensing.** Paul Graham essays are commonly used but their
   licensing status for redistribution is unclear. Alternative: use Project Gutenberg
   public domain text (e.g., "A Tale of Two Cities"). Leaning toward: Gutenberg text
   for cleaner licensing, with a script to optionally use custom corpus.

4. **VRAM polling accuracy.** `ollama.ps()` reports model VRAM, not total process VRAM.
   KV cache memory may or may not be included in this number depending on Ollama version.
   Need to verify empirically. If inaccurate, fall back to `nvidia-smi` process memory
   queries (Linux/Windows via `pynvml`).

5. **Interaction with weight quantization.** Should cache sweeps hold weight quantization
   constant? Yes — the sweep should test one `(model, weight_quant)` pair across cache
   types. But we could also offer a full matrix mode: `weight_quants x cache_types x
   context_lengths`. Leaning toward: single weight quant per sweep (simpler), with the
   option to run multiple sweeps and compare.
