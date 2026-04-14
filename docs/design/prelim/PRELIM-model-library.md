# PRELIM: Model Library — Test Matrix for 16GB VRAM Benchmarking

**Status:** Prelim
**Date:** 2026-04-13
**Depends on:** `PRELIM-kv-cache-compression.md` (quantization testing overlaps)

## Problem

The benchmark framework is implemented (suites, runner, evaluator, tool-use, routing
discovery) but has no defined model selection. Without a principled model matrix,
benchmarking risks either testing too many similar models (wasting time) or missing
architecturally interesting comparisons (wasting the framework's analytical power).

Preliminary PoC work (4 prompts, manual evaluation) showed surprising results:
1. gemma4:e4b (MoE, ~4B active) outperformed all models including 14B and 26B dense
2. qwen2:7b scored *worse* than qwen2:3b (inverse scaling)
3. Coder fine-tuning helped only on pure coding; disappeared on reasoning tasks
4. Parameter count was a poor predictor of output quality

These findings need validation at scale (139 prompts across 3 suites).

## Current State

### Hardware
| Component | Spec |
|---|---|
| GPU | AMD Radeon RX 9070 XT (RDNA 4, gfx1201, 16GB GDDR6) |
| RAM | 64 GB DDR5 |
| OS | Windows 11 |
| Ollama | v0.20.5 |
| GPU backend | ROCm via manual rocblas library copy (56 gfx1201 .hsaco files) |

### ROCm Workaround Status (verified 2026-04-13)
The gfx1201 rocblas libraries are in place at:
`C:\Users\micmo\AppData\Local\Programs\Ollama\lib\ollama\rocm\rocblas\library\`

56 gfx1201-specific kernel files present, copied from ROCm HIP SDK 7.1 at:
`C:\Program Files\AMD\ROCm\7.1\bin\rocblas\library\`

**Working:** gemma4:e4b loads 43/43 layers to GPU, generates at 140 tok/s.
All models in the library except qwen3.5:9b work with this fix.

**Not working:** qwen3.5:9b crashes with `rocBLAS error: hipErrorInvalidDeviceFunction`
on `SOLVE_TRI` (triangular solve kernel). This is not a detection issue — the GPU
is found and layers are allocated. The `qwen35` architecture uses a compute operation
that has no compiled gfx1201 kernel in the rocblas library. This is an upstream
Ollama/llama.cpp issue, not fixable by copying additional SDK files.

**Vulkan alternative: TESTED 2026-04-13 — DOES NOT WORK.**
`OLLAMA_VULKAN=1` was tested via `setx` (persistent user env var). The Ollama server
hung during GPU discovery and never became responsive. This matches upstream reports
(ROCm#5812, ollama#13908) of Vulkan initialization hangs on RDNA 4. The env var was
removed from registry after testing. Neither ROCm nor Vulkan backends work for
qwen3.5:9b on this hardware.

### Locally Available Models (as of 2026-04-13)
```
qwen3.5:9b                         6.6 GB   (GPU broken — ROCm SOLVE_TRI kernel missing)
deepseek-r1:14b                    9.0 GB   (evaluator judge — exclude from test matrix)
qwen2.5-coder:14b-instruct-q8_0   15 GB
gemma4:26b                         17 GB
gemma4:e4b                         9.6 GB
deepseek-r1:8b                     5.2 GB
qwen2.5-coder:14b                  9.0 GB
qwen2.5:7b                         4.7 GB
qwen2.5:3b                         1.9 GB
```

### Models to Pull
```
gemma4:e2b          ~7.2 GB   (smallest Gemma 4 — floor test)
qwen3:8b            ~5 GB     (best tool-calling stability per community)
phi4:14b            ~8 GB     (STEM/math reasoning champion)
qwen2.5-coder:7b    ~4.5 GB   (midpoint in Qwen coder scaling)
qwen2.5-coder:3b    ~2 GB     (small-model floor)
```

## Design

### Model Selection Principles

Each model in the matrix must test at least one of these hypotheses:

1. **Size scaling** — Does capability increase with parameters within a family?
2. **Architecture comparison** — MoE vs dense at similar active-parameter counts
3. **Specialist vs generalist** — Does code fine-tuning help or hurt?
4. **Quantization tradeoff** — Q4 vs Q8 quality and speed on the same model
5. **Reasoning overhead** — Is thinking-mode latency justified?
6. **Tool-use frontier** — Which small models can reliably call tools?
7. **Inverse scaling** — Where do larger models perform worse?

### Tier 1 — Primary Matrix (run all 3 suites, 3 repeats)

These models have either strong prior signal from PoC or strong community evidence.
This is the minimum viable benchmark.

| Model | Params | Architecture | VRAM | Hypothesis |
|---|---|---|---|---|
| `gemma4:e4b` | ~4.5B active | Dense | ~6 GB | PoC champion — validate at 139 prompts |
| `qwen2.5-coder:14b` (Q4_K_M) | 14B | Dense | ~9 GB | PoC fastest coder |
| `qwen2.5-coder:14b` (Q8_0) | 14B | Dense | ~15 GB | Quantization ceiling — Q8 vs Q4 |
| `qwen3:8b` | 8B | Dense | ~5 GB | "Most stable tool calling" — tool-use suite |
| `phi4:14b` | 14B | Dense | ~8 GB | STEM reasoning (80.4% MATH) — reasoning suite |

**5 models x 3 suites x 3 repeats = 45 runs**

### Tier 2 — Scaling and Architecture (run all 3 suites, 1 repeat)

These fill out the scaling curves and test architectural hypotheses.

| Model | Params | Architecture | VRAM | Hypothesis |
|---|---|---|---|---|
| `gemma4:e2b` | ~2.3B active | Dense | ~4 GB | MoE scaling floor — where does Gemma 4 fall off? |
| `gemma4:26b` | ~3.8B active / 26B total | MoE | ~17 GB | Wider-expert MoE — PoC showed it scored below e4b |
| `qwen2.5-coder:7b` | 7B | Dense | ~4.5 GB | Qwen coder midpoint |
| `qwen2.5-coder:3b` | 3B | Dense | ~2 GB | Small-model floor — tool-use capabilities? |
| `deepseek-r1:8b` | 8B | Dense (reasoning) | ~5 GB | Thinking overhead vs. well-specified tasks |

**5 models x 3 suites x 1 repeat = 15 runs**

### Tier 3 — Specialist vs Generalist (coding-basics only, 1 repeat)

Isolates the fine-tuning hypothesis on the subset of prompts where it matters.

| Model | Params | Architecture | VRAM | Hypothesis |
|---|---|---|---|---|
| `qwen2.5:7b` | 7B | Dense (base) | ~4.7 GB | Base vs coder at same size |
| `qwen2.5:3b` | 3B | Dense (base) | ~1.9 GB | Base vs coder at same size |

**2 models x 1 suite x 1 repeat = 2 runs**

### Tier 4 — Conditional (if Vulkan workaround succeeds)

| Model | Params | Architecture | VRAM | Hypothesis |
|---|---|---|---|---|
| `qwen3.5:9b` (/nothink) | 9B | Dense | ~6.6 GB | PoC showed strong coding — could challenge gemma4:e4b |

**Status: BLOCKED.** Both ROCm (SOLVE_TRI kernel crash) and Vulkan (server hangs
during GPU discovery) fail on this hardware. Requires upstream fix in Ollama or ROCm.
Monitor ollama/ollama#10430 and ROCm/ROCm#5812 for updates.

### Excluded Models

| Model | Reason |
|---|---|
| `deepseek-r1:14b` | Default evaluator judge — testing it creates self-preference bias |
| `qwen2:3b`, `qwen2:7b` | Superseded by qwen2.5 and qwen3 — old architecture |
| `llama3.1:8b`, `llama3.2:3b` | No prior signal; lower priority than models with PoC data |
| `mistral:7b` | Architecturally similar to llama; no unique hypothesis to test |

### Run Strategy

**Phase 1: Tier 1 with `--repeats 3`** (45 runs)

Establishes baseline with statistical rigor. Verify determinism first — if any
model shows divergent outputs across repeats at temperature=0/seed=42, investigate
before expanding.

Expected duration: ~3-5 hours depending on model load times.

**Phase 2: Tier 2 with `--repeats 1`** (15 runs)

Fills out scaling curves. Use `--resume` if Phase 1 was interrupted.

**Phase 3: Tier 3 specialist comparison** (2 runs)

Quick targeted test — only coding-basics suite.

**Phase 4: Tier 4 Vulkan test** (if applicable)

Test `OLLAMA_VULKAN=1` with a simple prompt first. If it works, run full Tier 1
protocol (3 suites x 3 repeats).

### Evaluator Configuration

Judge model: `deepseek-r1:14b` (local Ollama, no API costs)

This model is excluded from the test matrix to avoid self-preference bias.
The deepseek-r1 family uses a different architecture than all test models,
reducing cross-family evaluation bias.

### VRAM Co-residency for Routing

With 16GB total VRAM, co-residency testing (from profiler.py) can verify:
- Two 7B Q4 models (~4GB each) can co-reside
- One 14B Q4 (~9GB) + one 3B (~2GB) can co-reside
- gemma4:e4b (~6GB) + qwen2.5-coder:3b (~2GB) can co-reside
- gemma4:e4b + qwen2.5-coder:14b Q4 will NOT co-reside (~15GB)

This informs whether the routing architecture from the PoC (gemma4:e4b default +
qwen2.5-coder:14b for pure coding) requires model swapping or can run hot.

## Implementation Phases

### Phase 1: Pull Missing Models

**Goal:** Get all Tier 1-3 models available locally.

**Changes:**
- Pull: `gemma4:e2b`, `qwen3:8b`, `phi4:14b`, `qwen2.5-coder:7b`, `qwen2.5-coder:3b`
- Verify each loads to GPU with `--verbose`

**Validation:** `ollama list` shows all models; each produces GPU-accelerated
inference (check for `offloaded N/N layers to GPU` in server logs).

### Phase 2: Run Tier 1 Benchmark

**Goal:** Complete 45 runs across 5 models x 3 suites x 3 repeats.

**Changes:**
- Run via CLI: `ollama-bench run -s suites/coding-basics.yaml -m gemma4:e4b --repeats 3`
- Repeat for each model and suite combination

**Validation:**
- All result JSON files written to `results/`
- 143 prompts x 5 models x 3 repeats = 2145 prompt results
- Determinism check: repeat outputs should be identical for temperature=0/seed=42

### Phase 3: Run Tier 2-3 Benchmarks

**Goal:** Complete scaling and specialist comparison runs.

**Validation:**
- Tier 2: 15 result files
- Tier 3: 2 result files
- Scaling curves computable from merged results

### Phase 4: Analysis

**Goal:** Produce comparative analysis answering the 7 hypotheses.

**Changes:**
- Evaluate all runs: `ollama-bench evaluate -r <result> -e deepseek-r1:14b`
- Compare: `ollama-bench compare -r <result1> -r <result2>`
- Route analysis: `ollama-bench analyze-routes -r <routing-results>`

**Validation:**
- Scorecards for all Tier 1 runs (with CIs from 3 repeats)
- Paired comparison for each model pair on shared prompts
- Routing analysis identifying where small models substitute

## Files Involved

| File | Change |
|---|---|
| This document | Reference — no code changes |
| `suites/coding-basics.yaml` | May need `contamination_risk` tags added to prompts |
| `suites/routing-discovery.yaml` | May need `contamination_risk` tags added |
| `suites/tool-use.yaml` | No changes expected |

## Open Questions

1. **Vulkan viability for qwen3.5:9b** — Does `OLLAMA_VULKAN=1` bypass the
   `SOLVE_TRI` kernel crash? If yes, how does Vulkan inference speed compare
   to ROCm for the other models? Would switching the entire benchmark to Vulkan
   be more consistent than mixing backends?

2. **gemma4:26b VRAM** — At 17GB download size, does it actually fit in 16GB VRAM
   for inference? The PoC ran it, but KV cache overhead at 256K context might
   cause partial offload. Need to verify layers offloaded.

3. **phi4-reasoning:14b vs phi4:14b** — The reasoning variant adds chain-of-thought
   like deepseek-r1. Worth testing as a separate entry, or is one phi4 enough?
   Leaning toward just `phi4:14b` since we already test reasoning overhead
   with `deepseek-r1:8b`.

4. **qwen3:8b thinking mode** — Qwen 3 defaults to thinking-enabled. Should we
   test both `/think` and `/nothink` modes as separate matrix entries? This would
   add one more model to Tier 1 but directly tests the reasoning overhead hypothesis.
