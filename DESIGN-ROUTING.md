# Scale-Aware Prompt Routing — Design Document

Extends the ollama-bench framework to answer a specific empirical question before committing
to building a routing system.

**Depends on:** DESIGN.md (core benchmarking), DESIGN-SANDBOX.md (harness + sandbox layers)

---

## The Question

> For a given local model roster (e.g., 3B, 7B, 14B, 32B variants available via Ollama),
> does scale-aware prompting — adapting prompt strategy to model size — produce measurable
> quality or efficiency gains compared to universally prompting the strongest available model?

If yes: build a routing system that exploits this.
If no: use the strongest model for everything and save the complexity.

We are **not** designing the routing system yet. We are designing the benchmarking apparatus
to answer the question empirically, for a specific user's hardware and model roster.

---

## Motivation

Recent research (Hakim 2026, arXiv:2604.00025v1) demonstrates that on 7.7% of standard
benchmark problems, larger language models underperform smaller ones by 28.4 percentage
points. The mechanism is scale-dependent verbosity — large models overthink problems that
have straightforward solutions. Brevity constraints reverse this effect, improving large model
accuracy by 26pp and completely inverting the performance hierarchy on math and science
benchmarks.

The practical implication: **universal prompting leaves performance on the table.** A system
that matches prompt strategy to model scale could simultaneously improve quality (by
unlocking masked capabilities in large models) and reduce cost (by identifying when smaller
models suffice).

But this is an empirical claim that depends on:
- Which models you have available
- What quantization levels you're running
- What problem types you care about
- Whether the effect survives non-greedy decoding
- Whether the routing overhead eats the savings

ollama-bench should be able to measure all of this.

---

## What We Need to Measure

### Dimension 1: Prompt strategy × model scale

For each problem, run every model under multiple prompt strategies:

| Strategy | Description | System prompt or constraint |
|---|---|---|
| `universal` | No special instructions. The baseline. | (none) |
| `brevity` | Constrain response length. | "Answer concisely in under 50 words." |
| `direct` | Answer only, no reasoning. | "Respond with only the final answer." |
| `cot` | Explicit chain-of-thought elicitation. | "Think step by step." |
| `structured` | Force a specific output format. | "Respond in JSON: {answer, confidence}" |

This produces a matrix: `model × problem × strategy → (accuracy, tokens, latency)`.

The paper tested `universal`, `brevity`, and `direct`. We add `cot` and `structured`
because they're common in agentic workflows and may interact with scale differently.

> **Deferred: Caveman strategies.** Caveman-style token compression (`caveman`,
> `caveman-ultra`) inspired by github.com/JuliusBrussee/caveman is a promising
> addition but deferred until the core 5 strategies are validated. Key open question:
> smaller local models may not have the same RLHF-induced verbosity as frontier models,
> so compression gains could be smaller or accuracy impact worse. Revisit after initial
> routing discovery runs produce data.

### Dimension 2: Problem characteristics

Not all problems are equal. The paper found that inverse scaling concentrates on specific
problem types (math, science) and is absent on others (reading comprehension). We need to
tag problems with characteristics that might predict which strategy works:

- **Category** (already in suite schema): coding, reasoning, cross-domain
- **Difficulty** (already in suite schema): easy, medium, hard
- **Answer type** (new): factual, numeric, code, explanation, open-ended
- **Reasoning depth** (new): shallow (lookup/recall), medium (1-3 step), deep (multi-step)

These tags let us slice results to find patterns: "on numeric problems with shallow
reasoning, the 3B model with direct prompting beats the 32B model with universal prompting."

### Dimension 3: Cost

Quality alone doesn't justify routing — you need quality *per unit cost*. On local hardware,
cost has three components:

| Cost component | How to measure | Where it comes from |
|---|---|---|
| **Tokens generated** | `eval_count` from Ollama response | Already captured in run results |
| **Latency** | `total_duration` | Already captured |
| **VRAM residency** | Model size × time loaded | New: needs Ollama `/api/ps` polling |
| **Model swap time** | Time to load/unload a model | New: empirical measurement |

Model swap time is especially important for local routing. If swapping from a 32B to a 3B
model takes 8 seconds, and the 3B model saves 2 seconds on the task, the swap wasn't worth
it. ollama-bench should measure swap times as part of system profiling.

### Dimension 4: Consistency

The paper uses `temperature: 0` (greedy decoding), which is deterministic but may overstate
the verbosity effect. We should also measure with sampling enabled:

| Run type | Settings | Purpose |
|---|---|---|
| Deterministic | `temperature: 0, seed: 42` | Reproducible, comparable to the paper |
| Sampled (low) | `temperature: 0.3, seed: 42` | Closer to typical deployment |
| Sampled (high) | `temperature: 0.7, seed: 42` | Creative tasks, worst case for routing |

If the effect disappears under sampling, routing is less valuable for real deployment.

---

## Suite Design for Routing Discovery

A new suite type specifically designed to explore the prompt × model × scale space.

```yaml
# suites/routing-discovery.yaml
suite:
  name: "Routing Discovery"
  version: "1.0"
  description: >
    Systematic exploration of prompt strategy × model scale interactions.
    Run across your full model roster to discover routing opportunities.

defaults:
  options:
    temperature: 0
    seed: 42
    top_p: 1
    num_predict: 2048
    num_ctx: 4096

# Define prompt strategies as reusable templates
strategies:
  universal: {}
  brevity:
    system_message: "Answer concisely in under 50 words. Be direct."
  direct:
    system_message: "Respond with only the final answer. No explanation."
  cot:
    system_message: "Think step by step, then give your final answer."
  structured:
    system_message: "Respond in JSON format: {\"answer\": \"...\", \"reasoning\": \"...\"}"

prompts:
  - id: "math-percentage"
    category: reasoning
    difficulty: easy
    answer_type: numeric
    reasoning_depth: shallow
    tags: [math, arithmetic]
    expected_answer: "36"
    messages:
      - role: user
        content: "What is 15% of 240?"

  - id: "code-fizzbuzz"
    category: coding
    difficulty: easy
    answer_type: code
    reasoning_depth: medium
    tags: [python, loops]
    messages:
      - role: user
        content: "Write a Python FizzBuzz function for 1-100."

  # ... more problems across categories, difficulties, answer types
```

### How the runner handles strategy expansion

When the runner encounters a routing-discovery suite, it expands each prompt into
`len(strategies)` variants automatically:

```
For each model in roster:
  For each prompt in suite:
    For each strategy in suite.strategies:
      - Prepend strategy.system_message (if any) to messages
      - Run against model with suite defaults
      - Tag result with (model, prompt_id, strategy_name)
```

This is a multiplicative expansion: 10 problems × 5 strategies × 4 models = 200 runs.
The runner should estimate total runs and time before starting, since this gets large fast.

---

## Output: The Routing Analysis

The evaluator gains a new analysis mode that consumes routing-discovery results and produces
a routing analysis — not a scorecard.

### Raw performance matrix

For each `(model, problem, strategy)` triple:

```json
{
  "model": "qwen2.5:7b",
  "prompt_id": "math-percentage",
  "strategy": "brevity",
  "correct": true,
  "tokens_generated": 12,
  "latency_ms": 340,
  "tokens_per_second": 35.3
}
```

### Aggregated findings

Roll up the raw matrix into actionable patterns:

```json
{
  "routing_analysis": {
    "timestamp": "2026-04-11T14:00:00Z",
    "models_tested": ["qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b"],
    "strategies_tested": ["universal", "brevity", "direct", "cot", "structured"],

    "headline": {
      "inverse_scaling_detected": true,
      "inverse_scaling_rate": 0.083,
      "problems_where_routing_helps": 12,
      "problems_total": 50,
      "max_quality_gain_pp": 18.5,
      "max_cost_reduction_pct": 72.3,
      "routing_worthwhile": true
    },

    "best_route_per_problem": [
      {
        "prompt_id": "math-percentage",
        "best_model": "qwen2.5:3b",
        "best_strategy": "direct",
        "accuracy": 1.0,
        "tokens": 8,
        "vs_default": {
          "default_model": "qwen2.5:32b",
          "default_strategy": "universal",
          "default_accuracy": 0.8,
          "default_tokens": 145,
          "quality_delta_pp": 20.0,
          "token_savings_pct": 94.5
        }
      }
    ],

    "patterns": [
      {
        "description": "Numeric/shallow problems: small model + direct prompt outperforms large + universal",
        "affected_problems": ["math-percentage", "math-discount", "unit-conversion"],
        "recommended_route": {"model_size": "<=7B", "strategy": "direct"},
        "confidence": "high",
        "evidence_count": 8
      },
      {
        "description": "Complex coding: largest model + universal prompt is best, no routing benefit",
        "affected_problems": ["code-rest-api", "code-refactor"],
        "recommended_route": {"model_size": "max", "strategy": "universal"},
        "confidence": "high",
        "evidence_count": 5
      }
    ],

    "verdict": {
      "routing_recommended": true,
      "estimated_quality_improvement_pp": 6.2,
      "estimated_token_savings_pct": 38.0,
      "caveat": "Based on 50 problems with greedy decoding. Re-run with sampling to validate."
    }
  }
}
```

### The headline question answered

The `headline.routing_worthwhile` flag is the go/no-go signal. It considers:

1. **Inverse scaling rate** — what fraction of problems show the effect?
2. **Effect magnitude** — how many percentage points does routing gain?
3. **Cost savings** — how much cheaper is the routed path?
4. **Consistency** — does the effect hold across problem categories?
5. **Swap overhead** — do model swap times eat the savings?

If the answer is "no, just use the biggest model," that's a valid and useful finding.
The benchmarking framework has done its job by providing the evidence.

---

## CLI Interface

```bash
# Run routing discovery across your model roster
ollama-bench discover-routes \
  --suite suites/routing-discovery.yaml \
  --model qwen2.5:3b \
  --model qwen2.5:7b \
  --model qwen2.5:14b \
  --model qwen2.5:32b

# Analyze results and produce routing analysis
ollama-bench analyze-routes \
  --results results/routing-discovery-*.json

# Quick summary: is routing worth it for my setup?
ollama-bench analyze-routes --results results/routing-discovery-*.json --summary
```

The `discover-routes` command is a specialized runner mode that handles the strategy
expansion automatically. The `analyze-routes` command consumes the expanded results and
produces the routing analysis.

---

## System Profiling

Before running routing discovery, the framework should profile the local system to understand
the cost model. This is a prerequisite — without it, we can't assess whether routing saves
time or just shifts where time is spent.

```bash
ollama-bench profile --model qwen2.5:3b --model qwen2.5:7b --model qwen2.5:32b
```

Measures:
- **Model load time**: how long to load each model into VRAM from cold
- **Model swap time**: how long to unload model A and load model B
- **Inference baseline**: tokens/sec for each model on a standard prompt
- **VRAM usage**: how much VRAM each model consumes
- **Concurrent capacity**: can any models coexist in VRAM?

### Ollama concurrency and its impact on routing cost

Ollama supports loading multiple models simultaneously when VRAM permits. This is controlled
by two environment variables:

- **`OLLAMA_MAX_LOADED_MODELS`** — maximum number of models resident in VRAM at once.
  Default is 1 (unload before loading the next). Setting this higher enables zero-swap-cost
  routing between co-resident models.
- **`OLLAMA_NUM_PARALLEL`** — parallel request slots per loaded model. Relevant for
  throughput but not directly for routing.

This fundamentally changes the routing cost model. With 16GB VRAM:

| Configuration | Co-resident models | Swap cost | Routing overhead |
|---|---|---|---|
| Default (`MAX_LOADED_MODELS=1`) | Only one at a time | 1-9s per swap | Significant — must justify per swap |
| `MAX_LOADED_MODELS=2` | e.g., 3B + 7B (~7GB) | Zero between co-resident pair | Low — route freely between the pair |
| `MAX_LOADED_MODELS=3` | e.g., 3B + 3B + 7B | Zero within the trio | Minimal |

The sweet spot for routing: keep your small/medium models co-resident for free routing
between them, and only escalate to the large model (requiring a swap) when quality demands
it. This is a "hot tier / cold tier" pattern:

- **Hot tier**: models that fit in VRAM together, zero routing cost
- **Cold tier**: larger models that require a swap, route to only when the hot tier can't
  handle the problem

The profiler should detect Ollama's concurrency settings and test actual co-residency by
loading model combinations and verifying via `GET /api/ps` (which returns currently loaded
models and their VRAM usage). Advertised model size doesn't account for KV cache overhead
during inference, so empirical measurement is necessary.

**Important**: the profiler should measure VRAM usage under inference load, not just at idle.
A model that appears to fit at load time may cause OOM or CPU fallback when the KV cache
grows during a long generation. The profiler runs a standard prompt at `num_ctx` length to
measure peak VRAM usage per model, then tests co-residency under realistic conditions.

Output:

```json
{
  "system_profile": {
    "gpu": "AMD Radeon RX 9070 XT",
    "vram_total_gb": 16,
    "ollama_version": "0.6.2",
    "ollama_max_loaded_models": 3,
    "models": {
      "qwen2.5:3b":  {"vram_gb": 2.1, "vram_peak_gb": 2.8,  "load_time_s": 1.2,  "tok_per_sec": 85.3},
      "qwen2.5:7b":  {"vram_gb": 4.8, "vram_peak_gb": 5.9,  "load_time_s": 2.8,  "tok_per_sec": 52.1},
      "qwen2.5:32b": {"vram_gb": 14.2, "vram_peak_gb": 15.6, "load_time_s": 8.4,  "tok_per_sec": 18.7}
    },
    "swap_times_s": {
      "3b→7b": 3.1,
      "3b→32b": 8.9,
      "7b→32b": 9.2,
      "32b→7b": 3.4,
      "32b→3b": 1.8
    },
    "coexistence": {
      "3b+7b":  {"fits": true,  "combined_peak_gb": 8.7,  "headroom_gb": 7.3},
      "3b+32b": {"fits": false, "combined_peak_gb": 18.4, "headroom_gb": -2.4},
      "7b+32b": {"fits": false, "combined_peak_gb": 21.5, "headroom_gb": -5.5}
    },
    "recommended_hot_tier": ["qwen2.5:3b", "qwen2.5:7b"],
    "cold_tier": ["qwen2.5:32b"]
  }
}
```

This profile feeds into the routing analysis — a route that saves 50 tokens but requires an
8-second model swap is worse than just using the loaded model.

---

## Relationship to Other Design Documents

```
DESIGN.md                    DESIGN-SANDBOX.md              DESIGN-ROUTING.md
(core benchmarking)          (sandbox + harness + tools)     (this document)
                                                            
Define → Run → Evaluate      Harness: agent loop            Discover: strategy × model matrix
                             Sandbox: isolated execution     Analyze: is routing worth it?
                                                            Profile: system cost model
                                                            
         └──────── all three share the suite schema, runner, and result formats ────────┘
```

The routing discovery suite is a specialized use of the existing runner. It doesn't require
the sandbox or harness — it's text-in, text-out benchmarking with systematic prompt strategy
variation. The harness becomes relevant later *if* routing proves worthwhile and we want to
build an agent that dynamically routes during a multi-step task.

---

## What This Document Explicitly Does NOT Design

- **A routing engine.** We don't know if routing is worth it yet. Build the measurement first.
- **A prompt optimization system.** We test a fixed set of strategies, not an open-ended search.
- **A model recommendation service.** The analysis is specific to your hardware and roster.
- **Online/adaptive routing.** Everything here is offline benchmarking. Online routing is a
  separate (much harder) problem that we'd only tackle after proving the offline case.

The goal is to produce evidence, not infrastructure. If the evidence says "routing helps,"
the next step is a separate design doc for the routing engine itself, informed by the
empirical patterns this framework discovers.

---

## Design Decisions

- **Strategy set**: user-defined strategies via the `strategies` dict in suite YAML. Ship with five defaults (universal, brevity, direct, cot, structured). Caveman-style strategies deferred (see above).
- **Evaluation method for routing discovery**: deterministic where possible — `expected_answer` for exact-match and regex; frontier-model evaluation only for open-ended prompts where automated checking isn't feasible. This is cheaper and enables more experiments.

## Open Questions

- **Statistical rigor**: with `temperature: 0`, each (model, problem, strategy) triple
  produces one deterministic result. With sampling, we'd want multiple runs per triple to
  measure variance. How many runs per triple? 3? 5? 10? The combinatorial expansion gets
  expensive fast.

- **Incremental discovery**: if you add a new model to your roster, you shouldn't have to
  re-run the entire matrix. The runner should support incremental runs that fill in gaps in
  an existing result set.
