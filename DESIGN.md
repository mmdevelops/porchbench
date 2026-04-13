# ollama-bench — Design Document

A lightweight Python framework for deterministic benchmarking of local LLMs via Ollama,
with an optional frontier-model evaluation pass.

## Goals

1. **Reproducibility** — identical inputs produce identical runs across machines and time
2. **Structured I/O** — machine-readable input (test suites) and output (results) from day one
3. **Metrics capture** — both quality (via evaluation) and throughput (tokens/sec, latency)
4. **Extensibility** — clean enough to reuse the core model-call abstraction in an agentic workflow later
5. **Simplicity** — minimal dependencies, no framework magic, easy to hack on

## Non-goals (for now)

- GUI / dashboard (CLI-first; analysis can happen downstream)
- Multi-GPU / distributed inference
- Training or fine-tuning
- Real-time streaming evaluation

---

## Architecture Overview

```
┌─────────────┐      ┌─────────────┐      ┌──────────────┐
│  Test Suite  │─────▶│   Runner    │─────▶│  Run Result  │
│  (YAML/JSON) │      │  (Python)   │      │   (JSON)     │
└─────────────┘      └──────┬──────┘      └──────┬───────┘
                            │                     │
                       Ollama API            ┌────▼───────┐
                     (localhost:11434)       │  Evaluator  │
                                             │ (frontier)  │
                                             └────┬───────┘
                                                  │
                                            ┌─────▼──────┐
                                            │  Scorecard  │
                                            │   (JSON)    │
                                            └────────────┘
```

**Three independent stages, each runnable standalone:**

1. **Define** — author a test suite file
2. **Run** — execute it against one or more Ollama models → produces a run result
3. **Evaluate** — score the run result via a frontier model → produces a scorecard

---

## Stage 1: Test Suite Schema

Test suites are YAML files (human-authored, version-controlled).

```yaml
# suites/coding-basics.yaml
suite:
  name: "Coding Basics"
  version: "1.0"
  description: "Fundamental coding tasks across languages"
  categories:
    - coding
    - reasoning

defaults:
  options:
    temperature: 0
    seed: 42
    top_p: 1
    num_predict: 2048
    num_ctx: 4096

prompts:
  - id: "fizzbuzz-python"
    category: coding
    difficulty: easy
    tags: [python, loops, conditionals]
    messages:
      - role: user
        content: |
          Write a Python function that prints numbers 1 to 100.
          For multiples of 3 print "Fizz", for multiples of 5 print "Buzz",
          for multiples of both print "FizzBuzz".
    # Optional: per-prompt option overrides
    options:
      num_predict: 1024

  - id: "binary-search-explain"
    category: reasoning
    difficulty: medium
    tags: [algorithms, explanation]
    messages:
      - role: user
        content: |
          Explain how binary search works step by step,
          then implement it in a language of your choice.
          Analyze its time and space complexity.

  - id: "rest-api-design"
    category: cross-domain
    difficulty: hard
    tags: [architecture, http, databases]
    messages:
      - role: user
        content: |
          Design a REST API for a library management system.
          Include endpoints, data models, error handling strategy,
          and explain your choices.
```

### Suite schema fields

| Field | Required | Description |
|---|---|---|
| `suite.name` | yes | Human-readable name |
| `suite.version` | yes | Semver string for tracking prompt evolution |
| `suite.description` | no | What this suite tests |
| `suite.categories` | no | Top-level category tags |
| `defaults.options` | yes | Ollama model options applied to every prompt |
| `prompts[].id` | yes | Unique identifier, used as key in results |
| `prompts[].category` | yes | One of: `coding`, `reasoning`, `cross-domain` |
| `prompts[].difficulty` | yes | `easy`, `medium`, `hard` |
| `prompts[].tags` | no | Freeform tags for filtering/grouping |
| `prompts[].messages` | yes | Chat message list (`role` + `content`) |
| `prompts[].options` | no | Per-prompt overrides merged over defaults |

### Why `messages` (chat format) instead of a flat `prompt` string

- Matches Ollama's `/api/chat` endpoint directly
- Supports multi-turn prompts (follow-up questions, system prompts) without schema changes
- This is the same structure an agentic loop would use — conversation history as a list

---

## Stage 2: Runner

### Responsibilities

1. Load and validate the test suite
2. For each model × prompt combination:
   - Call Ollama `/api/chat` with deterministic params
   - Capture the full response including all timing metadata
3. Write a self-describing result file

### Determinism strategy

| Parameter | Value | Why |
|---|---|---|
| `temperature` | `0` | Eliminates sampling randomness |
| `seed` | fixed int (default `42`) | Seeds the RNG for any remaining stochasticity |
| `top_p` | `1` | No nucleus sampling truncation |

These are set in suite `defaults.options` and can be overridden per-prompt if needed (e.g., to
deliberately test creativity at `temperature: 0.7`).

### CLI interface

```bash
# Run a suite against one model
ollama-bench run --suite suites/coding-basics.yaml --model qwen2.5-coder:7b

# Run against multiple models
ollama-bench run --suite suites/coding-basics.yaml --model qwen2.5-coder:7b --model deepseek-coder-v2:16b

# Run a single prompt by ID (useful during development)
ollama-bench run --suite suites/coding-basics.yaml --model qwen2.5-coder:7b --prompt-id fizzbuzz-python

# Override Ollama host (default: http://localhost:11434)
ollama-bench run --suite suites/coding-basics.yaml --model qwen2.5-coder:7b --host http://192.168.1.50:11434
```

### Run result output

Each run produces a JSON file in `results/`:

```
results/
  2026-04-10T18-30-00_coding-basics_qwen2.5-coder-7b.json
```

### Run result schema

```json
{
  "run": {
    "id": "uuid",
    "timestamp": "2026-04-10T18:30:00Z",
    "suite": {
      "name": "Coding Basics",
      "version": "1.0",
      "file": "suites/coding-basics.yaml",
      "sha256": "abc123..."
    },
    "model": {
      "name": "qwen2.5-coder:7b",
      "details": {
        "format": "gguf",
        "family": "qwen2",
        "parameter_size": "7.6B",
        "quantization_level": "Q4_K_M"
      }
    },
    "system": {
      "ollama_version": "0.6.2",
      "gpu": "AMD Radeon RX 9070 XT",
      "vram_gb": 16,
      "os": "Windows 11"
    }
  },
  "results": [
    {
      "prompt_id": "fizzbuzz-python",
      "category": "coding",
      "difficulty": "easy",
      "tags": ["python", "loops", "conditionals"],
      "options_used": {
        "temperature": 0,
        "seed": 42,
        "top_p": 1,
        "num_predict": 1024,
        "num_ctx": 4096
      },
      "request": {
        "messages": [
          {"role": "user", "content": "Write a Python function..."}
        ]
      },
      "response": {
        "message": {
          "role": "assistant",
          "content": "Here's a Python function..."
        },
        "done_reason": "stop"
      },
      "metrics": {
        "prompt_eval_count": 52,
        "prompt_eval_duration_ns": 312000000,
        "eval_count": 187,
        "eval_duration_ns": 4210000000,
        "total_duration_ns": 4580000000,
        "tokens_per_second": 44.42,
        "time_to_first_token_ms": 312.0
      }
    }
  ],
  "summary": {
    "total_prompts": 3,
    "completed": 3,
    "failed": 0,
    "total_duration_s": 38.2,
    "avg_tokens_per_second": 41.7
  }
}
```

### Computed metrics

These are derived from Ollama's raw response fields:

```
tokens_per_second    = eval_count / (eval_duration_ns / 1e9)
time_to_first_token  = prompt_eval_duration_ns / 1e6  (ms)
total_time           = total_duration_ns / 1e9  (s)
```

---

## Stage 3: Evaluator

A separate command that reads a run result and scores each response via a frontier model.

### Why a separate stage

- Decoupled from the runner — you can re-evaluate old runs with updated rubrics
- The frontier model call is expensive; you don't want to re-run Ollama just to tweak scoring
- Different evaluator backends (Claude API, local model, manual) can share the same input

### Evaluation rubric

The evaluator sends each prompt + response to the frontier model along with a scoring rubric:

```yaml
# rubrics/default.yaml
rubric:
  name: "Default Coding Rubric"
  version: "1.0"

criteria:
  - name: correctness
    weight: 0.35
    description: "Does the code work? Would it produce the correct output?"
    scale: 1-5

  - name: reasoning
    weight: 0.25
    description: "Is the explanation clear and logically sound?"
    scale: 1-5

  - name: completeness
    weight: 0.20
    description: "Are edge cases addressed? Is the solution thorough?"
    scale: 1-5

  - name: code_quality
    weight: 0.10
    description: "Is the code clean, idiomatic, and well-structured?"
    scale: 1-5

  - name: domain_knowledge
    weight: 0.10
    description: "Does the response demonstrate understanding of the broader domain?"
    scale: 1-5
```

### Evaluator CLI

```bash
# Evaluate a run result
ollama-bench evaluate --result results/2026-04-10_coding-basics_qwen2.5-coder-7b.json

# Use a custom rubric
ollama-bench evaluate --result results/... --rubric rubrics/strict.yaml

# Compare two models side-by-side
ollama-bench compare --results results/model_a.json results/model_b.json
```

### Scorecard output

```json
{
  "evaluation": {
    "run_id": "uuid-of-original-run",
    "evaluator": "claude-sonnet-4-6",
    "rubric": "Default Coding Rubric v1.0",
    "timestamp": "2026-04-10T19:00:00Z"
  },
  "scores": [
    {
      "prompt_id": "fizzbuzz-python",
      "criteria": {
        "correctness":      { "score": 5, "rationale": "Function produces correct output..." },
        "reasoning":        { "score": 4, "rationale": "Clear explanation but..." },
        "completeness":     { "score": 4, "rationale": "..." },
        "code_quality":     { "score": 5, "rationale": "..." },
        "domain_knowledge": { "score": 3, "rationale": "..." }
      },
      "weighted_score": 4.35,
      "summary": "Strong implementation with minor gaps in explanation."
    }
  ],
  "aggregate": {
    "overall_weighted": 3.87,
    "by_category": {
      "coding": 4.10,
      "reasoning": 3.65,
      "cross-domain": 3.40
    },
    "by_difficulty": {
      "easy": 4.50,
      "medium": 3.80,
      "hard": 3.20
    }
  }
}
```

---

## Project Structure

```
ollama-bench/
├── pyproject.toml              # package config, dependencies, CLI entry point
├── README.md
├── suites/                     # test suite definitions
│   ├── coding-basics.yaml
│   ├── reasoning.yaml
│   └── cross-domain.yaml
├── rubrics/                    # evaluation rubrics
│   └── default.yaml
├── results/                    # run outputs (gitignored)
├── scorecards/                 # evaluation outputs (gitignored)
└── src/
    └── ollama_bench/
        ├── __init__.py
        ├── cli.py              # click/typer CLI entry point
        ├── schemas.py          # pydantic models for all data structures
        ├── suite.py            # suite loading + validation
        ├── runner.py           # orchestrates runs against Ollama
        ├── client.py           # thin wrapper around ollama-python
        ├── evaluator.py        # frontier model evaluation pass
        ├── metrics.py          # derived metric calculations
        └── compare.py          # model comparison utilities
```

### Dependencies

| Package | Purpose |
|---|---|
| `ollama` | Official Ollama Python client |
| `pydantic` | Schema validation for suites, results, scorecards |
| `pyyaml` | Suite and rubric file parsing |
| `typer` | CLI framework (lightweight, type-hint driven) |
| `anthropic` | Claude API for the evaluation pass |
| `rich` | Terminal output formatting (progress bars, tables) |

---

## Agentic Extension Points

These are **not built now** but the design deliberately leaves room for them:

| Future capability | What enables it |
|---|---|
| Multi-turn conversations | `messages` is already a list; append assistant/user turns |
| Tool use / function calling | Add `tools` field to prompt schema; parse tool calls from response |
| Chained reasoning | Runner already returns structured output; chain step N's output as step N+1's input |
| Self-correction loops | Compare response against expected output; re-prompt on failure |
| Parallel model queries | `client.py` already supports async via `ollama.AsyncClient` |

The core loop — **load structured input → call model → capture structured output** — is the
same whether you're benchmarking or running an agent. The benchmark runner is the simplest
version of that loop.

---

## Implementation Order

1. **`schemas.py`** — define pydantic models (Suite, Prompt, RunResult, Metrics)
2. **`client.py`** — wrap `ollama.chat()` with deterministic defaults, return typed result
3. **`suite.py`** — load YAML, validate against schema
4. **`runner.py`** — iterate suite × models, call client, collect results
5. **`cli.py`** — wire up `run` command
6. **`metrics.py`** — derived calculations (tokens/sec, aggregates)
7. **`evaluator.py`** — frontier model scoring
8. **`compare.py`** — cross-model comparison views

Each step is independently testable. Steps 1-5 get you a working end-to-end run.
Steps 6-8 add analysis capabilities.

---

## Open Questions

- **Suite versioning strategy**: do we version suites semantically, or hash the content and treat any change as a new version? (Leaning toward: both — semver for humans, content hash for machines)
- **Result storage**: flat JSON files are fine to start. If the number of runs grows, consider SQLite.
- **Evaluator model choice**: Claude via API is the obvious pick for quality, but should we also support using a strong local model (e.g., a large Qwen) as a cheaper evaluator for rapid iteration?
- **Execution isolation**: should we restart Ollama between models to ensure clean VRAM state? Probably yes for rigorous benchmarking.
