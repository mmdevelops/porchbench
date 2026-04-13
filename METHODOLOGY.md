# Benchmarking Methodology

Standards and practices for rigorous LLM evaluation in ollama-bench. Informed by
academic literature, established benchmark frameworks (HELM, lm-evaluation-harness,
OpenLLM Leaderboard v2), and Anthropic's agent evaluation guide.

**Depends on:** DESIGN.md (core framework), DESIGN-ROUTING.md (routing discovery)

---

## Statistical Rigor

### Sample sizes

- **Minimum 100 examples per task category** for reasonable statistical power.
  Detecting small differences (1-2% accuracy) requires 1,000+ examples. Sample
  size requirement grows quadratically with the inverse of the minimum detectable
  effect.
- For routing discovery, the multiplicative expansion (problems x strategies x models)
  naturally produces large sample counts per cell, but each cell may have only one
  observation at temperature=0. This is acceptable for deterministic greedy decoding
  but insufficient for sampled runs.

*Reference: Wolfe (2025), "Applying Statistics to LLM Evaluations"*

### Confidence intervals and model comparison

- Report **standard errors and confidence intervals** alongside all scores.
- Use **paired difference analysis** for model comparisons: analyze question-level
  score differences rather than comparing separate confidence intervals. This
  exploits positive correlations between model performances on the same questions
  and is far more statistically efficient.
- For correlated question groups (e.g., multi-part problems), use cluster-adjusted
  standard errors (can increase uncertainty by 3x vs. naive estimates).
- CLT-based confidence intervals are unreliable when n < 100. Use bootstrap
  confidence intervals for smaller samples.

*Reference: Wolfe (2025); Artificial Analysis Intelligence Index v4.0 methodology*

### Repeated runs

- **Run 3 repeats minimum** even at temperature=0 with fixed seed.
- Local single-GPU inference is the best case for determinism (no distributed
  parallelism introducing floating-point non-determinism), but verify empirically.
- With greedy decoding, prediction interval width <= 0.01 is achievable within
  3 repeats for most models on large benchmarks.
- If repeats produce non-identical outputs at temperature=0, document this as a
  finding — it indicates non-determinism in the model or runtime.

*Reference: Belem et al. (2024), "Towards Reproducible LLM Evaluation" (arXiv 2410.03492);
Song et al. (2025), "Evaluation of LLMs Should Not Ignore Non-Determinism" (NAACL 2025)*

---

## Quantization

Treat quantization as a **first-class independent variable**, not a footnote.

| Level | Typical impact | Notes |
|---|---|---|
| Q8_0 | ~lossless (perplexity +0.01) | Reference point |
| Q5_K_M | 95-99% of baseline | Production sweet spot |
| Q4_K_M | 3-6% general degradation, up to 10-20% on instruction-following and multilingual | Most common for VRAM-constrained setups |

### Rules

- **Never compare across quantization levels without explicit acknowledgment.** A Q4_K_M
  32B model vs. a Q8 7B model is comparing two variables simultaneously (size and precision).
  Valid for "what works best on my hardware" but not for isolating model capability.
- **Record the exact quantization level** in every run result. The model details from
  `ollama.show()` include `quantization_level`.
- **When routing discovery compares model scales**, hold quantization constant or treat it
  as a separate experimental dimension.

*Reference: Ionio.ai (2025), "Benchmarking Quantized LLMs: What Works Best for Real Tasks?"*

---

## Evaluation Methods

### When to use automated (deterministic) evaluation

Preferred for:
- Tasks with objectively correct answers: math, factual recall, multiple-choice
- Code correctness: **use test execution (pass@k), not text similarity**
- Format compliance: JSON validity, structure matching
- Routing discovery: `expected_answer` with exact-match or regex extraction

pass@k measures the probability that at least 1 of k generated samples passes all
unit tests. This is the gold standard for code evaluation, aligning with the sandbox
design's `expected_outcome` validation.

*Reference: Chen et al. (2021), "Evaluating Large Language Models Trained on Code" (HumanEval)*

### When to use LLM-as-judge evaluation

Necessary for:
- Open-ended responses (explanations, creative tasks, design decisions)
- Quality assessment beyond correctness (clarity, completeness, reasoning quality)
- Tool-use evaluation (efficiency, error recovery, tool selection)

### LLM-as-judge debiasing

Three documented biases must be mitigated:

**Position bias** — judges favor responses in specific positions (~40% inconsistency
when positions are swapped).
- *Mitigation:* Evaluate both orderings (A,B) and (B,A); only count consistent wins.

**Verbosity bias** — judges favor longer responses regardless of quality (~15% inflation).
- *Mitigation:* Use rubric scales with explicit conciseness criteria. Penalize unnecessary
  verbosity in the scoring prompt.

**Self-preference bias** — LLMs rate outputs resembling their own training distribution
higher (GPT-4 bias score: 0.520). Root cause is perplexity preference, not self-recognition.
- *Mitigation:* Use a different model family as the judge than the models being evaluated.
  Ensemble across multiple judge models from different families.

*References: Panickssery et al. (2024), "Self-Preference Bias in LLM-as-a-Judge" (arXiv 2410.21819);
Kim et al. (2025), "A Systematic Study of Position Bias in LLM-as-a-Judge" (IJCNLP 2025)*

---

## Scoring and Aggregation

### Normalized scoring

When aggregating across benchmarks of different difficulty, normalize each to a 0-100
scale where **random baseline = 0** and **perfect = 100**. This prevents easier
benchmarks from dominating the aggregate score.

This follows the OpenLLM Leaderboard v2 methodology.

### Weighted composites

The existing rubric design (DESIGN.md) uses weighted criteria with 1-5 scales. When
computing weighted scores:
- Report both the composite and per-criterion scores
- Include the rubric version in the scorecard (rubric evolution = new evaluation, not
  comparable to old scores)

---

## Suite Design

### Contamination awareness

Public benchmark problems (FizzBuzz, common algorithms) are likely in most models'
training data. This isn't disqualifying — it's useful for comparability with published
results — but it must be accounted for.

- **Include known benchmark problems** for comparability with published model evaluations
- **Include novel/original problems** for contamination-free measurement
- **Tag prompts** with `contamination_risk: high | medium | low` so analysis can filter
- **For routing discovery specifically**, contamination is less of a concern because
  we're comparing prompt strategies on the same problem, not absolute model capability.
  If a model has memorized the answer, it should get it right under all strategies —
  which is still useful signal for routing.

*Reference: Xu et al. (2024), "Benchmark Data Contamination of Large Language Models:
A Survey" (arXiv 2406.04244)*

### Prompt design principles

From HELM and Anthropic's eval guide:

- **Unambiguous** — domain experts should reach identical verdicts on correctness
- **Balanced** — test both positive and negative cases; avoid one-sided optimization
- **Grounded** — has either a deterministic correct answer or a clear rubric
- **Diverse** — covers multiple difficulty levels, categories, and answer types
- **Boundary-probing** — include tasks at the boundary where models begin to fail;
  tasks that are trivially easy or impossibly hard provide no signal

### Ceiling effect awareness

Benchmarks saturate as models improve. MMLU went from discriminating to useless
(88-93% for frontier models) within 3 years. Design suites with a difficulty range
that includes tasks current local models cannot solve, and expect to replace or extend
the suite as capabilities improve.

---

## Reproducibility Checklist

Every run result should capture sufficient metadata for exact reproduction:

| Field | Source | Why |
|---|---|---|
| Model name + tag | `ollama.show()` | Identifies the exact model file |
| Quantization level | `ollama.show().details.quantization_level` | Affects quality and speed |
| Model file SHA | `ollama.show()` digest | Detects silent model updates |
| Ollama server version | `GET /api/version` | Runtime behavior may vary |
| Suite file SHA256 | Computed at load time | Detects prompt changes |
| Suite semver | `suite.version` | Human-readable version |
| GPU model | System detection | Affects inference speed |
| VRAM total | System detection | Affects model loading strategy |
| OS | System detection | Runtime environment |
| Temperature, seed, top_p | From options | Determinism parameters |
| `num_ctx` | From options | Context window affects quality and VRAM |
| `num_predict` | From options | Max output length |

Most of this is already in the run result schema (DESIGN.md). The additions are
model file SHA (from `ollama.show()` digest) and Ollama server version.

---

## Ollama-Specific Implementation Notes

### Tool calling caveats

- `tool_calls[].function.arguments` is a **parsed dict** in Ollama, not a JSON string
  (differs from OpenAI — no `json.loads()` needed)
- Tool result messages use a `tool_name` field, not `tool_call_id` (differs from OpenAI)
- **`tool_choice` is not supported** — cannot force the model to use tools or suppress
  tool use. The harness must handle models responding with text when tools were expected.
- Small models (<7B) are generally unreliable for tool calling. Document this as a known
  limitation; tool-use benchmarks should focus on 7B+ models.
- Context window of 32k+ improves tool calling reliability.

### Server version detection

The `ollama` Python client does not expose server version. Use a direct HTTP call:

```python
import httpx
response = httpx.get(f"{host}/api/version")
version = response.json()["version"]
```

### Model loading metrics

Ollama returns `load_duration` in the chat response — the time spent loading the model
into memory for that request. This is 0 when the model is already loaded (hot) and
significant (1-9 seconds) on a cold start. Useful for profiling and routing cost
estimation.

---

## Key References

### Evaluation methodology
- Hendrycks et al. (2021), "Measuring Massive Multitask Language Understanding" (MMLU)
- Chen et al. (2021), "Evaluating Large Language Models Trained on Code" (HumanEval)
- Liang et al. (2022), "Holistic Evaluation of Language Models" (HELM)
- Jain et al. (2024), "LiveCodeBench: Holistic and Contamination Free Evaluation"
- Anthropic (2025), "Demystifying Evals for AI Agents"

### Scale-aware evaluation
- McKenzie et al. (2023), "Inverse Scaling: When Bigger Isn't Better" (arXiv 2306.09479)
- Hakim (2026), "Brevity Constraints Reverse Performance Hierarchies" (arXiv 2604.00025)
- Ong et al. (2024), "RouteLLM: Learning to Route LLMs with Preference Data" (ICLR 2025)

### Reproducibility and statistics
- Belem et al. (2024), "Towards Reproducible LLM Evaluation" (arXiv 2410.03492)
- Song et al. (2025), "Evaluation of LLMs Should Not Ignore Non-Determinism" (NAACL 2025)
- Wolfe (2025), "Applying Statistics to LLM Evaluations"

### Biases in LLM-as-judge
- Panickssery et al. (2024), "Self-Preference Bias in LLM-as-a-Judge" (arXiv 2410.21819)
- Kim et al. (2025), "Systematic Study of Position Bias" (IJCNLP 2025)

### Data contamination
- Xu et al. (2024), "Benchmark Data Contamination Survey" (arXiv 2406.04244)
