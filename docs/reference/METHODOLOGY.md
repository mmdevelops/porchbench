# Benchmarking Methodology

Standards and practices for rigorous LLM evaluation in porchbench. Informed by
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

**How porchbench implements this**

- `porchbench compare` runs a paired test on per-prompt quality score differences
  between two runs — Wilcoxon signed-rank for n >= 6, paired t for 2 <= n <= 5 —
  and reports a bootstrap CI on the mean difference and a Cohen's dz effect size.
- `porchbench leaderboard` is a **descriptive ranking of weighted means**; it does not
  run a significance test or attach CIs to the ranking itself. To judge whether a
  leaderboard gap reflects a real quality difference, run `porchbench compare` on the
  two underlying runs.

**p-value caveat.** Without a scipy dependency, porchbench approximates the t-tail
with a standard-normal tail. The normal has lighter tails than the t-distribution,
so this approximation *understates* p (anti-conservative) when applied at small df.
porchbench therefore gates paired-t p-values to df >= 30 (n >= 31); below that threshold
`p_value` and `significant` are reported as `null` and the bootstrap CI on the mean
difference plus the Cohen's dz effect size carry the inference. The Wilcoxon
signed-rank path uses its own asymptotic normal approximation of the W statistic,
which is textbook-standard for n >= 10 and reasonable for n >= 6.

**Bootstrap reproducibility.** The bootstrap resampler is seeded at `42` by default,
so `porchbench compare` on identical inputs produces byte-identical `p_value`, CI
bounds, and effect size across invocations. Override with `porchbench compare --seed N`
or the `PORCHBENCH_SEED` environment variable to probe sensitivity: re-running with
a handful of different seeds and confirming the CI bounds and Cohen's dz don't swing
materially is a cheap check that the 10,000-resample bootstrap has converged for
your data. At the default resample count the percentile CI is typically stable to
four decimals across seeds; if it isn't, that's a signal the sample is too small
or too skewed for percentile bootstrap and the effect size should carry more of
the inference than the CI edges.

*References: Wolfe (2025), "Applying Statistics to LLM Evaluations"; Artificial
Analysis Intelligence Index v4.0 methodology*

### Repeated runs

- **Run 3 repeats minimum** even at temperature=0 with fixed seed.
- Local single-GPU inference is the best case for determinism (no distributed
  parallelism introducing floating-point non-determinism), but verify empirically.
- With greedy decoding, prediction interval width <= 0.01 is achievable within
  3 repeats for most models on large benchmarks.
- If repeats produce non-identical outputs at temperature=0, document this as a
  finding — it indicates non-determinism in the model or runtime.

*Reference: Belem et al. (2024), "Towards Reproducible LLM Evaluation" (arXiv 2410.03492);
Song et al. (2025), "Evaluation of LLMs Should Not Ignore Non-Determinism" (NAACL 2025);
Khatchadourian & Franco (2025), "LLM Output Drift: Cross-Provider Validation & Mitigation
for Financial Workflows" (arXiv 2511.07585) — empirical cross-provider output-consistency
measurement at T=0 ranged 12.5%-100%, motivating "verify empirically" rather than "assume
deterministic at T=0"*

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

## KV Cache Compression

Treat KV cache type as a **first-class independent variable**, separate from weight
quantization. Weight quantization (Q4_K_M, Q8, etc.) compresses model parameters;
KV cache compression reduces the memory used to store attention context during inference.
They are orthogonal — a Q4_K_M model can use f16, q8_0, or q4_0 KV cache independently.

| KV Cache Type | Compression | Impact | Notes |
|---|---|---|---|
| f16 | 1x (baseline) | Reference quality | Ollama default |
| q8_0 | 2x | ~lossless | Safe for all context lengths |
| q4_0 | 4x | Measurable degradation at long contexts | May affect retrieval accuracy |
| tq3 | ~5x | Under evaluation | TurboQuant, not yet merged in Ollama |
| tq4 | ~4x | Under evaluation | TurboQuant, not yet merged in Ollama |

### Rules

- **Never compare runs with different KV cache types without explicit acknowledgment.**
  KV cache compression affects memory capacity, throughput, and potentially accuracy.
  A q4_0 cache run at 128K context vs. an f16 cache run at 32K context confounds cache
  type with context length.
- **Record the KV cache type** in every run result. The framework captures
  `OLLAMA_KV_CACHE_TYPE` from the server environment. When unset, Ollama defaults to f16.
- **Hold KV cache type constant** when comparing models or quantization levels, unless
  KV cache compression is the variable under study.
- **Test accuracy at multiple context lengths** when evaluating a new cache type.
  Degradation often appears only at longer contexts (>32K tokens).

### Benchmarking KV cache types

To compare cache types, set `OLLAMA_KV_CACHE_TYPE` and restart the Ollama server
between runs:

```bash
# Baseline run (f16 cache). Use a long-context-heavy suite; the example
# below uses the shipped coding-basics suite, but for KV-cache sensitivity
# you'll want prompts that exercise the context window.
OLLAMA_KV_CACHE_TYPE=f16 ollama serve &
porchbench run --suite coding-basics --model qwen2.5:7b

# Compressed cache run
OLLAMA_KV_CACHE_TYPE=q4_0 ollama serve &
porchbench run --suite coding-basics --model qwen2.5:7b
```

The framework records the cache type in `system.kv_cache_type` for each run result,
enabling downstream comparison of identical model + prompt combinations under different
compression settings.

### Key metrics for KV cache evaluation

| Metric | What it measures | How to capture |
|---|---|---|
| Throughput (tok/s) | Decode speed impact | `metrics.tokens_per_second` (already captured) |
| Prefill speed | Prompt processing impact | `metrics.prompt_eval_duration` (already captured) |
| Peak VRAM | Memory savings | `ollama.ps()` size_vram (profiler captures this) |
| Max context length | Capacity ceiling | Binary search on `num_ctx` until OOM |
| Retrieval accuracy | Compression quality loss | Needle-in-a-Haystack tasks at varying depths |

### Limitations

- `OLLAMA_KV_CACHE_TYPE` is a **server-level** setting. It cannot be varied per-request
  or per-model (unless set in a Modelfile). Benchmarking across cache types requires
  server restarts.
- The framework detects cache type from the **local environment variable**. When
  benchmarking against a remote Ollama server, the detection will be inaccurate —
  record the cache type manually in such cases.
- KV cache compression interacts with `num_ctx`: a q4_0 cache at 128K may fit in VRAM
  where f16 would not. This is a feature, not a confound — but it must be documented
  when reporting results.

*References: Zandieh et al. (2026), "TurboQuant: Online Vector Quantization with
Near-optimal Distortion Rate" (arXiv 2504.19874); Liu et al. (2024), "KIVI:
A Tuning-Free Asymmetric 2bit Quantization for KV Cache" (arXiv 2402.02750)*

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

LLM judges exhibit several well-documented biases. This section describes which
ones porchbench currently mitigates, how, and which remain as known limitations that
users should be aware of when interpreting scorecards.

**What ships today**

- **Rubric-anchored absolute scoring.** Every judge prompt includes a calibration
  preamble with worked examples at multiple quality tiers (strong / adequate /
  weak) drawn from `src/porchbench/data/rubrics/calibration-examples.yaml`. The judge
  is instructed to use these as fixed anchors for the 1-5 scale rather than
  calibrating internally from the single response under review. Implemented in
  `evaluator.format_calibration_preamble`; applied in `evaluator.build_scoring_prompt`.
- **Contamination-aware aggregation.** Prompts carry a `contamination_risk` tag
  (`high` / `medium` / `low`); scorecards produce both raw aggregates and `*_clean`
  variants that exclude high-contamination prompts. This surfaces whether a model's
  edge comes from genuinely novel problems or from public benchmark items likely
  present in training data. See `AggregateScores` in `src/porchbench/schemas.py`.
- **Single-response absolute scoring.** porchbench does not score pairwise; the judge
  rates each response against the rubric in isolation. Position bias (which applies
  to pairwise judging) therefore does not arise in the current pipeline.

**Operator guidance (not automated)**

- **Judge-family separation.** To reduce self-preference bias, use a different
  model family for the judge than the models under test (e.g., a Gemma judge
  scoring Qwen responses, or vice versa). The default Ollama judge is
  `gemma4:e4b`; the default API / Claude-Code judge is Claude Sonnet. porchbench does
  **not** warn when the judge family overlaps with a model under test — that
  check is the user's responsibility.

**Known gaps — not implemented in v0.1**

- **Multi-judge ensemble.** All v0.1 evaluations use a single judge model.
  Averaging across multiple judge families would reduce both self-preference
  bias and idiosyncratic bias from any individual judge.
- **Verbosity-penalizing rubric criteria.** Shipped rubrics evaluate content
  quality, completeness, and reasoning, but do not include an explicit
  conciseness / brevity criterion. Judges may still favor longer responses on
  open-ended tasks at roughly the rates reported in the literature.
- **Pairwise ordering controls.** If pairwise judging is added in a future
  release, it will need position-swap evaluation (compare both A,B and B,A
  orderings; count only consistent wins). Not needed today because scoring is
  absolute.

*References: Panickssery et al. (2024), "Self-Preference Bias in LLM-as-a-Judge"
(arXiv 2410.21819); Kim et al. (2025), "A Systematic Study of Position Bias in
LLM-as-a-Judge" (IJCNLP 2025)*

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
| GPU model | `porchbench profile` (schema field exists in every run but is populated only when profiling) | Affects inference speed |
| VRAM total | `porchbench profile` (schema field exists in every run but is populated only when profiling) | Affects model loading strategy |
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
- Khatchadourian & Franco (2025), "LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows" (arXiv 2511.07585)
- Wolfe (2025), "Applying Statistics to LLM Evaluations"

### Biases in LLM-as-judge
- Panickssery et al. (2024), "Self-Preference Bias in LLM-as-a-Judge" (arXiv 2410.21819)
- Kim et al. (2025), "Systematic Study of Position Bias" (IJCNLP 2025)

### KV cache compression
- Zandieh et al. (2026), "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (arXiv 2504.19874, ICLR 2026)
- Liu et al. (2024), "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache" (arXiv 2402.02750, ICML 2024)
- Hooper et al. (2024), "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization" (arXiv 2401.18079, NeurIPS 2024)
- Kang et al. (2024), "GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM" (arXiv 2403.05527, ICML 2024)

### Data contamination
- Xu et al. (2024), "Benchmark Data Contamination Survey" (arXiv 2406.04244)
