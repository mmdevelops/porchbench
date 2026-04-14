---
name: evaluate-paired
description: Rigorous cross-model evaluation with statistical controls. Evaluates multiple models prompt-by-prompt with calibration, randomized order, and reliability measurement. Use when comparing models on the same suite and you need cross-comparable scorecards.
disable-model-invocation: true
---

# Paired Evaluation Protocol

Rigorous cross-model evaluation for benchmark comparison. You are the frontier
judge — apply rubrics consistently across models with statistical controls that
make scores cross-comparable.

This skill differs from `/evaluate` (single-model, fast) by adding: calibration
priming, prompt-by-prompt cross-model scoring, randomized order, blind evaluation,
and reliability diagnostics.

## Input

The user provides paths to 2+ run result JSON files from the SAME suite:
```
/evaluate-paired results/2026-04-14_cross-domain_gemma4-e4b.json results/2026-04-14_cross-domain_qwen2.5-coder-14b.json
```

The arguments are: `$ARGUMENTS`

## Setup

1. Read ALL result files from `$ARGUMENTS` (split on spaces)
2. Parse each as `RunResult` JSON (schema: `src/ollama_bench/schemas.py`)
3. **Verify prompt alignment**: all result files must contain the same set of
   `prompt_id` values. If mismatched, report which prompts are missing from which
   files and ask the user how to proceed (evaluate only the intersection, or abort).
4. Load rubric:
   - Read the suite YAML (at `run.suite.file` from any result) for a `rubric` field
   - If present, use `rubrics/{rubric}.yaml` for all prompts
   - Otherwise fall back to per-category matching (see `/evaluate` skill)
5. Load calibration examples from `rubrics/calibration-examples.yaml`
6. **Generate evaluation plan**:
   - List models, prompt count, category breakdown
   - Generate a randomized prompt order (use a fixed seed for reproducibility —
     seed = sum of ASCII values of all model names, so the order is deterministic
     for the same model set but varies across different comparisons)
   - Generate a randomized model order **per prompt** (different shuffle per prompt,
     same seed basis)
   - Present the plan and ask user to confirm

## Phase 1: Calibration Priming

Before scoring any benchmark responses, review the calibration examples:

1. Read `rubrics/calibration-examples.yaml`
2. Select the calibration set matching the rubric being used (e.g., `coding` for
   coding-basics, `cross-domain-science` for the cross-domain science suite,
   `reasoning` for reasoning-focused prompts). If the suite mixes categories,
   use the set matching the dominant category.
3. For each tier (strong, adequate, weak):
   - Read the response summary and the assigned scores
   - Internalize the scale: what does a 5 look like? A 3? A 1?
3. State aloud: "Calibration reviewed. Scale anchors: [brief restatement of what
   each tier looks like for this rubric]."

**Do not re-score the calibration examples.** They anchor the scale — reviewing
them primes consistent scoring. This is the single highest-impact control
(AutoRubric, 2025: +3pp accuracy with few-shot calibration).

## Phase 2: Prompt-by-Prompt Evaluation

Process prompts in the randomized order from Setup step 6.

For each prompt, evaluate ALL models' responses before moving to the next prompt.
This is the core methodological control — it ensures your calibration for "what
does a good answer to this specific prompt look like" is identical across models.

### For each prompt:

#### Step 1: Read the prompt
- Read the original prompt (`request.messages`) — same across all models
- Read `expected_answer` correctness hints
- Read the rubric criteria and weights

#### Step 2: Evaluate each model's response (in randomized order)

For each model (in the per-prompt randomized order):

**a. Read the response**
- Read `response.message.content`
- Note `done_reason` (stop vs length)
- **Do NOT look at the model name** while reasoning about quality. The model
  name is recorded for scorecard assembly but should not influence scoring.
  Refer to responses as "Response A", "Response B", etc. during analysis.

**b. Execute code (for prompts with implementation requirements)**
- Extract code blocks and run via Bash
- Check: runs without errors, correct output, edge case handling
- Record execution results — these are hard evidence for correctness scoring

**c. Evidence-based reasoning (BEFORE scoring)**
For each criterion in the rubric:
- What the response does well
- What it gets wrong or misses (specific, line-level)
- How the expected_answer anchors apply
- If truncated, what's missing and whether it's material

Write this reasoning out. Reasoning BEFORE score — do not score first and
rationalize after.

**d. Score each criterion (1-5)**
```
1 — Fundamentally wrong or missing. Would not pass basic review.
2 — Partially addresses the criterion but has significant errors or gaps.
3 — Adequate. Meets basic requirements but lacks depth or has minor errors.
4 — Good. Solid work with minor issues. Would pass code review.
5 — Excellent. Thorough, correct, and demonstrates genuine understanding.
```

**e. Compute weighted score** (round to 2 decimal places)

**f. Write one-sentence summary** capturing the key quality signal.

**Rationale specificity rule:** Rationales must name specific failures. "Domain
knowledge is weak" is useless. "Claims high GC → higher mutation rate, which is
backwards — GC-rich regions are more stable" is actionable.

#### Step 3: Brief cross-model note (optional)

After scoring all models for this prompt, optionally note the most striking
difference: "Model A caught the CpG depletion mechanism; Models B and C both
missed it." This is for the comparison summary, not for adjusting scores.

### Batching

Process 2-3 prompts per batch (all models per prompt = one batch). This keeps
context manageable:
- 3 prompts × 3 models × ~2K tokens per evaluation = ~18K tokens per batch
- After each batch, present the scores to the user as a progress table
- Ask user to confirm before continuing (default: continue unless they intervene)

Between batches, briefly re-read the calibration tier summaries if the session
is getting long (>15 prompts scored). This prevents late-session drift.

## Phase 3: Scorecard Assembly

After all prompts are scored, assemble one scorecard JSON per model.

Use Bash to run a Python script that:
1. Takes the scored data as a Python literal
2. Computes per-model aggregates (overall, by_category, by_difficulty, normalized, clean)
3. Writes each scorecard to `scorecards/{timestamp}_{run_id_first_8}.json`

Scorecard schema must match `src/ollama_bench/schemas.py` Scorecard model.
Set `evaluator` field to `"claude-code/claude-opus-4-6/paired"` to distinguish
from single-model /evaluate scorecards.

```json
{
  "evaluation": {
    "run_id": "<from result file>",
    "evaluator": "claude-code/claude-opus-4-6/paired",
    "rubric": "<rubric name>",
    "timestamp": "<ISO 8601>"
  },
  "scores": [...],
  "aggregate": {
    "overall_weighted": 0.0,
    "by_category": {},
    "by_difficulty": {},
    "overall_normalized": 0.0,
    "by_difficulty_normalized": {},
    "overall_weighted_clean": 0.0,
    "by_category_clean": {},
    "by_difficulty_clean": {}
  }
}
```

## Phase 4: Comparison Summary

After writing scorecards, present a cross-model comparison:

### 4a. Score comparison table
```
Prompt ID          | Category           | Model A | Model B | Model C | Gap
sec-timing-oracle  | security           | 4.35    | 3.20    | 2.10    | 2.25
bio-sequence-align | biology            | 4.80    | 3.65    | 4.10    | 1.15
...
OVERALL            |                    | 4.12    | 3.45    | 3.10    | 1.02
```

### 4b. Per-domain breakdown
Show mean scores per model per category. Identify which domains differentiate
models most (largest spread) and which are similar across models.

### 4c. Notable findings
- Prompts where model rankings differ from overall ranking (Model B usually
  worst but best on this specific prompt — routing signal)
- Domain-specific strengths/weaknesses per model
- Traps that caught specific models (factual errors from expected_answer)
- Any truncation effects

### 4d. Statistical diagnostics

Run these checks and report results:

**Score distribution check:**
- Per model: mean, std, min, max across prompts
- Flag if any model has std < 0.5 (scores too compressed — not using the scale)
- Flag if any model has all scores in a 1-point range

**Criterion independence check:**
- For each pair of criteria, compute Pearson correlation across all (prompt, model)
  scores
- Flag if any pair has r > 0.85 (halo effect — criteria not differentiating)

**Position-in-batch check:**
- For each model, compute correlation between evaluation order (1st, 2nd, 3rd
  model evaluated for a prompt) and score
- Flag if |r| > 0.3 (systematic order bias)

Report these diagnostics to the user after the comparison table.

## Phase 5: Reliability Measurement (separate session)

**This phase runs in a SEPARATE session** — do not combine with Phases 1-4.
The user will invoke /evaluate-paired again with the same files and add
"--reliability" or similar indication.

When the user requests reliability measurement:

1. Read the existing scorecards from Phase 3
2. Sample 30% of (prompt, model) pairs randomly (minimum 6, seed from model names)
3. Re-evaluate those pairs following the same protocol (calibration, evidence-first,
   same rubric)
4. Compute agreement between original and re-evaluation:

**ICC(3,1) — Intraclass Correlation, two-way mixed, single measures, consistency:**
```python
# For each re-evaluated item, you have (score_original, score_retest)
# ICC(3,1) = (MS_rows - MS_error) / (MS_rows + MS_error)
# where MS comes from a two-way ANOVA on items × sessions
```

**Per-criterion ICC:**
Compute ICC separately for each rubric criterion. This reveals which dimensions
are most/least stable. Correctness should be high (code works or doesn't);
domain_knowledge may be lower (more subjective).

**Interpretation:**
| ICC | Reliability | Action |
|-----|------------|--------|
| > 0.90 | Excellent | Scores are highly trustworthy |
| 0.75 - 0.90 | Good | Scores are usable for comparison |
| 0.60 - 0.75 | Moderate | Investigate unstable criteria, consider simplifying rubric |
| < 0.60 | Poor | Scores not reliable enough for cross-model claims. Revise protocol. |

Report: overall ICC, per-criterion ICC, and any items where retest score differs
by > 1.5 points (flag for investigation).

## Methodology Notes

- **Blind evaluation**: Do not reference model names during scoring. Use
  "Response A/B/C" labels. Model identity is only attached during scorecard
  assembly. (Practical limit: response style can leak identity — thinking blocks
  for r1, response length patterns. This is unavoidable but not a major concern
  for pointwise scoring.)

- **Evidence before scoring** reduces post-hoc rationalization and improves
  agreement with human evaluators (Arize AI, 2025).

- **Pointwise scoring is more robust than pairwise** for your use case.
  Pairwise preferences flip ~35% of the time with distractor features vs ~9%
  for pointwise (COLM 2025). Since quality gaps between 7B-14B models are
  substantial, pointwise absolute scores are the right choice.

- **One (prompt, model) evaluation per reasoning block** prevents cross-response
  contamination. Do not score two models' responses to the same prompt in a single
  reasoning chain — finish scoring one, write the result, then start the next.

- **Calibration source disclosure**: The calibration examples in
  `rubrics/calibration-examples.yaml` were scored by Claude Opus 4.6 (chatbot),
  the same model family as the evaluator. This provides intra-rater scale
  consistency but is not an independent validity check. Note this limitation
  when reporting results.

- **Expected_answer hints are hard anchors** — if the hint says "must use XOR
  accumulator, NOT early return" and the response uses early return, that's a
  correctness deduction regardless of code elegance.

- **Contamination awareness**: Prompts tagged `contamination_risk: high` get
  scored normally but excluded from `_clean` aggregates. Note contamination
  risk in summaries but don't deduct.

## When NOT to use this skill

- **Single model evaluation**: Use `/evaluate` instead. This skill's overhead
  (calibration, cross-model batching, reliability check) is not justified for
  one model.
- **Quick iteration on prompts**: Use `/evaluate` to check if a prompt
  differentiates. Use this skill for the final rigorous comparison.
- **Fewer than 2 models**: This skill requires 2+ result files.
