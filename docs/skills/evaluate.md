---
name: evaluate
description: Frontier-model evaluation of benchmark results. Scores each prompt response against category-specific rubrics with chain-of-thought reasoning. Outputs a scorecard JSON compatible with feral compare and analysis tools. Use when the user wants to evaluate benchmark run results.
disable-model-invocation: true
---

# Evaluate Benchmark Results

Score benchmark run results using frontier-model judgment. You are the evaluator —
apply rubrics rigorously with evidence-based reasoning.

## Input

The user provides a path to a run result JSON file:
```
/evaluate results/2026-04-14T01-28-42_coding-basics_gemma4-e4b.json
```

The argument is: `$ARGUMENTS`

## Setup

1. Read the result file at the path provided in `$ARGUMENTS`
2. Parse it as a `RunResult` JSON (schema: `src/feral/schemas.py`)
3. Load rubrics from `rubrics/` directory:
   - First read the suite YAML (at `run.suite.file`) and check for a `rubric`
     field in the `suite` header. If present, use `rubrics/{rubric}.yaml` for
     ALL prompts in the run — this overrides per-category matching.
   - Otherwise, match by category:
     - `rubrics/coding.yaml` → category "coding"
     - `rubrics/reasoning.yaml` → category "reasoning"
     - `rubrics/cross-domain.yaml` → category "cross-domain"
     - `rubrics/default.yaml` → fallback for unmatched categories
4. Load calibration examples from `rubrics/calibration-examples.yaml` (if it
   exists). Select the calibration set matching the rubric being used:
   - `coding` set → for coding-basics or coding-heavy suites
   - `reasoning` set → for reasoning-focused prompts
   - `cross-domain-science` set → for the cross-domain science suite
   - `cross-domain` set → for coding-basics cross-domain prompts (architecture)
   - If the suite mixes categories, read the set matching the dominant category
   Review the three-tier examples (strong/adequate/weak) to anchor the 1-5 scale.
   This is the single highest-impact accuracy control (AutoRubric, 2025: +3pp
   with few-shot calibration). State briefly: "Calibration reviewed — a 5 looks
   like [X], a 3 looks like [Y], a 1 looks like [Z]."
5. Count total prompts to evaluate — include both `done_reason: "stop"` and
   `done_reason: "length"` (truncated but still has content)
6. Present the evaluation plan to the user:
   - Model name, suite name, prompt count
   - Category breakdown (how many coding, reasoning, cross-domain)
   - Ask user to confirm before proceeding

## Pre-scan

Before scoring, scan all responses to identify:
- **Truncated responses** (`done_reason: "length"`) — list them upfront so you
  know where to expect completeness gaps
- **Coding prompts with testable code** — flag these for execution verification
- **Category distribution** — know which rubric applies to each batch

## Evaluation Protocol

Process prompts in batches of 3-5 to manage context. Smaller batches reduce
scoring drift from context accumulation (Chroma context rot study: accuracy
drops >30% for content in mid-context positions).

### Triage: fast-track vs deep-review

Not every prompt needs the same depth of analysis:

- **Fast-track** (clean 5/5 cases): If the response is obviously correct, complete,
  and well-written (e.g., fizzbuzz), score quickly with brief rationale. Don't
  write a paragraph of evidence for a textbook solution.
- **Deep-review** (ambiguous or buggy cases): If something looks off, slow down.
  Trace the code mentally or execute it. Write detailed evidence. These are the
  scores that matter most for cross-model differentiation.

Spend your analysis budget where it creates signal, not on confirming obvious passes.

### Step 1: Read the prompt and response
- Read the original prompt (`request.messages`)
- Read the model's response (`response.message.content`)
- Note if truncated (`done_reason: "length"`)
- Read the `expected_answer` correctness hints if present

### Step 2: Select the rubric
- If the suite declares a `rubric` field, use that rubric for all prompts
- Otherwise, match `category` to the appropriate rubric file
- Coding rubric: correctness(0.40), completeness(0.20), code_quality(0.20),
  reasoning(0.10), domain_knowledge(0.10)
- Reasoning rubric: accuracy(0.35), clarity(0.30), depth(0.20), completeness(0.15)
- Cross-domain rubric: domain_knowledge(0.30), design_quality(0.25),
  reasoning(0.25), completeness(0.15), correctness(0.05)
- Cross-domain science rubric: domain_knowledge(0.30), correctness(0.25),
  reasoning(0.25), completeness(0.15), design_quality(0.05)

### Step 3: For coding prompts — EXECUTE the code

This is a major advantage over chatbot-based evaluation. Use the Bash tool to
run the model's code and verify it works:

```python
# Extract the code block from the response and run it
python -c "<model's code here>"
```

Check:
- Does it run without errors?
- Does it produce correct output for the examples given?
- Does it handle edge cases mentioned in the expected_answer?

If execution reveals a bug that looks correct on visual inspection (e.g., the
Fibonacci iterator producing 0,1,2,3,5 instead of 0,1,1,2,3,5), this is a
critical finding. Score correctness accordingly and note the specific failure
in the rationale.

Only skip execution for prompts where the code is clearly a design/architecture
response (e.g., class design without runnable main), or for non-coding categories.

### Step 4: Evidence-based reasoning (BEFORE scoring)
For each criterion in the rubric, reason through the evidence:

- **What the response does well** on this criterion
- **What the response gets wrong or misses** — be specific (line-level if possible)
- **How the correctness hints apply** — does the response meet the ground truth requirements?
- **If truncated** — note what appears to be missing and whether the truncation
  materially affects this criterion

Write this reasoning out. The reasoning comes BEFORE the score — do not score
first and rationalize after. This is the most important methodological point.

### Step 5: Score each criterion
After reasoning, assign a score on the 1-5 scale:

```
1 — Fundamentally wrong or missing. Would not pass basic review.
2 — Partially addresses the criterion but has significant errors or gaps.
3 — Adequate. Meets basic requirements but lacks depth or has minor errors.
4 — Good. Solid work with minor issues. Would pass code review.
5 — Excellent. Thorough, correct, and demonstrates genuine understanding.
```

**Use the full scale.** LLM judges exhibit central tendency bias — compressing
scores into a narrow 3-4 range (IRT reliability study, 2025). A response that
ignores half the prompt requirements is a 1-2, not a 3. A response that nails
every requirement with genuine insight is a 5, not a 4. If your scores for a
run cluster within a 1.5-point range, you are likely under-differentiating.

### Step 6: Compute weighted score
Calculate the weighted score for this prompt using the rubric weights.
Round to 2 decimal places.

### Step 7: Write a one-sentence summary
Capture the key quality signal — what stood out (good or bad).

**Rationale specificity rule:** Rationales must be specific enough to be useful
for cross-model comparison. "Sequence has bug" is useless. "Produces 0,1,2,3,5
instead of 0,1,1,2,3,5 due to state update skipping second iteration" is
actionable. When deducting, name the specific failure.

## Output Format

After evaluating all prompts, write a scorecard JSON file to the `scorecards/`
directory. The file must match the `Scorecard` schema in `src/feral/schemas.py`:

```json
{
  "evaluation": {
    "run_id": "<from result file>",
    "evaluator": "claude-code/claude-opus-4-6",
    "rubric": "category-aware (Coding Rubric, Reasoning Rubric, Cross-Domain Rubric)",
    "timestamp": "<ISO 8601>"
  },
  "scores": [
    {
      "prompt_id": "code-fizzbuzz",
      "criteria": {
        "correctness": {"score": 5, "rationale": "..."},
        "completeness": {"score": 4, "rationale": "..."}
      },
      "weighted_score": 4.35,
      "summary": "..."
    }
  ],
  "aggregate": {
    "overall_weighted": 4.12,
    "by_category": {"coding": 4.25, "reasoning": 3.90},
    "by_difficulty": {"easy": 4.50, "medium": 4.10, "hard": 3.80},
    "overall_normalized": 78.0,
    "by_difficulty_normalized": {"easy": 87.5, "medium": 77.5, "hard": 70.0},
    "overall_weighted_clean": 4.05,
    "by_category_clean": {},
    "by_difficulty_clean": {}
  }
}
```

Use Bash to run a short Python script that computes aggregates from your scored
data and writes the scorecard JSON. Build the scores list as a Python literal
in the script — do not write a separate helper file. Example pattern:

```python
python -c "
import json, os
from datetime import datetime, timezone
scores = [...]  # your scored data
# ... compute aggregates ...
os.makedirs('scorecards', exist_ok=True)
# ... write JSON ...
"
```

Naming convention: `scorecards/{timestamp}_{run_id_first_8_chars}.json`

## Methodology Notes

- **You are a different model family** from all test subjects (Ollama models).
  This eliminates self-preference bias entirely.
- **Evidence before scoring** reduces variance and increases agreement with
  human evaluators (Arize AI, 2025).
- **Correctness hints are hard anchors** — if the `expected_answer` says the
  function must check divisibility by 15 before 3 or 5, and the response
  doesn't, that's a correctness deduction regardless of how elegant the code is.
- **Truncated responses** — evaluate what's present, but note the truncation.
  Deduct on completeness only if the missing content would have been material.
  Do not deduct on correctness for code that was cut off mid-function.
- **Contamination awareness** — prompts tagged `contamination_risk: high` may
  get artificially good responses due to training data memorization. Note this
  in the summary but do not deduct — contamination filtering happens at
  aggregation time, not scoring time. `contamination_risk: None` means "not
  tagged" — treat as non-contaminated for clean scoring.

## Aggregation

After scoring all prompts, compute aggregates:

- **overall_weighted**: mean of all weighted_scores
- **by_category**: mean weighted_score per category
- **by_difficulty**: mean weighted_score per difficulty
- **overall_normalized**: normalize each difficulty mean from 1-5 scale to 0-100
  (where 1→0, 5→100), then average across difficulties (equal weight per level)
- **Contamination-filtered (_clean)**: exclude prompts with `contamination_risk: "high"`,
  recompute means

## Post-Hoc Diagnostics

After scoring all prompts, run these quick checks before presenting results:

- **Score distribution**: compute mean, std, min, max of weighted_scores. Flag if
  std < 0.5 (scores too compressed — central tendency bias likely active) or if
  all scores fall within a 1.5-point range.
- **Criterion independence**: for each pair of criteria, note if scores move in
  lockstep across all prompts. If two criteria always get the same score, the
  halo effect may be dominating — one criterion is not adding signal. Report
  any suspicious pairs.

These are quick mental checks, not formal statistics. Note findings when
presenting results.

## Presenting Results

After writing the scorecard, present a summary table to the user showing:
- Per-prompt scores (prompt_id, category, difficulty, weighted_score, summary)
- Aggregate scores (overall, by category, by difficulty)
- Normalized scores
- Clean vs unfiltered comparison
- Any notable findings (e.g., patterns in failures, truncation effects, domain gaps)
- Diagnostic flags (score compression, criterion lockstep) if any
