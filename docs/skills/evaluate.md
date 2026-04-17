---
name: evaluate
description: Frontier-model evaluation of benchmark results. Scores each prompt response against category-specific rubrics with chain-of-thought reasoning. Outputs a scorecard JSON compatible with porchbench compare and analysis tools. Use when the user wants to evaluate benchmark run results.
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

### Step 1: Extract compact evaluation data

Run the extraction command to produce a lightweight file with only the fields
needed for scoring. This avoids repeated partial reads of a large result file.

```bash
python -m porchbench eval-extract "$ARGUMENTS" --output .claude/eval-data.json
```

Then **read `.claude/eval-data.json`** to get:
- `header`: run_id, model_name, suite_name, suite_file, total_prompts,
  truncated_count, categories, difficulties
- `prompts`: list of `{prompt_id, category, difficulty, done_reason,
  contamination_risk, prompt_text, response_text, expected_answer}`

This replaces all partial reads of the original result JSON. You should not
need to read the original result file again during scoring.

### Step 2: Load rubrics

Rubrics live under `src/porchbench/data/rubrics/` in the repo (and ship bundled
with the installed wheel). A project-local `./rubrics/` directory, if present,
overrides the packaged copies.

- First read the suite YAML (at `header.suite_file`) and check for a `rubric`
  field in the `suite` header. If present, use that rubric name (e.g.
  `coding`, `cross-domain-science`) for ALL prompts in the run — this
  overrides per-category matching.
- Otherwise, match by category:
  - `coding.yaml` → category "coding"
  - `reasoning.yaml` → category "reasoning"
  - `cross-domain.yaml` → category "cross-domain"
  - `default.yaml` → fallback for unmatched categories

### Step 3: Load calibration examples

Load calibration examples from `calibration-examples.yaml` alongside the
resolved rubric (i.e. `src/porchbench/data/rubrics/calibration-examples.yaml` for
packaged runs, or `./rubrics/calibration-examples.yaml` for project-local) if it
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

### Step 4: Initialize scores file

Delete any stale scores file from a previous run:
```bash
rm -f .claude/eval-scores.jsonl
```

### Step 5: Present evaluation plan

Present the plan to the user using data from the extracted header:
- Model name, suite name, prompt count
- Category breakdown (how many coding, reasoning, cross-domain)
- Truncated responses
- Ask user to confirm before proceeding

## Pre-scan

Before scoring, scan the extracted prompts to identify:
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

Read each prompt from the extracted eval data (`.claude/eval-data.json`).
The fields you need are already extracted:
- `prompt_text` — the original prompt
- `response_text` — the model's response
- `done_reason` — "stop" or "length" (truncated)
- `expected_answer` — correctness hints if present

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

### Step 8: Stream the score to disk

After scoring each prompt, immediately append the score to the JSONL file.
Run a short Python snippet:

```python
python -c "
from porchbench.evaluator import append_score
from porchbench.schemas import PromptScore, CriterionScore
append_score(PromptScore(
    prompt_id='<id>',
    criteria={
        '<criterion>': CriterionScore(score=<N>, rationale='<text>'),
        ...
    },
    weighted_score=<float>,
    summary='<text>'
), '.claude/eval-scores.jsonl')
"
```

This persists progress incrementally. If the evaluation is interrupted,
completed scores are preserved and you can resume by checking what prompt_ids
are already in the JSONL.

**Important**: Keep rationale strings short (1-2 sentences) and avoid special
characters that break shell quoting (use straight quotes, no backslashes).
If a rationale is complex, simplify it to the key finding.

## Finalize: Write the Scorecard

After scoring all prompts, run the finalize command. This reads the JSONL
scores, loads the original result file for category/difficulty metadata,
computes all aggregates, and writes the scorecard JSON:

```bash
python -m porchbench eval-finalize "$ARGUMENTS" \
    --scores .claude/eval-scores.jsonl \
    --evaluator "claude-code/claude-opus-4-6" \
    --rubric "<rubric description>"
```

This replaces all inline Python for aggregation and scorecard writing.
The finalize command handles:
- Reading scores from the JSONL file
- Computing overall, by-category, by-difficulty means
- Normalizing to 0-100 scale
- Filtering contamination_risk: "high" for clean aggregates
- Writing timestamped scorecard to `scorecards/`

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

## Post-Hoc Diagnostics

After scoring all prompts and before presenting results, run these quick checks:

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

## Cleanup

After successful evaluation, clean up the working files:
```bash
rm -f .claude/eval-data.json .claude/eval-scores.jsonl
```
