"""Frontier model evaluation pass.

Sends each prompt+response pair from a run result to a frontier model
(Claude via Anthropic API) along with a scoring rubric. Produces a
Scorecard with per-prompt scores and aggregate breakdowns.

Methodology notes (see METHODOLOGY.md):
- Uses a different model family as judge than the models under test
  to avoid self-preference bias.
- Rubric-based absolute scoring on 1-5 scales with explicit criteria
  descriptions to reduce verbosity bias.
- Position bias is less of a concern here (single response, not pairwise)
  but becomes relevant if we add pairwise comparison later.
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from ollama_bench.schemas import (
    AggregateScores,
    Criterion,
    CriterionScore,
    EvaluationMetadata,
    PromptResult,
    PromptScore,
    Rubric,
    RubricMetadata,
    RunResult,
    Scorecard,
)

console = Console()

DEFAULT_EVALUATOR_MODEL = "claude-sonnet-4-6-20250514"


def load_rubric(path: str | Path) -> Rubric:
    """Load and validate a rubric YAML file."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Rubric.model_validate(data)


def build_scoring_prompt(
    prompt_result: PromptResult,
    rubric: Rubric,
) -> str:
    """Construct the evaluation prompt sent to the frontier model.

    Includes the original user prompt, the model's response, and the
    rubric criteria with scoring instructions.
    """
    criteria_block = "\n".join(
        f"- **{c.name}** (weight {c.weight}, scale {c.scale}): {c.description}"
        for c in rubric.criteria
    )

    user_messages = "\n\n".join(
        f"[{m.role}]: {m.content}" for m in prompt_result.request.messages
    )

    return f"""You are an expert evaluator assessing the quality of an AI model's response.

## Original Prompt
{user_messages}

## Model Response
{prompt_result.response.message.content}

## Scoring Rubric: {rubric.rubric.name} v{rubric.rubric.version}

Score the response on each criterion using the specified scale. Be rigorous and precise.

{criteria_block}

## Instructions

For each criterion, provide:
1. A numeric score on the specified scale
2. A brief rationale (1-2 sentences) justifying the score

Then provide a one-sentence overall summary.

Respond in this exact JSON format (no markdown fencing):
{{
  "criteria": {{
    "{rubric.criteria[0].name}": {{"score": <int>, "rationale": "<string>"}},
    ...one entry per criterion...
  }},
  "summary": "<one sentence overall assessment>"
}}"""


async def score_prompt(
    prompt_result: PromptResult,
    rubric: Rubric,
    client: anthropic.AsyncAnthropic,
    model: str = DEFAULT_EVALUATOR_MODEL,
) -> PromptScore:
    """Score a single prompt result against a rubric via the frontier model."""
    scoring_prompt = build_scoring_prompt(prompt_result, rubric)

    response = await client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": scoring_prompt}],
    )

    response_text = response.content[0].text
    parsed = _parse_scoring_response(response_text, rubric)

    # Compute weighted score
    weight_map = {c.name: c.weight for c in rubric.criteria}
    weighted = sum(
        parsed[name].score * weight_map.get(name, 0)
        for name in parsed
    )

    return PromptScore(
        prompt_id=prompt_result.prompt_id,
        criteria=parsed,
        weighted_score=round(weighted, 2),
        summary=_extract_summary(response_text),
    )


def _parse_scoring_response(
    text: str,
    rubric: Rubric,
) -> dict[str, CriterionScore]:
    """Parse the frontier model's JSON response into CriterionScores.

    Handles common LLM output quirks: markdown fencing, trailing commas.
    """
    # Strip markdown code fencing if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last lines (fencing)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    parsed = json.loads(cleaned)
    criteria_data = parsed.get("criteria", {})

    result: dict[str, CriterionScore] = {}
    for criterion in rubric.criteria:
        name = criterion.name
        if name in criteria_data:
            entry = criteria_data[name]
            result[name] = CriterionScore(
                score=int(entry["score"]),
                rationale=str(entry.get("rationale", "")),
            )
        else:
            # Criterion missing from response — record as 0 with note
            result[name] = CriterionScore(
                score=0,
                rationale="Criterion not scored by evaluator.",
            )

    return result


def _extract_summary(text: str) -> str:
    """Extract the summary field from the scoring response."""
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)
        parsed = json.loads(cleaned)
        return parsed.get("summary", "")
    except (json.JSONDecodeError, KeyError):
        return ""


def compute_aggregates(
    scores: list[PromptScore],
    results: list[PromptResult],
) -> AggregateScores:
    """Compute aggregate scores broken down by category and difficulty."""
    if not scores:
        return AggregateScores(overall_weighted=0.0)

    overall = sum(s.weighted_score for s in scores) / len(scores)

    # Build lookup from prompt_id to result metadata
    result_map = {r.prompt_id: r for r in results}

    # Group by category
    by_cat: dict[str, list[float]] = {}
    by_diff: dict[str, list[float]] = {}
    for s in scores:
        r = result_map.get(s.prompt_id)
        if r:
            by_cat.setdefault(r.category, []).append(s.weighted_score)
            by_diff.setdefault(r.difficulty, []).append(s.weighted_score)

    return AggregateScores(
        overall_weighted=round(overall, 2),
        by_category={k: round(sum(v) / len(v), 2) for k, v in by_cat.items()},
        by_difficulty={k: round(sum(v) / len(v), 2) for k, v in by_diff.items()},
    )


async def evaluate_run(
    run_result: RunResult,
    rubric: Rubric,
    evaluator_model: str = DEFAULT_EVALUATOR_MODEL,
    api_key: str | None = None,
) -> Scorecard:
    """Score all prompts in a run result. Returns a complete Scorecard."""
    client = anthropic.AsyncAnthropic(api_key=api_key) if api_key else anthropic.AsyncAnthropic()

    # Filter to completed prompts (skip failures)
    completed = [
        r for r in run_result.results
        if r.response.done_reason == "stop"
    ]

    scores: list[PromptScore] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating", total=len(completed))

        for prompt_result in completed:
            try:
                score = await score_prompt(prompt_result, rubric, client, evaluator_model)
                scores.append(score)
                progress.console.print(
                    f"  {prompt_result.prompt_id}: "
                    f"[green]{score.weighted_score:.2f}[/green]"
                )
            except Exception as exc:
                console.print(
                    f"  [red]{prompt_result.prompt_id}: evaluation failed — {exc}[/red]"
                )
            progress.advance(task)

    aggregates = compute_aggregates(scores, completed)

    rubric_label = f"{rubric.rubric.name} v{rubric.rubric.version}"

    return Scorecard(
        evaluation=EvaluationMetadata(
            run_id=run_result.run.id,
            evaluator=evaluator_model,
            rubric=rubric_label,
        ),
        scores=scores,
        aggregate=aggregates,
    )


def write_scorecard(scorecard: Scorecard, output_dir: str | Path = "scorecards") -> Path:
    """Write a scorecard to a timestamped JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = scorecard.evaluation.timestamp.strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"{ts}_{scorecard.evaluation.run_id[:8]}.json"

    path = output_dir / filename
    path.write_text(scorecard.model_dump_json(indent=2), encoding="utf-8")
    return path
