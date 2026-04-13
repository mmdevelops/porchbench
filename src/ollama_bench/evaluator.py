"""LLM-as-judge evaluation pass.

Sends each prompt+response pair from a run result to an evaluator model
along with a scoring rubric. Supports two backends:

- **ollama** (default): uses a local Ollama model as judge. Free, fast,
  good for iteration. Use a different model family than the models under
  test to avoid self-preference bias (e.g., judge qwen with deepseek-r1).
- **api**: uses Claude via Anthropic API. Higher quality but costs per token.
  Requires ANTHROPIC_API_KEY.

Methodology notes (see METHODOLOGY.md):
- Rubric-based absolute scoring on 1-5 scales with explicit criteria
  descriptions to reduce verbosity bias.
- Self-preference bias mitigated by using a different model family as judge.
- Position bias is less of a concern for single-response scoring but becomes
  relevant if pairwise comparison is added later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

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
    CriterionScore,
    EvaluationMetadata,
    Message,
    ModelOptions,
    PromptResult,
    PromptScore,
    Rubric,
    RubricMetadata,
    RunResult,
    Scorecard,
)

console = Console()

DEFAULT_OLLAMA_EVALUATOR = "deepseek-r1:14b"


# ---------------------------------------------------------------------------
# Backend protocol — any callable that takes a prompt string and returns text
# ---------------------------------------------------------------------------


class EvalBackend(Protocol):
    async def generate(self, prompt: str) -> str: ...


class OllamaEvalBackend:
    """Evaluator backend using a local Ollama model."""

    def __init__(self, model: str, host: str | None = None):
        self.model = model
        self.host = host

    async def generate(self, prompt: str) -> str:
        from ollama_bench import client

        messages = [Message(role="user", content=prompt)]
        options = ModelOptions(temperature=0, seed=42, num_predict=2048, num_ctx=8192)
        response = await client.chat(messages, self.model, options, host=self.host)
        return response.message.content or ""


class AnthropicEvalBackend:
    """Evaluator backend using Claude via Anthropic API."""

    def __init__(self, model: str = "claude-sonnet-4-6-20250514", api_key: str | None = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for API evaluation. "
                "Install it with: pip install anthropic"
            )
        self.model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key) if api_key else anthropic.AsyncAnthropic()

    async def generate(self, prompt: str) -> str:
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


# ---------------------------------------------------------------------------
# Rubric loading
# ---------------------------------------------------------------------------


def load_rubric(path: str | Path) -> Rubric:
    """Load and validate a rubric YAML file."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Rubric.model_validate(data)


# ---------------------------------------------------------------------------
# Scoring prompt construction
# ---------------------------------------------------------------------------


def build_scoring_prompt(
    prompt_result: PromptResult,
    rubric: Rubric,
) -> str:
    """Construct the evaluation prompt sent to the judge model."""
    criteria_block = "\n".join(
        f"- **{c.name}** (weight {c.weight}, scale {c.scale}): {c.description}"
        for c in rubric.criteria
    )

    user_messages = "\n\n".join(
        f"[{m.role}]: {m.content}" for m in prompt_result.request.messages
    )

    criteria_json = ", ".join(
        f'"{c.name}": {{"score": <int>, "rationale": "<string>"}}'
        for c in rubric.criteria
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

You MUST respond with valid JSON and nothing else. No markdown fencing, no extra text.

{{"criteria": {{{criteria_json}}}, "summary": "<one sentence overall assessment>"}}"""


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------


async def score_prompt(
    prompt_result: PromptResult,
    rubric: Rubric,
    backend: EvalBackend,
) -> PromptScore:
    """Score a single prompt result against a rubric via the evaluator backend."""
    scoring_prompt = build_scoring_prompt(prompt_result, rubric)
    response_text = await backend.generate(scoring_prompt)

    parsed = _parse_scoring_response(response_text, rubric)

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
    """Parse the judge model's JSON response into CriterionScores.

    Handles common LLM output quirks: markdown fencing, thinking tags,
    text before/after JSON.
    """
    cleaned = _extract_json(text)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Return zeros if we can't parse
        return {
            c.name: CriterionScore(score=0, rationale="Failed to parse evaluator response.")
            for c in rubric.criteria
        }

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
            result[name] = CriterionScore(
                score=0,
                rationale="Criterion not scored by evaluator.",
            )

    return result


def _extract_json(text: str) -> str:
    """Extract JSON from LLM output that may contain extra text.

    Handles: markdown fencing, <think> tags (deepseek-r1), preamble text.
    """
    cleaned = text.strip()

    # Strip <think>...</think> blocks (deepseek-r1 reasoning)
    import re
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

    # Strip markdown code fencing
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    # Try to find JSON object if there's surrounding text
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        if start >= 0:
            # Find matching closing brace
            depth = 0
            for i in range(start, len(cleaned)):
                if cleaned[i] == "{":
                    depth += 1
                elif cleaned[i] == "}":
                    depth -= 1
                    if depth == 0:
                        cleaned = cleaned[start:i + 1]
                        break

    return cleaned


def _extract_summary(text: str) -> str:
    """Extract the summary field from the scoring response."""
    try:
        cleaned = _extract_json(text)
        parsed = json.loads(cleaned)
        return parsed.get("summary", "")
    except (json.JSONDecodeError, KeyError):
        return ""


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def compute_aggregates(
    scores: list[PromptScore],
    results: list[PromptResult],
) -> AggregateScores:
    """Compute aggregate scores broken down by category and difficulty."""
    if not scores:
        return AggregateScores(overall_weighted=0.0)

    overall = sum(s.weighted_score for s in scores) / len(scores)

    result_map = {r.prompt_id: r for r in results}

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


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------


async def evaluate_run(
    run_result: RunResult,
    rubric: Rubric,
    backend: EvalBackend,
    evaluator_label: str = "",
) -> Scorecard:
    """Score all prompts in a run result. Returns a complete Scorecard."""
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
                score = await score_prompt(prompt_result, rubric, backend)
                scores.append(score)
                progress.console.print(
                    f"  {prompt_result.prompt_id}: "
                    f"[green]{score.weighted_score:.2f}[/green]"
                )
            except Exception as exc:
                console.print(
                    f"  [red]{prompt_result.prompt_id}: evaluation failed -- {exc}[/red]"
                )
            progress.advance(task)

    aggregates = compute_aggregates(scores, completed)

    rubric_label = f"{rubric.rubric.name} v{rubric.rubric.version}"

    return Scorecard(
        evaluation=EvaluationMetadata(
            run_id=run_result.run.id,
            evaluator=evaluator_label,
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
