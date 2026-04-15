"""LLM-as-judge evaluation pass.

Sends each prompt+response pair from a run result to an evaluator model
along with a scoring rubric. Supports three backends:

- **ollama** (default): uses a local Ollama model as judge. Free, fast,
  good for iteration. Use a different model family than the models under
  test to avoid self-preference bias (e.g., judge qwen with deepseek-r1).
- **api**: uses Claude via Anthropic API. Higher quality but costs per token.
  Requires ANTHROPIC_API_KEY.
- **claude-code**: uses Claude Code CLI (claude -p). Frontier quality,
  uses existing subscription, no per-token API cost.

Methodology notes (see METHODOLOGY.md):
- Rubric-based absolute scoring on 1-5 scales with explicit criteria
  descriptions to reduce verbosity bias.
- Self-preference bias mitigated by using a different model family as judge.
- Position bias is less of a concern for single-response scoring but becomes
  relevant if pairwise comparison is added later.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Protocol

import yaml
from pydantic import BaseModel
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from feral.schemas import (
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

DEFAULT_OLLAMA_EVALUATOR = "gemma4:e4b"

# Per-backend default models — used when --evaluator is not explicitly set
EVAL_BACKEND_DEFAULTS: dict[str, str] = {
    "ollama": "gemma4:e4b",
    "api": "claude-sonnet-4-6-20250514",
    "claude-code": "sonnet",
}


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
        from feral import client

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


class ClaudeCodeEvalBackend:
    """Evaluator backend using Claude Code CLI (claude -p).

    Uses the user's Claude Code subscription — no per-token API cost.
    Requires 'claude' on PATH.
    """

    def __init__(self, model: str = "sonnet", timeout_s: int = 120):
        self.model = model
        self.timeout_s = timeout_s

    async def generate(self, prompt: str) -> str:
        cmd = [
            "claude", "-p",
            "--model", self.model,
            "--output-format", "text",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(
                f"claude -p timed out after {self.timeout_s}s"
            )

        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"claude -p failed (exit {proc.returncode}): {err_msg}"
            )

        return stdout.decode("utf-8")


# ---------------------------------------------------------------------------
# Rubric loading
# ---------------------------------------------------------------------------


def load_rubric(path: str | Path) -> Rubric:
    """Load and validate a rubric YAML file."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Rubric.model_validate(data)


# Mapping from prompt category to rubric filename (without .yaml)
_CATEGORY_RUBRIC_MAP = {
    "coding": "coding",
    "reasoning": "reasoning",
    "cross-domain": "cross-domain",
}


def load_rubric_dir(rubric_dir: str | Path) -> dict[str, Rubric]:
    """Load all rubric YAML files from a directory, keyed by category.

    Returns a dict mapping category name to Rubric. Files are matched
    by the _CATEGORY_RUBRIC_MAP. A 'default' key is added if default.yaml
    exists, used as fallback for unmatched categories.
    """
    rubric_dir = Path(rubric_dir)
    rubrics: dict[str, Rubric] = {}

    for category, filename in _CATEGORY_RUBRIC_MAP.items():
        path = rubric_dir / f"{filename}.yaml"
        if path.exists():
            rubrics[category] = load_rubric(path)

    default_path = rubric_dir / "default.yaml"
    if default_path.exists():
        rubrics["default"] = load_rubric(default_path)

    return rubrics


def select_rubric(
    category: str,
    rubrics: dict[str, Rubric],
    fallback: Rubric | None = None,
) -> Rubric:
    """Pick the best rubric for a prompt category."""
    if category in rubrics:
        return rubrics[category]
    if "default" in rubrics:
        return rubrics["default"]
    if fallback is not None:
        return fallback
    raise ValueError(f"No rubric found for category '{category}' and no fallback available")


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

    # Include correctness hints if available
    reference_block = ""
    if prompt_result.expected_answer:
        reference_block = f"""
## Reference (Correctness Guide)
The following describes what a correct response should include. Use this
to verify factual accuracy and completeness — do not penalize alternative
valid approaches that meet these criteria.

{prompt_result.expected_answer}
"""

    return f"""You are an expert evaluator assessing the quality of an AI model's response.

## Original Prompt
{user_messages}

## Model Response
{prompt_result.response.message.content}
{reference_block}
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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def normalize_score(raw: float, scale_min: float = 1.0, scale_max: float = 5.0) -> float:
    """Normalize a raw score to 0-100 where scale_min→0 and scale_max→100."""
    if scale_max == scale_min:
        return 0.0
    return max(0.0, (raw - scale_min) / (scale_max - scale_min) * 100)


def compute_aggregates(
    scores: list[PromptScore],
    results: list[PromptResult],
) -> AggregateScores:
    """Compute aggregate scores with normalization and contamination filtering."""
    if not scores:
        return AggregateScores(overall_weighted=0.0)

    overall = _mean([s.weighted_score for s in scores])

    result_map = {r.prompt_id: r for r in results}

    by_cat: dict[str, list[float]] = {}
    by_diff: dict[str, list[float]] = {}
    by_cat_clean: dict[str, list[float]] = {}
    by_diff_clean: dict[str, list[float]] = {}
    clean_scores: list[float] = []

    for s in scores:
        r = result_map.get(s.prompt_id)
        if r:
            by_cat.setdefault(r.category, []).append(s.weighted_score)
            by_diff.setdefault(r.difficulty, []).append(s.weighted_score)

            if r.contamination_risk != "high":
                clean_scores.append(s.weighted_score)
                by_cat_clean.setdefault(r.category, []).append(s.weighted_score)
                by_diff_clean.setdefault(r.difficulty, []).append(s.weighted_score)

    by_diff_raw = {k: round(_mean(v), 2) for k, v in by_diff.items()}

    # Normalized: difficulty-weighted average (equal weight per difficulty level)
    by_diff_norm = {k: round(normalize_score(_mean(v)), 2) for k, v in by_diff.items()}
    overall_norm = round(_mean(list(by_diff_norm.values())), 2) if by_diff_norm else None

    return AggregateScores(
        overall_weighted=round(overall, 2),
        by_category={k: round(_mean(v), 2) for k, v in by_cat.items()},
        by_difficulty=by_diff_raw,
        overall_normalized=overall_norm,
        by_difficulty_normalized=by_diff_norm,
        overall_weighted_clean=round(_mean(clean_scores), 2) if clean_scores else None,
        by_category_clean={k: round(_mean(v), 2) for k, v in by_cat_clean.items()},
        by_difficulty_clean={k: round(_mean(v), 2) for k, v in by_diff_clean.items()},
    )


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------


async def evaluate_run(
    run_result: RunResult,
    rubric: Rubric,
    backend: EvalBackend,
    evaluator_label: str = "",
    rubrics_by_category: dict[str, Rubric] | None = None,
) -> Scorecard:
    """Score all prompts in a run result. Returns a complete Scorecard.

    When rubrics_by_category is provided, each prompt is scored with the
    rubric matching its category. Falls back to the rubric parameter for
    unmatched categories.
    """
    # Include responses that completed (stop) or were truncated (length)
    # — truncated responses still have real content worth evaluating
    completed = [
        r for r in run_result.results
        if r.response.done_reason in ("stop", "length", None)
        and r.response.message.content
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
                if rubrics_by_category:
                    prompt_rubric = select_rubric(
                        prompt_result.category, rubrics_by_category, fallback=rubric
                    )
                else:
                    prompt_rubric = rubric

                score = await score_prompt(prompt_result, prompt_rubric, backend)
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
    if rubrics_by_category:
        cat_names = [r.rubric.name for r in rubrics_by_category.values()]
        rubric_label = f"category-aware ({', '.join(cat_names)})"

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


# ---------------------------------------------------------------------------
# Claude Code /evaluate skill helpers
#
# These functions support the interactive /evaluate workflow where Claude Code
# scores prompts one at a time and streams results to disk, rather than the
# automated evaluate_run() pipeline above.
# ---------------------------------------------------------------------------


class EvalPromptSummary(BaseModel):
    """Compact representation of a prompt+response for evaluation.

    Strips metrics, options, and raw Ollama fields to reduce context
    consumption when reading responses during interactive evaluation.
    """

    prompt_id: str
    category: str
    difficulty: str
    done_reason: str | None = None
    contamination_risk: str | None = None
    prompt_text: str  # flattened from request.messages
    response_text: str  # from response.message.content
    expected_answer: str | None = None


class EvalRunHeader(BaseModel):
    """Run-level metadata extracted alongside prompt summaries."""

    run_id: str
    model_name: str
    suite_name: str
    suite_file: str
    total_prompts: int
    truncated_count: int
    categories: dict[str, int]
    difficulties: dict[str, int]


class EvalData(BaseModel):
    """Complete pre-extracted evaluation data: header + compact prompts."""

    header: EvalRunHeader
    prompts: list[EvalPromptSummary]


def extract_eval_data(result_path: str | Path) -> EvalData:
    """Pre-extract compact evaluation data from a RunResult JSON file.

    Reads the full result file once and returns a lightweight EvalData
    object containing only the fields needed for scoring. This avoids
    repeated partial reads of a large result file during evaluation.
    """
    result_path = Path(result_path)
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    run_result = RunResult.model_validate(raw)

    prompts: list[EvalPromptSummary] = []
    categories: dict[str, int] = {}
    difficulties: dict[str, int] = {}
    truncated = 0

    for r in run_result.results:
        if not r.response.message.content:
            continue

        prompt_text = "\n\n".join(
            f"[{m.role}]: {m.content}" for m in r.request.messages
        )

        prompts.append(EvalPromptSummary(
            prompt_id=r.prompt_id,
            category=r.category,
            difficulty=r.difficulty,
            done_reason=r.response.done_reason,
            contamination_risk=r.contamination_risk,
            prompt_text=prompt_text,
            response_text=r.response.message.content,
            expected_answer=r.expected_answer,
        ))

        categories[r.category] = categories.get(r.category, 0) + 1
        difficulties[r.difficulty] = difficulties.get(r.difficulty, 0) + 1
        if r.response.done_reason == "length":
            truncated += 1

    header = EvalRunHeader(
        run_id=run_result.run.id,
        model_name=run_result.run.model.name,
        suite_name=run_result.run.suite.name,
        suite_file=run_result.run.suite.file,
        total_prompts=len(prompts),
        truncated_count=truncated,
        categories=categories,
        difficulties=difficulties,
    )

    return EvalData(header=header, prompts=prompts)


def append_score(score: PromptScore, scores_path: str | Path) -> None:
    """Append a single PromptScore as one JSON line to a JSONL file.

    Called after scoring each prompt so progress is persisted to disk
    incrementally. If the evaluation is interrupted, completed scores
    are preserved.
    """
    scores_path = Path(scores_path)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scores_path, "a", encoding="utf-8") as f:
        f.write(score.model_dump_json() + "\n")


def load_scores(scores_path: str | Path) -> list[PromptScore]:
    """Read all PromptScores from a JSONL file written by append_score()."""
    scores_path = Path(scores_path)
    scores: list[PromptScore] = []
    for line in scores_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            scores.append(PromptScore.model_validate_json(line))
    return scores


def build_scorecard_from_scores(
    scores_path: str | Path,
    result_path: str | Path,
    evaluator: str = "claude-code/claude-opus-4-6",
    rubric_label: str = "",
    output_dir: str | Path = "scorecards",
) -> Path:
    """Read streamed scores + original results, compute aggregates, write scorecard.

    This is the final step of the /evaluate skill workflow:
    1. extract_eval_data() ran at the start to produce compact eval data
    2. Scores were streamed to scores_path via append_score() during evaluation
    3. This function reads them back, computes aggregates, and writes the
       final scorecard JSON.
    """
    scores = load_scores(scores_path)
    raw = json.loads(Path(result_path).read_text(encoding="utf-8"))
    run_result = RunResult.model_validate(raw)

    aggregates = compute_aggregates(scores, run_result.results)

    scorecard = Scorecard(
        evaluation=EvaluationMetadata(
            run_id=run_result.run.id,
            evaluator=evaluator,
            rubric=rubric_label,
        ),
        scores=scores,
        aggregate=aggregates,
    )

    return write_scorecard(scorecard, output_dir)
