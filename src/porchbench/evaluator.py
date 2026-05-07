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
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from porchbench._console import ensure_unicode_stdout
from porchbench.schemas import (
    AggregateScores,
    CriterionScore,
    EvalData,
    EvalPromptSummary,
    EvalRunHeader,
    EvaluationMetadata,
    ModelOptions,
    PromptResult,
    PromptScore,
    Rubric,
    RunResult,
    Scorecard,
)

console = Console()

# Per-backend default judge models. Cloud backends pin to a stable Anthropic-
# managed name. Ollama is intentionally absent — local model availability varies
# per machine, so the CLI prompts the user to pick (and persists the choice to
# .env) instead of hardcoding a model that may not be pulled.
EVAL_BACKEND_DEFAULTS: dict[str, str] = {
    "api": "claude-sonnet-4-6",
    "claude-code": "sonnet",
}

# Context window for the Ollama judge. Exposed at module level because the
# VRAM cofit preflight in `cli.py` needs it to size headroom for the judge's
# KV cache.
EVALUATOR_NUM_CTX = 32768


# ---------------------------------------------------------------------------
# Backend protocol — any callable that takes a prompt string and returns text
# ---------------------------------------------------------------------------


class EvalBackend(Protocol):
    async def generate(self, prompt: str) -> str: ...


class OllamaEvalBackend:
    """Evaluator backend using a local Ollama model."""

    def __init__(self, model: str, host: str | None = None):
        from porchbench.backend import OllamaBackend

        self.model = model
        self.host = host
        # One backend (and one AsyncClient connection pool) for the
        # lifetime of this evaluator. A 100-prompt eval used to spin up
        # 100 backends and 100 pools, paying TLS/setup per call and
        # leaking sockets to TIME_WAIT.
        self._backend = OllamaBackend(host=host)

    async def generate(self, prompt: str) -> str:
        options = ModelOptions(
            temperature=0, seed=42, num_predict=2048, num_ctx=EVALUATOR_NUM_CTX,
        )
        result = await self._backend.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            options=options,
        )
        return result.content


class AnthropicEvalBackend:
    """Evaluator backend using Claude via Anthropic API."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
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
        except TimeoutError:
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


_BACKEND_FACTORIES: dict[str, type] = {
    "ollama": OllamaEvalBackend,
    "api": AnthropicEvalBackend,
    "anthropic": AnthropicEvalBackend,
    "claude-code": ClaudeCodeEvalBackend,
}


def make_backend(name: str, **kwargs) -> EvalBackend:
    """Construct an evaluator backend by name without importing each class.

    Accepts "ollama", "api" (alias "anthropic"), and "claude-code". Keyword
    arguments are forwarded to the backend constructor; see each class for
    its accepted parameters (e.g. `model`, `host`, `api_key`, `timeout_s`).

    Raises ValueError on an unknown name.
    """
    try:
        cls = _BACKEND_FACTORIES[name]
    except KeyError:
        valid = ", ".join(sorted(set(_BACKEND_FACTORIES)))
        raise ValueError(
            f"Unknown evaluator backend {name!r}. Valid names: {valid}"
        )
    return cls(**kwargs)


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
# Calibration examples
# ---------------------------------------------------------------------------


def load_calibration_examples(path: str | Path) -> dict[str, list[dict]]:
    """Load calibration examples from YAML, keyed by rubric name.

    Returns e.g. {"coding": [...], "reasoning": [...], "cross-domain-science": [...]}.
    Returns empty dict if the file doesn't exist or can't be parsed.
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data.get("calibration", {})
    except Exception:
        return {}


# Mapping from rubric name to calibration key. The calibration YAML uses
# rubric-style keys; suite rubric hints may differ slightly.
_CALIBRATION_KEY_MAP = {
    "coding": "coding",
    "reasoning": "reasoning",
    "cross-domain": "cross-domain",
    "cross-domain-science": "cross-domain-science",
    "default": "coding",  # fallback
}


def _resolve_calibration_key(rubric: Rubric) -> str:
    """Determine which calibration set matches a rubric."""
    name = rubric.rubric.name.lower().replace(" rubric", "").replace(" ", "-")
    return _CALIBRATION_KEY_MAP.get(name, name)


def format_calibration_preamble(
    calibration_data: dict[str, list[dict]],
    rubric: Rubric,
) -> str:
    """Format calibration examples as a few-shot preamble for the scoring prompt.

    Selects the calibration set matching the rubric, then formats each tier
    (strong, adequate, weak) as a scored example with rationale.
    """
    key = _resolve_calibration_key(rubric)
    examples = calibration_data.get(key)
    if not examples:
        return ""

    lines = [
        "## Calibration Examples",
        "",
        "Before scoring, review these reference examples to anchor the 1-5 scale.",
        "Each shows a scored response at a different quality tier.",
        "",
    ]

    for ex in examples:
        tier = ex.get("tier", "unknown")
        scores = ex.get("scores", {})
        weighted = ex.get("weighted_score", "n/a")
        rationale = ex.get("rationale", "").strip()
        prompt_summary = ex.get("prompt_summary", "").strip()
        response_summary = ex.get("response_summary", "").strip()

        score_strs = [f"{k}: {v}" for k, v in scores.items()]

        lines.append(f"### {tier.title()}")
        lines.append(f"**Prompt:** {prompt_summary}")
        lines.append(f"**Response:** {response_summary}")
        lines.append(f"**Scores:** {', '.join(score_strs)} → weighted {weighted}")
        lines.append(f"**Why:** {rationale}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("Now score the following response using the same scale anchoring.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scoring prompt construction
# ---------------------------------------------------------------------------


def _format_scoring_prompt(
    formatted_user_prompt: str,
    response_text: str,
    rubric: Rubric,
    expected_answer: str | None = None,
    calibration_preamble: str = "",
) -> str:
    """String-only core of the scoring prompt template.

    Shared between `build_scoring_prompt` (PromptResult-shaped input from
    the batch pipeline) and `evaluate_single` (raw strings from external
    consumers like the agent harness bridge). Keeps the wording and JSON
    schema in one place.
    """
    criteria_block = "\n".join(
        f"- **{c.name}** (weight {c.weight}, scale {c.scale}): {c.description}"
        for c in rubric.criteria
    )

    criteria_json = ", ".join(
        f'"{c.name}": {{"score": <int>, "rationale": "<string>"}}'
        for c in rubric.criteria
    )

    reference_block = ""
    if expected_answer:
        reference_block = f"""
## Reference (Correctness Guide)
The following describes what a correct response should include. Use this
to verify factual accuracy and completeness — do not penalize alternative
valid approaches that meet these criteria.

{expected_answer}
"""

    return f"""You are an expert evaluator assessing the quality of an AI model's response.
{calibration_preamble}
## Original Prompt
{formatted_user_prompt}

## Model Response
{response_text}
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


def build_scoring_prompt(
    prompt_result: PromptResult,
    rubric: Rubric,
    calibration_preamble: str = "",
) -> str:
    """Construct the evaluation prompt sent to the judge model."""
    user_messages = "\n\n".join(
        f"[{m.role}]: {m.content}" for m in prompt_result.request.messages
    )
    return _format_scoring_prompt(
        formatted_user_prompt=user_messages,
        response_text=prompt_result.response.message.content,
        rubric=rubric,
        expected_answer=prompt_result.expected_answer,
        calibration_preamble=calibration_preamble,
    )


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------


async def score_prompt(
    prompt_result: PromptResult,
    rubric: Rubric,
    backend: EvalBackend,
    calibration_preamble: str = "",
) -> PromptScore:
    """Score a single prompt result against a rubric via the evaluator backend."""
    scoring_prompt = build_scoring_prompt(prompt_result, rubric, calibration_preamble)
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


async def evaluate_single(
    prompt_text: str,
    response_text: str,
    rubric: Rubric,
    backend: EvalBackend,
    *,
    prompt_id: str = "single",
    expected_answer: str | None = None,
    calibration_preamble: str = "",
) -> PromptScore:
    """Score one (prompt, response) pair against a rubric.

    Convenience entry point for callers that have a single response in
    memory and don't want to round-trip through RunResult JSON files —
    e.g. the agent-harness bridge scoring a single agent turn against a
    research rubric.

    `prompt_text` is the user prompt as it would appear at `[user]:` in
    the conversation; `response_text` is the assistant reply being
    judged. Pass `expected_answer` to surface a correctness reference to
    the judge for factual prompts.

    Returns a `PromptScore` with per-criterion scores, a weighted
    aggregate, and a one-sentence summary. Raises whatever the backend
    raises on transport / API failure.
    """
    ensure_unicode_stdout()

    scoring_prompt = _format_scoring_prompt(
        formatted_user_prompt=f"[user]: {prompt_text}",
        response_text=response_text,
        rubric=rubric,
        expected_answer=expected_answer,
        calibration_preamble=calibration_preamble,
    )
    judge_response = await backend.generate(scoring_prompt)
    parsed = _parse_scoring_response(judge_response, rubric)

    weight_map = {c.name: c.weight for c in rubric.criteria}
    weighted = sum(
        parsed[name].score * weight_map.get(name, 0)
        for name in parsed
    )

    return PromptScore(
        prompt_id=prompt_id,
        criteria=parsed,
        weighted_score=round(weighted, 2),
        summary=_extract_summary(judge_response),
    )


def evaluate_single_sync(
    prompt_text: str,
    response_text: str,
    rubric: Rubric,
    backend: EvalBackend,
    *,
    prompt_id: str = "single",
    expected_answer: str | None = None,
    calibration_preamble: str = "",
) -> PromptScore:
    """Synchronous wrapper around `evaluate_single`.

    Convenience for scripts and bridge code that aren't async-native.
    Internally calls `asyncio.run`, which means each call opens and
    closes a fresh event loop.

    **Backend reuse caveat:** `OllamaEvalBackend` (and any backend
    holding an `httpx.AsyncClient`) keys its connection pool by the
    running event loop. Reusing the same backend instance across
    multiple `evaluate_single_sync` calls is supported because the
    backend rebuilds its client on loop change, but you'll pay TLS /
    pool setup on every call. For batch workloads, prefer the async
    `evaluate_single` from inside one `asyncio.run` so the pool is
    reused.
    """
    return asyncio.run(evaluate_single(
        prompt_text=prompt_text,
        response_text=response_text,
        rubric=rubric,
        backend=backend,
        prompt_id=prompt_id,
        expected_answer=expected_answer,
        calibration_preamble=calibration_preamble,
    ))


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
        lines = [line for line in lines if not line.strip().startswith("```")]
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
    calibration_data: dict[str, list[dict]] | None = None,
) -> Scorecard:
    """Score all prompts in a run result. Returns a complete Scorecard.

    When rubrics_by_category is provided, each prompt is scored with the
    rubric matching its category. Falls back to the rubric parameter for
    unmatched categories. When calibration_data is provided, few-shot
    calibration examples are prepended to each scoring prompt.
    """
    ensure_unicode_stdout()

    # Include every non-errored response. Truncated responses still have real
    # content worth evaluating; empty responses (e.g. reasoning-mode models
    # that exhaust num_predict inside <think> without emitting an answer) are
    # scored 0 below so aggregate comparisons stay consistent across models
    # rather than silently dropping truncated prompts from the mean.
    scorable = [
        r for r in run_result.results
        if r.response.done_reason in ("stop", "length", None)
    ]

    if not scorable:
        # Tool-use and routing-discovery suites finish with `done_reason` set
        # to a harness `stopped_reason` ("done", "max_tool_calls", etc.) that
        # falls outside the inference filter above. Those suites use
        # deterministic validators (`validation_passed`) and aren't meant for
        # LLM-judging, so silent 0/0 results are by design — but the
        # difference matters for callers who can't tell "everything failed"
        # from "nothing was eligible". Be loud about it.
        console.print(
            "[yellow]No prompts eligible for LLM-judge eval — "
            "this run contains only harness/tool-use results which use "
            "deterministic validators. Inspect `validation_passed` and "
            "`tool_use_metrics` on the RunResult instead, or run a "
            "coding / reasoning / cross-domain suite for LLM-judge "
            "scoring.[/yellow]"
        )

    # Pre-compute calibration preambles per rubric to avoid reformatting each prompt
    _preamble_cache: dict[str, str] = {}

    def _get_preamble(prompt_rubric: Rubric) -> str:
        if not calibration_data:
            return ""
        key = _resolve_calibration_key(prompt_rubric)
        if key not in _preamble_cache:
            _preamble_cache[key] = format_calibration_preamble(calibration_data, prompt_rubric)
        return _preamble_cache[key]

    scores: list[PromptScore] = []

    # Use the ASCII "line" spinner (|/-\) instead of the default braille
    # dots so the Progress header is renderable on cp1252 / non-Unicode
    # captured streams even before ensure_unicode_stdout takes effect.
    with Progress(
        SpinnerColumn(spinner_name="line"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating", total=len(scorable))

        for prompt_result in scorable:
            if rubrics_by_category:
                prompt_rubric = select_rubric(
                    prompt_result.category, rubrics_by_category, fallback=rubric
                )
            else:
                prompt_rubric = rubric

            if not prompt_result.response.message.content:
                # No answer reached the evaluator — score 0 across criteria so
                # aggregates stay honest. Common cause on reasoning-mode
                # models: `num_predict` exhausted inside <think> before any
                # user-facing answer was emitted.
                reason = (
                    "truncated before answer emitted (try think: false or a larger num_predict)"
                    if prompt_result.response.done_reason == "length"
                    else "empty response from model"
                )
                zero_score = PromptScore(
                    prompt_id=prompt_result.prompt_id,
                    criteria={
                        c.name: CriterionScore(score=0, rationale=reason)
                        for c in prompt_rubric.criteria
                    },
                    weighted_score=0.0,
                    summary=reason,
                )
                scores.append(zero_score)
                progress.console.print(
                    f"  {prompt_result.prompt_id}: "
                    f"[red]0.00[/red] [dim]({reason})[/dim]"
                )
                progress.advance(task)
                continue

            try:
                preamble = _get_preamble(prompt_rubric)
                score = await score_prompt(prompt_result, prompt_rubric, backend, preamble)
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

    aggregates = compute_aggregates(scores, scorable)

    rubric_label = f"{rubric.rubric.name} v{rubric.rubric.version}"
    if rubrics_by_category:
        cat_names = [r.rubric.name for r in rubrics_by_category.values()]
        rubric_label = f"category-aware ({', '.join(cat_names)})"

    return Scorecard(
        evaluation=EvaluationMetadata(
            run_id=run_result.run.id,
            evaluator=evaluator_label,
            rubric=rubric_label,
            model_name=run_result.run.model.name,
            suite_name=run_result.run.suite.name,
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


async def batch_evaluate_results(
    result_paths: list[Path],
    eval_backend: EvalBackend,
    backend_label: str,
    output_dir: Path,
    explicit_rubric_path: Path | None = None,
    rubrics_by_category: dict[str, Rubric] | None = None,
    skip_scored: bool = False,
) -> list[tuple[str, str, float | None]]:
    """Score N result files sequentially with one evaluator backend instance.

    Shared orchestration used by the `evaluate` CLI batch mode and by
    `run --evaluate`'s post-run phase. Rubric + calibration are cached
    across results so a shared rubric only loads once.

    Returns a list of (run_label, status, overall_score) tuples, one per
    input path. Status is one of:

    - `"scored"`: at least one prompt was LLM-judged; `overall_score` is set.
    - `"skipped"`: a prior scorecard already exists and `skip_scored=True`.
    - `"no_eligible"`: the run had no prompts eligible for LLM-judging — see
      eligibility rules below. A scorecard is still written (with empty
      `scores`) so downstream tooling can distinguish this case from a
      genuine 0.0 score; `overall_score` is None.
    - `"failed"`: load or evaluation error; the result is logged and the
      batch continues with the next file.

    Eligibility — which prompts get LLM-judged:

    - **Inference suites** (coding, reasoning, cross-domain): all prompts
      whose `done_reason` is `"stop"`, `"length"`, or None. Truncated
      responses are still scored (with a 0 if no answer was emitted) so
      aggregates stay honest.
    - **Tool-use and routing-discovery suites**: zero prompts are
      LLM-judged. These run through the harness, which sets `done_reason`
      to a `stopped_reason` ("done", "max_tool_calls", "max_turns",
      "error") that falls outside the inference filter. They use
      deterministic validators (`PromptResult.validation_passed` and
      `tool_use_metrics`) instead, and surface as `status="no_eligible"`.

    Errors on one result don't abort the batch — they're reported and the
    next result proceeds.
    """
    from porchbench.assets import find_rubric
    from porchbench.errors import UserError, load_json_model

    rubric_cache: dict[Path, tuple[Path, Rubric, dict]] = {}

    def resolve_for(run_result: RunResult) -> tuple[Path, Rubric, dict]:
        rpath = explicit_rubric_path
        if rpath is None:
            hint = run_result.run.suite.rubric
            rpath = find_rubric(hint) if hint else find_rubric("default")
        if rpath not in rubric_cache:
            loaded = load_rubric(rpath)
            cal_file = rpath.parent / "calibration-examples.yaml"
            cal = load_calibration_examples(cal_file) if cal_file.exists() else {}
            rubric_cache[rpath] = (rpath, loaded, cal)
        return rubric_cache[rpath]

    summary: list[tuple[str, str, float | None]] = []

    for idx, rp in enumerate(result_paths, 1):
        prefix = f"({idx}/{len(result_paths)}) " if len(result_paths) > 1 else ""
        console.print(f"{prefix}[bold]{rp.name}[/bold]")

        try:
            run_result = load_json_model(rp, RunResult, "run result")
        except UserError as exc:
            console.print(f"  [red]load failed: {exc}[/red]")
            summary.append((rp.name, "failed", None))
            continue

        run_label = f"{run_result.run.model.name} ({run_result.run.id[:8]})"

        if skip_scored:
            existing = list(output_dir.glob(f"*_{run_result.run.id[:8]}.json"))
            if existing:
                console.print(f"  [yellow]skipped (scorecard exists: {existing[0].name})[/yellow]")
                summary.append((run_label, "skipped", None))
                continue

        try:
            _, rubric, calibration_data = resolve_for(run_result)
        except Exception as exc:
            console.print(f"  [red]rubric resolution failed: {exc}[/red]")
            summary.append((run_label, "failed", None))
            continue

        try:
            scorecard = await evaluate_run(
                run_result, rubric, eval_backend,
                evaluator_label=backend_label,
                rubrics_by_category=rubrics_by_category,
                calibration_data=calibration_data or None,
            )
            written = write_scorecard(scorecard, output_dir)
        except Exception as exc:
            console.print(f"  [red]evaluation failed: {exc}[/red]")
            summary.append((run_label, "failed", None))
            continue

        # Distinguish "no prompts were eligible for LLM-judge eval" (e.g.
        # tool-use / routing-discovery results) from a real 0.0 score so
        # consumers don't misread the second case as 'everything failed'.
        if not scorecard.scores:
            console.print(
                f"  [yellow]no eligible prompts — scorecard written ({written.name})[/yellow]"
            )
            summary.append((run_label, "no_eligible", None))
            continue

        overall = scorecard.aggregate.overall_weighted
        console.print(f"  [green]scored — {overall:.2f} → {written.name}[/green]")
        summary.append((run_label, "scored", overall))

    return summary


def batch_evaluate_results_sync(
    result_paths: list[Path],
    eval_backend: EvalBackend,
    backend_label: str,
    output_dir: Path,
    explicit_rubric_path: Path | None = None,
    rubrics_by_category: dict[str, Rubric] | None = None,
    skip_scored: bool = False,
) -> list[tuple[str, str, float | None]]:
    """Synchronous wrapper around `batch_evaluate_results`.

    Convenience for scripts and bridge code that aren't async-native.
    Internally calls `asyncio.run`, which opens and closes a fresh event
    loop. Status values include the new `"no_eligible"` (run had no
    LLM-judgeable prompts — typically tool-use / routing-discovery
    results scored by deterministic validators).

    **Backend reuse caveat:** see `evaluate_single_sync`. Each call to
    this function spins up a fresh event loop; a backend with a cached
    `httpx.AsyncClient` will rebuild its pool on each call.
    """
    return asyncio.run(batch_evaluate_results(
        result_paths=result_paths,
        eval_backend=eval_backend,
        backend_label=backend_label,
        output_dir=output_dir,
        explicit_rubric_path=explicit_rubric_path,
        rubrics_by_category=rubrics_by_category,
        skip_scored=skip_scored,
    ))


# ---------------------------------------------------------------------------
# Claude Code /evaluate skill helpers
#
# These functions support the interactive /evaluate workflow where Claude Code
# scores prompts one at a time and streams results to disk, rather than the
# automated evaluate_run() pipeline above.
# ---------------------------------------------------------------------------


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
            model_name=run_result.run.model.name,
            suite_name=run_result.run.suite.name,
        ),
        scores=scores,
        aggregate=aggregates,
    )

    return write_scorecard(scorecard, output_dir)
