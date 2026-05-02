"""Routing discovery: strategy expansion, correctness checking, and analysis.

Implements the routing discovery pipeline from DESIGN-ROUTING.md:
1. Expand prompts across strategies (routes discover)
2. Check correctness against expected answers
3. Analyze the result matrix to find routing patterns (routes analyze)
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from rich.console import Console

from porchbench.assets import porchbench_version
from porchbench.backend import InferenceBackend
from porchbench.display import format_validation_badge
from porchbench.schemas import (
    BestRoute,
    DefaultComparison,
    Message,
    ModelOptions,
    PromptResult,
    RequestData,
    ResponseData,
    ResponseMessage,
    RoutingAnalysis,
    RoutingCell,
    RoutingHeadline,
    RoutingPattern,
    RoutingVerdict,
    RunMetadata,
    RunResult,
    RunSummary,
    Suite,
    SuiteReference,
    compute_derived_metrics,
)
from porchbench.suite import resolve_messages, resolve_options

console = Console()


# ---------------------------------------------------------------------------
# Strategy expansion
# ---------------------------------------------------------------------------


def count_discovery_runs(suite: Suite, models: list[str]) -> int:
    """Calculate total runs for a routing discovery: prompts × strategies × models."""
    n_strategies = max(len(suite.strategies), 1)
    return len(suite.prompts) * n_strategies * len(models)


async def _run_tool_use_discovery_cell(
    prompt,
    model: str,
    options: ModelOptions,
    messages: list[Message],
    strategy_name: str,
    suite_dir: Path | None,
    backend: InferenceBackend,
) -> PromptResult:
    """Run a single tool-use prompt for routing discovery and package as PromptResult."""
    from porchbench.runner import harness_result_to_prompt_result
    from porchbench.tool_runner import run_tool_use_prompt

    result = await run_tool_use_prompt(
        prompt=prompt,
        model=model,
        options=options,
        messages=messages,
        suite_dir=suite_dir,
        backend=backend,
    )
    return harness_result_to_prompt_result(
        prompt, options, messages, result, strategy=strategy_name,
    )


async def run_discovery(
    suite: Suite,
    suite_ref: SuiteReference,
    models: list[str],
    backend: InferenceBackend,
    output_dir: str | Path = "results",
    on_cell_complete: Callable[[str, bool], None] | None = None,
    suite_dir: Path | None = None,
) -> list[RunResult]:
    """Run routing discovery: every prompt × strategy × model.

    Produces one RunResult per model, with each PromptResult tagged
    with its strategy name and correctness check.
    """
    results_per_model: list[RunResult] = []

    strategies = suite.strategies if suite.strategies else {"universal": None}

    for model_name in models:
        console.print(f"\n[bold]Model: {model_name}[/bold]")

        from porchbench.runner import get_model_info_safe, get_system_info

        # Wrapped fetch — a transient connection error here (e.g. user
        # restarted Ollama between models) would otherwise propagate out
        # of asyncio.run() and kill the whole multi-model run, losing
        # every model scheduled after this one. The stub fallback hands
        # control to the per-cell try/except, which records each
        # downstream chat() failure as an error PromptResult and writes
        # the partial RunResult JSON for this model before moving on.
        model_info = await get_model_info_safe(model_name, backend)

        run_meta = RunMetadata(
            suite=suite_ref,
            model=model_info,
            system=await get_system_info(backend),
            porchbench_version=porchbench_version(),
        )

        prompt_results: list[PromptResult] = []
        failed = 0
        run_start = time.monotonic()

        cells_per_model = len(suite.prompts) * max(len(strategies), 1)
        cell_idx = 0

        for prompt in suite.prompts:
            options = resolve_options(suite.defaults.options, prompt)

            for strategy_name, strategy in strategies.items():
                sys_msg = strategy.system_message if strategy else None
                messages = resolve_messages(prompt, system_message=sys_msg)

                cell_idx += 1
                progress = f"[{cell_idx}/{cells_per_model}]"
                label = f"{prompt.id}/{strategy_name}"
                try:
                    if prompt.mode == "tool-use":
                        pr = await _run_tool_use_discovery_cell(
                            prompt, model_name, options, messages,
                            strategy_name, suite_dir, backend,
                        )
                        correct = pr.validation_passed
                    else:
                        chat_result = await backend.chat(
                            messages=[{"role": m.role, "content": m.content} for m in messages],
                            model=model_name,
                            options=options,
                        )

                        metrics = compute_derived_metrics(chat_result.metrics)

                        response_data = ResponseData(
                            message=ResponseMessage(
                                role=chat_result.role,
                                content=chat_result.content,
                            ),
                            done_reason=chat_result.done_reason,
                        )

                        correct = check_correctness(
                            chat_result.content,
                            prompt.expected_answer,
                        )

                        pr = PromptResult(
                            prompt_id=prompt.id,
                            category=prompt.category,
                            difficulty=prompt.difficulty,
                            tags=prompt.tags,
                            contamination_risk=prompt.contamination_risk,
                            options_used=options,
                            request=RequestData(messages=messages),
                            response=response_data,
                            metrics=metrics,
                            strategy=strategy_name,
                            correct=correct,
                            expected_answer=prompt.expected_answer,
                        )

                    prompt_results.append(pr)

                    status = "[green]ok[/green]" if correct else (
                        "[yellow]?[/yellow]" if correct is None else "[red]FAIL[/red]"
                    )
                    val_badge = format_validation_badge(pr)
                    console.print(f"  {progress} {label}: {status}{val_badge}")

                    if on_cell_complete:
                        on_cell_complete(label, True)

                except Exception as exc:
                    failed += 1
                    console.print(f"  [red]{progress} {label}: error — {exc}[/red]")

                    prompt_results.append(PromptResult(
                        prompt_id=prompt.id,
                        category=prompt.category,
                        difficulty=prompt.difficulty,
                        tags=prompt.tags,
                        contamination_risk=prompt.contamination_risk,
                        options_used=options,
                        request=RequestData(messages=messages),
                        response=ResponseData(
                            message=ResponseMessage(content=""),
                            done_reason=f"error: {exc}",
                        ),
                        strategy=strategy_name,
                        correct=False,
                        expected_answer=prompt.expected_answer,
                    ))

                    if on_cell_complete:
                        on_cell_complete(label, False)

        elapsed = time.monotonic() - run_start
        tps_values = [
            r.metrics.tokens_per_second
            for r in prompt_results
            if r.metrics.tokens_per_second is not None
        ]

        run_result = RunResult(
            run=run_meta,
            results=prompt_results,
            summary=RunSummary(
                total_prompts=len(prompt_results),
                completed=len(prompt_results) - failed,
                failed=failed,
                total_duration_s=round(elapsed, 2),
                avg_tokens_per_second=(
                    round(sum(tps_values) / len(tps_values), 2) if tps_values else None
                ),
            ),
        )

        # Write per-model result
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = run_result.run.timestamp.strftime("%Y-%m-%dT%H-%M-%S")
        model_slug = model_name.replace(":", "-").replace("/", "-")
        filename = f"{ts}_routing-discovery_{model_slug}.json"
        path = out_dir / filename
        path.write_text(run_result.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"  [green]Written: {path}[/green]")

        results_per_model.append(run_result)

    return results_per_model


# ---------------------------------------------------------------------------
# Correctness checking
# ---------------------------------------------------------------------------


def check_correctness(response_text: str, expected_answer: str | None) -> bool | None:
    """Check if a response contains the expected answer.

    Returns True/False for prompts with expected_answer, None for open-ended prompts.
    Uses numeric parsing for numbers, case-insensitive substring match otherwise.
    """
    if expected_answer is None:
        return None

    expected = expected_answer.strip()
    text = response_text.strip()

    if not text:
        return False

    # Try numeric comparison first
    expected_num = _parse_number(expected)
    if expected_num is not None:
        return _check_numeric(text, expected_num)

    # Fall back to case-insensitive substring match
    return expected.lower() in text.lower()


def _parse_number(s: str) -> float | None:
    """Try to parse a string as a number."""
    try:
        return float(s)
    except ValueError:
        return None


def _check_numeric(text: str, expected: float) -> bool:
    """Check if any number in the response matches the expected value."""
    # Extract all numbers from the response
    numbers = re.findall(r'-?\d+\.?\d*', text)
    for n_str in numbers:
        try:
            n = float(n_str)
            # Exact match for integers, close match for floats
            if expected == int(expected):
                if n == expected:
                    return True
            else:
                if abs(n - expected) < 0.01:
                    return True
        except ValueError:
            continue
    return False


# ---------------------------------------------------------------------------
# Routing analysis
# ---------------------------------------------------------------------------


def build_routing_matrix(runs: list[RunResult]) -> list[RoutingCell]:
    """Extract the flat routing matrix from multiple run results."""
    cells: list[RoutingCell] = []
    for run in runs:
        model_name = run.run.model.name
        for r in run.results:
            if r.strategy is None:
                continue
            latency_ms = (
                r.metrics.total_duration / 1e6
                if r.metrics.total_duration is not None else None
            )
            cells.append(RoutingCell(
                model=model_name,
                prompt_id=r.prompt_id,
                strategy=r.strategy,
                correct=r.correct,
                tokens_generated=r.metrics.eval_count,
                latency_ms=latency_ms,
                tokens_per_second=r.metrics.tokens_per_second,
            ))
    return cells


def analyze_routes(
    runs: list[RunResult],
    default_strategy: str = "universal",
) -> RoutingAnalysis:
    """Produce a routing analysis from routing discovery runs.

    Identifies best routes per problem, detects patterns, and produces
    the headline go/no-go verdict.
    """
    matrix = build_routing_matrix(runs)

    models_tested = sorted({c.model for c in matrix})
    strategies_tested = sorted({c.strategy for c in matrix})
    prompt_ids = sorted({c.prompt_id for c in matrix})

    # Identify the "default" model (largest by parameter size, or last in list)
    default_model = _identify_default_model(runs)

    # Group cells by prompt_id
    by_prompt: dict[str, list[RoutingCell]] = defaultdict(list)
    for c in matrix:
        by_prompt[c.prompt_id].append(c)

    # Find best route per problem and compare to default
    best_routes: list[BestRoute] = []
    routing_helps_count = 0

    for pid in prompt_ids:
        cells = by_prompt[pid]
        best = _find_best_cell(cells)
        default_cell = _find_cell(cells, default_model, default_strategy)

        vs_default = None
        if default_cell and best:
            quality_delta = None
            if best.correct is not None and default_cell.correct is not None:
                quality_delta = (
                    (1.0 if best.correct else 0.0) - (1.0 if default_cell.correct else 0.0)
                ) * 100.0

            token_savings = None
            if (best.tokens_generated is not None and default_cell.tokens_generated is not None
                    and default_cell.tokens_generated > 0):
                token_savings = (
                    (1 - best.tokens_generated / default_cell.tokens_generated) * 100.0
                )

            vs_default = DefaultComparison(
                default_model=default_model,
                default_strategy=default_strategy,
                default_correct=default_cell.correct,
                default_tokens=default_cell.tokens_generated,
                quality_delta_pp=round(quality_delta, 1) if quality_delta is not None else None,
                token_savings_pct=round(token_savings, 1) if token_savings is not None else None,
            )

            # Routing helps if best route differs from default AND is better
            if best and (best.model != default_model or best.strategy != default_strategy):
                if quality_delta is not None and quality_delta > 0:
                    routing_helps_count += 1
                elif quality_delta == 0 and token_savings is not None and token_savings > 10:
                    routing_helps_count += 1

        best_routes.append(BestRoute(
            prompt_id=pid,
            best_model=best.model if best else default_model,
            best_strategy=best.strategy if best else default_strategy,
            correct=best.correct if best else None,
            tokens=best.tokens_generated if best else None,
            vs_default=vs_default,
        ))

    # Detect inverse scaling (smaller model outperforms larger on correctness)
    inverse_scaling_count = _count_inverse_scaling(matrix, default_model, default_strategy)

    # Detect patterns by grouping prompts with similar best routes
    patterns = _detect_patterns(best_routes, runs)

    # Compute headline
    max_quality_gain = max(
        (br.vs_default.quality_delta_pp for br in best_routes
         if br.vs_default and br.vs_default.quality_delta_pp is not None),
        default=None,
    )
    max_cost_reduction = max(
        (br.vs_default.token_savings_pct for br in best_routes
         if br.vs_default and br.vs_default.token_savings_pct is not None),
        default=None,
    )

    total_problems = len(prompt_ids)
    inverse_rate = inverse_scaling_count / total_problems if total_problems > 0 else 0

    routing_worthwhile = (
        routing_helps_count >= 3
        and routing_helps_count / total_problems >= 0.1
    )

    headline = RoutingHeadline(
        inverse_scaling_detected=inverse_scaling_count > 0,
        inverse_scaling_rate=round(inverse_rate, 3),
        problems_where_routing_helps=routing_helps_count,
        problems_total=total_problems,
        max_quality_gain_pp=round(max_quality_gain, 1) if max_quality_gain is not None else None,
        max_cost_reduction_pct=round(max_cost_reduction, 1) if max_cost_reduction is not None else None,
        routing_worthwhile=routing_worthwhile,
    )

    # Compute aggregate improvement estimates
    quality_gains = [
        br.vs_default.quality_delta_pp for br in best_routes
        if br.vs_default and br.vs_default.quality_delta_pp is not None
        and br.vs_default.quality_delta_pp > 0
    ]
    token_savings_list = [
        br.vs_default.token_savings_pct for br in best_routes
        if br.vs_default and br.vs_default.token_savings_pct is not None
        and br.vs_default.token_savings_pct > 0
    ]

    verdict = RoutingVerdict(
        routing_recommended=routing_worthwhile,
        estimated_quality_improvement_pp=(
            round(sum(quality_gains) / len(quality_gains), 1) if quality_gains else None
        ),
        estimated_token_savings_pct=(
            round(sum(token_savings_list) / len(token_savings_list), 1) if token_savings_list else None
        ),
        caveat=f"Based on {total_problems} problems with greedy decoding. Re-run with sampling to validate.",
    )

    return RoutingAnalysis(
        models_tested=models_tested,
        strategies_tested=strategies_tested,
        headline=headline,
        best_route_per_problem=best_routes,
        patterns=patterns,
        verdict=verdict,
        matrix=matrix,
    )


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def _identify_default_model(runs: list[RunResult]) -> str:
    """Pick the default model (largest by parameter_size, fallback to last)."""
    best_name = runs[-1].run.model.name
    best_size = 0.0
    for run in runs:
        size_str = run.run.model.details.parameter_size or "0"
        size = _parse_param_size(size_str)
        if size > best_size:
            best_size = size
            best_name = run.run.model.name
    return best_name


def _parse_param_size(s: str) -> float:
    """Parse '7.6B' or '3.1B' into a float."""
    match = re.match(r'([\d.]+)\s*[BbMm]?', s)
    if match:
        return float(match.group(1))
    return 0.0


def _find_best_cell(cells: list[RoutingCell]) -> RoutingCell | None:
    """Find the best cell: prioritize correctness, then fewest tokens."""
    if not cells:
        return None
    # Sort: correct first (True > False > None), then fewest tokens
    def sort_key(c: RoutingCell) -> tuple:
        correct_rank = 0 if c.correct is True else (1 if c.correct is None else 2)
        tokens = c.tokens_generated if c.tokens_generated is not None else float('inf')
        return (correct_rank, tokens)
    return min(cells, key=sort_key)


def _find_cell(
    cells: list[RoutingCell], model: str, strategy: str
) -> RoutingCell | None:
    """Find a specific (model, strategy) cell."""
    for c in cells:
        if c.model == model and c.strategy == strategy:
            return c
    return None


def _count_inverse_scaling(
    matrix: list[RoutingCell], largest_model: str, strategy: str
) -> int:
    """Count problems where a smaller model outperforms the largest on the same strategy."""
    # Group by (prompt_id, strategy)
    by_ps: dict[tuple[str, str], dict[str, RoutingCell]] = defaultdict(dict)
    for c in matrix:
        by_ps[(c.prompt_id, c.strategy)][c.model] = c

    count = 0
    for (pid, strat), model_cells in by_ps.items():
        if strat != strategy:
            continue
        largest_cell = model_cells.get(largest_model)
        if largest_cell is None or largest_cell.correct is not True:
            for m, cell in model_cells.items():
                if m != largest_model and cell.correct is True:
                    count += 1
                    break

    return count


def _detect_patterns(
    best_routes: list[BestRoute],
    runs: list[RunResult],
) -> list[RoutingPattern]:
    """Group best routes by model+strategy combination to find patterns."""
    # Build a prompt metadata lookup
    prompt_meta: dict[str, dict] = {}
    for run in runs:
        for r in run.results:
            if r.prompt_id not in prompt_meta:
                prompt_meta[r.prompt_id] = {
                    "category": r.category,
                    "difficulty": r.difficulty,
                    "tags": r.tags,
                }

    # Group by (best_model, best_strategy)
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for br in best_routes:
        groups[(br.best_model, br.best_strategy)].append(br.prompt_id)

    patterns: list[RoutingPattern] = []
    for (model, strategy), pids in groups.items():
        if len(pids) < 2:
            continue

        # Describe the pattern based on common characteristics
        categories = {prompt_meta.get(p, {}).get("category", "unknown") for p in pids}
        difficulties = {prompt_meta.get(p, {}).get("difficulty", "unknown") for p in pids}

        cat_str = "/".join(sorted(categories))
        diff_str = "/".join(sorted(difficulties))
        desc = f"{cat_str} ({diff_str}): best with {model} + {strategy}"

        confidence = "high" if len(pids) >= 5 else ("medium" if len(pids) >= 3 else "low")

        patterns.append(RoutingPattern(
            description=desc,
            affected_problems=sorted(pids),
            recommended_route={"model": model, "strategy": strategy},
            confidence=confidence,
            evidence_count=len(pids),
        ))

    # Sort by evidence count descending
    patterns.sort(key=lambda p: p.evidence_count, reverse=True)
    return patterns
