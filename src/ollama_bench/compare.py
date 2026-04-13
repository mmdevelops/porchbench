"""Cross-model comparison utilities.

Loads multiple run results, aligns them by prompt_id, and produces
side-by-side comparison tables for throughput metrics. When scorecards
are available, includes quality scores in the comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ollama_bench.metrics import describe, extract_tokens_per_second, summarize_run
from ollama_bench.schemas import PromptResult, RunResult, Scorecard

console = Console()


def load_run_result(path: str | Path) -> RunResult:
    """Load a run result JSON file."""
    path = Path(path)
    return RunResult.model_validate_json(path.read_text(encoding="utf-8"))


def load_scorecard(path: str | Path) -> Scorecard:
    """Load a scorecard JSON file."""
    path = Path(path)
    return Scorecard.model_validate_json(path.read_text(encoding="utf-8"))


def align_results(runs: list[RunResult]) -> dict[str, list[PromptResult | None]]:
    """Align prompt results across runs by prompt_id.

    Returns a dict mapping prompt_id → list of PromptResult (one per run,
    None if that run didn't include the prompt).
    """
    # Collect all prompt IDs in order of first appearance
    all_ids: list[str] = []
    seen: set[str] = set()
    for run in runs:
        for r in run.results:
            if r.prompt_id not in seen:
                all_ids.append(r.prompt_id)
                seen.add(r.prompt_id)

    # Build lookup per run
    lookups = []
    for run in runs:
        by_id = {r.prompt_id: r for r in run.results}
        lookups.append(by_id)

    aligned: dict[str, list[PromptResult | None]] = {}
    for pid in all_ids:
        aligned[pid] = [lookup.get(pid) for lookup in lookups]

    return aligned


def print_comparison_table(runs: list[RunResult], scorecards: list[Scorecard | None] | None = None) -> None:
    """Print a rich table comparing models across prompts.

    Shows tokens/sec and total tokens for each model on each prompt,
    plus quality scores if scorecards are provided.
    """
    if not runs:
        return

    model_names = [r.run.model.name for r in runs]
    aligned = align_results(runs)

    # Build score lookups if scorecards provided
    score_lookups: list[dict[str, float]] = []
    if scorecards:
        for sc in scorecards:
            if sc is not None:
                score_lookups.append({s.prompt_id: s.weighted_score for s in sc.scores})
            else:
                score_lookups.append({})
    has_scores = bool(score_lookups) and any(score_lookups)

    # Per-prompt comparison
    table = Table(title="Per-Prompt Comparison", title_style="bold")
    table.add_column("Prompt", style="bold")
    for name in model_names:
        table.add_column(f"{name}\ntok/s", justify="right")
        table.add_column(f"{name}\ntokens", justify="right")
        if has_scores:
            table.add_column(f"{name}\nscore", justify="right")

    for pid, results_row in aligned.items():
        row: list[str] = [pid]
        for i, pr in enumerate(results_row):
            if pr is not None and pr.metrics.tokens_per_second is not None:
                row.append(f"{pr.metrics.tokens_per_second:.1f}")
            else:
                row.append("-")

            if pr is not None and pr.metrics.eval_count is not None:
                row.append(str(pr.metrics.eval_count))
            else:
                row.append("-")

            if has_scores:
                score = score_lookups[i].get(pid) if i < len(score_lookups) else None
                row.append(f"{score:.2f}" if score is not None else "-")

        table.add_row(*row)

    console.print(table)
    console.print()

    # Summary comparison
    summary_table = Table(title="Model Summary", title_style="bold")
    summary_table.add_column("Metric", style="dim")
    for name in model_names:
        summary_table.add_column(name, justify="right")

    # Tokens per second
    tps_row = ["Avg tok/s"]
    for run in runs:
        tps = describe(extract_tokens_per_second(run.results))
        tps_row.append(f"{tps.mean:.1f}" if tps else "-")
    summary_table.add_row(*tps_row)

    # Total tokens generated
    tok_row = ["Total tokens"]
    for run in runs:
        total = sum(
            r.metrics.eval_count for r in run.results if r.metrics.eval_count is not None
        )
        tok_row.append(str(total))
    summary_table.add_row(*tok_row)

    # Total duration
    dur_row = ["Total time (s)"]
    for run in runs:
        dur_row.append(f"{run.summary.total_duration_s:.1f}")
    summary_table.add_row(*dur_row)

    # Completed/failed
    comp_row = ["Completed"]
    for run in runs:
        comp_row.append(f"{run.summary.completed}/{run.summary.total_prompts}")
    summary_table.add_row(*comp_row)

    # Quantization
    quant_row = ["Quantization"]
    for run in runs:
        quant_row.append(run.run.model.details.quantization_level or "-")
    summary_table.add_row(*quant_row)

    # Aggregate quality scores if available
    if has_scores:
        score_row = ["Avg score"]
        for i, sc in enumerate(scorecards or []):
            if sc is not None:
                score_row.append(f"{sc.aggregate.overall_weighted:.2f}")
            else:
                score_row.append("-")
        summary_table.add_row(*score_row)

    console.print(summary_table)
