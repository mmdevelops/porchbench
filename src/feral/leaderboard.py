"""Leaderboard: rank models from comparable scorecards.

Loads scorecards from a directory or explicit paths, groups by rubric,
and produces ranked tables with per-category/difficulty breakdowns
plus best/worst prompt highlights per model.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from feral.schemas import Scorecard

console = Console()


def load_scorecard(path: str | Path) -> Scorecard:
    path = Path(path)
    return Scorecard.model_validate_json(path.read_text(encoding="utf-8"))


def discover_scorecards(directory: str | Path) -> list[Scorecard]:
    """Load all scorecard JSON files from a directory."""
    directory = Path(directory)
    scorecards: list[Scorecard] = []
    for p in sorted(directory.glob("*.json")):
        try:
            scorecards.append(load_scorecard(p))
        except Exception:
            # Skip non-scorecard JSON files silently
            pass
    return scorecards


def filter_comparable(
    scorecards: list[Scorecard],
    strict: bool = False,
) -> list[Scorecard]:
    """Filter scorecards to a comparable set sharing the same rubric.

    Picks the rubric with the most scorecards. When strict=True, also
    requires the same evaluator. Warns on evaluator mismatches otherwise.
    """
    if not scorecards:
        return []

    # Group by rubric
    by_rubric: dict[str, list[Scorecard]] = defaultdict(list)
    for sc in scorecards:
        by_rubric[sc.evaluation.rubric].append(sc)

    # Pick the largest group
    best_rubric = max(by_rubric, key=lambda r: len(by_rubric[r]))
    group = by_rubric[best_rubric]

    if len(by_rubric) > 1:
        other_rubrics = [r for r in by_rubric if r != best_rubric]
        console.print(
            f"[yellow]Filtered to rubric '{best_rubric}' ({len(group)} scorecards). "
            f"Excluded {sum(len(by_rubric[r]) for r in other_rubrics)} scorecards "
            f"with different rubrics: {', '.join(repr(r) for r in other_rubrics)}[/yellow]"
        )

    # Check evaluator consistency
    evaluators = {sc.evaluation.evaluator for sc in group}
    if len(evaluators) > 1:
        if strict:
            # Pick the most common evaluator
            eval_counts: dict[str, int] = defaultdict(int)
            for sc in group:
                eval_counts[sc.evaluation.evaluator] += 1
            best_eval = max(eval_counts, key=lambda e: eval_counts[e])
            group = [sc for sc in group if sc.evaluation.evaluator == best_eval]
            console.print(
                f"[yellow]--strict: filtered to evaluator '{best_eval}' "
                f"({len(group)} scorecards). "
                f"Excluded scorecards with evaluators: "
                f"{', '.join(repr(e) for e in evaluators if e != best_eval)}[/yellow]"
            )
        else:
            console.print(
                f"[yellow]Warning: mixed evaluators in this rubric group: "
                f"{', '.join(repr(e) for e in sorted(evaluators))}. "
                f"Scores may not be directly comparable. Use --strict to enforce "
                f"same evaluator.[/yellow]"
            )

    return group


def _model_label(sc: Scorecard) -> str:
    """Best available label for the model in a scorecard."""
    return sc.evaluation.model_name or sc.evaluation.run_id[:8]


def print_leaderboard(scorecards: list[Scorecard], top_n: int = 3) -> None:
    """Print ranked leaderboard table and best/worst prompts per model."""
    if not scorecards:
        console.print("[yellow]No comparable scorecards found.[/yellow]")
        return

    # Sort by overall score descending
    ranked = sorted(scorecards, key=lambda sc: sc.aggregate.overall_weighted, reverse=True)

    rubric = ranked[0].evaluation.rubric
    evaluators = sorted({sc.evaluation.evaluator for sc in ranked})
    eval_label = evaluators[0] if len(evaluators) == 1 else f"{len(evaluators)} evaluators"

    # Collect all category and difficulty keys across all scorecards
    all_categories: list[str] = []
    all_difficulties: list[str] = []
    for sc in ranked:
        for cat in sc.aggregate.by_category:
            if cat not in all_categories:
                all_categories.append(cat)
        for diff in sc.aggregate.by_difficulty:
            if diff not in all_difficulties:
                all_difficulties.append(diff)

    # --- Main ranking table ---
    title = f"Leaderboard — {rubric}"
    table = Table(title=title, title_style="bold")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Model", style="bold")
    table.add_column("Overall", justify="right", style="bold")
    for cat in all_categories:
        table.add_column(cat, justify="right")
    for diff in all_difficulties:
        table.add_column(diff, justify="right")

    for rank, sc in enumerate(ranked, 1):
        agg = sc.aggregate
        row: list[str] = [
            str(rank),
            _model_label(sc),
            f"{agg.overall_weighted:.2f}",
        ]
        for cat in all_categories:
            val = agg.by_category.get(cat)
            row.append(f"{val:.2f}" if val is not None else "-")
        for diff in all_difficulties:
            val = agg.by_difficulty.get(diff)
            row.append(f"{val:.2f}" if val is not None else "-")
        table.add_row(*row)

    console.print(table)
    console.print(f"  Evaluator: {eval_label}   Scorecards: {len(ranked)}")
    console.print()

    # --- Best/worst prompts per model ---
    if top_n <= 0:
        return

    bw_table = Table(title="Best & Worst Prompts by Model", title_style="bold")
    bw_table.add_column("Model", style="bold")
    bw_table.add_column("Best", style="green")
    bw_table.add_column("Worst", style="red")

    for sc in ranked:
        if not sc.scores:
            continue
        sorted_scores = sorted(sc.scores, key=lambda s: s.weighted_score, reverse=True)
        best = sorted_scores[:top_n]
        worst = sorted_scores[-top_n:]

        best_str = "\n".join(f"{s.prompt_id} ({s.weighted_score:.2f})" for s in best)
        worst_str = "\n".join(f"{s.prompt_id} ({s.weighted_score:.2f})" for s in worst)

        bw_table.add_row(_model_label(sc), best_str, worst_str)

    console.print(bw_table)
