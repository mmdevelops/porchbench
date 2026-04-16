"""Leaderboard: rank models from comparable scorecards.

Loads scorecards from a directory or explicit paths, groups by rubric,
and produces ranked tables with per-category/difficulty breakdowns
plus best/worst prompt highlights per model.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pydantic
from rich.console import Console
from rich.table import Table

from feral.schemas import Scorecard

console = Console()

# Sentinel rationales written by evaluator.py on parse/scoring failures.
PARSE_FAIL_RATIONALE = "Failed to parse evaluator response."
MISSING_CRITERION_RATIONALE = "Criterion not scored by evaluator."


# ---------------------------------------------------------------------------
# Loading and discovery
# ---------------------------------------------------------------------------


def load_scorecard(path: str | Path) -> Scorecard:
    path = Path(path)
    return Scorecard.model_validate_json(path.read_text(encoding="utf-8"))


def discover_scorecards(
    directory: str | Path, verbose: bool = False,
) -> list[Scorecard]:
    """Load all scorecard JSON files from a directory.

    Reports skipped files with categorized reasons instead of failing silently.
    """
    directory = Path(directory)
    scorecards: list[Scorecard] = []
    skipped: dict[str, list[str]] = {
        "corrupt": [],
        "schema": [],
        "read_error": [],
        "other": [],
    }

    json_files = sorted(directory.glob("*.json"))
    for p in json_files:
        try:
            scorecards.append(load_scorecard(p))
        except json.JSONDecodeError:
            skipped["corrupt"].append(p.name)
        except pydantic.ValidationError:
            skipped["schema"].append(p.name)
        except (PermissionError, OSError):
            skipped["read_error"].append(p.name)
        except Exception:
            skipped["other"].append(p.name)

    total_skipped = sum(len(v) for v in skipped.values())
    if total_skipped > 0:
        parts = []
        if skipped["corrupt"]:
            parts.append(f"{len(skipped['corrupt'])} corrupt")
        if skipped["schema"]:
            parts.append(f"{len(skipped['schema'])} schema mismatch")
        if skipped["read_error"]:
            parts.append(f"{len(skipped['read_error'])} unreadable")
        if skipped["other"]:
            parts.append(f"{len(skipped['other'])} other")
        console.print(
            f"[yellow]Skipped {total_skipped} of {len(json_files)} files "
            f"({', '.join(parts)})[/yellow]"
        )
        if verbose:
            for category, names in skipped.items():
                for name in names:
                    console.print(f"  [dim]{name}: {category}[/dim]")

    return scorecards


# ---------------------------------------------------------------------------
# Rubric normalization and grouping
# ---------------------------------------------------------------------------


def _normalize_rubric(rubric: str) -> str:
    """Normalize rubric string for grouping. Strips version, qualifiers, whitespace."""
    s = rubric.strip().lower()
    s = re.sub(r"\s+v\d+(\.\d+)*$", "", s)  # trailing " v1.0"
    s = re.sub(r"\s*\(.*?\)\s*$", "", s)  # trailing parentheticals
    s = re.sub(r"\s+", " ", s).strip()
    return s


def group_scorecards(scorecards: list[Scorecard]) -> dict[str, list[Scorecard]]:
    """Group scorecards by normalized rubric name."""
    groups: dict[str, list[Scorecard]] = defaultdict(list)
    for sc in scorecards:
        key = _normalize_rubric(sc.evaluation.rubric)
        groups[key].append(sc)
    return dict(groups)


def filter_comparable(
    scorecards: list[Scorecard],
    strict: bool = False,
) -> list[Scorecard]:
    """Filter scorecards to a comparable set sharing the same rubric.

    Groups by normalized rubric, picks the largest group. Warns when exact
    rubric strings differ within a group (version changes). When strict=True,
    also requires the same evaluator.
    """
    if not scorecards:
        return []

    groups = group_scorecards(scorecards)

    # Pick the largest group
    best_key = max(groups, key=lambda k: len(groups[k]))
    group = groups[best_key]

    if len(groups) > 1:
        excluded = sum(len(v) for k, v in groups.items() if k != best_key)
        other_keys = [k for k in groups if k != best_key]
        console.print(
            f"[yellow]Filtered to rubric '{best_key}' ({len(group)} scorecards). "
            f"Excluded {excluded} scorecards with different rubrics: "
            f"{', '.join(repr(k) for k in other_keys)}[/yellow]"
        )

    # Warn on rubric version differences within the group
    exact_rubrics = {sc.evaluation.rubric for sc in group}
    if len(exact_rubrics) > 1:
        console.print(
            f"[yellow]Note: rubric group contains variant labels: "
            f"{', '.join(repr(r) for r in sorted(exact_rubrics))}. "
            f"Scores may reflect different rubric versions.[/yellow]"
        )

    # Check evaluator consistency
    evaluators = {sc.evaluation.evaluator for sc in group}
    if len(evaluators) > 1:
        if strict:
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


# ---------------------------------------------------------------------------
# Model label resolution
# ---------------------------------------------------------------------------

_model_name_cache: dict[str, str] = {}


def _build_model_name_cache(result_dir: Path) -> None:
    """Scan result files once to map run_id -> model name."""
    if _model_name_cache or not result_dir.is_dir():
        return
    for p in result_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            run = data.get("run", {})
            run_id = run.get("id")
            model_name = run.get("model", {}).get("name")
            if run_id and model_name:
                _model_name_cache[run_id] = model_name
        except Exception:
            continue


def _model_label(sc: Scorecard, result_dir: Path | None = None) -> str:
    """Best available label for the model in a scorecard.

    Priority: model_name > cross-reference run_id in result files >
    suite_name/run_id[:8] > run_id[:8].
    """
    if sc.evaluation.model_name:
        return sc.evaluation.model_name

    if result_dir:
        _build_model_name_cache(result_dir)
        cached = _model_name_cache.get(sc.evaluation.run_id)
        if cached:
            return cached

    if sc.evaluation.suite_name:
        return f"{sc.evaluation.suite_name}/{sc.evaluation.run_id[:8]}"

    return sc.evaluation.run_id[:8]


# ---------------------------------------------------------------------------
# Parse failure detection
# ---------------------------------------------------------------------------


def _count_parse_failures(sc: Scorecard) -> tuple[int, int]:
    """Count prompts with parse failures and missing criteria.

    Returns (fully_failed, partially_failed) counts.
    """
    full_fail = 0
    partial_fail = 0
    for ps in sc.scores:
        rationales = [cs.rationale for cs in ps.criteria.values()]
        if all(r == PARSE_FAIL_RATIONALE for r in rationales):
            full_fail += 1
        elif any(
            r in (PARSE_FAIL_RATIONALE, MISSING_CRITERION_RATIONALE)
            for r in rationales
        ):
            partial_fail += 1
    return full_fail, partial_fail


def _clean_overall(sc: Scorecard) -> float | None:
    """Compute overall score excluding fully parse-failed prompts.

    Returns None if there are no failures (clean == raw).
    """
    clean_scores = [
        ps.weighted_score
        for ps in sc.scores
        if not all(
            cs.rationale == PARSE_FAIL_RATIONALE
            for cs in ps.criteria.values()
        )
    ]
    if len(clean_scores) == len(sc.scores):
        return None  # no failures
    if not clean_scores:
        return None  # all failed
    return sum(clean_scores) / len(clean_scores)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_leaderboard(
    scorecards: list[Scorecard],
    top_n: int = 3,
    result_dir: Path | None = None,
) -> None:
    """Print ranked leaderboard table and best/worst prompts per model."""
    if not scorecards:
        console.print("[yellow]No comparable scorecards found.[/yellow]")
        return

    # Sort by overall score descending
    ranked = sorted(
        scorecards, key=lambda sc: sc.aggregate.overall_weighted, reverse=True,
    )

    rubric = ranked[0].evaluation.rubric
    evaluators = sorted({sc.evaluation.evaluator for sc in ranked})
    eval_label = (
        evaluators[0] if len(evaluators) == 1 else f"{len(evaluators)} evaluators"
    )

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

    # Check if any scorecard has parse failures (determines whether to show Flags column)
    has_any_failures = any(
        _count_parse_failures(sc) != (0, 0) for sc in ranked
    )

    # --- Main ranking table ---
    title = f"Leaderboard — {rubric}"
    table = Table(title=title, title_style="bold")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Model", style="bold")
    table.add_column("Overall", justify="right", style="bold")
    if has_any_failures:
        table.add_column("Flags", justify="left")
    for cat in all_categories:
        table.add_column(cat, justify="right")
    for diff in all_difficulties:
        table.add_column(diff, justify="right")

    for rank, sc in enumerate(ranked, 1):
        agg = sc.aggregate
        label = _model_label(sc, result_dir)

        # Overall score with clean score annotation
        clean = _clean_overall(sc)
        if clean is not None:
            overall_str = f"{agg.overall_weighted:.2f} ({clean:.2f} clean)"
        else:
            overall_str = f"{agg.overall_weighted:.2f}"

        row: list[str] = [str(rank), label, overall_str]

        if has_any_failures:
            full_fail, partial_fail = _count_parse_failures(sc)
            flags: list[str] = []
            if full_fail:
                flags.append(f"[red]{full_fail} parse fail[/red]")
            if partial_fail:
                flags.append(f"[yellow]{partial_fail} partial[/yellow]")
            row.append(", ".join(flags) if flags else "[green]ok[/green]")

        for cat in all_categories:
            val = agg.by_category.get(cat)
            row.append(f"{val:.2f}" if val is not None else "-")
        for diff in all_difficulties:
            val = agg.by_difficulty.get(diff)
            row.append(f"{val:.2f}" if val is not None else "-")
        table.add_row(*row)

    console.print(table)
    console.print(f"  Evaluator: {eval_label}   Scorecards: {len(ranked)}")

    if has_any_failures:
        console.print(
            "  [dim]Flags: 'parse fail' = evaluator returned invalid JSON "
            "(all criteria scored 0). 'partial' = some criteria missing. "
            "'clean' score excludes fully failed prompts.[/dim]"
        )

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
        sorted_scores = sorted(
            sc.scores, key=lambda s: s.weighted_score, reverse=True,
        )
        best = sorted_scores[:top_n]
        worst = sorted_scores[-top_n:]

        best_str = "\n".join(
            f"{s.prompt_id} ({s.weighted_score:.2f})" for s in best
        )
        worst_str = "\n".join(
            f"{s.prompt_id} ({s.weighted_score:.2f})" for s in worst
        )

        bw_table.add_row(_model_label(sc, result_dir), best_str, worst_str)

    console.print(bw_table)
