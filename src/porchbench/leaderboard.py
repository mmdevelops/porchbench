"""Leaderboard: rank models from comparable scorecards.

Loads scorecards from a directory or explicit paths, groups by rubric,
and produces ranked tables with per-category/difficulty breakdowns
plus best/worst prompt highlights per model.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pydantic
from rich.console import Console
from rich.table import Table

from porchbench.schemas import Scorecard

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
    evaluator: str | None = None,
) -> list[Scorecard]:
    """Filter scorecards to a comparable set sharing the same rubric.

    Groups by normalized rubric, picks the largest group. Warns when exact
    rubric strings differ within a group (version changes). When strict=True,
    also requires the same evaluator (auto-picks the largest evaluator
    sub-group). When `evaluator` is provided, filters to that exact evaluator
    label instead of auto-picking — implies strict.
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

    # Explicit --evaluator takes priority: pin to that judge regardless of size
    if evaluator is not None:
        if evaluator not in evaluators:
            console.print(
                f"[red]No scorecards found with evaluator '{evaluator}'. "
                f"Available evaluators in this rubric group: "
                f"{', '.join(repr(e) for e in sorted(evaluators))}[/red]"
            )
            return []
        excluded_evals = [e for e in evaluators if e != evaluator]
        group = [sc for sc in group if sc.evaluation.evaluator == evaluator]
        if excluded_evals:
            console.print(
                f"[yellow]--evaluator: filtered to '{evaluator}' "
                f"({len(group)} scorecards). Excluded scorecards with "
                f"evaluators: {', '.join(repr(e) for e in excluded_evals)}[/yellow]"
            )
    elif len(evaluators) > 1:
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
                f"{', '.join(repr(e) for e in evaluators if e != best_eval)}. "
                f"Use --evaluator <label> to pick a different judge.[/yellow]"
            )
        else:
            console.print(
                f"[yellow]Warning: mixed evaluators in this rubric group: "
                f"{', '.join(repr(e) for e in sorted(evaluators))}. "
                f"Scores may not be directly comparable. Use --strict to enforce "
                f"same evaluator (auto-picks largest), or --evaluator <label> "
                f"to pin a specific judge.[/yellow]"
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
# Model-level aggregation (collapses repeats into one row per model)
# ---------------------------------------------------------------------------


@dataclass
class ModelRow:
    """One model's aggregated view across any number of scorecards (repeats)."""

    label: str
    n: int
    mean_overall: float
    overall_range: float  # max - min; 0.0 when identical or n==1
    mean_clean: float | None  # None when no repeat had parse failures
    cat_means: dict[str, float] = field(default_factory=dict)
    diff_means: dict[str, float] = field(default_factory=dict)
    total_full_fail: int = 0
    total_partial_fail: int = 0
    pooled_prompt_means: dict[str, float] = field(default_factory=dict)


def aggregate_by_model(
    scorecards: list[Scorecard],
    result_dir: Path | None = None,
) -> list[ModelRow]:
    """Group scorecards by model label and aggregate across repeats.

    Multiple scorecards for the same model (e.g. from `overnight --repeats N`)
    collapse into a single ModelRow with mean scores and a repeat count.
    Per-prompt best/worst lookups pool scores by prompt_id across repeats and
    average — so a prompt that was consistently easy for a model stays ranked
    as "best" even if one judge call produced noise.
    """
    groups: dict[str, list[Scorecard]] = defaultdict(list)
    for sc in scorecards:
        groups[_model_label(sc, result_dir)].append(sc)

    rows: list[ModelRow] = []
    for label, group in groups.items():
        overalls = [sc.aggregate.overall_weighted for sc in group]
        mean_overall = sum(overalls) / len(overalls)
        overall_range = max(overalls) - min(overalls) if len(overalls) > 1 else 0.0

        cleans = [c for c in (_clean_overall(sc) for sc in group) if c is not None]
        mean_clean = sum(cleans) / len(cleans) if cleans else None

        cat_keys = {c for sc in group for c in sc.aggregate.by_category}
        cat_means = {
            cat: sum(
                sc.aggregate.by_category[cat] for sc in group if cat in sc.aggregate.by_category
            ) / sum(1 for sc in group if cat in sc.aggregate.by_category)
            for cat in cat_keys
        }

        diff_keys = {d for sc in group for d in sc.aggregate.by_difficulty}
        diff_means = {
            diff: sum(
                sc.aggregate.by_difficulty[diff] for sc in group if diff in sc.aggregate.by_difficulty
            ) / sum(1 for sc in group if diff in sc.aggregate.by_difficulty)
            for diff in diff_keys
        }

        total_full = 0
        total_partial = 0
        for sc in group:
            full, partial = _count_parse_failures(sc)
            total_full += full
            total_partial += partial

        prompt_scores: dict[str, list[float]] = defaultdict(list)
        for sc in group:
            for ps in sc.scores:
                prompt_scores[ps.prompt_id].append(ps.weighted_score)
        pooled = {pid: sum(vals) / len(vals) for pid, vals in prompt_scores.items()}

        rows.append(ModelRow(
            label=label,
            n=len(group),
            mean_overall=mean_overall,
            overall_range=overall_range,
            mean_clean=mean_clean,
            cat_means=cat_means,
            diff_means=diff_means,
            total_full_fail=total_full,
            total_partial_fail=total_partial,
            pooled_prompt_means=pooled,
        ))

    return rows


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


_SPARK_CHARS = " ▁▂▃▄▅▆▇█"


def _sparkline(counts: list[int]) -> str:
    """Render bucket counts as a Unicode sparkline; zero-count buckets use '·'."""
    max_c = max(counts)
    if max_c == 0:
        return "·" * len(counts)
    return "".join(
        "·" if c == 0 else _SPARK_CHARS[max(1, round(c / max_c * 8))]
        for c in counts
    )


def _bucket_weighted_scores(scores: list[float]) -> list[int]:
    """Bin 0–5 weighted scores into [0,1), [1,2), [2,3), [3,4), [4,5] counts."""
    buckets = [0, 0, 0, 0, 0]
    for s in scores:
        idx = max(0, min(int(s), 4))
        buckets[idx] += 1
    return buckets


def print_score_distribution(rows: list[ModelRow]) -> None:
    """Show per-model histogram of prompt-level weighted scores.

    Surfaces distribution shape — ceiling-stuck, long-tailed, or bimodal —
    beyond what a single mean reveals.
    """
    table = Table(title="Score Distribution by Model", title_style="bold")
    table.add_column("Model", style="bold")
    for bucket_label in ("0–1", "1–2", "2–3", "3–4", "4–5"):
        table.add_column(bucket_label, justify="right")
    table.add_column("Shape", justify="left")

    any_rendered = False
    for row in rows:
        scores = list(row.pooled_prompt_means.values())
        if not scores:
            continue
        buckets = _bucket_weighted_scores(scores)
        model_label = f"{row.label} (n={row.n})" if row.n > 1 else row.label
        table.add_row(model_label, *[str(b) for b in buckets], _sparkline(buckets))
        any_rendered = True

    if not any_rendered:
        return
    console.print(table)
    console.print(
        "  [dim]Counts of prompts by mean weighted score. "
        "Shape is a sparkline across the five buckets (· = empty).[/dim]"
    )
    console.print()


def print_differentiating_prompts(rows: list[ModelRow], top_n: int = 5) -> None:
    """Rank shared prompts by score spread across models.

    Hides prompts where models tied (gap == 0), surfacing only where they
    diverge. Requires at least two models; does nothing otherwise.
    """
    if len(rows) < 2:
        return

    per_prompt: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        for pid, score in row.pooled_prompt_means.items():
            per_prompt[pid][row.label] = score

    n_models = len(rows)
    shared = {pid: s for pid, s in per_prompt.items() if len(s) == n_models}
    diverging = {
        pid: s for pid, s in shared.items()
        if max(s.values()) - min(s.values()) > 0
    }
    if not diverging:
        if shared:
            console.print(
                "  [dim]No differentiating prompts: all models tied on every "
                "shared prompt.[/dim]"
            )
            console.print()
        else:
            console.print(
                "[yellow]No prompts scored by all models; "
                "differentiator table skipped.[/yellow]"
            )
        return

    ranked = sorted(
        diverging.items(),
        key=lambda kv: max(kv[1].values()) - min(kv[1].values()),
        reverse=True,
    )[:top_n]

    table = Table(
        title=f"Top {top_n} Differentiating Prompts (largest score gap)",
        title_style="bold",
    )
    table.add_column("Prompt", style="bold")
    for row in rows:
        model_label = f"{row.label} (n={row.n})" if row.n > 1 else row.label
        table.add_column(model_label, justify="right")
    table.add_column("Gap", justify="right", style="magenta")

    for pid, scores in ranked:
        vals = list(scores.values())
        gap = max(vals) - min(vals)
        cells = [pid]
        for row in rows:
            s = scores[row.label]
            if gap > 0 and s == max(vals):
                cells.append(f"[green]{s:.2f}[/green]")
            elif gap > 0 and s == min(vals):
                cells.append(f"[red]{s:.2f}[/red]")
            else:
                cells.append(f"{s:.2f}")
        cells.append(f"{gap:.2f}")
        table.add_row(*cells)

    console.print(table)
    console.print(
        "  [dim]Prompts where model scores diverge most. "
        "Green = top score, red = bottom score.[/dim]"
    )
    console.print()


def print_leaderboard(
    scorecards: list[Scorecard],
    top_n: int = 3,
    result_dir: Path | None = None,
) -> None:
    """Print ranked leaderboard, score distribution, and prompt-level drill-down.

    Multiple scorecards for the same model (e.g. from --repeats) collapse into
    a single row with mean scores, a ±half-range spread indicator, and an
    n=<count> badge. The drill-down is a differentiator table (prompts with the
    largest score gap across models) when ≥2 models are present, falling back
    to per-model best/worst when only one model is reported.
    """
    if not scorecards:
        console.print("[yellow]No comparable scorecards found.[/yellow]")
        return

    rows = aggregate_by_model(scorecards, result_dir)
    rows.sort(key=lambda r: r.mean_overall, reverse=True)

    rubric = scorecards[0].evaluation.rubric
    evaluators = sorted({sc.evaluation.evaluator for sc in scorecards})
    eval_label = (
        evaluators[0] if len(evaluators) == 1 else f"{len(evaluators)} evaluators"
    )

    all_categories: list[str] = []
    all_difficulties: list[str] = []
    for row in rows:
        for cat in row.cat_means:
            if cat not in all_categories:
                all_categories.append(cat)
        for diff in row.diff_means:
            if diff not in all_difficulties:
                all_difficulties.append(diff)

    has_any_failures = any(
        row.total_full_fail or row.total_partial_fail for row in rows
    )
    has_any_repeats = any(row.n > 1 for row in rows)

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

    for rank, row in enumerate(rows, 1):
        label = f"{row.label} (n={row.n})" if row.n > 1 else row.label

        overall_parts = [f"{row.mean_overall:.2f}"]
        if row.n > 1 and row.overall_range > 0:
            overall_parts.append(f"±{row.overall_range / 2:.2f}")
        if row.mean_clean is not None:
            overall_parts.append(f"({row.mean_clean:.2f} clean)")
        overall_str = " ".join(overall_parts)

        table_row: list[str] = [str(rank), label, overall_str]

        if has_any_failures:
            flags: list[str] = []
            if row.total_full_fail:
                flags.append(f"[red]{row.total_full_fail} parse fail[/red]")
            if row.total_partial_fail:
                flags.append(f"[yellow]{row.total_partial_fail} partial[/yellow]")
            table_row.append(", ".join(flags) if flags else "[green]ok[/green]")

        for cat in all_categories:
            val = row.cat_means.get(cat)
            table_row.append(f"{val:.2f}" if val is not None else "-")
        for diff in all_difficulties:
            val = row.diff_means.get(diff)
            table_row.append(f"{val:.2f}" if val is not None else "-")
        table.add_row(*table_row)

    console.print(table)
    console.print(
        f"  Evaluator: {eval_label}   "
        f"Models: {len(rows)}   Scorecards: {len(scorecards)}"
    )

    if has_any_repeats:
        console.print(
            "  [dim]n = scorecards per model; scores are means across repeats. "
            "'±' shows half the range across repeats.[/dim]"
        )
    if has_any_failures:
        console.print(
            "  [dim]Flags: 'parse fail' = evaluator returned invalid JSON "
            "(all criteria scored 0). 'partial' = some criteria missing. "
            "'clean' score excludes fully failed prompts.[/dim]"
        )

    console.print()

    if top_n <= 0:
        return

    print_score_distribution(rows)

    if len(rows) >= 2:
        print_differentiating_prompts(rows)
        return

    bw_table = Table(title="Best & Worst Prompts by Model", title_style="bold")
    bw_table.add_column("Model", style="bold")
    bw_table.add_column("Best", style="green")
    bw_table.add_column("Worst", style="red")

    for row in rows:
        if not row.pooled_prompt_means:
            continue
        sorted_ids = sorted(
            row.pooled_prompt_means.items(), key=lambda kv: kv[1], reverse=True,
        )
        best = sorted_ids[:top_n]
        worst = sorted_ids[-top_n:]

        best_str = "\n".join(f"{pid} ({score:.2f})" for pid, score in best)
        worst_str = "\n".join(f"{pid} ({score:.2f})" for pid, score in worst)

        label = f"{row.label} (n={row.n})" if row.n > 1 else row.label
        bw_table.add_row(label, best_str, worst_str)

    console.print(bw_table)
