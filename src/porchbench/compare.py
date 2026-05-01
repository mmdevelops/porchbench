"""Cross-model comparison utilities.

Loads multiple run results, aligns them by prompt_id, and produces
side-by-side comparison tables for throughput metrics. When scorecards
are available, includes quality scores in the comparison.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from porchbench.metrics import describe, extract_tokens_per_second
from porchbench.schemas import ModelOptions, PromptResult, RunResult, Scorecard
from porchbench.statistics import PairedTestResult, paired_comparison

console = Console()


def disambiguate_model_names(runs: list[RunResult]) -> list[str]:
    """Return per-run column labels with a suffix added when names collide.

    Same-model selections (e.g. two `gemma4:e2b` runs differing only in timestamp
    or `--set` overrides) produce identical column headers in the comparison
    tables. The Options row helps when the runs differ in resolved options, but
    can't tell apart two runs of the same model with the same options. Append
    `·HH:MM` from the run timestamp; if that still collides within a duplicate
    group, fall back to `·{run_id[:4]}`.
    """
    names = [r.run.model.name for r in runs]
    counts: dict[str, int] = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1

    labels = list(names)
    for name, count in counts.items():
        if count < 2:
            continue
        group_indices = [i for i, n in enumerate(names) if n == name]
        ts_suffixes = [runs[i].run.timestamp.strftime("%H:%M") for i in group_indices]
        if len(set(ts_suffixes)) == count:
            for idx, suffix in zip(group_indices, ts_suffixes):
                labels[idx] = f"{name}·{suffix}"
        else:
            for idx in group_indices:
                labels[idx] = f"{name}·{runs[idx].run.id[:4]}"
    return labels


def _format_options(opts: ModelOptions) -> str:
    """Compact display of resolved-option values that differ from class defaults.

    Defaults (temperature=0, num_ctx=4096, etc.) are uninteresting in a
    comparison context — what matters is what the user/suite changed. Extras
    (anything in `extra="allow"` like `think=false`) are always shown since
    they're definitionally non-default. Returns '(defaults)' when the run
    used everything as-is.
    """
    defaults = ModelOptions()
    parts = []
    for field, value in opts.model_dump().items():
        if value != getattr(defaults, field, None):
            parts.append(f"{field}={value}")
    return ", ".join(parts) if parts else "(defaults)"


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


def print_comparison_table(
    runs: list[RunResult],
    scorecards: list[Scorecard | None] | None = None,
    seed: int | None = 42,
) -> None:
    """Print a rich table comparing models across prompts.

    Shows tokens/sec and total tokens for each model on each prompt,
    plus quality scores if scorecards are provided.
    """
    if not runs:
        return

    model_names = disambiguate_model_names(runs)
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

    # Detect tool-use runs by presence of validator outcomes — when ANY run
    # carries validation_passed data, surface it as the leftmost data column
    # (the most actionable signal for tool-use comparisons; tokens/tok/s are
    # secondary when the question is "did the model do the task at all").
    has_validation = any(
        any(r.validation_passed is not None for r in run.results)
        for run in runs
    )

    # Per-prompt comparison — metrics grouped with models side-by-side
    table = Table(title="Per-Prompt Comparison", title_style="bold")
    table.add_column("Prompt", style="bold")

    # Validation columns (one per model) — leftmost data when applicable
    if has_validation:
        for name in model_names:
            table.add_column(f"{name}\nvalid", justify="center")
    # Tokens columns (one per model)
    for name in model_names:
        table.add_column(f"{name}\ntokens", justify="right")
    # Time columns (one per model)
    for name in model_names:
        table.add_column(f"{name}\nsecs", justify="right")
    # Tok/s columns (one per model)
    for name in model_names:
        table.add_column(f"{name}\ntok/s", justify="right")
    # Score columns (one per model, if available)
    if has_scores:
        for name in model_names:
            table.add_column(f"{name}\nscore", justify="right")

    for pid, results_row in aligned.items():
        row: list[str] = [pid]

        # Validation outcome (pass / fail / dash for non-validator prompts)
        if has_validation:
            for pr in results_row:
                if pr is None or pr.validation_passed is None:
                    row.append("-")
                elif pr.validation_passed:
                    row.append("[green]pass[/green]")
                else:
                    row.append("[yellow]fail[/yellow]")

        # Tokens
        for pr in results_row:
            if pr is not None and pr.metrics.eval_count is not None:
                row.append(str(pr.metrics.eval_count))
            else:
                row.append("-")

        # Time (seconds)
        for pr in results_row:
            if pr is not None and pr.metrics.total_duration is not None:
                row.append(f"{pr.metrics.total_duration / 1e9:.1f}")
            else:
                row.append("-")

        # Tok/s
        for pr in results_row:
            if pr is not None and pr.metrics.tokens_per_second is not None:
                row.append(f"{pr.metrics.tokens_per_second:.1f}")
            else:
                row.append("-")

        # Scores
        if has_scores:
            for i, pr in enumerate(results_row):
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
        tps = describe(extract_tokens_per_second(run.results), seed=seed)
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

    # Validation pass rate (when any run has validator outcomes)
    if has_validation:
        val_row = ["Validation"]
        for run in runs:
            scored = [r for r in run.results if r.validation_passed is not None]
            if scored:
                passed = sum(1 for r in scored if r.validation_passed)
                val_row.append(f"{passed}/{len(scored)}")
            else:
                val_row.append("-")
        summary_table.add_row(*val_row)

    # Resolved options (non-default values from the first prompt's options_used).
    # Disambiguates same-model runs that differ in `--set` or suite-level
    # overrides (e.g. think=false vs default) — the column header alone
    # gives no clue when comparing two gemma4:e2b runs.
    opts_row = ["Options"]
    for run in runs:
        if run.results:
            opts_row.append(_format_options(run.results[0].options_used))
        else:
            opts_row.append("-")
    summary_table.add_row(*opts_row)

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

    # Paired statistical comparison (when exactly 2 models)
    if len(runs) == 2 and has_scores and len(score_lookups) == 2:
        result = compare_models_paired(
            aligned, score_lookups[0], score_lookups[1], seed=seed
        )
        if result:
            _print_paired_result(model_names[0], model_names[1], result)


def compare_models_paired(
    aligned: dict[str, list[PromptResult | None]],
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    seed: int | None = 42,
) -> PairedTestResult | None:
    """Run a paired statistical test on quality scores for two models.

    Only includes prompts where both models have scores. The seed is
    forwarded to the CI's bootstrap path for reproducibility.
    """
    vals_a: list[float] = []
    vals_b: list[float] = []
    for pid in aligned:
        if pid in scores_a and pid in scores_b:
            vals_a.append(scores_a[pid])
            vals_b.append(scores_b[pid])

    if len(vals_a) < 2:
        return None
    return paired_comparison(vals_a, vals_b, seed=seed)


def _print_paired_result(name_a: str, name_b: str, result: PairedTestResult) -> None:
    """Print a paired comparison result."""
    console.print()
    table = Table(title=f"Paired Comparison: {name_a} vs {name_b}", title_style="bold")
    table.add_column("Metric", style="dim")
    table.add_column("Value")

    table.add_row("Test", result.test_name)
    table.add_row("Paired prompts", str(result.n_pairs))

    direction = name_a if result.mean_difference > 0 else name_b
    table.add_row("Mean difference", f"{result.mean_difference:+.4f} (favors {direction})")
    if result.p_value is None:
        table.add_row("p-value", "[dim]n too small (rely on CI / effect size)[/dim]")
        table.add_row("Significant (p<0.05)", "[dim]n/a[/dim]")
    else:
        table.add_row("p-value", f"{result.p_value:.4f}")
        sig_str = "[green]Yes[/green]" if result.significant else "[yellow]No[/yellow]"
        table.add_row("Significant (p<0.05)", sig_str)
    table.add_row("Effect size (Cohen's dz)", f"{result.effect_size:.3f} ({result.effect_magnitude})")

    if result.ci:
        table.add_row("95% CI on difference", f"[{result.ci.ci_lower:.4f}, {result.ci.ci_upper:.4f}]")

    console.print(table)
