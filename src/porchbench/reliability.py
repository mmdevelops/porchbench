"""Judge-reliability analysis: ICC and agreement companions from scorecards.

Two input shapes produce the ratings matrix (targets = prompts, raters =
repeated judge samples):

- **One scorecard with per-sample data** (`evaluate --judge-samples k`, the
  mean-of-k pipeline): the matrix comes from `PromptScore.samples` — one
  evaluate pass is enough.
- **Several scorecards of the same run** (k separate `evaluate` passes, e.g.
  with different seeds): each scorecard is one rater column, matched by
  prompt_id.

ICC alone cannot distinguish a noisy judge from a compressed (saturated)
suite — restricted range crashes ICC even when repeats agree within
rounding. The report therefore pairs ICC with variance-independent
companions (% within tolerance, MAE across samples) and a between-prompt
spread diagnostic. Interpret ICC against its 95% CI lower bound per
Koo & Li (2016): <0.5 poor, 0.5-0.75 moderate, 0.75-0.9 good, >0.9
excellent.
"""

from __future__ import annotations

import statistics as _stats
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from porchbench.schemas import Scorecard
from porchbench.statistics import ICCResult, icc_absolute_agreement, pct_within

console = Console()

# Koo & Li (2016) gate used by the discrimination-suite validation protocol:
# the ICC 95% CI lower bound must clear this for the judge to count as
# reliably "good" on the suite.
ICC_GATE_LOWER_BOUND = 0.75


@dataclass(frozen=True)
class ReliabilityReport:
    """Reliability metrics for one ratings matrix."""

    icc: ICCResult | None
    pct_within_tol: float | None
    tolerance: float
    mae_across_samples: float | None
    between_prompt_sd: float | None
    n_prompts: int
    k_raters: int
    excluded_prompts: list[str]  # prompts without sample data (e.g. zero-scored)

    @property
    def gate_passed(self) -> bool | None:
        """ICC CI lower bound > 0.75, per the validation-protocol gate.

        None when ICC is degenerate/unavailable — the gate is then
        undecidable from this data (deterministic judge or no variance).
        """
        if self.icc is None or self.icc.degenerate or self.icc.ci_lower is None:
            return None
        return self.icc.ci_lower > ICC_GATE_LOWER_BOUND


def matrix_from_samples(
    scorecard: Scorecard,
    criterion: str | None = None,
) -> tuple[list[list[float]], list[str], list[str]]:
    """Build (matrix, prompt_ids, excluded) from per-sample scorecard data.

    Rows are prompts that carry `samples`; column j is judge sample j.
    `criterion=None` uses per-sample weighted scores; a criterion name uses
    that criterion's per-sample scores. Prompts without samples (zero-scored
    truncations, or scorecards predating mean-of-k) are excluded and named.
    """
    matrix: list[list[float]] = []
    prompt_ids: list[str] = []
    excluded: list[str] = []
    k = None
    for score in scorecard.scores:
        if not score.samples:
            excluded.append(score.prompt_id)
            continue
        if k is None:
            k = len(score.samples)
        if len(score.samples) != k:
            excluded.append(score.prompt_id)
            continue
        if criterion is None:
            row = [s.weighted_score for s in score.samples]
        else:
            if any(criterion not in s.criteria_scores for s in score.samples):
                excluded.append(score.prompt_id)
                continue
            row = [s.criteria_scores[criterion] for s in score.samples]
        matrix.append(row)
        prompt_ids.append(score.prompt_id)
    return matrix, prompt_ids, excluded


def matrix_from_scorecards(
    scorecards: list[Scorecard],
) -> tuple[list[list[float]], list[str], list[str]]:
    """Build (matrix, prompt_ids, excluded) treating each scorecard as a rater.

    For the k-separate-evaluate-passes protocol. Prompts must appear in
    every scorecard to form a row; others are excluded and named.
    """
    lookups = [
        {s.prompt_id: s.weighted_score for s in sc.scores} for sc in scorecards
    ]
    all_ids: list[str] = []
    seen: set[str] = set()
    for sc in scorecards:
        for s in sc.scores:
            if s.prompt_id not in seen:
                all_ids.append(s.prompt_id)
                seen.add(s.prompt_id)

    matrix: list[list[float]] = []
    prompt_ids: list[str] = []
    excluded: list[str] = []
    for pid in all_ids:
        if all(pid in lk for lk in lookups):
            matrix.append([lk[pid] for lk in lookups])
            prompt_ids.append(pid)
        else:
            excluded.append(pid)
    return matrix, prompt_ids, excluded


def analyze_matrix(
    matrix: list[list[float]],
    excluded: list[str],
    tolerance: float = 1.0,
) -> ReliabilityReport:
    """Compute the full reliability report for a ratings matrix."""
    n = len(matrix)
    k = len(matrix[0]) if matrix else 0

    icc = icc_absolute_agreement(matrix) if n >= 2 and k >= 2 else None
    within = pct_within(matrix, tolerance) if matrix else None

    mae: float | None = None
    if matrix and k >= 2:
        deviations = [
            abs(v - _stats.mean(row)) for row in matrix for v in row
        ]
        mae = _stats.mean(deviations)

    between_sd: float | None = None
    if n >= 2:
        between_sd = _stats.stdev([_stats.mean(row) for row in matrix])

    return ReliabilityReport(
        icc=icc,
        pct_within_tol=within,
        tolerance=tolerance,
        mae_across_samples=mae,
        between_prompt_sd=between_sd,
        n_prompts=n,
        k_raters=k,
        excluded_prompts=excluded,
    )


def _koo_li_label(icc_value: float) -> str:
    if icc_value < 0.5:
        return "poor"
    if icc_value < 0.75:
        return "moderate"
    if icc_value < 0.9:
        return "good"
    return "excellent"


def print_reliability_report(
    report: ReliabilityReport,
    title: str,
) -> None:
    """Render one reliability report as a Rich table with gate readout."""
    table = Table(title=title, title_style="bold")
    table.add_column("Metric", style="dim")
    table.add_column("Value")

    table.add_row("Prompts x samples", f"{report.n_prompts} x {report.k_raters}")

    icc = report.icc
    if icc is None:
        table.add_row("ICC", "[yellow]unavailable (need >= 2 prompts and >= 2 samples)[/yellow]")
    elif icc.degenerate:
        table.add_row(
            "ICC",
            "[yellow]degenerate — zero variance (deterministic judge?); "
            "agreement is perfect but ICC is undefined. Raise --judge-temp "
            "or --judge-samples.[/yellow]",
        )
    else:
        label = _koo_li_label(icc.icc_single)
        table.add_row("ICC(A,1) single-sample", f"{icc.icc_single:.3f} ({label})")
        table.add_row(
            "  95% CI",
            f"[{icc.ci_lower:.3f}, {icc.ci_upper:.3f}]",
        )
        table.add_row("ICC(A,k) shipped mean-of-k", f"{icc.icc_mean_of_k:.3f}")

    if report.pct_within_tol is not None:
        table.add_row(
            f"Samples within ±{report.tolerance:g}",
            f"{report.pct_within_tol * 100:.1f}%",
        )
    if report.mae_across_samples is not None:
        table.add_row("MAE across samples", f"{report.mae_across_samples:.3f}")
    if report.between_prompt_sd is not None:
        table.add_row("Between-prompt SD", f"{report.between_prompt_sd:.3f}")

    gate = report.gate_passed
    if gate is None:
        table.add_row("Gate (CI lower > 0.75)", "[yellow]undecidable[/yellow]")
    elif gate:
        table.add_row("Gate (CI lower > 0.75)", "[green]PASS[/green]")
    else:
        table.add_row("Gate (CI lower > 0.75)", "[red]FAIL[/red]")

    console.print(table)

    # Disambiguation note: low ICC + high within-tolerance agreement means
    # the SUITE is compressed, not that the judge is noisy.
    if (
        icc is not None
        and not icc.degenerate
        and icc.icc_single is not None
        and icc.icc_single < 0.5
        and report.pct_within_tol is not None
        and report.pct_within_tol > 0.9
        and report.between_prompt_sd is not None
        and report.between_prompt_sd < 0.5
    ):
        console.print(
            "[yellow]Low ICC with high sample agreement and low between-prompt "
            "spread: this pattern means the suite is compressed (restricted "
            "range / saturation), not that the judge is unreliable.[/yellow]"
        )

    if report.excluded_prompts:
        console.print(
            f"[dim]Excluded {len(report.excluded_prompts)} prompt(s) without "
            f"comparable sample data: {', '.join(report.excluded_prompts[:5])}"
            f"{'…' if len(report.excluded_prompts) > 5 else ''}[/dim]"
        )
