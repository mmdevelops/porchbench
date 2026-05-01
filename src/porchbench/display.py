"""Shared display helpers for the CLI and routing-discovery output.

Lives below the engine modules (``runner``, ``routing``) and the CLI so
both can import without forming the ``routing → cli → routing`` cycle
that pulled the badge formatter out of ``cli.py``.
"""

from __future__ import annotations

from porchbench.schemas import PromptResult


def format_validation_badge(result: PromptResult | None) -> str:
    """Format the per-prompt validator outcome as a compact inline badge.

    Returns the empty string when the prompt has no validator (e.g. text-mode
    prompts) so non-tool-use suites are unaffected. For tool-use prompts the
    result is already in ``result.validation_passed`` from the per-prompt
    sandbox check that ran during inference — surfacing it inline lets users
    see pass/fail as it happens instead of only in the final summary.
    """
    if result is None:
        return ""
    passed = result.validation_passed
    if passed is None:
        return ""
    if passed:
        return " [bold green]\\[pass][/bold green]"
    reason = result.validation_reason or ""
    suffix = ""
    if reason:
        first_line = reason.splitlines()[0]
        # 120 chars keeps composite-validator chains like 'X exists (N bytes);
        # X missing required <field>' readable; ellipsis flags truncation so
        # users know to check the result JSON for the full reason.
        short_reason = first_line if len(first_line) <= 120 else first_line[:117] + "..."
        suffix = f": {short_reason}"
    return f" [bold yellow]\\[val-fail][/bold yellow]{suffix}"
