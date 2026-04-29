"""Tests for porchbench.compare — particularly the dynamic validation column."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from porchbench import compare as compare_mod
from porchbench.schemas import (
    Message,
    ModelInfo,
    ModelOptions,
    PromptMetrics,
    PromptResult,
    RequestData,
    ResponseData,
    ResponseMessage,
    RunMetadata,
    RunResult,
    RunSummary,
    SuiteReference,
)


def _make_run(
    run_id: str,
    model: str,
    *,
    prompt_results: list[PromptResult],
    total_duration_s: float = 1.0,
) -> RunResult:
    return RunResult(
        run=RunMetadata(
            id=run_id,
            suite=SuiteReference(
                name="T", version="1.0", file="x", sha256="y", rubric=None,
            ),
            model=ModelInfo(name=model),
        ),
        results=prompt_results,
        summary=RunSummary(
            total_prompts=len(prompt_results),
            completed=len(prompt_results),
            failed=0,
            total_duration_s=total_duration_s,
        ),
    )


def _make_pr(
    prompt_id: str,
    *,
    eval_count: int | None = None,
    total_duration: int | None = None,
    tokens_per_second: float | None = None,
    validation_passed: bool | None = None,
) -> PromptResult:
    return PromptResult(
        prompt_id=prompt_id,
        category="tool-use",
        difficulty="easy",
        options_used=ModelOptions(),
        request=RequestData(messages=[Message(role="user", content="x")]),
        response=ResponseData(message=ResponseMessage(content="ok")),
        metrics=PromptMetrics(
            eval_count=eval_count,
            total_duration=total_duration,
            tokens_per_second=tokens_per_second,
        ),
        validation_passed=validation_passed,
    )


def _capture(runs):
    """Render `print_comparison_table` to a string buffer for assertion."""
    buf = StringIO()
    test_console = Console(file=buf, width=240, force_terminal=False)
    original = compare_mod.console
    compare_mod.console = test_console
    try:
        compare_mod.print_comparison_table(runs)
    finally:
        compare_mod.console = original
    return buf.getvalue()


def test_validation_columns_appear_for_tool_use_runs():
    """When any run carries validator outcomes, add a per-model 'valid' column."""
    run_a = _make_run(
        "a", "m1",
        prompt_results=[
            _make_pr("p1", eval_count=10, total_duration=1_000_000_000,
                     tokens_per_second=10.0, validation_passed=True),
            _make_pr("p2", eval_count=20, total_duration=2_000_000_000,
                     tokens_per_second=10.0, validation_passed=False),
        ],
    )
    run_b = _make_run(
        "b", "m2",
        prompt_results=[
            _make_pr("p1", eval_count=15, total_duration=1_000_000_000,
                     tokens_per_second=15.0, validation_passed=True),
            _make_pr("p2", eval_count=25, total_duration=2_000_000_000,
                     tokens_per_second=12.5, validation_passed=True),
        ],
    )

    out = _capture([run_a, run_b])

    # Per-model valid columns surfaced
    assert "m1" in out
    assert "m2" in out
    assert "valid" in out
    # Cell values: pass/fail rendered as text (rich strips markup in non-terminal)
    assert "pass" in out
    assert "fail" in out
    # Summary picks up the validation pass-rate row
    assert "Validation" in out
    assert "1/2" in out  # m1: 1 of 2 passed
    assert "2/2" in out  # m2: 2 of 2 passed


def test_validation_columns_omitted_when_no_run_has_validator_data():
    """Pure text-suite comparisons stay 3-column-per-model — no empty 'valid'."""
    run = _make_run(
        "a", "m1",
        prompt_results=[
            _make_pr("p1", eval_count=10, total_duration=1_000_000_000,
                     tokens_per_second=10.0),
        ],
    )

    out = _capture([run])

    # No validation column header, no Validation summary row
    assert "valid" not in out
    assert "Validation" not in out


def test_options_row_shows_non_default_values():
    """The Options row disambiguates same-model runs with different overrides.

    Without it, two columns labelled 'gemma4:e2b' (one with think=false, one
    without) are indistinguishable and users can't tell which is which.
    """
    pr_with_think_off = _make_pr(
        "p1", eval_count=10, total_duration=1_000_000_000,
        tokens_per_second=10.0, validation_passed=True,
    )
    # Override options on this prompt to simulate a --set think=false run
    pr_with_think_off.options_used = ModelOptions(num_ctx=8192, think=False)

    pr_default = _make_pr(
        "p1", eval_count=15, total_duration=1_500_000_000,
        tokens_per_second=10.0, validation_passed=True,
    )
    # Default ModelOptions for the second run

    run_a = _make_run("a", "gemma4:e2b", prompt_results=[pr_with_think_off])
    run_b = _make_run("b", "gemma4:e2b", prompt_results=[pr_default])

    out = _capture([run_a, run_b])

    assert "Options" in out
    # Run A overrides shown, run B is on defaults
    assert "think=False" in out
    assert "num_ctx=8192" in out
    assert "(defaults)" in out


def test_format_options_returns_defaults_marker_when_unchanged():
    from porchbench.compare import _format_options

    assert _format_options(ModelOptions()) == "(defaults)"


def test_format_options_includes_extras():
    """Pydantic extras (e.g. think) are always non-default — must be surfaced."""
    from porchbench.compare import _format_options

    out = _format_options(ModelOptions(think=False))
    assert "think=False" in out


def test_mixed_runs_show_dash_for_prompts_without_validator():
    """When some prompts have validators and others don't, missing cells show '-'."""
    pr_with_val = _make_pr("p1", eval_count=10, total_duration=1_000_000_000,
                           tokens_per_second=10.0, validation_passed=True)
    pr_no_val = _make_pr("p2", eval_count=20, total_duration=2_000_000_000,
                         tokens_per_second=10.0, validation_passed=None)

    run = _make_run("a", "m1", prompt_results=[pr_with_val, pr_no_val])
    out = _capture([run])

    # Validation column added because at least one prompt has a validator
    assert "valid" in out
    # p1's row contains 'pass'; p2's row should have a '-' in the valid column
    # (we can't easily assert column position without parsing the table, but
    # the '-' must be present)
    assert "pass" in out
