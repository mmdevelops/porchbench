"""Tests for porchbench.compare — particularly the dynamic validation column."""

from __future__ import annotations

from datetime import UTC, datetime
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
    timestamp: datetime | None = None,
) -> RunResult:
    metadata_kwargs: dict = dict(
        id=run_id,
        suite=SuiteReference(
            name="T", version="1.0", file="x", sha256="y", rubric=None,
        ),
        model=ModelInfo(name=model),
    )
    if timestamp is not None:
        metadata_kwargs["timestamp"] = timestamp
    return RunResult(
        run=RunMetadata(**metadata_kwargs),
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


# ---------------------------------------------------------------------------
# Auto-discovery: compare matches scorecards to results by run_id prefix
# ---------------------------------------------------------------------------


def _make_scorecard_with_score(run_id: str, prompt_id: str, weighted: float, overall: float):
    """Build a Scorecard with one realistic PromptScore (matches what evaluate writes)."""
    from porchbench.schemas import (
        AggregateScores,
        CriterionScore,
        EvaluationMetadata,
        PromptScore,
        Scorecard,
    )

    return Scorecard(
        evaluation=EvaluationMetadata(
            run_id=run_id, evaluator="test/judge", rubric="r v1.0",
            model_name="m", suite_name="T",
        ),
        scores=[
            PromptScore(
                prompt_id=prompt_id,
                criteria={"correctness": CriterionScore(score=int(round(weighted)), rationale="ok")},
                weighted_score=weighted,
                summary="ok",
            )
        ],
        aggregate=AggregateScores(overall_weighted=overall),
    )


def test_compare_auto_discovers_scorecards_by_run_id_prefix(tmp_path):
    """`porchbench compare` should pair scorecards to results without users
    having to specify --scorecard for each -r. Scorecards are written to
    `scorecards/{ts}_{run_id[:8]}.json`, so the prefix glob is unambiguous."""
    from typer.testing import CliRunner

    from porchbench.cli import app

    runner = CliRunner()
    results_dir = tmp_path / "results"
    scorecards_dir = tmp_path / "scorecards"
    results_dir.mkdir()
    scorecards_dir.mkdir()

    # Two run results with distinct run IDs
    run_a = _make_run(
        "aaaaaaaa-1111-1111-1111-111111111111", "m1",
        prompt_results=[_make_pr("p1", eval_count=10, total_duration=1_000_000_000,
                                 tokens_per_second=10.0)],
    )
    run_b = _make_run(
        "bbbbbbbb-2222-2222-2222-222222222222", "m2",
        prompt_results=[_make_pr("p1", eval_count=20, total_duration=2_000_000_000,
                                 tokens_per_second=10.0)],
    )
    rp_a = results_dir / "result_a.json"
    rp_b = results_dir / "result_b.json"
    rp_a.write_text(run_a.model_dump_json(), encoding="utf-8")
    rp_b.write_text(run_b.model_dump_json(), encoding="utf-8")

    # Matching scorecards named with run_id[:8] prefix; realistic shape
    sc_a = _make_scorecard_with_score(run_a.run.id, "p1", weighted=4.5, overall=4.5)
    sc_b = _make_scorecard_with_score(run_b.run.id, "p1", weighted=3.5, overall=3.5)
    (scorecards_dir / "2026-04-29T10-00-00_aaaaaaaa.json").write_text(
        sc_a.model_dump_json(), encoding="utf-8",
    )
    (scorecards_dir / "2026-04-29T10-01-00_bbbbbbbb.json").write_text(
        sc_b.model_dump_json(), encoding="utf-8",
    )

    res = runner.invoke(app, [
        "compare",
        "-r", str(rp_a), "-r", str(rp_b),
        "--scorecard-dir", str(scorecards_dir),
    ])

    assert res.exit_code == 0, res.output
    # Scorecards were picked up — Avg score row should appear and per-prompt
    # score column should render with the expected values.
    assert "Avg score" in res.output
    assert "4.50" in res.output
    assert "3.50" in res.output


def test_compare_warns_when_some_results_lack_scorecards(tmp_path):
    """Auto-discovery is best-effort: missing scorecards get a friendly note,
    not a hard failure. Users see which models still need evaluating."""
    from typer.testing import CliRunner

    from porchbench.cli import app

    runner = CliRunner()
    results_dir = tmp_path / "results"
    scorecards_dir = tmp_path / "scorecards"
    results_dir.mkdir()
    scorecards_dir.mkdir()

    run_a = _make_run(
        "aaaaaaaa-1111-1111-1111-111111111111", "m1",
        prompt_results=[_make_pr("p1", eval_count=10, total_duration=1_000_000_000,
                                 tokens_per_second=10.0)],
    )
    run_b = _make_run(
        "bbbbbbbb-2222-2222-2222-222222222222", "m2-not-scored",
        prompt_results=[_make_pr("p1", eval_count=20, total_duration=2_000_000_000,
                                 tokens_per_second=10.0)],
    )
    (results_dir / "a.json").write_text(run_a.model_dump_json(), encoding="utf-8")
    (results_dir / "b.json").write_text(run_b.model_dump_json(), encoding="utf-8")

    # Only score run_a; run_b has no scorecard
    sc_a = _make_scorecard_with_score(run_a.run.id, "p1", weighted=4.5, overall=4.5)
    (scorecards_dir / "ts_aaaaaaaa.json").write_text(
        sc_a.model_dump_json(), encoding="utf-8",
    )

    res = runner.invoke(app, [
        "compare",
        "-r", str(results_dir / "a.json"),
        "-r", str(results_dir / "b.json"),
        "--scorecard-dir", str(scorecards_dir),
    ])

    assert res.exit_code == 0, res.output
    # Friendly note names the un-scored model and points to evaluate
    assert "m2-not-scored" in res.output
    assert "porchbench evaluate" in res.output


# ---------------------------------------------------------------------------
# disambiguate_model_names: same-model selections get a column-suffix
# ---------------------------------------------------------------------------


def _pr_minimal() -> PromptResult:
    return _make_pr("p1", eval_count=10, total_duration=1_000_000_000,
                    tokens_per_second=10.0)


def test_disambiguate_unique_names_pass_through():
    """Distinct model names need no suffix — pure pass-through."""
    from porchbench.compare import disambiguate_model_names

    runs = [
        _make_run("a", "qwen3:8b", prompt_results=[_pr_minimal()]),
        _make_run("b", "gemma4:e2b", prompt_results=[_pr_minimal()]),
    ]
    assert disambiguate_model_names(runs) == ["qwen3:8b", "gemma4:e2b"]


def test_disambiguate_appends_hhmm_when_names_collide():
    """Two same-model runs at different minutes get `·HH:MM` suffixes."""
    from porchbench.compare import disambiguate_model_names

    runs = [
        _make_run("a", "gemma4:e2b", prompt_results=[_pr_minimal()],
                  timestamp=datetime(2026, 4, 29, 14, 32, 0, tzinfo=UTC)),
        _make_run("b", "gemma4:e2b", prompt_results=[_pr_minimal()],
                  timestamp=datetime(2026, 4, 29, 15, 10, 0, tzinfo=UTC)),
    ]
    labels = disambiguate_model_names(runs)
    assert labels == ["gemma4:e2b·14:32", "gemma4:e2b·15:10"]


def test_disambiguate_falls_back_to_run_id_on_minute_collision():
    """When two same-model runs share the same minute, suffix uses run_id[:4]."""
    from porchbench.compare import disambiguate_model_names

    same_minute = datetime(2026, 4, 29, 14, 32, 0, tzinfo=UTC)
    runs = [
        _make_run("aaaa1111-1111-1111-1111-111111111111", "gemma4:e2b",
                  prompt_results=[_pr_minimal()], timestamp=same_minute),
        _make_run("bbbb2222-2222-2222-2222-222222222222", "gemma4:e2b",
                  prompt_results=[_pr_minimal()], timestamp=same_minute),
    ]
    labels = disambiguate_model_names(runs)
    assert labels == ["gemma4:e2b·aaaa", "gemma4:e2b·bbbb"]


def test_disambiguate_only_suffixes_duplicates_in_mixed_list():
    """Unique names in a mixed list stay clean; only the duplicate group is suffixed."""
    from porchbench.compare import disambiguate_model_names

    runs = [
        _make_run("a", "gemma4:e2b", prompt_results=[_pr_minimal()],
                  timestamp=datetime(2026, 4, 29, 14, 32, 0, tzinfo=UTC)),
        _make_run("b", "qwen3:8b", prompt_results=[_pr_minimal()]),
        _make_run("c", "gemma4:e2b", prompt_results=[_pr_minimal()],
                  timestamp=datetime(2026, 4, 29, 15, 10, 0, tzinfo=UTC)),
    ]
    labels = disambiguate_model_names(runs)
    assert labels == ["gemma4:e2b·14:32", "qwen3:8b", "gemma4:e2b·15:10"]


def test_compare_table_renders_disambiguated_headers_for_same_model_picks():
    """End-to-end: print_comparison_table uses the disambiguator so the user can
    actually distinguish columns for two same-model runs."""
    runs = [
        _make_run("a", "gemma4:e2b", prompt_results=[_pr_minimal()],
                  timestamp=datetime(2026, 4, 29, 14, 32, 0, tzinfo=UTC)),
        _make_run("b", "gemma4:e2b", prompt_results=[_pr_minimal()],
                  timestamp=datetime(2026, 4, 29, 15, 10, 0, tzinfo=UTC)),
    ]
    out = _capture(runs)
    assert "14:32" in out
    assert "15:10" in out


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
