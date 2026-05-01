"""Tests for routing discovery: correctness checking and analysis logic."""

import pytest

from porchbench.routing import (
    _find_best_cell,
    _parse_param_size,
    analyze_routes,
    build_routing_matrix,
    check_correctness,
)
from porchbench.schemas import (
    Message,
    ModelDetails,
    ModelInfo,
    ModelOptions,
    PromptMetrics,
    PromptResult,
    RequestData,
    ResponseData,
    ResponseMessage,
    RoutingCell,
    RunMetadata,
    RunResult,
    RunSummary,
    SuiteReference,
    SystemInfo,
)

# ---------------------------------------------------------------------------
# Correctness checking
# ---------------------------------------------------------------------------


class TestCheckCorrectness:
    def test_no_expected_answer(self):
        assert check_correctness("anything", None) is None

    def test_empty_response(self):
        assert check_correctness("", "42") is False

    def test_exact_numeric(self):
        assert check_correctness("The answer is 36.", "36") is True

    def test_numeric_not_found(self):
        assert check_correctness("The answer is 42.", "36") is False

    def test_float_match(self):
        assert check_correctness("The result is 3.14159", "3.14") is True

    def test_float_no_match(self):
        assert check_correctness("The result is 2.71", "3.14") is False

    def test_string_match_case_insensitive(self):
        assert check_correctness("paris is the capital", "Paris") is True

    def test_string_match_substring(self):
        assert check_correctness("The capital of France is Paris.", "Paris") is True

    def test_string_no_match(self):
        assert check_correctness("London is the capital", "Paris") is False

    def test_numeric_in_longer_number(self):
        # "36" should match even if embedded in text with other numbers
        assert check_correctness("Step 1: 240 * 0.15 = 36.0", "36") is True

    def test_negative_number(self):
        assert check_correctness("The answer is -5", "-5") is True

    def test_zero(self):
        assert check_correctness("The answer is 0", "0") is True

    def test_code_substring(self):
        assert check_correctness("def reverse(s):\n    return s[::-1]", "def reverse") is True

    def test_formula_match(self):
        assert check_correctness("Time complexity is O(n log n)", "O(n log n)") is True


# ---------------------------------------------------------------------------
# Parameter size parsing
# ---------------------------------------------------------------------------


class TestParseParamSize:
    def test_billions(self):
        assert _parse_param_size("7.6B") == pytest.approx(7.6)

    def test_small(self):
        assert _parse_param_size("3.1B") == pytest.approx(3.1)

    def test_no_suffix(self):
        assert _parse_param_size("14.8") == pytest.approx(14.8)

    def test_empty(self):
        assert _parse_param_size("") == 0.0

    def test_garbage(self):
        assert _parse_param_size("unknown") == 0.0


# ---------------------------------------------------------------------------
# Best cell selection
# ---------------------------------------------------------------------------


class TestFindBestCell:
    def test_correct_over_incorrect(self):
        cells = [
            RoutingCell(model="a", prompt_id="p", strategy="s1",
                        correct=False, tokens_generated=5),
            RoutingCell(model="b", prompt_id="p", strategy="s2",
                        correct=True, tokens_generated=100),
        ]
        best = _find_best_cell(cells)
        assert best.model == "b"  # correct wins even with more tokens

    def test_fewer_tokens_when_both_correct(self):
        cells = [
            RoutingCell(model="a", prompt_id="p", strategy="s1",
                        correct=True, tokens_generated=50),
            RoutingCell(model="b", prompt_id="p", strategy="s2",
                        correct=True, tokens_generated=5),
        ]
        best = _find_best_cell(cells)
        assert best.model == "b"

    def test_none_correct_is_middle(self):
        cells = [
            RoutingCell(model="a", prompt_id="p", strategy="s1",
                        correct=None, tokens_generated=10),
            RoutingCell(model="b", prompt_id="p", strategy="s2",
                        correct=True, tokens_generated=100),
            RoutingCell(model="c", prompt_id="p", strategy="s3",
                        correct=False, tokens_generated=5),
        ]
        best = _find_best_cell(cells)
        assert best.model == "b"  # True > None > False

    def test_empty_list(self):
        assert _find_best_cell([]) is None


# ---------------------------------------------------------------------------
# Routing analysis
# ---------------------------------------------------------------------------


def _make_run(model: str, param_size: str, results: list[dict]) -> RunResult:
    """Helper to build a RunResult for testing."""
    prompt_results = []
    for r in results:
        prompt_results.append(PromptResult(
            prompt_id=r["pid"],
            category=r.get("cat", "reasoning"),
            difficulty=r.get("diff", "easy"),
            options_used=ModelOptions(),
            request=RequestData(messages=[Message(role="user", content="test")]),
            response=ResponseData(message=ResponseMessage(content=r.get("content", ""))),
            metrics=PromptMetrics(
                eval_count=r.get("tokens", 10),
                eval_duration=1000000000,
                total_duration=2000000000,
            ),
            strategy=r.get("strategy", "universal"),
            correct=r.get("correct"),
            expected_answer=r.get("expected"),
        ))

    return RunResult(
        run=RunMetadata(
            suite=SuiteReference(name="Test", version="1.0", file="t.yaml", sha256="x"),
            model=ModelInfo(name=model,
                            details=ModelDetails(parameter_size=param_size)),
            system=SystemInfo(),
        ),
        results=prompt_results,
        summary=RunSummary(total_prompts=len(results), completed=len(results),
                           failed=0, total_duration_s=1.0),
    )


class TestAnalyzeRoutes:
    def test_basic_analysis(self):
        small = _make_run("small:3b", "3.1B", [
            {"pid": "p1", "strategy": "universal", "correct": True, "tokens": 100},
            {"pid": "p1", "strategy": "direct", "correct": True, "tokens": 5},
            {"pid": "p2", "strategy": "universal", "correct": False, "tokens": 50},
            {"pid": "p2", "strategy": "direct", "correct": False, "tokens": 3},
        ])
        large = _make_run("large:7b", "7.6B", [
            {"pid": "p1", "strategy": "universal", "correct": True, "tokens": 150},
            {"pid": "p1", "strategy": "direct", "correct": True, "tokens": 8},
            {"pid": "p2", "strategy": "universal", "correct": True, "tokens": 200},
            {"pid": "p2", "strategy": "direct", "correct": True, "tokens": 10},
        ])

        analysis = analyze_routes([small, large])

        assert analysis.headline.problems_total == 2
        assert len(analysis.models_tested) == 2
        assert "direct" in analysis.strategies_tested
        assert "universal" in analysis.strategies_tested
        assert len(analysis.best_route_per_problem) == 2

    def test_inverse_scaling_detection(self):
        # Small model gets p1 right, large model gets it wrong under universal
        small = _make_run("small:3b", "3.1B", [
            {"pid": "p1", "strategy": "universal", "correct": True, "tokens": 10},
        ])
        large = _make_run("large:7b", "7.6B", [
            {"pid": "p1", "strategy": "universal", "correct": False, "tokens": 100},
        ])

        analysis = analyze_routes([small, large])
        assert analysis.headline.inverse_scaling_detected is True
        assert analysis.headline.inverse_scaling_rate > 0

    def test_routing_matrix(self):
        run = _make_run("test:3b", "3.1B", [
            {"pid": "p1", "strategy": "universal", "correct": True, "tokens": 50},
            {"pid": "p1", "strategy": "direct", "correct": True, "tokens": 5},
        ])
        matrix = build_routing_matrix([run])
        assert len(matrix) == 2
        assert matrix[0].model == "test:3b"
        assert {c.strategy for c in matrix} == {"universal", "direct"}


# ---------------------------------------------------------------------------
# Tool-use discovery cell — metric merging
# ---------------------------------------------------------------------------


class TestToolUseDiscoveryCellMetrics:
    """Routing-discovery tool-use cells must propagate harness-aggregated
    timing into PromptResult.metrics. Without this the per-model summary
    line reports `0.0 avg tok/s` because every cell's tokens_per_second
    is None."""

    @pytest.mark.asyncio
    async def test_aggregated_metrics_flow_into_prompt_result(self):
        from unittest.mock import AsyncMock, patch

        from porchbench.harness.harness import HarnessResult, Outcome, ToolUseMetrics
        from porchbench.routing import _run_tool_use_discovery_cell

        prompt = _make_tool_use_discovery_prompt()
        harness_result = HarnessResult(
            transcript=[{"role": "assistant", "content": "ok"}],
            outcome=Outcome(),
            tool_use_metrics=ToolUseMetrics(),
            stopped_reason="done",
            aggregated_metrics=PromptMetrics(
                prompt_eval_count=200,
                eval_count=80,
                eval_duration=800_000_000,  # 0.8s
                tokens_per_second=100.0,
            ),
        )
        mock_return = {
            "harness_result": harness_result,
            "validation_passed": True,
            "validation_reason": "ok",
            "elapsed_ns": 5_000_000_000,
        }

        with patch(
            "porchbench.tool_runner.run_tool_use_prompt",
            new_callable=AsyncMock,
            return_value=mock_return,
        ):
            result = await _run_tool_use_discovery_cell(
                prompt, "m", ModelOptions(),
                [Message(role="user", content="x")],
                "universal", None, None,
            )

        # Wall-clock from elapsed_ns lands in total_duration
        assert result.metrics.total_duration == 5_000_000_000
        # Token counts and tok/s come from the harness aggregate — the
        # exact bug surfaced during UAT was these all reading None,
        # producing 0.0 avg tok/s in the per-model summary.
        assert result.metrics.eval_count == 80
        assert result.metrics.tokens_per_second == 100.0
        assert result.metrics.prompt_eval_count == 200


def _make_tool_use_discovery_prompt():
    """Minimal Prompt fixture for routing-discovery tool-use cells."""
    from porchbench.schemas import Prompt

    return Prompt(
        id="t1-read",
        category="tool-use",
        difficulty="easy",
        mode="tool-use",
        max_tool_calls=5,
        messages=[Message(role="user", content="x")],
    )


# ---------------------------------------------------------------------------
# `routes analyze` CLI: attribute access on RunResult
# ---------------------------------------------------------------------------


class TestRoutesAnalyzeCli:
    """Regression for AttributeError on non-routing detection.

    The CLI used `r.prompt_results` and `r.metadata.run_id` — neither field
    exists on RunResult (correct names are `results` and `run.id`). The
    error path was never exercised in tests; the first invocation against
    any result file crashed before the analyze logic could run.
    """

    def test_friendly_error_for_non_routing_result_file(self, tmp_path):
        """A run without strategy tags should produce a friendly message,
        not an AttributeError. This pins the field-access fix and the
        improved error that names the offending filename + model."""
        from typer.testing import CliRunner

        from porchbench.cli import app

        # A non-routing run: PromptResult.strategy is None on every result
        non_routing_run = _build_run("m1", strategy_assignments=[None])
        path = tmp_path / "non_routing.json"
        path.write_text(non_routing_run.model_dump_json(), encoding="utf-8")

        runner = CliRunner()
        res = runner.invoke(app, ["routes", "analyze", "-r", str(path)])

        # Command exits 1 with the friendly error, not with a Python traceback
        assert res.exit_code == 1
        assert "routes analyze requires results" in res.output
        assert "AttributeError" not in res.output
        # Error names the file and model so the user knows what to remove
        assert "non_routing.json" in res.output
        assert "m1" in res.output

    def test_single_model_routing_result_refused(self, tmp_path):
        """Routing analysis is fundamentally cross-model; one model must
        produce a clean refusal pointing at the fix, not a degenerate report
        ("Use largest model" verdict, 0/N routing-helps, etc.)."""
        from typer.testing import CliRunner

        from porchbench.cli import app

        run = _build_run(
            "only-model:7b",
            strategy_assignments=["universal", "cot"],
        )
        path = tmp_path / "single.json"
        path.write_text(run.model_dump_json(), encoding="utf-8")

        runner = CliRunner()
        res = runner.invoke(app, ["routes", "analyze", "-r", str(path)])

        assert res.exit_code == 1
        assert "needs at least 2 distinct models" in res.output
        assert "only-model:7b" in res.output

    def test_picker_filters_to_routing_discovery_files(self, tmp_path):
        """The interactive picker for `routes analyze` must hide non-routing
        result files so users can't accidentally pick a regular `run` output.
        Filter is filename pre-filter + content check (`results[0].strategy`
        non-null) — filename alone is ambiguous when a suite is literally
        named `routing-discovery` and run via plain `porchbench run`."""
        from typer.testing import CliRunner

        from porchbench.cli import app

        non_routing = _build_run("m1", strategy_assignments=[None])

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
            from pathlib import Path as _Path
            iso_results = _Path(cwd) / "results"
            iso_results.mkdir()
            # Non-routing filename — should be filtered by name
            (iso_results / "2026-04-30T01-00-00_coding-basics_m1.json").write_text(
                non_routing.model_dump_json(), encoding="utf-8",
            )
            res = runner.invoke(app, ["routes", "analyze"])

        assert res.exit_code == 1
        assert "No routing-discovery result files found" in res.output
        assert "AttributeError" not in res.output

    def test_picker_excludes_filename_match_with_no_strategy_tags(self, tmp_path):
        """When a suite literally named `routing-discovery` got run via plain
        `porchbench run`, the resulting filename matches `_routing-discovery_`
        but the contents have no strategy tags. The picker filter must drop
        these via the content check, not just trust the filename."""
        from typer.testing import CliRunner

        from porchbench.cli import app

        # Filename matches the routing-discovery substring, but contents
        # have NO strategy tags — produced by `porchbench run -s routing-discovery`
        non_routing_named_routing = _build_run("gemma4:e2b", strategy_assignments=[None])

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
            from pathlib import Path as _Path
            iso_results = _Path(cwd) / "results"
            iso_results.mkdir()
            (iso_results / "2026-04-28T16-49-00_routing-discovery_gemma4-e2b.json").write_text(
                non_routing_named_routing.model_dump_json(), encoding="utf-8",
            )
            res = runner.invoke(app, ["routes", "analyze"])

        # The file was filtered out (content check rejected it), so no
        # routing-discovery results are available — friendly empty message
        # rather than the picker happily showing it.
        assert res.exit_code == 1
        assert "No routing-discovery result files found" in res.output

    def test_default_strategy_flag_validates_against_runs(self, tmp_path):
        """An unknown --default-strategy must fail loudly with the available
        set, not silently zero every comparison."""
        from typer.testing import CliRunner

        from porchbench.cli import app

        run_a = _build_run("m1", strategy_assignments=["universal", "cot"])
        run_b = _build_run("m2", strategy_assignments=["universal", "cot"])
        for n, r in [("a", run_a), ("b", run_b)]:
            (tmp_path / f"{n}.json").write_text(r.model_dump_json(), encoding="utf-8")

        runner = CliRunner()
        res = runner.invoke(app, [
            "routes", "analyze",
            "-r", str(tmp_path / "a.json"),
            "-r", str(tmp_path / "b.json"),
            "--default-strategy", "made-up",
        ])

        assert res.exit_code == 1
        assert "--default-strategy 'made-up' not present" in res.output
        # Available strategies surfaced so user sees what to pick instead
        assert "universal" in res.output
        assert "cot" in res.output


def _build_run(model: str, *, strategy_assignments: list[str | None]) -> RunResult:
    """Build a minimal routing RunResult with one PromptResult per
    strategy assignment in the given list."""
    results = []
    for i, strat in enumerate(strategy_assignments):
        results.append(PromptResult(
            prompt_id=f"p{i}",
            category="reasoning",
            difficulty="easy",
            options_used=ModelOptions(),
            request=RequestData(messages=[Message(role="user", content="x")]),
            response=ResponseData(message=ResponseMessage(content="y")),
            metrics=PromptMetrics(),
            strategy=strat,
            correct=True,
        ))
    return RunResult(
        run=RunMetadata(
            suite=SuiteReference(name="T", version="1.0", file="x", sha256="y"),
            model=ModelInfo(name=model, details=ModelDetails()),
            system=SystemInfo(),
        ),
        results=results,
        summary=RunSummary(
            total_prompts=len(results), completed=len(results),
            failed=0, total_duration_s=1.0,
        ),
    )
