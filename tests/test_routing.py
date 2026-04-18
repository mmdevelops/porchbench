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
