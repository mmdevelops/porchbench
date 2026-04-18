"""Tests for leaderboard ranking, rubric grouping, and parse failure detection."""


import pytest

from porchbench.leaderboard import (
    MISSING_CRITERION_RATIONALE,
    PARSE_FAIL_RATIONALE,
    _clean_overall,
    _count_parse_failures,
    _normalize_rubric,
    aggregate_by_model,
    discover_scorecards,
    group_scorecards,
)
from porchbench.schemas import (
    AggregateScores,
    CriterionScore,
    EvaluationMetadata,
    PromptScore,
    Scorecard,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scorecard(
    rubric: str = "Test Rubric v1.0",
    evaluator: str = "test/evaluator",
    model_name: str | None = None,
    run_id: str = "abcdef01-0000-0000-0000-000000000000",
    scores: list[PromptScore] | None = None,
) -> Scorecard:
    """Build a minimal Scorecard for testing."""
    if scores is None:
        scores = [
            PromptScore(
                prompt_id="p1",
                criteria={
                    "correctness": CriterionScore(score=4, rationale="Good."),
                },
                weighted_score=4.0,
                summary="Fine.",
            )
        ]
    return Scorecard(
        evaluation=EvaluationMetadata(
            run_id=run_id,
            evaluator=evaluator,
            rubric=rubric,
            model_name=model_name,
        ),
        scores=scores,
        aggregate=AggregateScores(overall_weighted=4.0),
    )


# ---------------------------------------------------------------------------
# Rubric normalization
# ---------------------------------------------------------------------------


class TestNormalizeRubric:
    def test_strips_version(self):
        assert _normalize_rubric("Coding Rubric v1.0") == "coding rubric"

    def test_strips_multi_part_version(self):
        assert _normalize_rubric("My Rubric v2.1.3") == "my rubric"

    def test_strips_parenthetical(self):
        assert _normalize_rubric("cross-domain-science (suite-level override)") == "cross-domain-science"

    def test_strips_version_and_normalizes_case(self):
        assert _normalize_rubric("Cross-Domain Science Rubric v1.0") == "cross-domain science rubric"

    def test_complex_parenthetical(self):
        result = _normalize_rubric("category-aware (Coding Rubric, Reasoning Rubric)")
        assert result == "category-aware"

    def test_plain_name_unchanged(self):
        assert _normalize_rubric("default") == "default"

    def test_whitespace_collapsed(self):
        assert _normalize_rubric("  My   Rubric  ") == "my rubric"


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


class TestGroupScorecards:
    def test_single_group(self):
        sc1 = _make_scorecard(rubric="Coding Rubric v1.0")
        sc2 = _make_scorecard(rubric="Coding Rubric v2.0")
        groups = group_scorecards([sc1, sc2])
        # Both normalize to "coding rubric"
        assert len(groups) == 1
        assert len(list(groups.values())[0]) == 2

    def test_distinct_groups(self):
        sc1 = _make_scorecard(rubric="Coding Rubric v1.0")
        sc2 = _make_scorecard(rubric="Science Rubric v1.0")
        groups = group_scorecards([sc1, sc2])
        assert len(groups) == 2

    def test_empty_list(self):
        assert group_scorecards([]) == {}


# ---------------------------------------------------------------------------
# Parse failure detection
# ---------------------------------------------------------------------------


class TestCountParseFailures:
    def test_no_failures(self):
        sc = _make_scorecard()
        assert _count_parse_failures(sc) == (0, 0)

    def test_full_failure(self):
        scores = [
            PromptScore(
                prompt_id="p1",
                criteria={
                    "c1": CriterionScore(score=0, rationale=PARSE_FAIL_RATIONALE),
                    "c2": CriterionScore(score=0, rationale=PARSE_FAIL_RATIONALE),
                },
                weighted_score=0.0,
                summary="Failed.",
            )
        ]
        sc = _make_scorecard(scores=scores)
        full, partial = _count_parse_failures(sc)
        assert full == 1
        assert partial == 0

    def test_partial_failure(self):
        scores = [
            PromptScore(
                prompt_id="p1",
                criteria={
                    "c1": CriterionScore(score=4, rationale="Good."),
                    "c2": CriterionScore(score=0, rationale=MISSING_CRITERION_RATIONALE),
                },
                weighted_score=2.0,
                summary="Partial.",
            )
        ]
        sc = _make_scorecard(scores=scores)
        full, partial = _count_parse_failures(sc)
        assert full == 0
        assert partial == 1

    def test_mixed(self):
        scores = [
            PromptScore(
                prompt_id="p1",
                criteria={"c1": CriterionScore(score=0, rationale=PARSE_FAIL_RATIONALE)},
                weighted_score=0.0,
                summary="Fail.",
            ),
            PromptScore(
                prompt_id="p2",
                criteria={"c1": CriterionScore(score=5, rationale="Great.")},
                weighted_score=5.0,
                summary="OK.",
            ),
            PromptScore(
                prompt_id="p3",
                criteria={
                    "c1": CriterionScore(score=3, rationale="OK."),
                    "c2": CriterionScore(score=0, rationale=MISSING_CRITERION_RATIONALE),
                },
                weighted_score=1.5,
                summary="Partial.",
            ),
        ]
        sc = _make_scorecard(scores=scores)
        full, partial = _count_parse_failures(sc)
        assert full == 1
        assert partial == 1


# ---------------------------------------------------------------------------
# Clean overall score
# ---------------------------------------------------------------------------


class TestCleanOverall:
    def test_no_failures_returns_none(self):
        sc = _make_scorecard()
        assert _clean_overall(sc) is None

    def test_excludes_failed_prompts(self):
        scores = [
            PromptScore(
                prompt_id="p1",
                criteria={"c1": CriterionScore(score=0, rationale=PARSE_FAIL_RATIONALE)},
                weighted_score=0.0,
                summary="Fail.",
            ),
            PromptScore(
                prompt_id="p2",
                criteria={"c1": CriterionScore(score=4, rationale="Good.")},
                weighted_score=4.0,
                summary="OK.",
            ),
        ]
        sc = _make_scorecard(scores=scores)
        clean = _clean_overall(sc)
        assert clean == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscoverScorecards:
    def test_loads_valid_scorecards(self, tmp_path):
        sc = _make_scorecard()
        (tmp_path / "good.json").write_text(sc.model_dump_json())
        result = discover_scorecards(tmp_path)
        assert len(result) == 1

    def test_skips_corrupt_json(self, tmp_path, capsys):
        (tmp_path / "bad.json").write_text("{broken json")
        sc = _make_scorecard()
        (tmp_path / "good.json").write_text(sc.model_dump_json())
        result = discover_scorecards(tmp_path)
        assert len(result) == 1

    def test_skips_non_scorecard_json(self, tmp_path):
        (tmp_path / "other.json").write_text('{"not": "a scorecard"}')
        result = discover_scorecards(tmp_path)
        assert len(result) == 0

    def test_empty_directory(self, tmp_path):
        result = discover_scorecards(tmp_path)
        assert result == []


# ---------------------------------------------------------------------------
# Aggregation across repeats
# ---------------------------------------------------------------------------


def _make_scorecard_with_aggs(
    model_name: str,
    overall: float,
    by_category: dict[str, float] | None = None,
    by_difficulty: dict[str, float] | None = None,
    run_id: str | None = None,
    scores: list[PromptScore] | None = None,
) -> Scorecard:
    """Build a Scorecard with a known model name and configurable aggregates."""
    return Scorecard(
        evaluation=EvaluationMetadata(
            run_id=run_id or f"run-{model_name}-{overall}",
            evaluator="test/eval",
            rubric="Test Rubric v1.0",
            model_name=model_name,
        ),
        scores=scores or [],
        aggregate=AggregateScores(
            overall_weighted=overall,
            by_category=by_category or {},
            by_difficulty=by_difficulty or {},
        ),
    )


class TestAggregateByModel:
    def test_single_scorecard_per_model_preserves_scores(self):
        sc_a = _make_scorecard_with_aggs("model-a", 4.2, {"coding": 4.2}, {"easy": 4.5})
        sc_b = _make_scorecard_with_aggs("model-b", 3.8, {"coding": 3.8}, {"easy": 4.0})
        rows = aggregate_by_model([sc_a, sc_b])

        assert len(rows) == 2
        by_label = {r.label: r for r in rows}
        assert by_label["model-a"].n == 1
        assert by_label["model-a"].mean_overall == pytest.approx(4.2)
        assert by_label["model-a"].overall_range == 0.0
        assert by_label["model-a"].cat_means == {"coding": pytest.approx(4.2)}
        assert by_label["model-b"].mean_overall == pytest.approx(3.8)

    def test_identical_repeats_collapse_to_one_row(self):
        """The exact case from the user's bug report: two temp=0/seed=42 runs
        produce identical scores and should collapse into n=2 with no spread."""
        sc_1 = _make_scorecard_with_aggs(
            "qwen2.5:3b", 4.19, {"coding": 3.67}, run_id="run-1",
        )
        sc_2 = _make_scorecard_with_aggs(
            "qwen2.5:3b", 4.19, {"coding": 3.67}, run_id="run-2",
        )
        rows = aggregate_by_model([sc_1, sc_2])

        assert len(rows) == 1
        row = rows[0]
        assert row.label == "qwen2.5:3b"
        assert row.n == 2
        assert row.mean_overall == pytest.approx(4.19)
        assert row.overall_range == pytest.approx(0.0)
        assert row.cat_means == {"coding": pytest.approx(3.67)}

    def test_varying_repeats_produce_range_and_mean(self):
        sc_1 = _make_scorecard_with_aggs("qwen2.5:3b", 4.10, run_id="r1")
        sc_2 = _make_scorecard_with_aggs("qwen2.5:3b", 4.20, run_id="r2")
        sc_3 = _make_scorecard_with_aggs("qwen2.5:3b", 4.30, run_id="r3")
        rows = aggregate_by_model([sc_1, sc_2, sc_3])

        assert len(rows) == 1
        row = rows[0]
        assert row.n == 3
        assert row.mean_overall == pytest.approx(4.20)
        assert row.overall_range == pytest.approx(0.20)

    def test_pooled_prompt_means_average_across_repeats(self):
        """Best/worst lookups should average per-prompt scores across repeats,
        so one noisy judge call doesn't flip a prompt's ranking."""
        def _ps(prompt_id: str, weighted: float) -> PromptScore:
            return PromptScore(
                prompt_id=prompt_id,
                criteria={"c1": CriterionScore(score=int(weighted), rationale="ok")},
                weighted_score=weighted,
                summary="",
            )

        sc_1 = _make_scorecard_with_aggs(
            "model-x", 4.0, run_id="r1",
            scores=[_ps("p-easy", 5.0), _ps("p-hard", 2.0)],
        )
        sc_2 = _make_scorecard_with_aggs(
            "model-x", 4.0, run_id="r2",
            scores=[_ps("p-easy", 5.0), _ps("p-hard", 3.0)],
        )
        rows = aggregate_by_model([sc_1, sc_2])

        assert len(rows) == 1
        pooled = rows[0].pooled_prompt_means
        assert pooled["p-easy"] == pytest.approx(5.0)
        assert pooled["p-hard"] == pytest.approx(2.5)

    def test_parse_failures_sum_across_repeats(self):
        fail_score = PromptScore(
            prompt_id="p-fail",
            criteria={"c1": CriterionScore(score=0, rationale=PARSE_FAIL_RATIONALE)},
            weighted_score=0.0,
            summary="",
        )
        ok_score = PromptScore(
            prompt_id="p-ok",
            criteria={"c1": CriterionScore(score=4, rationale="good")},
            weighted_score=4.0,
            summary="",
        )
        sc_1 = _make_scorecard_with_aggs(
            "model-x", 2.0, run_id="r1", scores=[fail_score, ok_score],
        )
        sc_2 = _make_scorecard_with_aggs(
            "model-x", 2.0, run_id="r2", scores=[fail_score, ok_score],
        )
        rows = aggregate_by_model([sc_1, sc_2])

        assert len(rows) == 1
        assert rows[0].total_full_fail == 2  # one per repeat
        assert rows[0].mean_clean is not None  # at least one repeat had a clean variant

    def test_mean_clean_is_none_when_no_parse_failures(self):
        sc = _make_scorecard_with_aggs("model-x", 4.0, run_id="r1")
        rows = aggregate_by_model([sc])
        assert rows[0].mean_clean is None

    def test_different_models_stay_separate_even_with_identical_scores(self):
        sc_a = _make_scorecard_with_aggs("model-a", 4.0)
        sc_b = _make_scorecard_with_aggs("model-b", 4.0)
        rows = aggregate_by_model([sc_a, sc_b])
        assert len(rows) == 2
        assert {r.label for r in rows} == {"model-a", "model-b"}
        assert all(r.n == 1 for r in rows)

    def test_empty_input_returns_empty_list(self):
        assert aggregate_by_model([]) == []
