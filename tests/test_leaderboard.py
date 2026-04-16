"""Tests for leaderboard ranking, rubric grouping, and parse failure detection."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from feral.leaderboard import (
    MISSING_CRITERION_RATIONALE,
    PARSE_FAIL_RATIONALE,
    _clean_overall,
    _count_parse_failures,
    _normalize_rubric,
    discover_scorecards,
    group_scorecards,
    load_scorecard,
)
from feral.schemas import (
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
