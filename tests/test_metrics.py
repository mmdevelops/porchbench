"""Tests for metrics aggregation and descriptive statistics."""

import pytest

from porchbench.metrics import (
    describe,
    extract_tokens_per_second,
    extract_time_to_first_token,
    extract_total_tokens,
    filter_contamination,
    group_by_category,
    group_by_difficulty,
    summarize_run,
)
from porchbench.evaluator import compute_aggregates, normalize_score
from porchbench.schemas import (
    CriterionScore,
    Message,
    ModelDetails,
    ModelInfo,
    ModelOptions,
    PromptMetrics,
    PromptResult,
    PromptScore,
    RequestData,
    ResponseData,
    ResponseMessage,
    RunMetadata,
    RunResult,
    RunSummary,
    SuiteReference,
    SystemInfo,
)


# ---------------------------------------------------------------------------
# Descriptive stats
# ---------------------------------------------------------------------------


class TestDescribe:
    def test_basic(self):
        stats = describe([10.0, 20.0, 30.0])
        assert stats.count == 3
        assert stats.mean == pytest.approx(20.0)
        assert stats.median == pytest.approx(20.0)
        assert stats.min == pytest.approx(10.0)
        assert stats.max == pytest.approx(30.0)
        assert stats.stdev is not None

    def test_single_value(self):
        stats = describe([42.0])
        assert stats.count == 1
        assert stats.mean == pytest.approx(42.0)
        assert stats.stdev is None  # can't compute stdev with n=1

    def test_empty_list(self):
        assert describe([]) is None

    def test_as_dict(self):
        stats = describe([1.0, 2.0, 3.0])
        d = stats.as_dict()
        assert "count" in d
        assert "mean" in d
        assert d["count"] == 3

    def test_ci_included_for_multiple_values(self):
        stats = describe([10.0, 20.0, 30.0, 40.0, 50.0])
        assert stats.ci is not None
        assert stats.ci.mean == pytest.approx(30.0)
        assert stats.ci.ci_lower < 30.0
        assert stats.ci.ci_upper > 30.0
        d = stats.as_dict()
        assert "ci" in d

    def test_ci_none_for_single_value(self):
        stats = describe([42.0])
        assert stats.ci is None

    def test_ci_disabled(self):
        stats = describe([10.0, 20.0, 30.0], compute_ci=False)
        assert stats.ci is None


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _make_result(tps=None, ttft=None, tokens=None, category="coding", difficulty="easy",
                  contamination_risk=None, prompt_id="test"):
    return PromptResult(
        prompt_id=prompt_id,
        category=category,
        difficulty=difficulty,
        contamination_risk=contamination_risk,
        options_used=ModelOptions(),
        request=RequestData(messages=[Message(role="user", content="Hi")]),
        response=ResponseData(message=ResponseMessage(content="Hello")),
        metrics=PromptMetrics(
            tokens_per_second=tps,
            time_to_first_token_ms=ttft,
            eval_count=tokens,
        ),
    )


class TestExtractors:
    def test_extract_tps(self):
        results = [_make_result(tps=100.0), _make_result(tps=None), _make_result(tps=200.0)]
        values = extract_tokens_per_second(results)
        assert values == [100.0, 200.0]

    def test_extract_ttft(self):
        results = [_make_result(ttft=10.0), _make_result(ttft=20.0)]
        values = extract_time_to_first_token(results)
        assert values == [10.0, 20.0]

    def test_extract_total_tokens(self):
        results = [_make_result(tokens=50), _make_result(tokens=None), _make_result(tokens=100)]
        values = extract_total_tokens(results)
        assert values == [50, 100]


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


class TestGrouping:
    def test_group_by_category(self):
        results = [
            _make_result(category="coding"),
            _make_result(category="coding"),
            _make_result(category="reasoning"),
        ]
        groups = group_by_category(results)
        assert len(groups["coding"]) == 2
        assert len(groups["reasoning"]) == 1

    def test_group_by_difficulty(self):
        results = [
            _make_result(difficulty="easy"),
            _make_result(difficulty="hard"),
            _make_result(difficulty="easy"),
        ]
        groups = group_by_difficulty(results)
        assert len(groups["easy"]) == 2
        assert len(groups["hard"]) == 1


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------


class TestSummarizeRun:
    def test_basic_summary(self):
        run = RunResult(
            run=RunMetadata(
                suite=SuiteReference(name="T", version="1", file="t.yaml", sha256="x"),
                model=ModelInfo(name="test:3b",
                                details=ModelDetails(quantization_level="Q4_K_M")),
                system=SystemInfo(kv_cache_type="f16"),
            ),
            results=[
                _make_result(tps=100.0, ttft=10.0, tokens=50, category="coding"),
                _make_result(tps=200.0, ttft=20.0, tokens=100, category="reasoning"),
            ],
            summary=RunSummary(total_prompts=2, completed=2, failed=0,
                               total_duration_s=5.0, avg_tokens_per_second=150.0),
        )
        summary = summarize_run(run)
        assert summary["model"] == "test:3b"
        assert summary["quantization"] == "Q4_K_M"
        assert summary["kv_cache_type"] == "f16"
        assert summary["overall"]["tokens_per_second"]["mean"] == pytest.approx(150.0)
        assert "coding" in summary["by_category"]
        assert "reasoning" in summary["by_category"]


# ---------------------------------------------------------------------------
# Contamination filtering
# ---------------------------------------------------------------------------


class TestContaminationFiltering:
    def test_filter_excludes_high(self):
        results = [
            _make_result(prompt_id="p1", contamination_risk="high"),
            _make_result(prompt_id="p2", contamination_risk="low"),
            _make_result(prompt_id="p3", contamination_risk=None),
        ]
        filtered = filter_contamination(results, exclude="high")
        assert len(filtered) == 2
        assert all(r.contamination_risk != "high" for r in filtered)

    def test_filter_excludes_medium_and_high(self):
        results = [
            _make_result(prompt_id="p1", contamination_risk="high"),
            _make_result(prompt_id="p2", contamination_risk="medium"),
            _make_result(prompt_id="p3", contamination_risk="low"),
        ]
        filtered = filter_contamination(results, exclude="medium")
        assert len(filtered) == 1
        assert filtered[0].contamination_risk == "low"

    def test_filter_keeps_all_when_no_contamination(self):
        results = [_make_result(prompt_id="p1"), _make_result(prompt_id="p2")]
        filtered = filter_contamination(results, exclude="high")
        assert len(filtered) == 2


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_normalize_boundaries(self):
        assert normalize_score(1.0) == pytest.approx(0.0)
        assert normalize_score(5.0) == pytest.approx(100.0)
        assert normalize_score(3.0) == pytest.approx(50.0)

    def test_normalize_clamps_below_min(self):
        assert normalize_score(0.5) == pytest.approx(0.0)

    def test_normalize_custom_scale(self):
        assert normalize_score(50, scale_min=0, scale_max=100) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Aggregation with contamination + normalization
# ---------------------------------------------------------------------------


def _make_score(prompt_id, weighted_score):
    return PromptScore(
        prompt_id=prompt_id,
        criteria={"quality": CriterionScore(score=int(weighted_score), rationale="ok")},
        weighted_score=weighted_score,
        summary="ok",
    )


class TestComputeAggregates:
    def test_contamination_filtered_aggregates(self):
        results = [
            _make_result(prompt_id="p1", difficulty="easy", contamination_risk="high"),
            _make_result(prompt_id="p2", difficulty="easy", contamination_risk="low"),
            _make_result(prompt_id="p3", difficulty="hard", contamination_risk=None),
        ]
        scores = [
            _make_score("p1", 5.0),
            _make_score("p2", 3.0),
            _make_score("p3", 2.0),
        ]
        agg = compute_aggregates(scores, results)

        # Overall includes all
        assert agg.overall_weighted == pytest.approx(3.33, abs=0.01)
        # Clean excludes p1 (high contamination)
        assert agg.overall_weighted_clean == pytest.approx(2.5)
        assert "easy" in agg.by_category_clean or "coding" in agg.by_category_clean

    def test_normalized_scoring(self):
        results = [
            _make_result(prompt_id="p1", difficulty="easy"),
            _make_result(prompt_id="p2", difficulty="hard"),
        ]
        scores = [
            _make_score("p1", 5.0),  # easy=5.0 → normalized 100
            _make_score("p2", 1.0),  # hard=1.0 → normalized 0
        ]
        agg = compute_aggregates(scores, results)

        assert agg.by_difficulty_normalized["easy"] == pytest.approx(100.0)
        assert agg.by_difficulty_normalized["hard"] == pytest.approx(0.0)
        # Overall normalized = mean of difficulty-level normalized scores
        assert agg.overall_normalized == pytest.approx(50.0)

    def test_empty_scores(self):
        agg = compute_aggregates([], [])
        assert agg.overall_weighted == 0.0
        assert agg.overall_normalized is None
        assert agg.overall_weighted_clean is None
