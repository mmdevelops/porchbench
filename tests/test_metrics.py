"""Tests for metrics aggregation and descriptive statistics."""

import pytest

from ollama_bench.metrics import (
    describe,
    extract_tokens_per_second,
    extract_time_to_first_token,
    extract_total_tokens,
    group_by_category,
    group_by_difficulty,
    summarize_run,
)
from ollama_bench.schemas import (
    Message,
    ModelDetails,
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


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _make_result(tps=None, ttft=None, tokens=None, category="coding", difficulty="easy"):
    return PromptResult(
        prompt_id="test",
        category=category,
        difficulty=difficulty,
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
