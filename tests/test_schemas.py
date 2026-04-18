"""Tests for pydantic schema models: validation, serialization, computed metrics."""

import pytest
from pydantic import ValidationError

from porchbench.schemas import (
    AggregateScores,
    Criterion,
    CriterionScore,
    EvaluationMetadata,
    Message,
    ModelDetails,
    ModelInfo,
    ModelOptions,
    Prompt,
    PromptMetrics,
    PromptResult,
    PromptScore,
    RequestData,
    ResponseData,
    ResponseMessage,
    Rubric,
    RubricMetadata,
    RunMetadata,
    RunResult,
    RunSummary,
    Scorecard,
    Strategy,
    Suite,
    SuiteDefaults,
    SuiteMetadata,
    SuiteReference,
    SystemInfo,
    ToolUseMetricsData,
    compute_derived_metrics,
)

# ---------------------------------------------------------------------------
# Suite schemas
# ---------------------------------------------------------------------------


class TestSuite:
    def test_minimal_suite(self):
        suite = Suite(
            suite=SuiteMetadata(name="Test", version="1.0"),
            defaults=SuiteDefaults(options=ModelOptions()),
            prompts=[
                Prompt(
                    id="p1",
                    category="coding",
                    difficulty="easy",
                    messages=[Message(role="user", content="Hello")],
                )
            ],
        )
        assert suite.suite.name == "Test"
        assert len(suite.prompts) == 1

    def test_duplicate_prompt_ids_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate prompt IDs"):
            Suite(
                suite=SuiteMetadata(name="Test", version="1.0"),
                defaults=SuiteDefaults(options=ModelOptions()),
                prompts=[
                    Prompt(id="dup", category="coding", difficulty="easy",
                           messages=[Message(role="user", content="A")]),
                    Prompt(id="dup", category="coding", difficulty="easy",
                           messages=[Message(role="user", content="B")]),
                ],
            )

    def test_suite_with_strategies(self):
        suite = Suite(
            suite=SuiteMetadata(name="Routing", version="1.0"),
            defaults=SuiteDefaults(options=ModelOptions()),
            prompts=[
                Prompt(id="p1", category="coding", difficulty="easy",
                       messages=[Message(role="user", content="Hi")])
            ],
            strategies={"brevity": Strategy(system_message="Be brief.")},
        )
        assert "brevity" in suite.strategies
        assert suite.strategies["brevity"].system_message == "Be brief."

    def test_prompt_routing_extensions(self):
        p = Prompt(
            id="math-1", category="reasoning", difficulty="easy",
            messages=[Message(role="user", content="2+2?")],
            answer_type="numeric", reasoning_depth="shallow",
            expected_answer="4", contamination_risk="low",
        )
        assert p.answer_type == "numeric"
        assert p.expected_answer == "4"
        assert p.contamination_risk == "low"

    def test_prompt_sandbox_extensions(self):
        p = Prompt(
            id="tool-1", category="tool-use", difficulty="medium",
            messages=[Message(role="user", content="Sort the file")],
            mode="tool-use",
            tools=[{"type": "function", "function": {"name": "execute_code"}}],
            max_tool_calls=10,
        )
        assert p.mode == "tool-use"
        assert len(p.tools) == 1
        assert p.max_tool_calls == 10

    def test_model_options_extra_fields(self):
        opts = ModelOptions(temperature=0.7, seed=42, mirostat=2, repeat_penalty=1.1)
        dumped = opts.model_dump()
        assert dumped["mirostat"] == 2
        assert dumped["repeat_penalty"] == 1.1
        assert dumped["temperature"] == 0.7


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_compute_derived_metrics(self):
        raw = PromptMetrics(
            prompt_eval_count=52,
            prompt_eval_duration=312000000,
            eval_count=187,
            eval_duration=4210000000,
            total_duration=4580000000,
            load_duration=101000000,
        )
        computed = compute_derived_metrics(raw)
        assert computed.tokens_per_second == pytest.approx(44.42, rel=0.01)
        assert computed.time_to_first_token_ms == pytest.approx(312.0, rel=0.01)
        # Raw fields preserved
        assert computed.eval_count == 187
        assert computed.load_duration == 101000000

    def test_compute_derived_metrics_zero_duration(self):
        raw = PromptMetrics(eval_count=100, eval_duration=0)
        computed = compute_derived_metrics(raw)
        assert computed.tokens_per_second is None

    def test_compute_derived_metrics_missing_fields(self):
        raw = PromptMetrics()
        computed = compute_derived_metrics(raw)
        assert computed.tokens_per_second is None
        assert computed.time_to_first_token_ms is None

    def test_compute_derived_metrics_immutable(self):
        raw = PromptMetrics(eval_count=100, eval_duration=1000000000)
        computed = compute_derived_metrics(raw)
        # Original unchanged
        assert raw.tokens_per_second is None
        assert computed.tokens_per_second == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Run result roundtrip
# ---------------------------------------------------------------------------


class TestRunResult:
    def test_json_roundtrip(self):
        run = RunResult(
            run=RunMetadata(
                suite=SuiteReference(name="Test", version="1.0",
                                     file="test.yaml", sha256="abc"),
                model=ModelInfo(name="test:7b", digest="sha256:def",
                                details=ModelDetails(family="test",
                                                     quantization_level="Q4_K_M")),
                system=SystemInfo(ollama_version="0.20.5", os="Windows 11"),
            ),
            results=[
                PromptResult(
                    prompt_id="p1", category="coding", difficulty="easy",
                    options_used=ModelOptions(),
                    request=RequestData(messages=[Message(role="user", content="Hi")]),
                    response=ResponseData(message=ResponseMessage(content="Hello"),
                                          done_reason="stop"),
                    metrics=PromptMetrics(eval_count=10, eval_duration=100000000),
                ),
            ],
            summary=RunSummary(total_prompts=1, completed=1, failed=0,
                               total_duration_s=1.0, avg_tokens_per_second=100.0),
        )
        json_str = run.model_dump_json()
        restored = RunResult.model_validate_json(json_str)
        assert restored.run.model.name == "test:7b"
        assert restored.run.model.digest == "sha256:def"
        assert restored.results[0].prompt_id == "p1"
        assert restored.summary.completed == 1

    def test_prompt_result_routing_fields(self):
        pr = PromptResult(
            prompt_id="p1", category="coding", difficulty="easy",
            options_used=ModelOptions(),
            request=RequestData(messages=[Message(role="user", content="Hi")]),
            response=ResponseData(message=ResponseMessage(content="42")),
            strategy="direct", correct=True, expected_answer="42",
        )
        assert pr.strategy == "direct"
        assert pr.correct is True
        assert pr.expected_answer == "42"

    def test_prompt_result_tool_use_fields_default_none(self):
        pr = PromptResult(
            prompt_id="p1", category="coding", difficulty="easy",
            options_used=ModelOptions(),
            request=RequestData(messages=[Message(role="user", content="Hi")]),
            response=ResponseData(message=ResponseMessage(content="ok")),
        )
        assert pr.validation_passed is None
        assert pr.validation_reason is None
        assert pr.stopped_reason is None
        assert pr.tool_use_metrics is None

    def test_prompt_result_tool_use_fields_populated(self):
        metrics = ToolUseMetricsData(
            total_tool_calls=3,
            tool_call_breakdown={"execute_code": 2, "read_file": 1},
            errors_encountered=1,
            self_corrections=1,
            conversation_turns=4,
        )
        pr = PromptResult(
            prompt_id="t1-csv", category="tool-use", difficulty="easy",
            options_used=ModelOptions(),
            request=RequestData(messages=[Message(role="user", content="Sort the CSV")]),
            response=ResponseData(
                message=ResponseMessage(content="Done"),
                done_reason="done",
            ),
            validation_passed=True,
            validation_reason="CSV sorted correctly",
            stopped_reason="done",
            tool_use_metrics=metrics,
        )
        assert pr.validation_passed is True
        assert pr.stopped_reason == "done"
        assert pr.tool_use_metrics.total_tool_calls == 3
        assert pr.tool_use_metrics.tool_call_breakdown["execute_code"] == 2

    def test_prompt_result_tool_use_roundtrip_json(self):
        """Tool-use fields survive JSON serialization and deserialization."""
        metrics = ToolUseMetricsData(
            total_tool_calls=5,
            tool_call_breakdown={"execute_code": 3, "write_file": 2},
            errors_encountered=0,
            self_corrections=0,
            conversation_turns=6,
        )
        pr = PromptResult(
            prompt_id="t2-multi", category="tool-use", difficulty="medium",
            options_used=ModelOptions(),
            request=RequestData(messages=[Message(role="user", content="task")]),
            response=ResponseData(message=ResponseMessage(content="result")),
            validation_passed=False,
            validation_reason="Output mismatch",
            stopped_reason="max_tool_calls",
            tool_use_metrics=metrics,
        )
        json_str = pr.model_dump_json()
        restored = PromptResult.model_validate_json(json_str)
        assert restored.validation_passed is False
        assert restored.validation_reason == "Output mismatch"
        assert restored.stopped_reason == "max_tool_calls"
        assert restored.tool_use_metrics.total_tool_calls == 5
        assert restored.tool_use_metrics.tool_call_breakdown["write_file"] == 2

    def test_tool_use_metrics_data_defaults(self):
        m = ToolUseMetricsData()
        assert m.total_tool_calls == 0
        assert m.tool_call_breakdown == {}
        assert m.errors_encountered == 0

    def test_model_info_defaults(self):
        info = ModelInfo(name="test:7b")
        assert info.digest is None
        assert info.details.format is None
        assert info.details.quantization_level is None


# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------


class TestRubric:
    def test_valid_rubric(self):
        rubric = Rubric(
            rubric=RubricMetadata(name="Test", version="1.0"),
            criteria=[
                Criterion(name="a", weight=0.6, description="..."),
                Criterion(name="b", weight=0.4, description="..."),
            ],
        )
        assert len(rubric.criteria) == 2

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValidationError, match="weights sum"):
            Rubric(
                rubric=RubricMetadata(name="Bad", version="1.0"),
                criteria=[Criterion(name="x", weight=0.3, description="...")],
            )

    def test_weights_tolerance(self):
        # 0.35 + 0.25 + 0.20 + 0.10 + 0.10 = 1.00
        rubric = Rubric(
            rubric=RubricMetadata(name="Default", version="1.0"),
            criteria=[
                Criterion(name="a", weight=0.35, description="..."),
                Criterion(name="b", weight=0.25, description="..."),
                Criterion(name="c", weight=0.20, description="..."),
                Criterion(name="d", weight=0.10, description="..."),
                Criterion(name="e", weight=0.10, description="..."),
            ],
        )
        assert len(rubric.criteria) == 5


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------


class TestScorecard:
    def test_scorecard_structure(self):
        sc = Scorecard(
            evaluation=EvaluationMetadata(
                run_id="abc", evaluator="claude", rubric="Test v1.0",
            ),
            scores=[
                PromptScore(
                    prompt_id="p1",
                    criteria={"correctness": CriterionScore(score=5, rationale="Good")},
                    weighted_score=4.5,
                    summary="Strong.",
                )
            ],
            aggregate=AggregateScores(
                overall_weighted=4.5,
                by_category={"coding": 4.5},
                by_difficulty={"easy": 4.5},
            ),
        )
        assert sc.scores[0].criteria["correctness"].score == 5
        assert sc.aggregate.overall_weighted == 4.5
