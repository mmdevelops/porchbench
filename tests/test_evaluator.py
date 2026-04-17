"""Tests for evaluator backends, scoring prompt construction, and score parsing."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from porchbench.evaluator import (
    ClaudeCodeEvalBackend,
    _extract_json,
    _extract_summary,
    _parse_scoring_response,
    build_scoring_prompt,
    compute_aggregates,
    normalize_score,
    score_prompt,
)
from porchbench.schemas import (
    Criterion,
    CriterionScore,
    Message,
    ModelOptions,
    PromptMetrics,
    PromptResult,
    PromptScore,
    RequestData,
    ResponseData,
    ResponseMessage,
    Rubric,
    RubricMetadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rubric(**overrides) -> Rubric:
    defaults = dict(
        rubric=RubricMetadata(name="test-rubric", version="1.0"),
        criteria=[
            Criterion(name="correctness", weight=0.6, description="Is the answer correct?"),
            Criterion(name="clarity", weight=0.4, description="Is the answer clear?"),
        ],
    )
    defaults.update(overrides)
    return Rubric(**defaults)


def _make_prompt_result(**overrides) -> PromptResult:
    defaults = dict(
        prompt_id="p1",
        category="coding",
        difficulty="medium",
        options_used=ModelOptions(),
        request=RequestData(messages=[Message(role="user", content="Write hello world")]),
        response=ResponseData(message=ResponseMessage(content="print('hello world')")),
        metrics=PromptMetrics(),
    )
    defaults.update(overrides)
    return PromptResult(**defaults)


# ---------------------------------------------------------------------------
# ClaudeCodeEvalBackend
# ---------------------------------------------------------------------------


class TestClaudeCodeEvalBackend:
    def test_init_defaults(self):
        backend = ClaudeCodeEvalBackend()
        assert backend.model == "sonnet"
        assert backend.timeout_s == 120

    def test_init_custom(self):
        backend = ClaudeCodeEvalBackend(model="opus", timeout_s=300)
        assert backend.model == "opus"
        assert backend.timeout_s == 300

    @pytest.mark.asyncio
    async def test_generate_sends_prompt_via_stdin(self):
        backend = ClaudeCodeEvalBackend(model="sonnet")

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b'{"criteria": {}}', b"")
        mock_proc.returncode = 0

        with patch("porchbench.evaluator.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            result = await backend.generate("Score this response")

        # Verify the command
        args = mock_exec.call_args[0]
        assert args[0] == "claude"
        assert args[1] == "-p"
        assert "--model" in args
        assert "sonnet" in args
        assert "--output-format" in args
        assert "text" in args

        # Verify prompt was sent via stdin
        mock_proc.communicate.assert_awaited_once()
        stdin_data = mock_proc.communicate.call_args[1]["input"]
        assert stdin_data == b"Score this response"

        assert result == '{"criteria": {}}'

    @pytest.mark.asyncio
    async def test_generate_nonzero_exit_raises(self):
        backend = ClaudeCodeEvalBackend()

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"Error: rate limited")
        mock_proc.returncode = 1

        with patch("porchbench.evaluator.asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="claude -p failed"):
                await backend.generate("prompt")

    @pytest.mark.asyncio
    async def test_generate_timeout_raises(self):
        backend = ClaudeCodeEvalBackend(timeout_s=1)

        mock_proc = AsyncMock()
        mock_proc.communicate.side_effect = asyncio.TimeoutError()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with patch("porchbench.evaluator.asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="timed out"):
                await backend.generate("prompt")

        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_uses_custom_model(self):
        backend = ClaudeCodeEvalBackend(model="opus")

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"response", b"")
        mock_proc.returncode = 0

        with patch("porchbench.evaluator.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await backend.generate("prompt")

        args = mock_exec.call_args[0]
        model_idx = list(args).index("--model")
        assert args[model_idx + 1] == "opus"

    @pytest.mark.asyncio
    async def test_generate_handles_unicode_prompt(self):
        backend = ClaudeCodeEvalBackend()

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = ("résultat".encode("utf-8"), b"")
        mock_proc.returncode = 0

        with patch("porchbench.evaluator.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await backend.generate("évaluer cette réponse")

        stdin_data = mock_proc.communicate.call_args[1]["input"]
        assert stdin_data == "évaluer cette réponse".encode("utf-8")
        assert result == "résultat"


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_clean_json(self):
        raw = '{"criteria": {"correctness": {"score": 4, "rationale": "good"}}}'
        assert json.loads(_extract_json(raw)) == json.loads(raw)

    def test_markdown_fencing(self):
        raw = '```json\n{"score": 4}\n```'
        assert json.loads(_extract_json(raw)) == {"score": 4}

    def test_think_tags(self):
        raw = '<think>reasoning here</think>\n{"score": 4}'
        assert json.loads(_extract_json(raw)) == {"score": 4}

    def test_preamble_text(self):
        raw = 'Here is my evaluation:\n{"score": 4}'
        assert json.loads(_extract_json(raw)) == {"score": 4}


# ---------------------------------------------------------------------------
# Scoring prompt construction
# ---------------------------------------------------------------------------


class TestBuildScoringPrompt:
    def test_includes_user_prompt(self):
        pr = _make_prompt_result()
        rubric = _make_rubric()
        prompt = build_scoring_prompt(pr, rubric)
        assert "Write hello world" in prompt

    def test_includes_model_response(self):
        pr = _make_prompt_result()
        rubric = _make_rubric()
        prompt = build_scoring_prompt(pr, rubric)
        assert "print('hello world')" in prompt

    def test_includes_criteria(self):
        pr = _make_prompt_result()
        rubric = _make_rubric()
        prompt = build_scoring_prompt(pr, rubric)
        assert "correctness" in prompt
        assert "clarity" in prompt

    def test_includes_expected_answer_when_present(self):
        pr = _make_prompt_result(expected_answer="should print hello world to stdout")
        rubric = _make_rubric()
        prompt = build_scoring_prompt(pr, rubric)
        assert "Reference (Correctness Guide)" in prompt
        assert "should print hello world to stdout" in prompt

    def test_no_reference_section_without_expected_answer(self):
        pr = _make_prompt_result()
        rubric = _make_rubric()
        prompt = build_scoring_prompt(pr, rubric)
        assert "Reference (Correctness Guide)" not in prompt


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------


class TestParseScoring:
    def test_valid_response(self):
        rubric = _make_rubric()
        text = json.dumps({
            "criteria": {
                "correctness": {"score": 5, "rationale": "Perfect"},
                "clarity": {"score": 4, "rationale": "Good"},
            },
            "summary": "Well done",
        })
        result = _parse_scoring_response(text, rubric)
        assert result["correctness"].score == 5
        assert result["clarity"].score == 4

    def test_missing_criterion_returns_zero(self):
        rubric = _make_rubric()
        text = json.dumps({
            "criteria": {
                "correctness": {"score": 5, "rationale": "Fine"},
            },
            "summary": "ok",
        })
        result = _parse_scoring_response(text, rubric)
        assert result["clarity"].score == 0

    def test_unparseable_returns_zeros(self):
        rubric = _make_rubric()
        result = _parse_scoring_response("not json at all", rubric)
        assert result["correctness"].score == 0
        assert result["clarity"].score == 0


# ---------------------------------------------------------------------------
# Score prompt (end-to-end with mocked backend)
# ---------------------------------------------------------------------------


class TestScorePrompt:
    @pytest.mark.asyncio
    async def test_score_prompt_returns_weighted_score(self):
        rubric = _make_rubric()
        pr = _make_prompt_result()

        mock_backend = AsyncMock()
        mock_backend.generate.return_value = json.dumps({
            "criteria": {
                "correctness": {"score": 5, "rationale": "Correct"},
                "clarity": {"score": 4, "rationale": "Clear"},
            },
            "summary": "Good response",
        })

        result = await score_prompt(pr, rubric, mock_backend)
        assert result.prompt_id == "p1"
        # weighted = 5*0.6 + 4*0.4 = 3.0 + 1.6 = 4.6
        assert result.weighted_score == pytest.approx(4.6, abs=0.01)
        assert result.criteria["correctness"].score == 5


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestNormalizeScore:
    def test_min_maps_to_zero(self):
        assert normalize_score(1.0) == 0.0

    def test_max_maps_to_hundred(self):
        assert normalize_score(5.0) == 100.0

    def test_midpoint(self):
        assert normalize_score(3.0) == pytest.approx(50.0)


class TestComputeAggregates:
    def test_overall_weighted(self):
        scores = [
            PromptScore(prompt_id="p1", criteria={}, weighted_score=4.0, summary=""),
            PromptScore(prompt_id="p2", criteria={}, weighted_score=3.0, summary=""),
        ]
        results = [
            _make_prompt_result(prompt_id="p1", category="coding", difficulty="easy"),
            _make_prompt_result(prompt_id="p2", category="reasoning", difficulty="hard"),
        ]
        agg = compute_aggregates(scores, results)
        assert agg.overall_weighted == 3.5

    def test_by_category(self):
        scores = [
            PromptScore(prompt_id="p1", criteria={}, weighted_score=4.0, summary=""),
            PromptScore(prompt_id="p2", criteria={}, weighted_score=2.0, summary=""),
        ]
        results = [
            _make_prompt_result(prompt_id="p1", category="coding", difficulty="easy"),
            _make_prompt_result(prompt_id="p2", category="reasoning", difficulty="easy"),
        ]
        agg = compute_aggregates(scores, results)
        assert agg.by_category["coding"] == 4.0
        assert agg.by_category["reasoning"] == 2.0

    def test_contamination_filtering(self):
        scores = [
            PromptScore(prompt_id="p1", criteria={}, weighted_score=5.0, summary=""),
            PromptScore(prompt_id="p2", criteria={}, weighted_score=3.0, summary=""),
        ]
        results = [
            _make_prompt_result(prompt_id="p1", category="coding", difficulty="easy", contamination_risk="high"),
            _make_prompt_result(prompt_id="p2", category="coding", difficulty="easy", contamination_risk="low"),
        ]
        agg = compute_aggregates(scores, results)
        # Clean excludes p1
        assert agg.overall_weighted_clean == 3.0

    def test_empty_scores(self):
        agg = compute_aggregates([], [])
        assert agg.overall_weighted == 0.0


# ---------------------------------------------------------------------------
# /evaluate skill helpers: extract, stream, finalize
# ---------------------------------------------------------------------------


class TestExtractEvalData:
    def test_extracts_prompts_from_run_result(self, tmp_path):
        from porchbench.evaluator import extract_eval_data
        from porchbench.schemas import (
            RunMetadata,
            RunResult,
            RunSummary,
            SuiteReference,
            ModelInfo,
        )

        result = RunResult(
            run=RunMetadata(
                id="test-run-123",
                suite=SuiteReference(name="Test", version="1.0", file="suites/test.yaml", sha256="abc"),
                model=ModelInfo(name="test-model"),
            ),
            results=[
                _make_prompt_result(prompt_id="p1", category="coding", difficulty="easy"),
                _make_prompt_result(prompt_id="p2", category="reasoning", difficulty="hard"),
            ],
            summary=RunSummary(total_prompts=2, completed=2, failed=0, total_duration_s=10.0),
        )

        path = tmp_path / "result.json"
        path.write_text(result.model_dump_json(), encoding="utf-8")

        data = extract_eval_data(path)
        assert data.header.run_id == "test-run-123"
        assert data.header.model_name == "test-model"
        assert data.header.total_prompts == 2
        assert data.header.categories == {"coding": 1, "reasoning": 1}
        assert data.header.difficulties == {"easy": 1, "hard": 1}
        assert len(data.prompts) == 2
        assert data.prompts[0].prompt_id == "p1"
        assert "Write hello world" in data.prompts[0].prompt_text
        assert data.prompts[0].response_text == "print('hello world')"

    def test_counts_truncated(self, tmp_path):
        from porchbench.evaluator import extract_eval_data
        from porchbench.schemas import (
            RunMetadata, RunResult, RunSummary, SuiteReference, ModelInfo,
            ResponseData, ResponseMessage,
        )

        result = RunResult(
            run=RunMetadata(
                id="trunc-test",
                suite=SuiteReference(name="T", version="1.0", file="s.yaml", sha256="x"),
                model=ModelInfo(name="m"),
            ),
            results=[
                _make_prompt_result(
                    prompt_id="p1",
                    response=ResponseData(
                        message=ResponseMessage(content="partial..."),
                        done_reason="length",
                    ),
                ),
            ],
            summary=RunSummary(total_prompts=1, completed=1, failed=0, total_duration_s=1.0),
        )

        path = tmp_path / "result.json"
        path.write_text(result.model_dump_json(), encoding="utf-8")

        data = extract_eval_data(path)
        assert data.header.truncated_count == 1
        assert data.prompts[0].done_reason == "length"


class TestAppendAndLoadScores:
    def test_round_trip(self, tmp_path):
        from porchbench.evaluator import append_score, load_scores

        scores_path = tmp_path / "scores.jsonl"
        s1 = PromptScore(
            prompt_id="p1",
            criteria={"c": CriterionScore(score=4, rationale="good")},
            weighted_score=4.0,
            summary="solid",
        )
        s2 = PromptScore(
            prompt_id="p2",
            criteria={"c": CriterionScore(score=2, rationale="weak")},
            weighted_score=2.0,
            summary="poor",
        )

        append_score(s1, scores_path)
        append_score(s2, scores_path)

        loaded = load_scores(scores_path)
        assert len(loaded) == 2
        assert loaded[0].prompt_id == "p1"
        assert loaded[0].weighted_score == 4.0
        assert loaded[0].criteria["c"].rationale == "good"
        assert loaded[1].prompt_id == "p2"

    def test_creates_parent_dirs(self, tmp_path):
        from porchbench.evaluator import append_score

        scores_path = tmp_path / "deep" / "nested" / "scores.jsonl"
        s = PromptScore(prompt_id="p1", criteria={}, weighted_score=3.0, summary="ok")
        append_score(s, scores_path)
        assert scores_path.exists()


class TestBuildScorecardFromScores:
    def test_produces_valid_scorecard(self, tmp_path):
        from porchbench.evaluator import append_score, build_scorecard_from_scores
        from porchbench.schemas import (
            RunMetadata, RunResult, RunSummary, SuiteReference, ModelInfo,
        )

        # Write a result file
        result = RunResult(
            run=RunMetadata(
                id="finalize-test",
                suite=SuiteReference(name="T", version="1.0", file="s.yaml", sha256="x"),
                model=ModelInfo(name="m"),
            ),
            results=[
                _make_prompt_result(prompt_id="p1", category="coding", difficulty="easy"),
                _make_prompt_result(prompt_id="p2", category="reasoning", difficulty="hard"),
            ],
            summary=RunSummary(total_prompts=2, completed=2, failed=0, total_duration_s=5.0),
        )
        result_path = tmp_path / "result.json"
        result_path.write_text(result.model_dump_json(), encoding="utf-8")

        # Write scores
        scores_path = tmp_path / "scores.jsonl"
        append_score(
            PromptScore(prompt_id="p1", criteria={}, weighted_score=4.0, summary="good"),
            scores_path,
        )
        append_score(
            PromptScore(prompt_id="p2", criteria={}, weighted_score=2.0, summary="weak"),
            scores_path,
        )

        # Finalize
        out_dir = tmp_path / "scorecards"
        sc_path = build_scorecard_from_scores(
            scores_path, result_path,
            evaluator="test-eval", rubric_label="test-rubric",
            output_dir=out_dir,
        )

        assert sc_path.exists()
        sc = json.loads(sc_path.read_text(encoding="utf-8"))
        assert sc["evaluation"]["run_id"] == "finalize-test"
        assert sc["evaluation"]["evaluator"] == "test-eval"
        assert len(sc["scores"]) == 2
        assert sc["aggregate"]["overall_weighted"] == 3.0
        assert sc["aggregate"]["by_category"]["coding"] == 4.0
        assert sc["aggregate"]["by_category"]["reasoning"] == 2.0
