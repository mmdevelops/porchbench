"""Tests for evaluator backends, scoring prompt construction, and score parsing."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from porchbench.evaluator import (
    ClaudeCodeEvalBackend,
    _extract_json,
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
        mock_proc.communicate.side_effect = TimeoutError()
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
        mock_proc.communicate.return_value = ("résultat".encode(), b"")
        mock_proc.returncode = 0

        with patch("porchbench.evaluator.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await backend.generate("évaluer cette réponse")

        stdin_data = mock_proc.communicate.call_args[1]["input"]
        assert stdin_data == "évaluer cette réponse".encode()
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
# evaluate_run — empty/truncated content is scored 0, not silently dropped
# ---------------------------------------------------------------------------


class TestEmptyContentHandling:
    """Reasoning-mode models can exhaust num_predict inside <think> and emit
    zero user-facing content. Pre-fix these prompts were filtered from the
    scorecard entirely, inflating aggregate scores. Post-fix they score 0
    with a specific rationale, keeping cross-model aggregates honest."""

    @pytest.mark.asyncio
    async def test_length_truncated_empty_content_scores_zero(self):
        from porchbench.evaluator import evaluate_run
        from porchbench.schemas import RunMetadata, RunResult, RunSummary, SuiteReference, ModelInfo

        # Two prompts: one with real content, one truncated with empty content
        good = _make_prompt_result(
            prompt_id="p-good",
            response=ResponseData(
                message=ResponseMessage(content="real answer"),
                done_reason="stop",
            ),
        )
        empty = _make_prompt_result(
            prompt_id="p-empty",
            response=ResponseData(
                message=ResponseMessage(content=""),
                done_reason="length",  # exhausted num_predict
            ),
        )

        rr = RunResult(
            run=RunMetadata(
                suite=SuiteReference(name="test", version="1", file="test.yaml", sha256="x" * 64),
                model=ModelInfo(name="m", size=1, quantization="Q4", family="test", parameter_size="1B", digest="d"),
            ),
            results=[good, empty],
            summary=RunSummary(total_prompts=2, completed=2, failed=0, total_duration_s=1.0),
        )

        # Backend should only be called for the good prompt — empty is shortcut
        mock_backend = AsyncMock()
        mock_backend.generate.return_value = json.dumps({
            "criteria": {
                "correctness": {"score": 5, "rationale": "fine"},
                "clarity": {"score": 5, "rationale": "fine"},
            },
            "summary": "good",
        })

        scorecard = await evaluate_run(rr, _make_rubric(), mock_backend, evaluator_label="ollama/test")

        assert len(scorecard.scores) == 2
        good_score = next(s for s in scorecard.scores if s.prompt_id == "p-good")
        empty_score = next(s for s in scorecard.scores if s.prompt_id == "p-empty")

        assert good_score.weighted_score == pytest.approx(5.0)
        assert empty_score.weighted_score == 0.0
        assert "truncated" in empty_score.summary
        assert all(c.score == 0 for c in empty_score.criteria.values())
        # Backend called once for the good prompt; empty prompt shortcut to zero.
        assert mock_backend.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_stop_with_empty_content_also_scores_zero(self):
        """done_reason=stop + empty content = model returned nothing. Same zero
        treatment as length-truncation, but with a different rationale tag."""
        from porchbench.evaluator import evaluate_run
        from porchbench.schemas import RunMetadata, RunResult, RunSummary, SuiteReference, ModelInfo

        empty = _make_prompt_result(
            prompt_id="p-empty",
            response=ResponseData(
                message=ResponseMessage(content=""),
                done_reason="stop",
            ),
        )

        rr = RunResult(
            run=RunMetadata(
                suite=SuiteReference(name="test", version="1", file="test.yaml", sha256="x" * 64),
                model=ModelInfo(name="m", size=1, quantization="Q4", family="test", parameter_size="1B", digest="d"),
            ),
            results=[empty],
            summary=RunSummary(total_prompts=1, completed=1, failed=0, total_duration_s=1.0),
        )

        mock_backend = AsyncMock()
        scorecard = await evaluate_run(rr, _make_rubric(), mock_backend, evaluator_label="ollama/test")

        assert len(scorecard.scores) == 1
        assert scorecard.scores[0].weighted_score == 0.0
        assert "empty" in scorecard.scores[0].summary.lower()
        assert mock_backend.generate.call_count == 0

    @pytest.mark.asyncio
    async def test_errored_responses_still_excluded(self):
        """Runtime errors (done_reason starts with 'error:') are NOT scored — these
        indicate the inference itself failed, not a model producing a bad answer.
        Those stay filtered out."""
        from porchbench.evaluator import evaluate_run
        from porchbench.schemas import RunMetadata, RunResult, RunSummary, SuiteReference, ModelInfo

        errored = _make_prompt_result(
            prompt_id="p-error",
            response=ResponseData(
                message=ResponseMessage(content=""),
                done_reason="error: connection reset",
            ),
        )

        rr = RunResult(
            run=RunMetadata(
                suite=SuiteReference(name="test", version="1", file="test.yaml", sha256="x" * 64),
                model=ModelInfo(name="m", size=1, quantization="Q4", family="test", parameter_size="1B", digest="d"),
            ),
            results=[errored],
            summary=RunSummary(total_prompts=1, completed=0, failed=1, total_duration_s=1.0),
        )

        mock_backend = AsyncMock()
        scorecard = await evaluate_run(rr, _make_rubric(), mock_backend, evaluator_label="ollama/test")

        assert len(scorecard.scores) == 0
        assert mock_backend.generate.call_count == 0


# ---------------------------------------------------------------------------
# /evaluate skill helpers: extract, stream, finalize
# ---------------------------------------------------------------------------


class TestExtractEvalData:
    def test_extracts_prompts_from_run_result(self, tmp_path):
        from porchbench.evaluator import extract_eval_data
        from porchbench.schemas import (
            ModelInfo,
            RunMetadata,
            RunResult,
            RunSummary,
            SuiteReference,
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
            ModelInfo,
            ResponseData,
            ResponseMessage,
            RunMetadata,
            RunResult,
            RunSummary,
            SuiteReference,
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
            ModelInfo,
            RunMetadata,
            RunResult,
            RunSummary,
            SuiteReference,
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


# ---------------------------------------------------------------------------
# Public API additions: make_backend, evaluate_single (+ sync), suite slug,
# no-eligible path through evaluate_run (harness-shaped done_reasons).
# Driven by feedback from the agent-harness bridge integration.
# ---------------------------------------------------------------------------


class TestMakeBackend:
    def test_dispatches_ollama(self):
        from porchbench.evaluator import OllamaEvalBackend, make_backend

        backend = make_backend("ollama", model="llama3:8b")
        assert isinstance(backend, OllamaEvalBackend)
        assert backend.model == "llama3:8b"

    def test_dispatches_claude_code(self):
        from porchbench.evaluator import ClaudeCodeEvalBackend, make_backend

        backend = make_backend("claude-code", model="opus", timeout_s=60)
        assert isinstance(backend, ClaudeCodeEvalBackend)
        assert backend.model == "opus"
        assert backend.timeout_s == 60

    def test_anthropic_alias_for_api(self):
        from porchbench.evaluator import _BACKEND_FACTORIES

        assert _BACKEND_FACTORIES["api"] is _BACKEND_FACTORIES["anthropic"]

    def test_unknown_name_raises_with_valid_list(self):
        from porchbench.evaluator import make_backend

        with pytest.raises(ValueError, match="Unknown evaluator backend"):
            make_backend("openai", model="gpt-4")


class TestSuiteSlug:
    def test_slugify_lowercases_and_hyphenates(self):
        from porchbench.schemas import slugify_suite_name

        assert slugify_suite_name("Tool Use Discovery") == "tool-use-discovery"
        assert slugify_suite_name("Coding Basics") == "coding-basics"
        assert slugify_suite_name("already-lower") == "already-lower"

    def test_suite_reference_slug_property(self):
        from porchbench.schemas import SuiteReference

        ref = SuiteReference(
            name="Cross Domain Science",
            version="1.0",
            file="x.yaml",
            sha256="y" * 64,
        )
        assert ref.slug == "cross-domain-science"


class TestEvaluateSingle:
    """Public single-response entry point — bridges the agent-harness use case
    where consumers have one (prompt, response) pair in memory and want a
    PromptScore back without round-tripping through RunResult JSON."""

    @pytest.mark.asyncio
    async def test_returns_weighted_score_from_strings(self):
        from porchbench.evaluator import evaluate_single

        mock_backend = AsyncMock()
        mock_backend.generate.return_value = json.dumps({
            "criteria": {
                "correctness": {"score": 5, "rationale": "ok"},
                "clarity": {"score": 4, "rationale": "ok"},
            },
            "summary": "fine",
        })

        score = await evaluate_single(
            prompt_text="What is 2 + 2?",
            response_text="4",
            rubric=_make_rubric(),
            backend=mock_backend,
        )

        # weighted = 5*0.6 + 4*0.4 = 4.6
        assert score.weighted_score == pytest.approx(4.6, abs=0.01)
        assert score.criteria["correctness"].score == 5
        assert score.prompt_id == "single"

    @pytest.mark.asyncio
    async def test_passes_prompt_and_response_into_judge_prompt(self):
        from porchbench.evaluator import evaluate_single

        mock_backend = AsyncMock()
        mock_backend.generate.return_value = json.dumps({
            "criteria": {
                "correctness": {"score": 3, "rationale": ""},
                "clarity": {"score": 3, "rationale": ""},
            },
            "summary": "",
        })

        await evaluate_single(
            prompt_text="UNIQUE_PROMPT_MARKER",
            response_text="UNIQUE_RESPONSE_MARKER",
            rubric=_make_rubric(),
            backend=mock_backend,
            expected_answer="UNIQUE_REFERENCE_MARKER",
        )

        sent_prompt = mock_backend.generate.call_args[0][0]
        assert "UNIQUE_PROMPT_MARKER" in sent_prompt
        assert "UNIQUE_RESPONSE_MARKER" in sent_prompt
        assert "UNIQUE_REFERENCE_MARKER" in sent_prompt
        # The user-prompt section is tagged so the judge can distinguish
        # original prompt from response — match the existing convention.
        assert "[user]: UNIQUE_PROMPT_MARKER" in sent_prompt

    @pytest.mark.asyncio
    async def test_custom_prompt_id_propagates(self):
        from porchbench.evaluator import evaluate_single

        mock_backend = AsyncMock()
        mock_backend.generate.return_value = json.dumps({
            "criteria": {
                "correctness": {"score": 5, "rationale": ""},
                "clarity": {"score": 5, "rationale": ""},
            },
            "summary": "",
        })

        score = await evaluate_single(
            prompt_text="q", response_text="a",
            rubric=_make_rubric(), backend=mock_backend,
            prompt_id="harness-turn-7",
        )
        assert score.prompt_id == "harness-turn-7"


class TestEvaluateSingleSync:
    def test_sync_wrapper_returns_promptscore(self):
        from porchbench.evaluator import evaluate_single_sync

        # Sync wrapper opens its own asyncio.run; back the backend with a
        # plain async function rather than AsyncMock so it survives the
        # fresh loop.
        async def fake_generate(prompt: str) -> str:
            return json.dumps({
                "criteria": {
                    "correctness": {"score": 4, "rationale": ""},
                    "clarity": {"score": 4, "rationale": ""},
                },
                "summary": "",
            })

        class _Backend:
            generate = staticmethod(fake_generate)

        score = evaluate_single_sync(
            prompt_text="q", response_text="a",
            rubric=_make_rubric(), backend=_Backend(),
        )
        assert score.weighted_score == pytest.approx(4.0, abs=0.01)


class TestNoEligibleHarnessRun:
    """Tool-use / routing-discovery results have done_reason set from the
    harness stopped_reason ("done", "max_tool_calls", etc.) — those fall
    outside evaluate_run's inference filter and produce an empty scorecard.
    Verify the warning path and the "no_eligible" status propagation."""

    @pytest.mark.asyncio
    async def test_evaluate_run_emits_empty_scorecard_for_harness_results(self, capsys):
        from porchbench.evaluator import evaluate_run
        from porchbench.schemas import (
            ModelInfo, RunMetadata, RunResult, RunSummary, SuiteReference,
        )

        # All results have harness-style done_reasons; none are eligible.
        harness_done = _make_prompt_result(
            prompt_id="t1",
            response=ResponseData(
                message=ResponseMessage(content="ran tools"),
                done_reason="done",
            ),
        )
        harness_max_calls = _make_prompt_result(
            prompt_id="t2",
            response=ResponseData(
                message=ResponseMessage(content="hit cap"),
                done_reason="max_tool_calls",
            ),
        )

        rr = RunResult(
            run=RunMetadata(
                suite=SuiteReference(name="Tool Use Discovery", version="1", file="x.yaml", sha256="x" * 64),
                model=ModelInfo(name="m", size=1, quantization="Q4", family="test", parameter_size="1B", digest="d"),
            ),
            results=[harness_done, harness_max_calls],
            summary=RunSummary(total_prompts=2, completed=2, failed=0, total_duration_s=1.0),
        )

        mock_backend = AsyncMock()
        scorecard = await evaluate_run(rr, _make_rubric(), mock_backend, evaluator_label="ollama/test")

        assert scorecard.scores == []
        assert mock_backend.generate.call_count == 0
        # Warning surfaces eligibility info — harness results aren't a
        # silent 0/0 anymore.
        out = capsys.readouterr().out
        assert "No prompts eligible" in out
