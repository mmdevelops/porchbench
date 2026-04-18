"""Tests for runner dispatch: tool-use prompt routing, result packaging, incremental discovery."""

from unittest.mock import AsyncMock, patch

import pytest

from porchbench.harness.harness import HarnessResult, Outcome, ToolUseMetrics
from porchbench.runner import _run_tool_use_prompt, find_completed_prompt_ids, result_path_for
from porchbench.schemas import (
    Message,
    ModelInfo,
    ModelOptions,
    Prompt,
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


def _make_tool_use_prompt(**overrides) -> Prompt:
    defaults = dict(
        id="t1-read",
        category="tool-use",
        difficulty="easy",
        mode="tool-use",
        max_tool_calls=5,
        messages=[Message(role="user", content="Read data.txt")],
    )
    defaults.update(overrides)
    return Prompt(**defaults)


def _make_harness_result(**overrides) -> HarnessResult:
    defaults = dict(
        transcript=[
            {"role": "user", "content": "Read data.txt"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "read_file", "arguments": {"path": "data.txt"}}}
            ]},
            {"role": "tool", "content": "hello world"},
            {"role": "assistant", "content": "The file contains: hello world"},
        ],
        outcome=Outcome(),
        tool_use_metrics=ToolUseMetrics(
            total_tool_calls=1,
            tool_call_breakdown={"read_file": 1},
            conversation_turns=2,
        ),
        stopped_reason="done",
    )
    defaults.update(overrides)
    return HarnessResult(**defaults)


class TestToolUseDispatch:
    @pytest.mark.asyncio
    async def test_packages_harness_result_into_prompt_result(self):
        """_run_tool_use_prompt converts harness output to PromptResult with tool-use fields."""
        prompt = _make_tool_use_prompt()
        harness_result = _make_harness_result()
        messages = [Message(role="user", content="Read data.txt")]

        mock_return = {
            "harness_result": harness_result,
            "validation_passed": True,
            "validation_reason": "File content matches",
        }

        with patch(
            "porchbench.tool_runner.run_tool_use_prompt",
            new_callable=AsyncMock,
            return_value=mock_return,
        ):
            result = await _run_tool_use_prompt(
                prompt, "test-model:7b", ModelOptions(), messages, None, None,
            )

        assert result.prompt_id == "t1-read"
        assert result.category == "tool-use"
        assert result.validation_passed is True
        assert result.validation_reason == "File content matches"
        assert result.stopped_reason == "done"
        assert result.tool_use_metrics is not None
        assert result.tool_use_metrics.total_tool_calls == 1
        assert result.tool_use_metrics.tool_call_breakdown == {"read_file": 1}

    @pytest.mark.asyncio
    async def test_extracts_final_assistant_message(self):
        """Response content is taken from the last assistant message in transcript."""
        prompt = _make_tool_use_prompt()
        harness_result = _make_harness_result()
        messages = [Message(role="user", content="Read data.txt")]

        mock_return = {
            "harness_result": harness_result,
            "validation_passed": None,
            "validation_reason": "No expected outcome defined",
        }

        with patch(
            "porchbench.tool_runner.run_tool_use_prompt",
            new_callable=AsyncMock,
            return_value=mock_return,
        ):
            result = await _run_tool_use_prompt(
                prompt, "test-model:7b", ModelOptions(), messages, None, None,
            )

        assert result.response.message.content == "The file contains: hello world"

    @pytest.mark.asyncio
    async def test_handles_validation_failure(self):
        prompt = _make_tool_use_prompt()
        harness_result = _make_harness_result(stopped_reason="max_tool_calls")
        messages = [Message(role="user", content="Read data.txt")]

        mock_return = {
            "harness_result": harness_result,
            "validation_passed": False,
            "validation_reason": "Expected sorted output, got unsorted",
        }

        with patch(
            "porchbench.tool_runner.run_tool_use_prompt",
            new_callable=AsyncMock,
            return_value=mock_return,
        ):
            result = await _run_tool_use_prompt(
                prompt, "test-model:7b", ModelOptions(), messages, None, None,
            )

        assert result.validation_passed is False
        assert result.stopped_reason == "max_tool_calls"
        assert result.tool_use_metrics.conversation_turns == 2

    @pytest.mark.asyncio
    async def test_empty_transcript_yields_empty_content(self):
        """When transcript has no assistant messages, response content is empty."""
        prompt = _make_tool_use_prompt()
        harness_result = _make_harness_result(
            transcript=[{"role": "user", "content": "Read data.txt"}],
        )
        messages = [Message(role="user", content="Read data.txt")]

        mock_return = {
            "harness_result": harness_result,
            "validation_passed": None,
            "validation_reason": "No expected outcome defined",
        }

        with patch(
            "porchbench.tool_runner.run_tool_use_prompt",
            new_callable=AsyncMock,
            return_value=mock_return,
        ):
            result = await _run_tool_use_prompt(
                prompt, "test-model:7b", ModelOptions(), messages, None, None,
            )

        assert result.response.message.content == ""


# ---------------------------------------------------------------------------
# Incremental discovery
# ---------------------------------------------------------------------------


def _write_run_result(tmp_path, suite_name, model_name, prompt_ids, error_ids=None):
    """Write a minimal RunResult JSON to tmp_path for testing resume logic."""
    error_ids = error_ids or set()
    results = []
    for pid in prompt_ids:
        if pid in error_ids:
            done_reason = "error: timeout"
        else:
            done_reason = "stop"
        results.append(PromptResult(
            prompt_id=pid,
            category="coding",
            difficulty="easy",
            options_used=ModelOptions(),
            request=RequestData(messages=[Message(role="user", content="test")]),
            response=ResponseData(
                message=ResponseMessage(content="ok"),
                done_reason=done_reason,
            ),
            metrics=PromptMetrics(),
        ))

    suite_slug = suite_name.lower().replace(" ", "-")
    model_slug = model_name.replace(":", "-").replace("/", "-")

    run_result = RunResult(
        run=RunMetadata(
            suite=SuiteReference(name=suite_name, version="1.0", file="test.yaml", sha256="abc"),
            model=ModelInfo(name=model_name),
        ),
        results=results,
        summary=RunSummary(
            total_prompts=len(prompt_ids), completed=len(prompt_ids),
            failed=len(error_ids), total_duration_s=1.0,
        ),
    )

    path = tmp_path / f"2026-01-01T00-00-00_{suite_slug}_{model_slug}.json"
    path.write_text(run_result.model_dump_json(), encoding="utf-8")
    return path


class TestIncrementalDiscovery:
    def test_finds_completed_prompts(self, tmp_path):
        _write_run_result(tmp_path, "Test Suite", "model:7b", ["p1", "p2", "p3"])
        completed = find_completed_prompt_ids("Test Suite", "model:7b", tmp_path)
        assert completed == {"p1", "p2", "p3"}

    def test_excludes_errored_prompts(self, tmp_path):
        _write_run_result(tmp_path, "Test Suite", "model:7b", ["p1", "p2"], error_ids={"p2"})
        completed = find_completed_prompt_ids("Test Suite", "model:7b", tmp_path)
        assert completed == {"p1"}

    def test_no_match_returns_empty(self, tmp_path):
        _write_run_result(tmp_path, "Other Suite", "model:7b", ["p1"])
        completed = find_completed_prompt_ids("Test Suite", "model:7b", tmp_path)
        assert completed == set()

    def test_merges_across_multiple_files(self, tmp_path):
        _write_run_result(tmp_path, "Test Suite", "model:7b", ["p1", "p2"])
        # Write a second file with different prompts
        run2 = RunResult(
            run=RunMetadata(
                suite=SuiteReference(name="Test Suite", version="1.0", file="test.yaml", sha256="abc"),
                model=ModelInfo(name="model:7b"),
            ),
            results=[PromptResult(
                prompt_id="p3", category="coding", difficulty="easy",
                options_used=ModelOptions(),
                request=RequestData(messages=[Message(role="user", content="test")]),
                response=ResponseData(message=ResponseMessage(content="ok"), done_reason="stop"),
                metrics=PromptMetrics(),
            )],
            summary=RunSummary(total_prompts=1, completed=1, failed=0, total_duration_s=1.0),
        )
        path2 = tmp_path / "2026-01-02T00-00-00_test-suite_model-7b.json"
        path2.write_text(run2.model_dump_json(), encoding="utf-8")

        completed = find_completed_prompt_ids("Test Suite", "model:7b", tmp_path)
        assert completed == {"p1", "p2", "p3"}


class TestResultPathFor:
    """result_path_for must match the filename _write_result produces — overnight
    relies on this to locate result files post-inference for batch evaluation."""

    def _make_run_result(
        self, model: str = "m", suite_name: str = "T", repeat: int | None = None,
    ) -> RunResult:
        from datetime import UTC, datetime
        return RunResult(
            run=RunMetadata(
                timestamp=datetime(2026, 4, 18, 12, 0, 0, tzinfo=UTC),
                suite=SuiteReference(name=suite_name, version="1.0", file="s.yaml", sha256="x"),
                model=ModelInfo(name=model),
                repeat_index=repeat,
            ),
            results=[],
            summary=RunSummary(total_prompts=0, completed=0, failed=0, total_duration_s=0.0),
        )

    def test_deterministic_path(self, tmp_path):
        from porchbench.runner import result_path_for
        rr = self._make_run_result(model="qwen2.5:3b", suite_name="Coding Basics")
        path = result_path_for(rr, tmp_path)
        assert path == tmp_path / "2026-04-18T12-00-00_coding-basics_qwen2.5-3b.json"

    def test_includes_repeat_suffix(self, tmp_path):
        from porchbench.runner import result_path_for
        rr = self._make_run_result(model="qwen2.5:3b", suite_name="Coding Basics", repeat=2)
        path = result_path_for(rr, tmp_path)
        assert path.name.endswith("_repeat-2.json")

    def test_round_trips_with_writer(self, tmp_path):
        """The path computed independently must equal the path _write_result wrote to.
        This invariant is what lets overnight locate files without a return-value round-trip."""
        from porchbench.runner import _write_result, result_path_for
        rr = self._make_run_result(model="qwen2.5:3b", suite_name="Coding Basics")
        written = _write_result(rr, tmp_path)
        computed = result_path_for(rr, tmp_path)
        assert written == computed
        assert written.exists()
