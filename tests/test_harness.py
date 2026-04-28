"""Tests for the agent-loop harness.

Uses a scripted fake backend and a stub sandbox so the loop can be
exercised without spawning subprocesses or hitting Ollama. Covers the
four control-flow branches that carry the most real-world bug risk:
natural termination, max-tool-calls circuit breaker, self-correction
accounting, and unknown-tool dispatch.
"""

from __future__ import annotations

from typing import Any

import pytest

from porchbench.backend import ChatResult, ToolCall
from porchbench.harness.harness import Harness
from porchbench.schemas import ModelOptions, PromptMetrics


class ScriptedBackend:
    """InferenceBackend fake that returns pre-queued ChatResults in order.

    Each chat() call pops the next scripted response. An IndexError
    means the harness kept looping past where the test expected it to
    stop — surface that as a test failure, not a silent hang.
    """

    def __init__(self, responses: list[ChatResult]):
        self._responses = list(responses)
        self.call_count = 0
        self.last_tools: list[dict] | None = None

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        options: ModelOptions,
        tools: list[dict] | None = None,
    ) -> ChatResult:
        self.call_count += 1
        self.last_tools = tools
        return self._responses.pop(0)


class StubSandbox:
    """Minimal sandbox stand-in. _capture_outcome tolerates workdir=None."""

    workdir = None


def _chat_result(
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
) -> ChatResult:
    return ChatResult(
        content=content,
        role="assistant",
        done_reason="stop",
        metrics=PromptMetrics(),
        tool_calls=tool_calls,
    )


# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_harness_terminates_when_model_emits_no_tool_calls():
    """Happy path: one tool call, then text-only response -> stopped_reason='done'."""
    captured: dict[str, Any] = {}

    async def fake_tool(arg: str = "") -> str:
        captured["arg"] = arg
        return "tool output"

    backend = ScriptedBackend([
        _chat_result(
            content="Calling the tool.",
            tool_calls=[ToolCall(name="fake_tool", arguments={"arg": "hello"})],
        ),
        _chat_result(content="Tool said hello back."),
    ])

    harness = Harness(
        model="fake-model",
        sandbox=StubSandbox(),
        backend=backend,
        dispatch={"fake_tool": fake_tool},
        tools=[{"type": "function", "function": {"name": "fake_tool"}}],
    )

    result = await harness.run(messages=[{"role": "user", "content": "hi"}])

    assert result.stopped_reason == "done"
    assert backend.call_count == 2
    assert captured["arg"] == "hello"
    assert result.tool_use_metrics.total_tool_calls == 1
    assert result.tool_use_metrics.tool_call_breakdown == {"fake_tool": 1}
    assert result.tool_use_metrics.conversation_turns == 2

    roles = [m["role"] for m in result.transcript]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert result.transcript[-1]["content"] == "Tool said hello back."


def _chat_result_with_metrics(
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    *,
    prompt_eval_count: int | None = None,
    eval_count: int | None = None,
    prompt_eval_duration: int | None = None,
    eval_duration: int | None = None,
    peak_vram_bytes: int | None = None,
) -> ChatResult:
    return ChatResult(
        content=content,
        role="assistant",
        done_reason="stop",
        metrics=PromptMetrics(
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
            prompt_eval_duration=prompt_eval_duration,
            eval_duration=eval_duration,
            peak_vram_bytes=peak_vram_bytes,
        ),
        tool_calls=tool_calls,
    )


@pytest.mark.asyncio
async def test_harness_aggregates_metrics_across_turns():
    """Per-turn metrics from each chat() sum into HarnessResult.aggregated_metrics.

    Without aggregation, tool-use prompts produce all-None metrics in the
    runner's PromptResult — `compare`'s per-prompt table renders as all
    dashes, the duration estimator can't see tokens, and verbose run output
    can't show tok/s for tool-use.
    """
    async def fake_tool(arg: str = "") -> str:
        return "ok"

    backend = ScriptedBackend([
        _chat_result_with_metrics(
            content="Calling.",
            tool_calls=[ToolCall(name="fake_tool", arguments={"arg": "x"})],
            prompt_eval_count=100, eval_count=50,
            prompt_eval_duration=200_000_000, eval_duration=500_000_000,  # ns
            peak_vram_bytes=8 * 1024**3,
        ),
        _chat_result_with_metrics(
            content="Done.",
            prompt_eval_count=120, eval_count=30,
            prompt_eval_duration=180_000_000, eval_duration=300_000_000,
            peak_vram_bytes=9 * 1024**3,  # higher than first turn — should win
        ),
    ])

    harness = Harness(
        model="fake-model",
        sandbox=StubSandbox(),
        backend=backend,
        dispatch={"fake_tool": fake_tool},
        tools=[{"type": "function", "function": {"name": "fake_tool"}}],
    )

    result = await harness.run(messages=[{"role": "user", "content": "go"}])

    assert result.aggregated_metrics is not None
    agg = result.aggregated_metrics
    assert agg.prompt_eval_count == 220  # 100 + 120
    assert agg.eval_count == 80  # 50 + 30
    assert agg.prompt_eval_duration == 380_000_000
    assert agg.eval_duration == 800_000_000  # 500M + 300M ns
    # tokens_per_second = 80 tokens / 0.8s = 100
    assert agg.tokens_per_second == pytest.approx(100.0)
    # peak_vram is a high-water mark, not a sum
    assert agg.peak_vram_bytes == 9 * 1024**3


@pytest.mark.asyncio
async def test_harness_aggregates_metrics_when_no_tool_calls():
    """Single-turn run (model emits text immediately) still captures the one turn."""
    backend = ScriptedBackend([
        _chat_result_with_metrics(
            content="Direct answer.",
            prompt_eval_count=10, eval_count=5,
            prompt_eval_duration=50_000_000, eval_duration=100_000_000,
        ),
    ])

    harness = Harness(
        model="fake-model",
        sandbox=StubSandbox(),
        backend=backend,
        tools=[],
    )

    result = await harness.run(messages=[{"role": "user", "content": "hi"}])
    assert result.stopped_reason == "done"
    assert result.aggregated_metrics is not None
    assert result.aggregated_metrics.eval_count == 5
    assert result.aggregated_metrics.prompt_eval_count == 10


@pytest.mark.asyncio
async def test_harness_stops_at_max_tool_calls():
    """Circuit breaker: model keeps calling tools -> stopped_reason='max_tool_calls'."""

    async def looping_tool() -> str:
        return "keep going"

    looping_response = _chat_result(
        tool_calls=[ToolCall(name="looping_tool", arguments={})],
    )
    backend = ScriptedBackend([looping_response] * 5)

    harness = Harness(
        model="fake-model",
        sandbox=StubSandbox(),
        backend=backend,
        dispatch={"looping_tool": looping_tool},
        tools=[{"type": "function", "function": {"name": "looping_tool"}}],
    )

    result = await harness.run(
        messages=[{"role": "user", "content": "loop"}],
        max_tool_calls=3,
    )

    assert result.stopped_reason == "max_tool_calls"
    assert result.tool_use_metrics.total_tool_calls == 3


@pytest.mark.asyncio
async def test_harness_tracks_self_correction_after_error():
    """Handler raises on turn 1, succeeds on turn 2 -> errors_encountered=1, self_corrections=1.

    Note: errors_encountered counts harness-level failures (handler raise,
    unknown tool). Tool output strings that happen to start with "Error:"
    are still successful handler calls; they drive self_corrections via the
    string-marker path but do not increment errors_encountered.
    """
    call_num = 0

    async def flaky_tool() -> str:
        nonlocal call_num
        call_num += 1
        if call_num == 1:
            raise RuntimeError("first attempt failed")
        return "success"

    backend = ScriptedBackend([
        _chat_result(tool_calls=[ToolCall(name="flaky_tool", arguments={})]),
        _chat_result(tool_calls=[ToolCall(name="flaky_tool", arguments={})]),
        _chat_result(content="Got it on the second try."),
    ])

    harness = Harness(
        model="fake-model",
        sandbox=StubSandbox(),
        backend=backend,
        dispatch={"flaky_tool": flaky_tool},
        tools=[{"type": "function", "function": {"name": "flaky_tool"}}],
    )

    result = await harness.run(messages=[{"role": "user", "content": "try"}])

    assert result.stopped_reason == "done"
    assert result.tool_use_metrics.errors_encountered == 1
    assert result.tool_use_metrics.self_corrections == 1
    assert result.tool_use_metrics.total_tool_calls == 2


@pytest.mark.asyncio
async def test_harness_records_error_for_unknown_tool():
    """Unknown tool name -> error recorded, loop continues to natural end."""
    backend = ScriptedBackend([
        _chat_result(tool_calls=[ToolCall(name="does_not_exist", arguments={})]),
        _chat_result(content="Stopping."),
    ])

    harness = Harness(
        model="fake-model",
        sandbox=StubSandbox(),
        backend=backend,
        dispatch={},
        tools=[],
    )

    result = await harness.run(messages=[{"role": "user", "content": "x"}])

    assert result.stopped_reason == "done"
    assert result.tool_use_metrics.errors_encountered == 1
    tool_msg = next(m for m in result.transcript if m["role"] == "tool")
    assert "unknown tool" in tool_msg["content"]
