"""Agent loop harness: model <-> tools, to completion.

The harness sits between the runner (benchmark concerns) and the sandbox
(execution concerns). It has exactly one job: given a model, a set of
tools, and an initial prompt, run an agent loop to completion and return
the transcript.

The harness doesn't know it's being benchmarked. It just runs a task.
The runner scores the result. The sandbox executes code. No layer reaches
into another's responsibility.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from porchbench.backend import InferenceBackend
from porchbench.sandbox.base import (
    ExecutionRequest,
    Sandbox,
)
from porchbench.schemas import ModelOptions, PromptMetrics, compute_derived_metrics


_TOOL_CALL_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$")
_JSON_OPENER_RE = re.compile(r"[\{\[]")


def _has_tool_call_shape(obj: Any) -> bool:
    """Tool-call shape predicate: ``{"name": str, "arguments": dict}``."""
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("name"), str)
        and isinstance(obj.get("arguments"), dict)
    )


def looks_like_tool_call_json(content: str) -> bool:
    """True when assistant text content looks like an Ollama tool call emitted as text.

    Detects the protocol-adherence regression where a model knows it should call
    a tool but emits the call as JSON in ``content`` instead of routing through
    Ollama's structured ``tool_calls`` field. Recognised shapes:

    - bare object: ``{"name": ..., "arguments": ...}``
    - bare array: ``[{...}, {...}]``
    - concatenated objects: ``{...}\\n{...}`` (common qwen2.5-coder emission)
    - any of the above wrapped in code fences or surrounding prose

    Scans with ``JSONDecoder.raw_decode`` so concatenated and prose-embedded
    objects are peeled one-at-a-time rather than failing whole-string parse.
    Empty list ``[]`` deliberately does NOT register — relies on ``any([])``
    being False (NOT ``all``), pinned by tests.

    Used by the harness to populate ToolUseMetrics.tool_calls_via_text so users
    picking models can distinguish "Ollama tool-call regression" from "model
    too weak to drive tools."
    """
    if not content or not content.strip():
        return False

    stripped = _TOOL_CALL_FENCE_RE.sub("", content.strip()).strip()
    if not stripped:
        return False

    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(stripped):
        opener = _JSON_OPENER_RE.search(stripped, pos)
        if opener is None:
            return False
        start = opener.start()
        try:
            obj, end_offset = decoder.raw_decode(stripped[start:])
        except (json.JSONDecodeError, ValueError):
            pos = start + 1
            continue
        items = obj if isinstance(obj, list) else [obj]
        if any(_has_tool_call_shape(item) for item in items):
            return True
        pos = start + end_offset
    return False


@dataclass
class ToolUseMetrics:
    """Counts and breakdown of tool usage during a harness run."""

    total_tool_calls: int = 0
    tool_call_breakdown: dict[str, int] = field(default_factory=dict)
    errors_encountered: int = 0
    self_corrections: int = 0
    conversation_turns: int = 0
    tool_calls_via_text: int = 0


@dataclass
class Outcome:
    """Final state of the sandbox after the harness run."""

    files_produced: dict[str, dict[str, Any]] = field(default_factory=dict)
    exit_code: int | None = None


@dataclass
class HarnessResult:
    """Complete result of a harness run. The runner wraps this in its schema."""

    transcript: list[dict[str, Any]]
    outcome: Outcome
    tool_use_metrics: ToolUseMetrics
    stopped_reason: str  # "done" | "max_tool_calls" | "max_turns" | "error"
    aggregated_metrics: PromptMetrics | None = None  # summed across chat() turns


def _accumulate_metrics(acc: PromptMetrics, turn: PromptMetrics) -> PromptMetrics:
    """Sum raw Ollama timing fields across one harness turn into the accumulator.

    The harness makes N chat() calls per prompt (one per turn). Each call returns
    its own per-turn metrics; the runner needs a single per-prompt PromptMetrics.
    Raw token/duration fields sum naturally; tokens_per_second is derived from
    the totals afterwards. peak_vram_bytes takes the max because it's a
    high-water mark, not a flow rate.
    """
    return PromptMetrics(
        prompt_eval_count=(acc.prompt_eval_count or 0) + (turn.prompt_eval_count or 0)
        if (acc.prompt_eval_count is not None or turn.prompt_eval_count is not None)
        else None,
        prompt_eval_duration=(acc.prompt_eval_duration or 0) + (turn.prompt_eval_duration or 0)
        if (acc.prompt_eval_duration is not None or turn.prompt_eval_duration is not None)
        else None,
        eval_count=(acc.eval_count or 0) + (turn.eval_count or 0)
        if (acc.eval_count is not None or turn.eval_count is not None)
        else None,
        eval_duration=(acc.eval_duration or 0) + (turn.eval_duration or 0)
        if (acc.eval_duration is not None or turn.eval_duration is not None)
        else None,
        load_duration=(acc.load_duration or 0) + (turn.load_duration or 0)
        if (acc.load_duration is not None or turn.load_duration is not None)
        else None,
        peak_vram_bytes=max(
            acc.peak_vram_bytes or 0, turn.peak_vram_bytes or 0,
        ) or None,
    )


# Default tool dispatch: maps tool names to sandbox operations
def build_default_dispatch(sandbox: Sandbox) -> dict[str, Callable]:
    """Build the standard tool dispatch table backed by a sandbox."""

    async def execute_code(language: str = "python", code: str = "") -> str:
        result = await sandbox.execute(ExecutionRequest(code=code, language=language))
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        if result.timed_out:
            output += "\n[Execution timed out]"
        return output or "(no output)"

    async def read_file(path: str = "") -> str:
        try:
            return await sandbox.read_file(path)
        except FileNotFoundError:
            return f"Error: file not found: {path}"

    async def write_file(path: str = "", content: str = "") -> str:
        from porchbench.sandbox.base import FileContent
        await sandbox.write_files([FileContent(path=path, content=content)])
        return f"File written: {path}"

    return {
        "execute_code": execute_code,
        "read_file": read_file,
        "write_file": write_file,
    }


# Standard tool definitions for Ollama's tool-use API
STANDARD_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute code and return the output (stdout/stderr).",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "enum": ["python", "bash"],
                        "description": "Programming language to execute.",
                    },
                    "code": {
                        "type": "string",
                        "description": "The code to execute.",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
]


class Harness:
    """Runs an agent loop: model <-> tools, to completion."""

    def __init__(
        self,
        model: str,
        sandbox: Sandbox,
        backend: InferenceBackend,
        dispatch: dict[str, Callable] | None = None,
        tools: list[dict] | None = None,
    ):
        self.model = model
        self.sandbox = sandbox
        self.backend = backend
        self.dispatch = dispatch or build_default_dispatch(sandbox)
        self.tools = tools or STANDARD_TOOLS

    async def run(
        self,
        messages: list[dict[str, str]],
        options: ModelOptions | None = None,
        max_tool_calls: int = 10,
        max_turns: int = 20,
    ) -> HarnessResult:
        """Run the agent loop to completion."""
        opts = options or ModelOptions()

        transcript: list[dict] = list(messages)
        metrics = ToolUseMetrics()
        agg_metrics = PromptMetrics()
        tool_call_count = 0
        turn_count = 0
        last_error = False

        while turn_count < max_turns:
            turn_count += 1
            metrics.conversation_turns = turn_count

            result = await self.backend.chat(
                messages=transcript,
                model=self.model,
                options=opts,
                tools=self.tools,
            )
            agg_metrics = _accumulate_metrics(agg_metrics, result.metrics)

            tool_calls = result.tool_calls or []

            if not tool_calls:
                if looks_like_tool_call_json(result.content):
                    metrics.tool_calls_via_text += 1
                # Model responded with text, no tool calls -> done
                transcript.append({
                    "role": "assistant",
                    "content": result.content,
                })
                return HarnessResult(
                    transcript=transcript,
                    outcome=await self._capture_outcome(),
                    tool_use_metrics=metrics,
                    stopped_reason="done",
                    aggregated_metrics=compute_derived_metrics(agg_metrics),
                )

            # Process tool calls
            transcript.append({
                "role": "assistant",
                "content": result.content,
                "tool_calls": [
                    {"function": {"name": tc.name,
                                  "arguments": tc.arguments}}
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                tool_name = tc.name
                tool_args = tc.arguments

                tool_call_count += 1
                metrics.total_tool_calls = tool_call_count
                metrics.tool_call_breakdown[tool_name] = (
                    metrics.tool_call_breakdown.get(tool_name, 0) + 1
                )

                # Dispatch
                handler = self.dispatch.get(tool_name)
                if handler is None:
                    result_text = f"Error: unknown tool '{tool_name}'"
                    metrics.errors_encountered += 1
                else:
                    try:
                        result_text = await handler(**tool_args)
                    except Exception as exc:
                        result_text = f"Error: {exc}"
                        metrics.errors_encountered += 1

                # Track self-corrections (error followed by another attempt)
                if "Error" in result_text or "Traceback" in result_text:
                    if not last_error:
                        last_error = True
                    # else: consecutive errors, not a correction yet
                elif last_error:
                    metrics.self_corrections += 1
                    last_error = False
                else:
                    last_error = False

                transcript.append({
                    "role": "tool",
                    "content": result_text,
                })

                # Circuit breaker
                if tool_call_count >= max_tool_calls:
                    return HarnessResult(
                        transcript=transcript,
                        outcome=await self._capture_outcome(),
                        tool_use_metrics=metrics,
                        stopped_reason="max_tool_calls",
                        aggregated_metrics=compute_derived_metrics(agg_metrics),
                    )

        return HarnessResult(
            transcript=transcript,
            outcome=await self._capture_outcome(),
            tool_use_metrics=metrics,
            stopped_reason="max_turns",
            aggregated_metrics=compute_derived_metrics(agg_metrics),
        )

    async def _capture_outcome(self) -> Outcome:
        """Snapshot the sandbox state after the run."""
        files: dict[str, dict] = {}
        if hasattr(self.sandbox, "workdir") and self.sandbox.workdir:
            workdir = self.sandbox.workdir
            for path in workdir.rglob("*"):
                if path.is_file() and not path.name.startswith("_exec"):
                    rel = str(path.relative_to(workdir))
                    files[rel] = {
                        "exists": True,
                        "size_bytes": path.stat().st_size,
                    }
        return Outcome(files_produced=files)
