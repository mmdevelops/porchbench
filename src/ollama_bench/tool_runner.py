"""Tool-use runner: orchestrates sandbox + harness for tool-use prompts.

Bridges the gap between the runner (which iterates suites) and the
sandbox/harness layers. Handles fixture loading, harness execution,
outcome validation, and result packaging.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ollama_bench.harness import Harness, HarnessResult
from ollama_bench.sandbox import SubprocessSandbox, SandboxConfig
from ollama_bench.sandbox.base import FileContent
from ollama_bench.sandbox.validator_dispatch import build_validator, _ResponseContainsValidator
from ollama_bench.schemas import (
    Message,
    ModelOptions,
    Prompt,
)


async def run_tool_use_prompt(
    prompt: Prompt,
    model: str,
    options: ModelOptions,
    messages: list[Message],
    suite_dir: Path | None = None,
    host: str | None = None,
) -> dict[str, Any]:
    """Run a single tool-use prompt through the sandbox and harness.

    Returns a dict with:
        harness_result: HarnessResult from the agent loop
        validation_passed: bool | None
        validation_reason: str
        transcript: list[dict] (the full conversation)
    """
    sandbox_config_raw = prompt.sandbox or {}
    config = SandboxConfig(
        timeout_s=sandbox_config_raw.get("timeout_s", 30),
        memory_limit_mb=sandbox_config_raw.get("memory_limit_mb", 256),
        network_enabled=sandbox_config_raw.get("network_enabled", False),
    )

    sandbox = SubprocessSandbox()
    await sandbox.create(config)

    try:
        await _load_fixtures(sandbox, prompt, suite_dir)

        harness = Harness(model=model, sandbox=sandbox, host=host)

        harness_result = await harness.run(
            messages=[{"role": m.role, "content": m.content} for m in messages],
            options=options,
            max_tool_calls=prompt.max_tool_calls or 10,
        )

        validation_passed, validation_reason = await _validate_outcome(
            prompt, sandbox, harness_result
        )

        return {
            "harness_result": harness_result,
            "validation_passed": validation_passed,
            "validation_reason": validation_reason,
        }

    finally:
        await sandbox.destroy()


async def _load_fixtures(
    sandbox: SubprocessSandbox,
    prompt: Prompt,
    suite_dir: Path | None,
) -> None:
    """Load setup_files into the sandbox from the fixtures directory."""
    if not prompt.setup_files:
        return

    files = []
    for spec in prompt.setup_files:
        target_path = spec["path"]
        source = spec.get("source", "")

        if source and suite_dir:
            source_path = suite_dir / source
            if source_path.exists():
                content = source_path.read_text(encoding="utf-8")
                files.append(FileContent(path=target_path, content=content))
            else:
                raise FileNotFoundError(
                    f"Fixture not found: {source_path} (referenced by prompt {prompt.id})"
                )
        elif "content" in spec:
            files.append(FileContent(path=target_path, content=spec["content"]))

    if files:
        await sandbox.write_files(files)


async def _validate_outcome(
    prompt: Prompt,
    sandbox: SubprocessSandbox,
    harness_result: HarnessResult,
) -> tuple[bool | None, str]:
    """Run the expected_outcome validator against the sandbox state."""
    if not prompt.expected_outcome:
        return None, "No expected outcome defined"

    spec = prompt.expected_outcome
    if isinstance(spec, dict):
        validator = build_validator(spec)
    else:
        return None, "Invalid expected_outcome format"

    if validator is None:
        return None, "No validator configured"

    if isinstance(validator, _ResponseContainsValidator):
        last_assistant = ""
        for msg in reversed(harness_result.transcript):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_assistant = msg["content"]
                break
        return validator.check_response(last_assistant)

    return await validator.validate(sandbox)
