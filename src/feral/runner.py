"""Benchmark run orchestrator.

Iterates suite prompts against one or more models, calls the Ollama client,
collects results, computes metrics, and writes run result JSON files.
Errors on individual prompts are captured without aborting the run.
"""

from __future__ import annotations

import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from feral.assets import feral_version
from feral.backend import InferenceBackend, OllamaBackend
from feral.schemas import (
    Message,
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
    SystemInfo,
    Suite,
    SuiteReference,
    ToolUseMetricsData,
    compute_derived_metrics,
)
from feral.suite import resolve_messages, resolve_options


def find_completed_prompt_ids(
    suite_name: str,
    model: str,
    results_dir: Path,
) -> set[str]:
    """Scan results dir for prior runs of this suite+model and return completed prompt IDs."""
    suite_slug = suite_name.lower().replace(" ", "-")
    model_slug = model.replace(":", "-").replace("/", "-")
    pattern = f"*_{suite_slug}_{model_slug}*.json"

    completed: set[str] = set()
    for path in results_dir.glob(pattern):
        try:
            data = path.read_text(encoding="utf-8")
            run = RunResult.model_validate_json(data)
            for r in run.results:
                # Only count as completed if it didn't error
                if r.response.done_reason is None or not str(r.response.done_reason).startswith("error:"):
                    completed.add(r.prompt_id)
        except Exception:
            continue  # skip corrupt/incompatible files

    return completed

console = Console()


async def run_prompt(
    messages: list[Message],
    model: str,
    options: ModelOptions,
    backend: InferenceBackend,
    profile_vram: bool = False,
) -> tuple[ResponseData, PromptMetrics]:
    """Run a single prompt against a model. Returns response data and raw metrics."""
    peak_vram: int | None = None

    if profile_vram and isinstance(backend, OllamaBackend):
        from feral.profiler import measure_peak_vram

        async with measure_peak_vram(backend, model) as sample:
            result = await backend.chat(
                messages=[{"role": m.role, "content": m.content} for m in messages],
                model=model,
                options=options,
            )
        if sample.peak_bytes > 0:
            peak_vram = sample.peak_bytes
    else:
        result = await backend.chat(
            messages=[{"role": m.role, "content": m.content} for m in messages],
            model=model,
            options=options,
        )

    response_data = ResponseData(
        message=ResponseMessage(
            role=result.role,
            content=result.content,
        ),
        done_reason=result.done_reason,
    )

    metrics = compute_derived_metrics(result.metrics)
    if peak_vram is not None:
        metrics = metrics.model_copy(update={"peak_vram_bytes": peak_vram})

    return response_data, metrics


async def _run_tool_use_prompt(
    prompt,
    model: str,
    options: ModelOptions,
    messages: list[Message],
    suite_dir: Path | None,
    backend: InferenceBackend,
) -> PromptResult:
    """Dispatch a tool-use prompt through the sandbox harness and package the result."""
    from feral.tool_runner import run_tool_use_prompt

    result = await run_tool_use_prompt(
        prompt=prompt,
        model=model,
        options=options,
        messages=messages,
        suite_dir=suite_dir,
        backend=backend,
    )

    harness_result = result["harness_result"]

    # Extract the final assistant message from transcript as the response
    final_content = ""
    for msg in reversed(harness_result.transcript):
        if msg.get("role") == "assistant" and msg.get("content"):
            final_content = msg["content"]
            break

    return PromptResult(
        prompt_id=prompt.id,
        category=prompt.category,
        difficulty=prompt.difficulty,
        tags=prompt.tags,
        contamination_risk=prompt.contamination_risk,
        expected_answer=prompt.expected_answer,
        options_used=options,
        request=RequestData(messages=messages),
        response=ResponseData(
            message=ResponseMessage(content=final_content),
            done_reason=harness_result.stopped_reason,
        ),
        metrics=PromptMetrics(),
        validation_passed=result["validation_passed"],
        validation_reason=result["validation_reason"],
        stopped_reason=harness_result.stopped_reason,
        tool_use_metrics=ToolUseMetricsData(
            total_tool_calls=harness_result.tool_use_metrics.total_tool_calls,
            tool_call_breakdown=harness_result.tool_use_metrics.tool_call_breakdown,
            errors_encountered=harness_result.tool_use_metrics.errors_encountered,
            self_corrections=harness_result.tool_use_metrics.self_corrections,
            conversation_turns=harness_result.tool_use_metrics.conversation_turns,
        ),
    )


async def run_suite(
    suite: Suite,
    suite_ref: SuiteReference,
    model: str,
    backend: InferenceBackend,
    prompt_ids: list[str] | None = None,
    output_dir: str | Path = "results",
    on_prompt_complete: Callable[[str, bool], None] | None = None,
    suite_dir: Path | None = None,
    repeat_index: int | None = None,
    total_repeats: int | None = None,
    resume: bool = False,
    profile_vram: bool = False,
) -> RunResult:
    """Run a full suite against a single model.

    Args:
        suite: Validated suite definition.
        suite_ref: Suite reference with file path and hash.
        model: Model name (e.g. "qwen2.5-coder:7b").
        backend: Inference backend to use.
        prompt_ids: Optional filter — only run these prompt IDs.
        output_dir: Directory for writing result JSON.
        on_prompt_complete: Optional callback(prompt_id, success) for progress reporting.
        suite_dir: Directory containing the suite YAML (for resolving fixture paths).
        repeat_index: 1-based repeat number (None for single runs).
        total_repeats: Total number of repeats planned (None for single runs).

    Returns:
        The completed RunResult (also written to disk).
    """
    # Gather model and system metadata
    model_info = await _get_model_info_safe(model, backend)
    system_info = await _get_system_info(backend)

    run_meta = RunMetadata(
        suite=suite_ref,
        model=model_info,
        system=system_info,
        repeat_index=repeat_index,
        total_repeats=total_repeats,
        feral_version=feral_version(),
    )

    # Filter prompts if specific IDs requested
    prompts = suite.prompts
    if prompt_ids:
        id_set = set(prompt_ids)
        prompts = [p for p in prompts if p.id in id_set]
        missing = id_set - {p.id for p in prompts}
        if missing:
            console.print(f"[yellow]Warning: prompt IDs not found in suite: {missing}[/yellow]")

    # Resume: skip already-completed prompts
    if resume:
        already_done = find_completed_prompt_ids(suite_ref.name, model, Path(output_dir))
        before = len(prompts)
        prompts = [p for p in prompts if p.id not in already_done]
        skipped = before - len(prompts)
        if skipped:
            console.print(f"[dim]Resuming: skipping {skipped} already-completed prompts[/dim]")

    # Run each prompt
    results: list[PromptResult] = []
    failed_count = 0
    run_start = time.monotonic()

    for prompt in prompts:
        options = resolve_options(suite.defaults.options, prompt)
        messages = resolve_messages(prompt)

        try:
            if prompt.mode == "tool-use":
                result = await _run_tool_use_prompt(
                    prompt, model, options, messages, suite_dir, backend,
                )
            else:
                response_data, metrics = await run_prompt(
                    messages, model, options, backend=backend, profile_vram=profile_vram,
                )
                result = PromptResult(
                    prompt_id=prompt.id,
                    category=prompt.category,
                    difficulty=prompt.difficulty,
                    tags=prompt.tags,
                    contamination_risk=prompt.contamination_risk,
        expected_answer=prompt.expected_answer,
                    options_used=options,
                    request=RequestData(messages=messages),
                    response=response_data,
                    metrics=metrics,
                )

            results.append(result)
            if on_prompt_complete:
                on_prompt_complete(prompt.id, True, result)

        except Exception as exc:
            failed_count += 1
            console.print(f"[red]Error on prompt '{prompt.id}': {exc}[/red]")

            # Record the failure with empty response and metrics
            results.append(PromptResult(
                prompt_id=prompt.id,
                category=prompt.category,
                difficulty=prompt.difficulty,
                tags=prompt.tags,
                contamination_risk=prompt.contamination_risk,
        expected_answer=prompt.expected_answer,
                options_used=options,
                request=RequestData(messages=messages),
                response=ResponseData(
                    message=ResponseMessage(content=""),
                    done_reason=f"error: {exc}",
                ),
                metrics=PromptMetrics(),
            ))
            if on_prompt_complete:
                on_prompt_complete(prompt.id, False)

    run_elapsed = time.monotonic() - run_start

    # Compute summary
    tps_values = [
        r.metrics.tokens_per_second
        for r in results
        if r.metrics.tokens_per_second is not None
    ]
    summary = RunSummary(
        total_prompts=len(prompts),
        completed=len(prompts) - failed_count,
        failed=failed_count,
        total_duration_s=round(run_elapsed, 2),
        avg_tokens_per_second=round(sum(tps_values) / len(tps_values), 2) if tps_values else None,
    )

    run_result = RunResult(run=run_meta, results=results, summary=summary)

    # Write to disk
    output_path = _write_result(run_result, output_dir)
    console.print(f"[green]Results written to {output_path}[/green]")

    return run_result


def _write_result(run_result: RunResult, output_dir: str | Path) -> Path:
    """Serialize run result to a timestamped JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = run_result.run.timestamp.strftime("%Y-%m-%dT%H-%M-%S")
    suite_slug = run_result.run.suite.name.lower().replace(" ", "-")
    model_slug = run_result.run.model.name.replace(":", "-").replace("/", "-")
    repeat_suffix = f"_repeat-{run_result.run.repeat_index}" if run_result.run.repeat_index else ""
    filename = f"{ts}_{suite_slug}_{model_slug}{repeat_suffix}.json"

    path = output_dir / filename
    path.write_text(
        run_result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return path


async def _get_model_info_safe(model: str, backend: InferenceBackend) -> ModelInfo:
    """Fetch model info, falling back to just the name on error."""
    try:
        return await backend.get_model_info(model)
    except Exception as exc:
        console.print(f"[yellow]Warning: could not fetch model details: {exc}[/yellow]")
        return ModelInfo(name=model)


async def _get_system_info(backend: InferenceBackend) -> SystemInfo:
    """Gather system metadata for the run result."""
    healthy, label = await backend.get_server_health()
    kv_cache_type = os.environ.get("OLLAMA_KV_CACHE_TYPE")
    return SystemInfo(
        ollama_version=label,
        os=f"{platform.system()} {platform.release()}",
        kv_cache_type=kv_cache_type,
    )
