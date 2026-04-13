"""Benchmark run orchestrator.

Iterates suite prompts against one or more models, calls the Ollama client,
collects results, computes metrics, and writes run result JSON files.
Errors on individual prompts are captured without aborting the run.
"""

from __future__ import annotations

import json
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

from ollama_bench import client
from ollama_bench.schemas import (
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
    compute_derived_metrics,
)
from ollama_bench.suite import resolve_messages, resolve_options

console = Console()


async def run_prompt(
    messages: list[Message],
    model: str,
    options: ModelOptions,
    host: str | None = None,
) -> tuple[ResponseData, PromptMetrics]:
    """Run a single prompt against a model. Returns response data and raw metrics."""
    response = await client.chat(messages, model, options, host=host)

    response_data = ResponseData(
        message=ResponseMessage(
            role=response.message.role or "assistant",
            content=response.message.content or "",
        ),
        done_reason=getattr(response, "done_reason", None),
    )

    raw_metrics = client.extract_metrics(response)
    metrics = compute_derived_metrics(raw_metrics)

    return response_data, metrics


async def run_suite(
    suite: Suite,
    suite_ref: SuiteReference,
    model: str,
    host: str | None = None,
    prompt_ids: list[str] | None = None,
    output_dir: str | Path = "results",
    on_prompt_complete: Callable[[str, bool], None] | None = None,
) -> RunResult:
    """Run a full suite against a single model.

    Args:
        suite: Validated suite definition.
        suite_ref: Suite reference with file path and hash.
        model: Ollama model name (e.g. "qwen2.5-coder:7b").
        host: Ollama server URL. None for localhost default.
        prompt_ids: Optional filter — only run these prompt IDs.
        output_dir: Directory for writing result JSON.
        on_prompt_complete: Optional callback(prompt_id, success) for progress reporting.

    Returns:
        The completed RunResult (also written to disk).
    """
    # Gather model and system metadata
    model_info = await _get_model_info_safe(model, host)
    system_info = await _get_system_info(host)

    run_meta = RunMetadata(
        suite=suite_ref,
        model=model_info,
        system=system_info,
    )

    # Filter prompts if specific IDs requested
    prompts = suite.prompts
    if prompt_ids:
        id_set = set(prompt_ids)
        prompts = [p for p in prompts if p.id in id_set]
        missing = id_set - {p.id for p in prompts}
        if missing:
            console.print(f"[yellow]Warning: prompt IDs not found in suite: {missing}[/yellow]")

    # Run each prompt
    results: list[PromptResult] = []
    failed_count = 0
    run_start = time.monotonic()

    for prompt in prompts:
        options = resolve_options(suite.defaults.options, prompt)
        messages = resolve_messages(prompt)

        try:
            response_data, metrics = await run_prompt(messages, model, options, host=host)

            results.append(PromptResult(
                prompt_id=prompt.id,
                category=prompt.category,
                difficulty=prompt.difficulty,
                tags=prompt.tags,
                options_used=options,
                request=RequestData(messages=messages),
                response=response_data,
                metrics=metrics,
            ))
            if on_prompt_complete:
                on_prompt_complete(prompt.id, True)

        except Exception as exc:
            failed_count += 1
            console.print(f"[red]Error on prompt '{prompt.id}': {exc}[/red]")

            # Record the failure with empty response and metrics
            results.append(PromptResult(
                prompt_id=prompt.id,
                category=prompt.category,
                difficulty=prompt.difficulty,
                tags=prompt.tags,
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
    filename = f"{ts}_{suite_slug}_{model_slug}.json"

    path = output_dir / filename
    path.write_text(
        run_result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return path


async def _get_model_info_safe(model: str, host: str | None) -> ModelInfo:
    """Fetch model info, falling back to just the name on error."""
    try:
        return await client.get_model_info(model, host)
    except Exception as exc:
        console.print(f"[yellow]Warning: could not fetch model details: {exc}[/yellow]")
        return ModelInfo(name=model)


async def _get_system_info(host: str | None) -> SystemInfo:
    """Gather system metadata for the run result."""
    ollama_version = await client.get_server_version(host)
    return SystemInfo(
        ollama_version=ollama_version,
        os=f"{platform.system()} {platform.release()}",
        # GPU detection is platform-specific and best-effort.
        # For now we capture OS; GPU info can be added via ollama.ps() or nvidia-smi.
    )
