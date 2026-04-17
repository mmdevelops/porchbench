"""Overnight orchestration: unattended multi-suite benchmark runs.

Discovers suites, classifies them (standard vs routing discovery),
builds an execution plan, runs preflight checks, and executes with
error resilience. Designed for hobbyists queuing benchmarks overnight.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.table import Table

from porchbench.backend import InferenceBackend, OllamaBackend
from porchbench.profiler import detect_gpu
from porchbench.runner import run_suite
from porchbench.routing import count_discovery_runs, run_discovery
from porchbench.schemas import RunResult, Suite, SuiteReference
from porchbench.suite import discover_suites, load_suite, make_suite_reference

console = Console()

SECONDS_PER_PROMPT_DEFAULT = 30.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class OvernightTask:
    """A unit of work in the overnight plan."""

    suite_path: Path
    suite: Suite
    suite_ref: SuiteReference
    dispatch_type: str  # 'standard' | 'discovery'
    models: list[str]
    repeats: int
    prompt_count: int
    strategy_count: int
    run_count: int  # total inference calls


@dataclass
class OvernightResult:
    """Outcome of a single task execution."""

    task: OvernightTask
    model: str
    repeat: int | None
    success: bool
    error: str | None = None
    duration_s: float = 0.0
    eval_score: float | None = None  # weighted aggregate from post-run evaluation


# ---------------------------------------------------------------------------
# Suite discovery and classification
# ---------------------------------------------------------------------------



def classify_suite(suite: Suite) -> str:
    """Return 'discovery' if suite has strategies, else 'standard'."""
    return "discovery" if suite.strategies else "standard"


# ---------------------------------------------------------------------------
# Plan building
# ---------------------------------------------------------------------------


def build_plan(
    suite_paths: list[Path],
    models: list[str],
    repeats: int,
) -> list[OvernightTask]:
    """Load suites, classify them, and build an ordered task list."""
    tasks: list[OvernightTask] = []
    for path in suite_paths:
        suite = load_suite(path)
        suite_ref = make_suite_reference(path, suite)
        dispatch = classify_suite(suite)
        n_prompts = len(suite.prompts)
        n_strategies = max(len(suite.strategies), 1)

        if dispatch == "discovery":
            run_count = n_prompts * n_strategies * len(models)
            task_repeats = 1
        else:
            run_count = n_prompts * len(models) * repeats
            task_repeats = repeats

        tasks.append(OvernightTask(
            suite_path=path,
            suite=suite,
            suite_ref=suite_ref,
            dispatch_type=dispatch,
            models=models,
            repeats=task_repeats,
            prompt_count=n_prompts,
            strategy_count=n_strategies,
            run_count=run_count,
        ))
    return tasks


def estimate_duration(
    plan: list[OvernightTask],
    seconds_per_prompt: float = SECONDS_PER_PROMPT_DEFAULT,
) -> float:
    """Estimate total runtime in seconds."""
    return sum(task.run_count * seconds_per_prompt for task in plan)


def format_estimate(seconds: float) -> str:
    """Format seconds as a human-readable duration."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"~{hours}h {minutes}m"
    return f"~{minutes}m"


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------


async def check_ollama_health(backend: InferenceBackend) -> tuple[bool, str]:
    """Check if the inference backend is reachable."""
    return await backend.get_server_health()


async def check_gpu_status(backend: InferenceBackend, model: str) -> tuple[bool, str]:
    """Check GPU detection and acceleration.

    Runs a 1-token warmup inference then checks if the model loaded into VRAM.
    VRAM check requires an OllamaBackend; other backends skip it.
    """
    # Static GPU detection
    gpu_name, vram_gb = detect_gpu()
    if not gpu_name:
        return False, "No GPU detected"

    gpu_label = gpu_name
    if vram_gb:
        gpu_label += f" ({vram_gb:.0f} GB)"

    # Warmup: 1-token inference to force model load (timeout prevents hanging)
    try:
        from porchbench.schemas import ModelOptions

        await asyncio.wait_for(
            backend.chat(
                messages=[{"role": "user", "content": "ok"}],
                model=model,
                options=ModelOptions(num_predict=1),
            ),
            timeout=120,
        )
    except asyncio.TimeoutError:
        return False, f"GPU check failed — warmup timed out after 120s"
    except Exception as exc:
        return False, f"GPU check failed — warmup error: {exc}"

    # Check VRAM usage via Ollama ps() — only available on OllamaBackend
    if isinstance(backend, OllamaBackend):
        try:
            running = await backend.list_running_models()
            for m in running:
                if m.get("size_vram") and m["size_vram"] > 0:
                    vram_used_gb = m["size_vram"] / (1024**3)
                    return True, f"{gpu_label} — {vram_used_gb:.1f} GB VRAM in use"
            return False, f"{gpu_label} detected but model not using VRAM (CPU inference)"
        except Exception:
            return True, f"{gpu_label} detected (could not verify VRAM usage)"

    return True, f"{gpu_label} detected (VRAM check not available for this backend)"


async def run_preflight(
    backend: InferenceBackend,
    models: list[str],
) -> list[tuple[str, bool, str]]:
    """Run all preflight checks. Returns list of (name, passed, message)."""
    checks: list[tuple[str, bool, str]] = []

    ok, msg = await check_ollama_health(backend)
    checks.append(("Server", ok, msg))
    if not ok:
        return checks  # no point continuing

    ok, msg = await check_gpu_status(backend, models[0])
    checks.append(("GPU acceleration", ok, msg))

    return checks


# ---------------------------------------------------------------------------
# Plan display
# ---------------------------------------------------------------------------


def print_plan(plan: list[OvernightTask], models: list[str]) -> None:
    """Print the execution plan as a Rich table."""
    table = Table(title="Overnight Plan", title_style="bold")
    table.add_column("Suite", style="bold")
    table.add_column("Type")
    table.add_column("Prompts", justify="right")
    table.add_column("Strategies", justify="right")
    table.add_column("Repeats", justify="right")
    table.add_column("Total runs", justify="right")

    for task in plan:
        repeats_str = str(task.repeats) if task.dispatch_type == "standard" else "n/a"
        strategies_str = str(task.strategy_count) if task.dispatch_type == "discovery" else "-"
        table.add_row(
            task.suite.suite.name,
            task.dispatch_type,
            str(task.prompt_count),
            strategies_str,
            repeats_str,
            str(task.run_count),
        )

    console.print(table)
    console.print(f"Models: {', '.join(models)}")

    total_runs = sum(t.run_count for t in plan)
    estimate = estimate_duration(plan)
    console.print(f"Total inference calls: [bold]{total_runs}[/bold]")
    console.print(f"Estimated duration: [bold]{format_estimate(estimate)}[/bold]")
    console.print()


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


async def execute_plan(
    plan: list[OvernightTask],
    backend: InferenceBackend,
    output_dir: Path,
    resume: bool,
    verbose: bool,
    on_task_start: Callable[[OvernightTask, str, int | None], None] | None = None,
    on_task_done: Callable[[OvernightResult], None] | None = None,
    on_run_eval: Callable | None = None,
    profile_vram: bool = False,
) -> list[OvernightResult]:
    """Execute the overnight plan with error resilience.

    When on_run_eval is provided, it is called after each successful standard
    run with the RunResult. It should be an async callable returning a float
    (the aggregate eval score) or None on failure.
    """
    results: list[OvernightResult] = []

    for task in plan:
        if task.dispatch_type == "discovery":
            # Discovery: single call with all models
            if on_task_start:
                on_task_start(task, "(all models)", None)

            start = time.monotonic()
            try:
                await run_discovery(
                    suite=task.suite,
                    suite_ref=task.suite_ref,
                    models=task.models,
                    backend=backend,
                    output_dir=output_dir,
                    suite_dir=task.suite_path.parent,
                )
                result = OvernightResult(
                    task=task, model="(all)", repeat=None,
                    success=True, duration_s=time.monotonic() - start,
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                result = OvernightResult(
                    task=task, model="(all)", repeat=None,
                    success=False, error=str(exc),
                    duration_s=time.monotonic() - start,
                )

            results.append(result)
            if on_task_done:
                on_task_done(result)

        else:
            # Standard: loop model × repeat
            for model in task.models:
                for repeat_i in range(1, task.repeats + 1):
                    if on_task_start:
                        on_task_start(task, model, repeat_i)

                    start = time.monotonic()
                    run_result = None
                    try:
                        run_result = await run_suite(
                            suite=task.suite,
                            suite_ref=task.suite_ref,
                            model=model,
                            backend=backend,
                            output_dir=output_dir,
                            suite_dir=task.suite_path.parent,
                            repeat_index=repeat_i if task.repeats > 1 else None,
                            total_repeats=task.repeats if task.repeats > 1 else None,
                            resume=resume,
                            profile_vram=profile_vram,
                        )
                        result = OvernightResult(
                            task=task, model=model, repeat=repeat_i,
                            success=True, duration_s=time.monotonic() - start,
                        )
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        result = OvernightResult(
                            task=task, model=model, repeat=repeat_i,
                            success=False, error=str(exc),
                            duration_s=time.monotonic() - start,
                        )

                    # Post-run evaluation (skip on failure or discovery)
                    if on_run_eval and result.success and run_result is not None:
                        try:
                            result.eval_score = await on_run_eval(run_result)
                        except KeyboardInterrupt:
                            raise
                        except Exception as exc:
                            console.print(f"  [yellow]Eval failed: {exc}[/yellow]")

                    results.append(result)
                    if on_task_done:
                        on_task_done(result)

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(results: list[OvernightResult], total_elapsed: float) -> None:
    """Print a formatted summary of the overnight run."""
    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    seconds = int(total_elapsed % 60)
    time_str = f"{hours}h {minutes:02d}m {seconds:02d}s" if hours else f"{minutes}m {seconds:02d}s"

    console.print(f"\n[bold]=== Overnight Complete ({time_str}) ===[/bold]\n")

    # Group by suite
    by_suite: dict[str, list[OvernightResult]] = {}
    for r in results:
        name = r.task.suite.suite.name
        by_suite.setdefault(name, []).append(r)

    failures: list[OvernightResult] = []

    for suite_name, suite_results in by_suite.items():
        console.print(f"[bold]{suite_name}[/bold]:")
        for r in suite_results:
            repeat_str = f" repeat {r.repeat}" if r.repeat else ""
            if r.success:
                dur = f"{r.duration_s:.0f}s"
                eval_str = f"  score: {r.eval_score:.2f}" if r.eval_score is not None else ""
                console.print(f"  [green]OK[/green]  {r.model}{repeat_str} ({dur}){eval_str}")
            else:
                console.print(f"  [red]FAIL[/red] {r.model}{repeat_str}: {r.error}")
                failures.append(r)
        console.print()

    passed = sum(1 for r in results if r.success)
    total = len(results)
    console.print(f"Passed: {passed}/{total}")
    if failures:
        console.print(f"[red]Failures: {len(failures)}[/red]")
