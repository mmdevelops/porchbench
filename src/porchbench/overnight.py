"""Overnight orchestration: unattended multi-suite benchmark runs.

Discovers suites, classifies them (standard vs routing discovery),
builds an execution plan, runs preflight checks, and executes with
error resilience. Designed for hobbyists queuing benchmarks overnight.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from porchbench.backend import InferenceBackend, OllamaBackend
from porchbench.profiler import detect_gpu
from porchbench.routing import run_discovery
from porchbench.runner import run_suite
from porchbench.schemas import RunResult, Suite, SuiteReference
from porchbench.suite import load_suite, make_suite_reference

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
    result_path: Path | None = None  # on-disk path to the RunResult JSON (successful standard runs only)
    eval_score: float | None = None  # weighted aggregate from post-run evaluation (set later by post-run batch eval)


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
    option_overrides: dict[str, object] | None = None,
) -> list[OvernightTask]:
    """Load suites, classify them, and build an ordered task list.

    `option_overrides`, when provided, is layered onto each suite's
    `defaults.options` before tasks are built — used by the CLI's `--set`
    flag so users can flip e.g. `think=false` without editing suite YAML.
    """
    from porchbench.suite import apply_option_overrides

    tasks: list[OvernightTask] = []
    for path in suite_paths:
        suite = load_suite(path)
        if option_overrides:
            suite = apply_option_overrides(suite, option_overrides)
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
    """Estimate total runtime in seconds at a constant per-prompt rate.

    Pure helper. Use `estimate_duration_from_history` instead when you have
    access to a results directory — it produces a far better estimate by
    measuring the actual per-(model, suite) rate from past runs.
    """
    return sum(task.run_count * seconds_per_prompt for task in plan)


def _seconds_per_prompt_from_history(
    model: str, suite_slug: str, results_dir: Path,
) -> float | None:
    """Median per-prompt total_duration (seconds) for past (model, suite) runs.

    Reads matching `RunResult` JSON files in `results_dir` and aggregates the
    per-prompt `metrics.total_duration` field. Returns None when no matching
    run exists, or when matching runs contain no usable timing data. No
    filtering by hardware — assumes the results dir reflects runs on this
    machine; cross-machine result trees may yield estimates that don't
    match local performance.
    """
    if not results_dir.is_dir():
        return None
    model_slug = model.replace(":", "-").replace("/", "-")
    pattern = f"*_{suite_slug}_{model_slug}*.json"
    durations: list[float] = []
    for path in results_dir.glob(pattern):
        try:
            text = path.read_text(encoding="utf-8")
            rr = RunResult.model_validate_json(text)
        except Exception:
            continue
        for r in rr.results:
            if r.metrics.total_duration:
                durations.append(r.metrics.total_duration / 1e9)
    if not durations:
        return None
    return statistics.median(durations)


def estimate_single_suite_duration_from_history(
    models: list[str],
    suite_name: str,
    prompt_count: int,
    repeats: int,
    results_dir: Path,
) -> tuple[float, int, int]:
    """Estimate runtime for a single-suite run (`porchbench run`) from history.

    Returns `(total_seconds, calls_with_history, calls_total)`. Same partial-
    coverage semantics as `estimate_duration_from_history`: calls without a
    prior per-(model, suite) rate are excluded from the time and counted
    separately so the caller can render coverage honestly.
    """
    suite_slug = suite_name.lower().replace(" ", "-")
    per_model_calls = prompt_count * repeats
    total_seconds = 0.0
    calls_with_history = 0
    calls_total = 0
    for model in models:
        calls_total += per_model_calls
        rate = _seconds_per_prompt_from_history(model, suite_slug, results_dir)
        if rate is not None:
            total_seconds += per_model_calls * rate
            calls_with_history += per_model_calls
    return total_seconds, calls_with_history, calls_total


def estimate_duration_from_history(
    plan: list[OvernightTask], results_dir: Path,
) -> tuple[float, int, int]:
    """Estimate plan runtime from prior-run timings per (model, suite).

    Returns `(total_seconds, calls_with_history, calls_total)`. The estimate
    only covers calls for which a per-(model, suite) rate is available;
    calls without history are excluded from the time and surfaced via the
    coverage tuple so callers can render partial-coverage honestly rather
    than fabricating a number.
    """
    total_seconds = 0.0
    calls_with_history = 0
    calls_total = 0
    for task in plan:
        suite_slug = task.suite.suite.name.lower().replace(" ", "-")
        per_model_calls = task.run_count // max(len(task.models), 1)
        for model in task.models:
            calls_total += per_model_calls
            rate = _seconds_per_prompt_from_history(model, suite_slug, results_dir)
            if rate is not None:
                total_seconds += per_model_calls * rate
                calls_with_history += per_model_calls
    return total_seconds, calls_with_history, calls_total


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
    except TimeoutError:
        return False, "GPU check failed — warmup timed out after 120s"
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


async def _get_ollama_model_size_bytes(
    backend: OllamaBackend, model: str,
) -> int | None:
    """Return on-disk size in bytes for an Ollama model, or None if unknown."""
    from ollama import AsyncClient

    try:
        client = AsyncClient(host=backend.host)
        listing = await client.list()
        for m in listing.models:
            m_name = getattr(m, "model", "") or ""
            if m_name == model or m_name.startswith(f"{model}:") or model.startswith(m_name):
                size = getattr(m, "size", None)
                if isinstance(size, int) and size > 0:
                    return size
    except Exception:
        pass
    return None


async def check_vram_cofit(
    backend: InferenceBackend,
    target_models: list[str],
    eval_model: str,
) -> tuple[bool, str]:
    """Warn if target + eval model can't cofit in VRAM.

    Returns (ok, message). `ok=False` is advisory only — the caller should
    surface the warning but not block the run. Only meaningful on
    OllamaBackend where we can look up disk sizes.
    """
    if not isinstance(backend, OllamaBackend):
        return True, "cofit check not available for this backend"

    _gpu_name, vram_gb = detect_gpu()
    if not vram_gb:
        return True, "VRAM unknown — skipping cofit check"

    sizes_gb: dict[str, float] = {}
    for m in [*target_models, eval_model]:
        if m in sizes_gb:
            continue
        size_bytes = await _get_ollama_model_size_bytes(backend, m)
        if size_bytes is None:
            return True, f"could not determine size for {m} — skipping cofit check"
        sizes_gb[m] = size_bytes / (1024**3)

    eval_gb = sizes_gb[eval_model]
    # Leave ~1 GB headroom for KV cache + compute graph on each side
    HEADROOM_GB = 1.5

    # Worst-case: largest target model + eval model + headroom must fit in VRAM
    worst_target = max(target_models, key=lambda m: sizes_gb[m])
    worst_target_gb = sizes_gb[worst_target]
    combined = worst_target_gb + eval_gb + HEADROOM_GB

    if combined <= vram_gb:
        return True, (
            f"target + eval fit in VRAM "
            f"({worst_target_gb:.1f} + {eval_gb:.1f} GB + ~1.5 GB headroom ≤ {vram_gb:.1f} GB)"
        )

    return False, (
        f"target + eval don't cofit ({worst_target}={worst_target_gb:.1f} GB + "
        f"{eval_model}={eval_gb:.1f} GB + ~1.5 GB headroom = {combined:.1f} GB > "
        f"{vram_gb:.1f} GB). Ollama will swap between models at eval time. "
        f"Mitigations: --eval-backend claude-code / --eval-backend api (off-GPU judge); "
        f"smaller --eval-model; or drop --evaluate and run `porchbench evaluate -r results/*.json` later."
    )


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


def print_plan(
    plan: list[OvernightTask],
    models: list[str],
    results_dir: Path | None = None,
) -> None:
    """Print the execution plan as a Rich table.

    When `results_dir` is provided, the duration estimate is computed from
    prior runs of each (model, suite) pair found there. Calls without prior
    history are flagged in the coverage line rather than estimated against
    a fabricated default.
    """
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
    console.print(f"Total inference calls: [bold]{total_runs}[/bold]")

    if results_dir is not None:
        total_seconds, with_history, total_calls = estimate_duration_from_history(
            plan, results_dir,
        )
        if total_calls == 0:
            pass  # empty plan; nothing to estimate
        elif with_history == 0:
            console.print(
                "Estimated duration: [dim]no prior runs of these (model, suite) pairs "
                f"in {results_dir}/ — first run, no estimate[/dim]"
            )
        elif with_history == total_calls:
            console.print(
                f"Estimated duration: [bold]{format_estimate(total_seconds)}[/bold] "
                "[dim](median of prior runs)[/dim]"
            )
        else:
            console.print(
                f"Estimated duration: [bold]{format_estimate(total_seconds)}[/bold] "
                f"[dim]for {with_history}/{total_calls} calls "
                "(no history for the rest)[/dim]"
            )
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
    on_prompt_complete: Callable[..., None] | None = None,
    profile_vram: bool = False,
    heartbeat_s: float | None = None,
) -> list[OvernightResult]:
    """Execute the overnight plan with error resilience.

    Inference runs one model at a time to completion. Successful standard
    runs record their on-disk `result_path` so a post-phase evaluator can
    batch-score all results with a single judge model load, instead of
    thrashing between target-model and judge-model on every task.
    """
    from porchbench.runner import result_path_for

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
                            on_prompt_complete=on_prompt_complete,
                            heartbeat_s=heartbeat_s,
                        )
                        result = OvernightResult(
                            task=task, model=model, repeat=repeat_i,
                            success=True, duration_s=time.monotonic() - start,
                            result_path=result_path_for(run_result, output_dir),
                        )
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        result = OvernightResult(
                            task=task, model=model, repeat=repeat_i,
                            success=False, error=str(exc),
                            duration_s=time.monotonic() - start,
                        )

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
