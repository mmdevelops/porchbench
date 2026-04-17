"""CLI entry point for feral.

Uses typer for argument parsing and rich for terminal output.
Loads .env from the working directory for persistent configuration.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated, Optional

from dotenv import load_dotenv

load_dotenv()  # load .env before typer reads envvar defaults

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from feral.assets import (
    find_rubric,
    find_suite,
    resolve_rubric_dir,
    resolve_suite_dir,
)
from feral.backend import InferenceBackend, OllamaBackend, OpenAICompatBackend
from feral.errors import UserError, load_json_model
from feral.runner import run_suite
from feral.suite import load_suite, make_suite_reference
from feral.schemas import RunResult, Scorecard

app = typer.Typer(
    name="feral",
    help="Deterministic benchmarking of local LLMs.",
    no_args_is_help=True,
)
routes_app = typer.Typer(
    name="routes",
    help="Find which model handles each prompt type best.",
    no_args_is_help=True,
)
app.add_typer(routes_app)
console = Console()


def construct_backend(
    backend: str,
    host: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> InferenceBackend:
    """Build an InferenceBackend from CLI flags."""
    if backend == "ollama":
        return OllamaBackend(host=host)
    elif backend == "openai-compat":
        if not base_url:
            console.print("[red]--base-url is required for openai-compat backend.[/red]")
            raise typer.Exit(code=1)
        return OpenAICompatBackend(base_url=base_url, api_key=api_key or "not-needed")
    else:
        console.print(f"[red]Unknown backend: {backend}. Use 'ollama' or 'openai-compat'.[/red]")
        raise typer.Exit(code=1)


def check_server_or_exit(backend: InferenceBackend, backend_name: str) -> None:
    """Verify the inference server is reachable, exit with helpful message if not."""
    healthy, health_msg = asyncio.run(backend.get_server_health())
    if not healthy:
        console.print(f"[red]Cannot reach inference server: {health_msg}[/red]")
        if backend_name == "ollama":
            console.print(
                "\n[yellow]Troubleshooting:[/yellow]\n"
                "  1. Install Ollama from https://ollama.com\n"
                "  2. Start the server: [bold]ollama serve[/bold]\n"
                "  3. Pull a model: [bold]ollama pull <model>[/bold]"
            )
        raise typer.Exit(code=1)


def check_models_or_exit(
    backend: InferenceBackend, models: list[str], backend_name: str,
) -> None:
    """Verify all models exist. Hard exit for Ollama, soft warning for other backends."""
    for model in models:
        try:
            asyncio.run(backend.get_model_info(model))
        except LookupError:
            # Server confirmed model doesn't exist
            console.print(f"[red]Model not found: {model}[/red]")
            if backend_name == "ollama":
                console.print(f"  Run [bold]ollama pull {model}[/bold] to download it.")
            raise typer.Exit(code=1)
        except Exception:
            if backend_name == "ollama":
                console.print(f"[red]Model not found: {model}[/red]")
                console.print(f"  Run [bold]ollama pull {model}[/bold] to download it.")
                raise typer.Exit(code=1)
            # Non-Ollama backend couldn't verify — warn and continue
            console.print(
                f"[yellow]Could not verify model \"{model}\" "
                f"(server does not support model lookups)[/yellow]"
            )


@app.command()
def run(
    suite_path: Annotated[
        Optional[Path],
        typer.Option("--suite", "-s", help="Suite name (e.g. 'coding-basics') or path to a YAML file. Interactive picker if omitted."),
    ] = None,
    models: Annotated[
        Optional[list[str]],
        typer.Option("--model", "-m", help="Model name(s). Repeat for multiple. Interactive picker if omitted."),
    ] = None,
    prompt_ids: Annotated[
        Optional[list[str]],
        typer.Option("--prompt-id", "-p", help="Run only these prompt IDs."),
    ] = None,
    backend_name: Annotated[
        str,
        typer.Option("--backend", envvar="FERAL_BACKEND", help="Inference backend: 'ollama' (default) or 'openai-compat'."),
    ] = "ollama",
    host: Annotated[
        Optional[str],
        typer.Option("--host", "-H", envvar="OLLAMA_HOST", help="Ollama server URL."),
    ] = None,
    base_url: Annotated[
        Optional[str],
        typer.Option("--base-url", envvar="FERAL_BASE_URL", help="OpenAI-compat server URL."),
    ] = None,
    api_key: Annotated[
        Optional[str],
        typer.Option("--api-key", envvar="FERAL_API_KEY", help="API key for OpenAI-compat servers."),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for result JSON files."),
    ] = Path("results"),
    repeats: Annotated[
        int,
        typer.Option("--repeats", "-n", help="Number of times to repeat each run for determinism verification."),
    ] = 1,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Skip prompts already completed in prior runs of the same suite+model."),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show per-prompt metrics and response preview."),
    ] = False,
    profile_vram: Annotated[
        bool,
        typer.Option("--profile-vram", help="Poll VRAM usage during inference (Ollama only)."),
    ] = False,
) -> None:
    """Run a benchmark suite against one or more models."""
    interactive = models is None or suite_path is None

    # Interactive selection when args omitted (model first, then suite)
    if models is None:
        from feral.interactive import select_models
        backend = construct_backend(backend_name, host=host, base_url=base_url, api_key=api_key)
        check_server_or_exit(backend, backend_name)
        models = select_models(backend)
    if suite_path is None:
        from feral.interactive import select_suite
        suite_path = select_suite()
    if interactive:
        from feral.interactive import select_run_options
        opts = select_run_options(default_repeats=repeats)
        repeats = opts["repeats"]
        verbose = opts["verbose"]
        resume = opts["resume"]
        profile_vram = opts["profile_vram"]

    # Resolve bare names and relative paths against cwd/packaged defaults
    try:
        suite_path = find_suite(suite_path)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    # Load and validate suite
    try:
        suite = load_suite(suite_path)
    except Exception as exc:
        console.print(f"[red]Failed to load suite: {exc}[/red]")
        raise typer.Exit(code=1)

    suite_ref = make_suite_reference(suite_path, suite)

    console.print(f"Suite: [bold]{suite.suite.name}[/bold] v{suite.suite.version}")
    console.print(f"Prompts: {len(suite.prompts)}")
    console.print(f"Models: {', '.join(models)}")
    if repeats > 1:
        console.print(f"Repeats: {repeats}")

    if prompt_ids:
        console.print(f"Filter: {', '.join(prompt_ids)}")

    console.print()

    # Build backend and verify connectivity
    backend = construct_backend(backend_name, host=host, base_url=base_url, api_key=api_key)

    check_server_or_exit(backend, backend_name)
    check_models_or_exit(backend, models, backend_name)

    for model in models:
        for repeat_i in range(1, repeats + 1):
            repeat_label = f" (repeat {repeat_i}/{repeats})" if repeats > 1 else ""
            console.rule(f"[bold]{model}[/bold]{repeat_label}")

            prompt_count = len(prompt_ids) if prompt_ids else len(suite.prompts)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Running {model}", total=prompt_count)

                def on_complete(prompt_id: str, success: bool, result=None) -> None:
                    status = "[green]ok[/green]" if success else "[red]FAIL[/red]"
                    dur = result.metrics.total_duration if result else None
                    dur_str = f"{dur / 1e9:.1f}s" if dur else ""

                    if verbose and result:
                        tps = result.metrics.tokens_per_second
                        toks = result.metrics.eval_count
                        done = result.response.done_reason or "?"
                        tps_str = f"{tps:.1f} tok/s" if tps else "n/a"
                        toks_str = f"{toks} tokens" if toks else "n/a"
                        vram = result.metrics.peak_vram_bytes
                        vram_str = f", {vram / (1024**3):.2f}GB VRAM" if vram else ""
                        progress.console.print(
                            f"  {prompt_id}: {status}  "
                            f"[dim]{dur_str}, {toks_str}, {tps_str}, done={done}{vram_str}[/dim]"
                        )
                        preview = result.response.message.content[:200].replace("\n", " ")
                        progress.console.print(f"    [dim]{preview}...[/dim]")
                    else:
                        time_part = f"  [dim]{dur_str}[/dim]" if dur_str else ""
                        progress.console.print(f"  {prompt_id}: {status}{time_part}")
                    progress.advance(task)

                result = asyncio.run(
                    run_suite(
                        suite=suite,
                        suite_ref=suite_ref,
                        model=model,
                        backend=backend,
                        prompt_ids=prompt_ids,
                        output_dir=output_dir,
                        on_prompt_complete=on_complete,
                        suite_dir=suite_path.parent,
                        repeat_index=repeat_i if repeats > 1 else None,
                        total_repeats=repeats if repeats > 1 else None,
                        resume=resume,
                        profile_vram=profile_vram,
                    )
                )

            # Print summary table
            _print_summary(result)
            console.print()


def _print_summary(result) -> None:
    """Print a compact summary table for a completed run."""
    s = result.summary
    table = Table(title="Summary", show_header=False, title_style="bold")
    table.add_column("Metric", style="dim")
    table.add_column("Value")

    table.add_row("Completed", f"{s.completed}/{s.total_prompts}")
    if s.failed > 0:
        table.add_row("Failed", f"[red]{s.failed}[/red]")
    table.add_row("Total time", f"{s.total_duration_s:.1f}s")
    if s.avg_tokens_per_second is not None:
        table.add_row("Avg tokens/sec", f"{s.avg_tokens_per_second:.1f}")

    # Tool-use validation summary
    tool_results = [r for r in result.results if r.validation_passed is not None]
    if tool_results:
        passed = sum(1 for r in tool_results if r.validation_passed)
        table.add_row("Validation", f"{passed}/{len(tool_results)} passed")

    console.print(table)


@app.command()
def evaluate(
    result_path: Annotated[
        Optional[Path],
        typer.Option("--result", "-r", help="Path to a run result JSON file. Interactive picker if omitted."),
    ] = None,
    rubric_path: Annotated[
        Optional[Path],
        typer.Option("--rubric", help="Path to a rubric YAML file. Auto-resolved from run result if omitted."),
    ] = None,
    evaluator_model: Annotated[
        Optional[str],
        typer.Option("--evaluator", "-e", envvar="FERAL_EVAL_MODEL", help="Model to use as judge. Defaults per backend: ollama=gemma4:e4b, api=claude-sonnet-4-6, claude-code=sonnet."),
    ] = None,
    backend: Annotated[
        str,
        typer.Option("--backend", "-b", envvar="FERAL_EVAL_BACKEND", help="Evaluation backend: 'ollama' (default), 'api', or 'claude-code'."),
    ] = "ollama",
    host: Annotated[
        Optional[str],
        typer.Option("--host", "-H", help="Ollama server URL (for ollama backend)."),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for scorecard JSON files."),
    ] = Path("scorecards"),
    api_key: Annotated[
        Optional[str],
        typer.Option("--api-key", envvar="ANTHROPIC_API_KEY", help="Anthropic API key (for api backend)."),
    ] = None,
    rubric_dir: Annotated[
        Optional[Path],
        typer.Option("--rubric-dir", help="Directory of category-specific rubrics (coding.yaml, reasoning.yaml, cross-domain.yaml)."),
    ] = None,
    eval_timeout: Annotated[
        int,
        typer.Option("--eval-timeout", help="Timeout in seconds per prompt evaluation (claude-code backend)."),
    ] = 120,
) -> None:
    """Score model responses for quality using an LLM-as-judge evaluator."""
    from feral.evaluator import (
        EVAL_BACKEND_DEFAULTS,
        AnthropicEvalBackend,
        ClaudeCodeEvalBackend,
        OllamaEvalBackend,
        evaluate_run,
        load_calibration_examples,
        load_rubric,
        load_rubric_dir,
        write_scorecard,
    )

    # Interactive selection when args omitted
    if result_path is None:
        from feral.interactive import select_result
        result_path = select_result()

    # Resolve evaluator model: explicit --evaluator > env var > per-backend default
    if evaluator_model is None:
        evaluator_model = EVAL_BACKEND_DEFAULTS.get(backend, "gemma4:e4b")

    # Load inputs
    try:
        run_result = load_json_model(result_path, RunResult, "run result")
    except UserError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    # Resolve rubric: explicit --rubric > suite hint > default
    if rubric_path is None:
        suite_rubric_hint = run_result.run.suite.rubric
        if suite_rubric_hint:
            rubric_path = find_rubric(suite_rubric_hint)
            console.print(f"Rubric (from suite): [bold]{rubric_path}[/bold]")
        else:
            rubric_path = find_rubric("default")

    try:
        rubric = load_rubric(rubric_path)
    except Exception as exc:
        console.print(f"[red]Failed to load rubric: {exc}[/red]")
        raise typer.Exit(code=1)

    # Load category-specific rubrics if directory provided
    rubrics_by_category = None
    if rubric_dir:
        try:
            rubrics_by_category = load_rubric_dir(rubric_dir)
            console.print(f"Category rubrics: {', '.join(rubrics_by_category.keys())}")
        except Exception as exc:
            console.print(f"[yellow]Warning: could not load rubric dir: {exc}[/yellow]")

    # Load calibration examples for few-shot priming
    calibration_path = rubric_path.parent / "calibration-examples.yaml" if rubric_path else None
    calibration_data = load_calibration_examples(calibration_path) if calibration_path else {}
    if calibration_data:
        console.print(f"Calibration: {', '.join(calibration_data.keys())}")

    # Create backend
    if backend == "ollama":
        eval_backend = OllamaEvalBackend(model=evaluator_model, host=host)
        backend_label = f"ollama/{evaluator_model}"
    elif backend == "api":
        try:
            eval_backend = AnthropicEvalBackend(model=evaluator_model, api_key=api_key)
        except ImportError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1)
        backend_label = f"api/{evaluator_model}"
    elif backend == "claude-code":
        eval_backend = ClaudeCodeEvalBackend(model=evaluator_model, timeout_s=eval_timeout)
        backend_label = f"claude-code/{evaluator_model}"
    else:
        console.print(f"[red]Unknown backend: {backend}. Use 'ollama', 'api', or 'claude-code'.[/red]")
        raise typer.Exit(code=1)

    console.print(f"Run: [bold]{run_result.run.model.name}[/bold] ({run_result.run.id[:8]})")
    console.print(f"Rubric: {rubric.rubric.name} v{rubric.rubric.version}")
    console.print(f"Evaluator: {backend_label}")
    console.print(f"Prompts to score: {len(run_result.results)}")
    console.print()

    scorecard = asyncio.run(
        evaluate_run(
            run_result, rubric, eval_backend,
            evaluator_label=backend_label,
            rubrics_by_category=rubrics_by_category,
            calibration_data=calibration_data or None,
        )
    )

    path = write_scorecard(scorecard, output_dir)
    console.print(f"\n[green]Scorecard written to {path}[/green]")

    # Print aggregate scores
    agg = scorecard.aggregate
    table = Table(title="Aggregate Scores", show_header=False, title_style="bold")
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    table.add_row("Overall", f"{agg.overall_weighted:.2f}")
    for cat, score in agg.by_category.items():
        table.add_row(f"  {cat}", f"{score:.2f}")
    for diff, score in agg.by_difficulty.items():
        table.add_row(f"  {diff}", f"{score:.2f}")
    console.print(table)


@app.command()
def compare(
    result_paths: Annotated[
        Optional[list[Path]],
        typer.Option("--result", "-r", help="Run result JSON files to compare. Interactive picker if omitted."),
    ] = None,
    scorecard_paths: Annotated[
        Optional[list[Path]],
        typer.Option("--scorecard", help="Scorecard JSON files (same order as results)."),
    ] = None,
) -> None:
    """Compare metrics and scores across models side-by-side."""
    from feral.compare import print_comparison_table

    # Interactive selection when args omitted
    if result_paths is None:
        from feral.interactive import select_results
        result_paths = select_results()

    runs = []
    for p in result_paths:
        try:
            runs.append(load_json_model(p, RunResult, "run result"))
        except UserError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1)

    scorecards = None
    if scorecard_paths:
        scorecards = []
        for p in scorecard_paths:
            try:
                scorecards.append(load_json_model(p, Scorecard, "scorecard"))
            except UserError as exc:
                console.print(f"[yellow]Warning: {exc}[/yellow]")
                scorecards.append(None)

    print_comparison_table(runs, scorecards)


@routes_app.command("discover")
def discover_routes(
    suite_path: Annotated[
        Optional[Path],
        typer.Option("--suite", "-s", help="Suite name (e.g. 'routing-discovery') or path to a YAML file. Interactive picker if omitted."),
    ] = None,
    models: Annotated[
        Optional[list[str]],
        typer.Option("--model", "-m", help="Model name(s). Repeat for each. Interactive picker if omitted."),
    ] = None,
    backend_name: Annotated[
        str,
        typer.Option("--backend", envvar="FERAL_BACKEND", help="Inference backend: 'ollama' or 'openai-compat'."),
    ] = "ollama",
    host: Annotated[
        Optional[str],
        typer.Option("--host", "-H", envvar="OLLAMA_HOST", help="Ollama server URL."),
    ] = None,
    base_url: Annotated[
        Optional[str],
        typer.Option("--base-url", envvar="FERAL_BASE_URL", help="OpenAI-compat server URL."),
    ] = None,
    api_key: Annotated[
        Optional[str],
        typer.Option("--api-key", envvar="FERAL_API_KEY", help="API key for OpenAI-compat servers."),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for result JSON files."),
    ] = Path("results"),
) -> None:
    """Run all prompt x strategy x model combinations to map capabilities."""
    from feral.interactive import select_models, select_suite
    from feral.routing import count_discovery_runs, run_discovery

    if suite_path is None:
        suite_path = select_suite()
    if models is None:
        backend = construct_backend(backend_name, host=host, base_url=base_url, api_key=api_key)
        check_server_or_exit(backend, backend_name)
        models = select_models(backend)

    try:
        suite_path = find_suite(suite_path)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    try:
        suite = load_suite(suite_path)
    except Exception as exc:
        console.print(f"[red]Failed to load suite: {exc}[/red]")
        raise typer.Exit(code=1)

    suite_ref = make_suite_reference(suite_path, suite)
    total = count_discovery_runs(suite, models)

    console.print(f"Suite: [bold]{suite.suite.name}[/bold] v{suite.suite.version}")
    console.print(f"Prompts: {len(suite.prompts)}")
    console.print(f"Strategies: {len(suite.strategies) or 1}")
    console.print(f"Models: {', '.join(models)}")
    console.print(f"Total runs: [bold]{total}[/bold]")
    console.print()

    backend = construct_backend(backend_name, host=host, base_url=base_url, api_key=api_key)
    check_server_or_exit(backend, backend_name)
    check_models_or_exit(backend, models, backend_name)

    results = asyncio.run(
        run_discovery(
            suite, suite_ref, models, backend=backend, output_dir=output_dir,
            suite_dir=suite_path.parent,
        )
    )

    # Print summary per model
    for run in results:
        correct_count = sum(1 for r in run.results if r.correct is True)
        total_checked = sum(1 for r in run.results if r.correct is not None)
        console.print(
            f"\n[bold]{run.run.model.name}[/bold]: "
            f"{correct_count}/{total_checked} correct, "
            f"{run.summary.avg_tokens_per_second or 0:.1f} avg tok/s"
        )


@routes_app.command("analyze")
def analyze_routes_cmd(
    result_paths: Annotated[
        Optional[list[Path]],
        typer.Option("--result", "-r", help="Routing discovery result files. Interactive picker if omitted."),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for analysis output."),
    ] = Path("results"),
    summary_only: Annotated[
        bool,
        typer.Option("--summary", help="Print summary only, don't write full analysis."),
    ] = False,
) -> None:
    """Analyze discovery results to find optimal routing strategies."""
    from feral.routing import analyze_routes

    # Interactive selection when args omitted
    if result_paths is None:
        from feral.interactive import select_results
        result_paths = select_results()

    runs = []
    for p in result_paths:
        try:
            runs.append(load_json_model(p, RunResult, "run result"))
        except UserError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1)

    analysis = analyze_routes(runs)

    # Print headline
    h = analysis.headline
    console.print("\n[bold]Routing Analysis[/bold]")
    console.print(f"Models: {', '.join(analysis.models_tested)}")
    console.print(f"Strategies: {', '.join(analysis.strategies_tested)}")
    console.print(f"Problems: {h.problems_total}")
    console.print()

    worthwhile_str = (
        "[green]YES[/green]" if h.routing_worthwhile else "[yellow]NO[/yellow]"
    )
    console.print(f"Routing worthwhile: {worthwhile_str}")
    console.print(f"Inverse scaling detected: {h.inverse_scaling_detected} (rate: {h.inverse_scaling_rate:.1%})")
    console.print(f"Problems where routing helps: {h.problems_where_routing_helps}/{h.problems_total}")

    if h.max_quality_gain_pp is not None:
        console.print(f"Max quality gain: {h.max_quality_gain_pp:+.1f}pp")
    if h.max_cost_reduction_pct is not None:
        console.print(f"Max token savings: {h.max_cost_reduction_pct:.1f}%")

    # Print patterns
    if analysis.patterns:
        console.print("\n[bold]Patterns[/bold]")
        for p in analysis.patterns:
            console.print(f"  [{p.confidence}] {p.description} ({p.evidence_count} problems)")

    # Print verdict
    v = analysis.verdict
    console.print(f"\n[bold]Verdict[/bold]: {'Route' if v.routing_recommended else 'Use largest model'}")
    if v.estimated_quality_improvement_pp is not None:
        console.print(f"  Est. quality improvement: {v.estimated_quality_improvement_pp:+.1f}pp")
    if v.estimated_token_savings_pct is not None:
        console.print(f"  Est. token savings: {v.estimated_token_savings_pct:.1f}%")
    console.print(f"  Caveat: {v.caveat}")

    if not summary_only:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = analysis.timestamp.strftime("%Y-%m-%dT%H-%M-%S")
        path = output_dir / f"{ts}_routing-analysis.json"
        path.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"\n[green]Analysis written to {path}[/green]")


@app.command()
def profile(
    models: Annotated[
        Optional[list[str]],
        typer.Option("--model", "-m", help="Ollama model name(s) to profile. Interactive picker if omitted."),
    ] = None,
    backend_name: Annotated[
        str,
        typer.Option("--backend", envvar="FERAL_BACKEND", help="Inference backend (only 'ollama' supported for profiling)."),
    ] = "ollama",
    host: Annotated[
        Optional[str],
        typer.Option("--host", "-H", envvar="OLLAMA_HOST", help="Ollama server URL."),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for profile output."),
    ] = Path("results"),
) -> None:
    """Measure GPU memory, model load times, and swap costs (Ollama only)."""
    from feral.interactive import select_models
    from feral.profiler import profile_system, write_profile, print_profile_summary

    if backend_name != "ollama":
        console.print(
            f"[red]Profile requires Ollama backend (got '{backend_name}'). "
            f"VRAM and swap profiling are Ollama-specific.[/red]"
        )
        raise typer.Exit(code=1)

    backend = OllamaBackend(host=host)

    if models is None:
        check_server_or_exit(backend, "ollama")
        models = select_models(backend)

    sys_profile = asyncio.run(profile_system(models, backend=backend))

    path = write_profile(sys_profile, output_dir)
    console.print(f"\n[green]Profile written to {path}[/green]\n")

    print_profile_summary(sys_profile)


@app.command()
def leaderboard(
    scorecard_paths: Annotated[
        Optional[list[Path]],
        typer.Option("--scorecard", help="Scorecard JSON files. Repeat for each."),
    ] = None,
    scorecard_dir: Annotated[
        Path,
        typer.Option("--dir", "-d", help="Directory to scan for scorecard JSON files."),
    ] = Path("scorecards"),
    result_dir: Annotated[
        Path,
        typer.Option("--result-dir", help="Directory of run results for model name lookup."),
    ] = Path("results"),
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Require same evaluator in addition to same rubric."),
    ] = False,
    top_n: Annotated[
        int,
        typer.Option("--top-n", "-n", help="Number of best/worst prompts to show per model."),
    ] = 3,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show details of skipped files during discovery."),
    ] = False,
) -> None:
    """Rank scored models in a leaderboard table."""
    from feral.leaderboard import (
        discover_scorecards,
        filter_comparable,
        group_scorecards,
        print_leaderboard,
    )

    scorecards = []
    if scorecard_paths:
        for p in scorecard_paths:
            try:
                scorecards.append(load_json_model(p, Scorecard, "scorecard"))
            except UserError as exc:
                console.print(f"[red]{exc}[/red]")
                raise typer.Exit(code=1)
    else:
        if not scorecard_dir.is_dir():
            console.print(f"[red]Scorecard directory not found: {scorecard_dir}[/red]")
            console.print("  Run [bold]feral evaluate[/bold] on a run result first to produce a scorecard.")
            raise typer.Exit(code=1)
        scorecards = discover_scorecards(scorecard_dir, verbose=verbose)

    if not scorecards:
        console.print("[yellow]No scorecards found.[/yellow]")
        console.print("  Run [bold]feral evaluate[/bold] on a run result first to produce a scorecard.")
        raise typer.Exit(code=1)

    # Interactive rubric group selection when multiple groups exist
    groups = group_scorecards(scorecards)
    if len(groups) > 1 and scorecard_paths is None:
        from feral.interactive import select_rubric_group
        selected = select_rubric_group(groups)
    else:
        selected = scorecards

    # Evaluator consistency check + --strict filtering on the selected group
    comparable = filter_comparable(selected, strict=strict)

    print_leaderboard(comparable, top_n=top_n, result_dir=result_dir)


@app.command("eval-extract", hidden=True)
def eval_extract(
    result_path: Annotated[
        Path,
        typer.Argument(help="Path to a run result JSON file."),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output path for extracted data JSON. Defaults to .claude/eval-data.json."),
    ] = None,
) -> None:
    """Pre-extract compact evaluation data from a run result file.

    Reads the full result JSON once and writes a lightweight file containing
    only prompt text, response text, expected answers, and metadata. Used by
    the /evaluate skill to avoid repeated partial reads of large result files.
    """
    from feral.evaluator import extract_eval_data

    eval_data = extract_eval_data(result_path)

    out_path = output or Path(".claude/eval-data.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(eval_data.model_dump_json(indent=2), encoding="utf-8")

    console.print(f"[green]Extracted {eval_data.header.total_prompts} prompts to {out_path}[/green]")
    console.print(f"  Model: {eval_data.header.model_name}")
    console.print(f"  Suite: {eval_data.header.suite_name}")
    console.print(f"  Categories: {eval_data.header.categories}")
    console.print(f"  Difficulties: {eval_data.header.difficulties}")
    if eval_data.header.truncated_count:
        console.print(f"  [yellow]Truncated: {eval_data.header.truncated_count}[/yellow]")


@app.command("eval-finalize", hidden=True)
def eval_finalize(
    result_path: Annotated[
        Path,
        typer.Argument(help="Path to the original run result JSON file."),
    ],
    scores_path: Annotated[
        Path,
        typer.Option("--scores", "-s", help="Path to the JSONL scores file."),
    ] = Path(".claude/eval-scores.jsonl"),
    evaluator: Annotated[
        str,
        typer.Option("--evaluator", "-e", help="Evaluator label for the scorecard."),
    ] = "claude-code/claude-opus-4-6",
    rubric_label: Annotated[
        str,
        typer.Option("--rubric", help="Rubric description for the scorecard."),
    ] = "",
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for scorecard JSON files."),
    ] = Path("scorecards"),
) -> None:
    """Finalize evaluation: read streamed scores, compute aggregates, write scorecard.

    Reads the JSONL scores file produced during /evaluate scoring, combines
    with the original run result for category/difficulty metadata, computes
    all aggregates (by-category, by-difficulty, normalized, contamination-
    filtered), and writes the final scorecard JSON.
    """
    from feral.evaluator import build_scorecard_from_scores

    path = build_scorecard_from_scores(
        scores_path=scores_path,
        result_path=result_path,
        evaluator=evaluator,
        rubric_label=rubric_label,
        output_dir=output_dir,
    )
    console.print(f"[green]Scorecard written to {path}[/green]")


@app.command()
def overnight(
    models: Annotated[
        Optional[list[str]],
        typer.Option("--model", "-m", help="Model name(s). Repeat for multiple. Interactive picker if omitted."),
    ] = None,
    suite_paths: Annotated[
        Optional[list[Path]],
        typer.Option("--suite", "-s", help="Suite names or YAML paths. Repeat for multiple. Omit to auto-discover."),
    ] = None,
    repeats: Annotated[
        int,
        typer.Option("--repeats", "-n", help="Repeats per standard suite (discovery suites expand by strategy)."),
    ] = 3,
    backend_name: Annotated[
        str,
        typer.Option("--backend", envvar="FERAL_BACKEND", help="Inference backend: 'ollama' or 'openai-compat'."),
    ] = "ollama",
    host: Annotated[
        Optional[str],
        typer.Option("--host", "-H", envvar="OLLAMA_HOST", help="Ollama server URL."),
    ] = None,
    base_url: Annotated[
        Optional[str],
        typer.Option("--base-url", envvar="FERAL_BASE_URL", help="OpenAI-compat server URL."),
    ] = None,
    api_key: Annotated[
        Optional[str],
        typer.Option("--api-key", envvar="FERAL_API_KEY", help="API key for OpenAI-compat servers."),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for result JSON files."),
    ] = Path("results"),
    suite_dir: Annotated[
        Path | None,
        typer.Option(
            "--suite-dir",
            help="Directory to auto-discover suites from. Defaults to ./suites if present, else the packaged suites bundled with feral.",
        ),
    ] = None,
    do_profile: Annotated[
        bool,
        typer.Option("--profile", help="Run system profiling before benchmarks."),
    ] = False,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Skip already-completed runs."),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt."),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show per-prompt metrics."),
    ] = False,
    do_evaluate: Annotated[
        bool,
        typer.Option("--evaluate", help="Score each run after inference completes."),
    ] = False,
    eval_backend: Annotated[
        str,
        typer.Option("--eval-backend", envvar="FERAL_EVAL_BACKEND", help="Evaluation backend: ollama, api, or claude-code."),
    ] = "ollama",
    eval_model: Annotated[
        Optional[str],
        typer.Option("--eval-model", envvar="FERAL_EVAL_MODEL", help="Judge model. Defaults per backend."),
    ] = None,
    rubric_path: Annotated[
        Optional[Path],
        typer.Option("--rubric", help="Rubric YAML for evaluation. Auto-resolved from suite if omitted."),
    ] = None,
    rubric_dir: Annotated[
        Optional[Path],
        typer.Option("--rubric-dir", help="Directory of category-specific rubrics."),
    ] = None,
    eval_timeout: Annotated[
        int,
        typer.Option("--eval-timeout", help="Timeout in seconds per prompt evaluation (claude-code backend)."),
    ] = 120,
    profile_vram: Annotated[
        bool,
        typer.Option("--profile-vram", help="Poll VRAM usage during inference (Ollama only)."),
    ] = False,
) -> None:
    """Queue multiple suites and models for unattended batch benchmarking."""
    import time as _time

    from feral.overnight import (
        build_plan,
        execute_plan,
        print_plan,
        print_summary,
    )
    from feral.suite import discover_suites

    interactive = models is None or suite_paths is None

    # Interactive selection when args omitted
    if models is None:
        from feral.interactive import select_models
        backend = construct_backend(backend_name, host=host, base_url=base_url, api_key=api_key)
        check_server_or_exit(backend, backend_name)
        models = select_models(backend)

    # 1. Discover or use provided suites
    if suite_paths:
        paths = [find_suite(p) for p in suite_paths]
    else:
        from feral.interactive import select_suites
        paths = select_suites(resolve_suite_dir(suite_dir))

    # Interactive options screen
    if interactive:
        from feral.interactive import select_overnight_options
        opts = select_overnight_options(default_repeats=repeats)
        repeats = opts["repeats"]
        do_evaluate = opts["evaluate"]
        do_profile = opts["profile"]
        profile_vram = opts["profile_vram"]
        resume = opts["resume"]
        verbose = opts["verbose"]

    # 2. Build plan
    try:
        plan = build_plan(paths, models, repeats)
    except Exception as exc:
        console.print(f"[red]Failed to build plan: {exc}[/red]")
        raise typer.Exit(code=1)

    # 3. Display plan
    console.print()
    print_plan(plan, models)

    # 4. Preflight checks
    backend = construct_backend(backend_name, host=host, base_url=base_url, api_key=api_key)

    console.print("[bold]Preflight checks[/bold]")

    with console.status("  Checking server connectivity..."):
        server_ok, server_msg = asyncio.run(backend.get_server_health())
    status = "[green]PASS[/green]" if server_ok else "[red]FAIL[/red]"
    console.print(f"  {status} Server: {server_msg}")

    if not server_ok:
        console.print("\n[red]Inference server not reachable. Aborting.[/red]")
        raise typer.Exit(code=1)

    from feral.overnight import check_gpu_status

    with console.status(f"  Warming up {models[0]} (loading model into VRAM)..."):
        gpu_ok, gpu_msg = asyncio.run(check_gpu_status(backend, models[0]))
    status = "[green]PASS[/green]" if gpu_ok else "[red]FAIL[/red]"
    console.print(f"  {status} GPU acceleration: {gpu_msg}")

    console.print()

    # 5. Confirm
    if not yes:
        if not typer.confirm("Start overnight run?"):
            raise typer.Exit()

    # 6. Optional profiling (Ollama only)
    if do_profile:
        if not isinstance(backend, OllamaBackend):
            console.print("[yellow]Warning: --profile skipped — requires Ollama backend.[/yellow]")
        else:
            console.rule("[bold]System Profile[/bold]")
            from feral.profiler import profile_system, write_profile, print_profile_summary

            sys_profile = asyncio.run(profile_system(models, backend=backend))
            path = write_profile(sys_profile, output_dir)
            console.print(f"[green]Profile written to {path}[/green]\n")
            print_profile_summary(sys_profile)
            console.print()

    # 7. Build eval callback if --evaluate is set
    on_run_eval = None
    if do_evaluate:
        from feral.evaluator import (
            EVAL_BACKEND_DEFAULTS,
            AnthropicEvalBackend,
            ClaudeCodeEvalBackend,
            OllamaEvalBackend,
            evaluate_run,
            load_calibration_examples,
            load_rubric,
            load_rubric_dir,
            write_scorecard,
        )

        resolved_eval_model = eval_model or EVAL_BACKEND_DEFAULTS.get(eval_backend, "gemma4:e4b")

        if eval_backend == "ollama":
            eval_be = OllamaEvalBackend(model=resolved_eval_model, host=host)
        elif eval_backend == "api":
            eval_be = AnthropicEvalBackend(model=resolved_eval_model)
        elif eval_backend == "claude-code":
            eval_be = ClaudeCodeEvalBackend(model=resolved_eval_model, timeout_s=eval_timeout)
        else:
            console.print(f"[red]Unknown eval backend: {eval_backend}[/red]")
            raise typer.Exit(code=1)

        eval_backend_label = f"{eval_backend}/{resolved_eval_model}"

        # Pre-load category rubrics (falls through to packaged defaults if no override)
        eval_rubrics_by_cat = load_rubric_dir(resolve_rubric_dir(rubric_dir))

        console.print(f"[bold]Post-run evaluation:[/bold] {eval_backend_label}")
        console.print()

        async def on_run_eval(run_result):
            """Evaluate a completed run and write its scorecard."""
            # Resolve rubric from suite hint or explicit flag
            suite_hint = run_result.run.suite.rubric
            if rubric_path:
                r_path = rubric_path
            elif suite_hint:
                r_path = find_rubric(suite_hint)
            else:
                r_path = find_rubric("default")

            r = load_rubric(r_path)
            # Calibration examples live alongside the resolved rubric —
            # packaged when rubric is packaged, project-local when overridden.
            cal_data = load_calibration_examples(r_path.parent / "calibration-examples.yaml")
            scorecard = await evaluate_run(
                run_result, r, eval_be,
                evaluator_label=eval_backend_label,
                rubrics_by_category=eval_rubrics_by_cat,
                calibration_data=cal_data or None,
            )
            write_scorecard(scorecard)
            console.print(
                f"  [green]Eval:[/green] {scorecard.aggregate.overall_weighted:.2f} "
                f"({r.rubric.name})"
            )
            return scorecard.aggregate.overall_weighted

    # 8. Execute
    console.rule("[bold]Running benchmarks[/bold]")
    start = _time.monotonic()

    def on_start(task, model, repeat):
        repeat_str = f" repeat {repeat}/{task.repeats}" if repeat else ""
        console.rule(f"[bold]{task.suite.suite.name}[/bold] / {model}{repeat_str}")

    def on_done(result):
        if result.success:
            eval_str = f"  score: {result.eval_score:.2f}" if result.eval_score is not None else ""
            console.print(f"  [green]Done[/green] ({result.duration_s:.0f}s){eval_str}")
        else:
            console.print(f"  [red]Failed: {result.error}[/red]")

    try:
        results = asyncio.run(
            execute_plan(
                plan=plan,
                backend=backend,
                output_dir=output_dir,
                resume=resume,
                verbose=verbose,
                on_task_start=on_start,
                on_task_done=on_done,
                on_run_eval=on_run_eval,
                profile_vram=profile_vram,
            )
        )
    except KeyboardInterrupt:
        elapsed = _time.monotonic() - start
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        console.print(f"Elapsed: {elapsed:.0f}s")
        raise typer.Exit(code=130)

    elapsed = _time.monotonic() - start

    # 8. Summary
    print_summary(results, elapsed)


def main() -> None:
    """Entry point wrapper that turns Ctrl+C into a clean exit.

    Without this, aborting a beaupy picker raises KeyboardInterrupt and
    users see a Python traceback. Exit 130 is the shell convention for
    SIGINT-style termination.
    """
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
