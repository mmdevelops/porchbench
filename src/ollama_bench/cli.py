"""CLI entry point for ollama-bench.

Uses typer for argument parsing and rich for terminal output.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Optional

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

from ollama_bench.runner import run_suite
from ollama_bench.suite import load_suite, make_suite_reference
from ollama_bench.schemas import RunResult

app = typer.Typer(
    name="ollama-bench",
    help="Deterministic benchmarking of local LLMs via Ollama.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    suite_path: Annotated[
        Path,
        typer.Option("--suite", "-s", help="Path to a test suite YAML file."),
    ],
    models: Annotated[
        list[str],
        typer.Option("--model", "-m", help="Ollama model name(s). Repeat for multiple models."),
    ],
    prompt_ids: Annotated[
        Optional[list[str]],
        typer.Option("--prompt-id", "-p", help="Run only these prompt IDs."),
    ] = None,
    host: Annotated[
        Optional[str],
        typer.Option("--host", "-H", help="Ollama server URL."),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for result JSON files."),
    ] = Path("results"),
) -> None:
    """Run a benchmark suite against one or more Ollama models."""
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

    if prompt_ids:
        console.print(f"Filter: {', '.join(prompt_ids)}")

    console.print()

    # Run each model
    for model in models:
        console.rule(f"[bold]{model}[/bold]")

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

            def on_complete(prompt_id: str, success: bool) -> None:
                status = "[green]ok[/green]" if success else "[red]FAIL[/red]"
                progress.console.print(f"  {prompt_id}: {status}")
                progress.advance(task)

            result = asyncio.run(
                run_suite(
                    suite=suite,
                    suite_ref=suite_ref,
                    model=model,
                    host=host,
                    prompt_ids=prompt_ids,
                    output_dir=output_dir,
                    on_prompt_complete=on_complete,
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

    console.print(table)


@app.command()
def evaluate(
    result_path: Annotated[
        Path,
        typer.Option("--result", "-r", help="Path to a run result JSON file."),
    ],
    rubric_path: Annotated[
        Path,
        typer.Option("--rubric", "-R", help="Path to a rubric YAML file."),
    ] = Path("rubrics/default.yaml"),
    evaluator_model: Annotated[
        str,
        typer.Option("--evaluator", "-e", help="Model to use as judge."),
    ] = "deepseek-r1:14b",
    backend: Annotated[
        str,
        typer.Option("--backend", "-b", help="Evaluation backend: 'ollama' (default) or 'api'."),
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
) -> None:
    """Score a run result using an LLM judge (local Ollama model or Claude API)."""
    from ollama_bench.evaluator import (
        AnthropicEvalBackend,
        OllamaEvalBackend,
        evaluate_run,
        load_rubric,
        write_scorecard,
    )

    # Load inputs
    try:
        run_data = result_path.read_text(encoding="utf-8")
        run_result = RunResult.model_validate_json(run_data)
    except Exception as exc:
        console.print(f"[red]Failed to load run result: {exc}[/red]")
        raise typer.Exit(code=1)

    try:
        rubric = load_rubric(rubric_path)
    except Exception as exc:
        console.print(f"[red]Failed to load rubric: {exc}[/red]")
        raise typer.Exit(code=1)

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
    else:
        console.print(f"[red]Unknown backend: {backend}. Use 'ollama' or 'api'.[/red]")
        raise typer.Exit(code=1)

    console.print(f"Run: [bold]{run_result.run.model.name}[/bold] ({run_result.run.id[:8]})")
    console.print(f"Rubric: {rubric.rubric.name} v{rubric.rubric.version}")
    console.print(f"Evaluator: {backend_label}")
    console.print(f"Prompts to score: {len(run_result.results)}")
    console.print()

    scorecard = asyncio.run(
        evaluate_run(run_result, rubric, eval_backend, evaluator_label=backend_label)
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
        list[Path],
        typer.Option("--result", "-r", help="Run result JSON files to compare. Repeat for each."),
    ],
    scorecard_paths: Annotated[
        Optional[list[Path]],
        typer.Option("--scorecard", "-S", help="Scorecard JSON files (same order as results)."),
    ] = None,
) -> None:
    """Compare results across models side-by-side."""
    from ollama_bench.compare import load_run_result, load_scorecard, print_comparison_table

    runs = []
    for p in result_paths:
        try:
            runs.append(load_run_result(p))
        except Exception as exc:
            console.print(f"[red]Failed to load {p}: {exc}[/red]")
            raise typer.Exit(code=1)

    scorecards = None
    if scorecard_paths:
        scorecards = []
        for p in scorecard_paths:
            try:
                scorecards.append(load_scorecard(p))
            except Exception as exc:
                console.print(f"[yellow]Warning: could not load scorecard {p}: {exc}[/yellow]")
                scorecards.append(None)

    print_comparison_table(runs, scorecards)


@app.command("discover-routes")
def discover_routes(
    suite_path: Annotated[
        Path,
        typer.Option("--suite", "-s", help="Path to a routing discovery suite YAML."),
    ],
    models: Annotated[
        list[str],
        typer.Option("--model", "-m", help="Ollama model name(s). Repeat for each."),
    ],
    host: Annotated[
        Optional[str],
        typer.Option("--host", "-H", help="Ollama server URL."),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for result JSON files."),
    ] = Path("results"),
) -> None:
    """Run routing discovery: every prompt x strategy x model."""
    from ollama_bench.routing import count_discovery_runs, run_discovery

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

    results = asyncio.run(
        run_discovery(suite, suite_ref, models, host=host, output_dir=output_dir)
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


@app.command("analyze-routes")
def analyze_routes_cmd(
    result_paths: Annotated[
        list[Path],
        typer.Option("--result", "-r", help="Routing discovery result JSON files."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for analysis output."),
    ] = Path("results"),
    summary_only: Annotated[
        bool,
        typer.Option("--summary", help="Print summary only, don't write full analysis."),
    ] = False,
) -> None:
    """Analyze routing discovery results and produce a routing analysis."""
    from ollama_bench.routing import analyze_routes

    runs = []
    for p in result_paths:
        try:
            data = p.read_text(encoding="utf-8")
            runs.append(RunResult.model_validate_json(data))
        except Exception as exc:
            console.print(f"[red]Failed to load {p}: {exc}[/red]")
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
        list[str],
        typer.Option("--model", "-m", help="Ollama model name(s) to profile."),
    ],
    host: Annotated[
        Optional[str],
        typer.Option("--host", "-H", help="Ollama server URL."),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for profile output."),
    ] = Path("results"),
) -> None:
    """Profile local system: model load times, VRAM, swap costs, co-residency."""
    from ollama_bench.profiler import profile_system, write_profile, print_profile_summary

    sys_profile = asyncio.run(profile_system(models, host=host))

    path = write_profile(sys_profile, output_dir)
    console.print(f"\n[green]Profile written to {path}[/green]\n")

    print_profile_summary(sys_profile)


if __name__ == "__main__":
    app()
