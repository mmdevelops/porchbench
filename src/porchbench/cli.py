"""CLI entry point for porchbench.

Uses typer for argument parsing and rich for terminal output.
Loads .env from the working directory for persistent configuration.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated

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

from porchbench.assets import (
    find_rubric,
    find_suite,
    resolve_rubric_dir,
    resolve_suite_dir,
)
from porchbench.backend import InferenceBackend, OllamaBackend, OpenAICompatBackend
from porchbench.errors import UserError, load_json_model
from porchbench.runner import result_path_for, run_suite
from porchbench.schemas import RunResult, Scorecard
from porchbench.suite import load_suite, make_suite_reference

app = typer.Typer(
    name="porchbench",
    help="Deterministic benchmarking of local LLMs.",
    no_args_is_help=True,
)
routes_app = typer.Typer(
    name="routes",
    help="Find which model handles each prompt type best.",
    no_args_is_help=True,
)
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


def parse_set_overrides(items: list[str] | None) -> dict[str, object]:
    """Parse repeated `--set KEY=VALUE` flags into a typed override dict.

    Values are interpreted with YAML rules so `false`/`true`/`null`/ints/floats
    round-trip to the right Python types. Used to override `defaults.options` on
    the loaded suite without editing the suite YAML.
    """
    import yaml

    if not items:
        return {}
    out: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(
                f"--set requires KEY=VALUE format, got: {item!r}",
                param_hint="--set",
            )
        key, _, raw_value = item.partition("=")
        key = key.strip()
        if not key:
            raise typer.BadParameter(
                f"--set: empty key in {item!r}",
                param_hint="--set",
            )
        try:
            out[key] = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise typer.BadParameter(
                f"--set {key}: could not parse value {raw_value!r}: {exc}",
                param_hint="--set",
            )
    return out


def resolve_eval_model_or_exit(
    backend_name: str,
    explicit_model: str | None,
    backend: InferenceBackend | None,
    *,
    interactive: bool,
) -> str:
    """Resolve which model to use as the LLM-as-judge evaluator.

    Precedence: explicit (--eval-model or PORCHBENCH_EVAL_MODEL env) > stable
    cloud-backend default > interactive picker (ollama). For ollama with no
    explicit model, prompts the user to pick from currently-available models
    and offers to persist the choice to ./.env so future runs skip the prompt.
    """
    from porchbench.evaluator import EVAL_BACKEND_DEFAULTS

    if explicit_model:
        return explicit_model

    if backend_name in EVAL_BACKEND_DEFAULTS:
        return EVAL_BACKEND_DEFAULTS[backend_name]

    if not interactive:
        console.print(
            f"[red]No evaluator model specified for {backend_name} backend.[/red]\n"
            f"Pass [bold]--eval-model <name>[/bold] or set "
            f"[bold]PORCHBENCH_EVAL_MODEL[/bold] in .env."
        )
        raise typer.Exit(code=1)

    if not isinstance(backend, OllamaBackend):
        console.print(
            f"[red]Cannot pick evaluator interactively for {backend_name} backend.[/red]\n"
            f"Pass --eval-model <name>."
        )
        raise typer.Exit(code=1)

    from porchbench.interactive import select_evaluator_model

    chosen = select_evaluator_model(backend)

    if typer.confirm(f"Save '{chosen}' as default evaluator in .env?", default=True):
        _persist_eval_model_default(chosen)

    return chosen


def _model_family(model: str) -> str:
    """Best-effort 'family' identifier for an Ollama model name.

    Takes the first colon-separated segment, lowercased, so 'gemma4:e4b' and
    'gemma4:e2b' share the family 'gemma4'. Used to flag same-family judge
    setups; intentionally exact-match on the segment rather than fuzzy across
    generations (gemma4 vs gemma3 stay distinct), to avoid crying wolf on
    every cross-version comparison.
    """
    return model.split(":", 1)[0].strip().lower()


def warn_if_same_family_judge(target_model: str, eval_model: str) -> None:
    """Print a one-line warning if the LLM-as-judge is from the target's model family.

    Same-family judges over-rate same-family responses (Panickssery et al. 2024).
    Soft warning, not a block — the user may have valid reasons (calibration
    test, only one judge available). Only fires when both segments resolve to
    the same family root; pure cloud-vs-local pairs (ollama target judged by
    Anthropic) never trigger.
    """
    if not target_model or not eval_model:
        return
    if _model_family(target_model) != _model_family(eval_model):
        return
    console.print(
        f"  [yellow]WARN[/yellow] Evaluator '{eval_model}' is same-family as "
        f"target '{target_model}'.\n"
        f"       Same-family judges over-rate same-family responses "
        f"(Panickssery et al. 2024); scores may be inflated.\n"
        f"       Consider a different-family judge for cross-checks."
    )


def _persist_eval_model_default(model: str, dotenv_path: Path = Path(".env")) -> None:
    """Upsert PORCHBENCH_EVAL_MODEL=<model> in ./.env, creating the file if absent."""
    from dotenv import set_key

    dotenv_path.touch(exist_ok=True)
    set_key(str(dotenv_path), "PORCHBENCH_EVAL_MODEL", model, quote_mode="never")
    console.print(f"[green]Saved PORCHBENCH_EVAL_MODEL={model} to {dotenv_path}[/green]")


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


def check_tool_support_or_exit(
    backend: InferenceBackend,
    models: list[str],
    suite,
    backend_name: str,
) -> None:
    """Hard-fail if the suite needs tool-calling and any chosen model lacks it.

    Without this, models like medgemma or pure-vision variants get queued
    against `tool-use` and waste minutes producing a 0/19 score because they
    can't issue tool calls at all. Capability is read from Ollama's
    `client.show(model).capabilities` array — only checked for ollama; other
    backends silently skip (we don't have a portable capability probe).
    """
    if backend_name != "ollama":
        return
    if not isinstance(backend, OllamaBackend):
        return
    if not any(getattr(p, "mode", "text") == "tool-use" for p in suite.prompts):
        return

    from ollama import AsyncClient

    async def _caps(model: str) -> list[str]:
        client = AsyncClient(host=backend.host)
        info = await client.show(model)
        if hasattr(info, "model_dump"):
            info = info.model_dump()
        return info.get("capabilities") or []

    for model in models:
        try:
            caps = asyncio.run(_caps(model))
        except Exception as exc:
            console.print(
                f"[yellow]Could not verify tool capability for {model} "
                f"({type(exc).__name__}: {exc}). Continuing.[/yellow]"
            )
            continue
        if "tools" not in caps:
            console.print(
                f"[red]Model '{model}' does not support tool calling "
                f"(capabilities: {caps}).[/red]\n"
                f"This suite requires tool-use. Pick a model with 'tools' "
                f"in its capabilities, or run a non-tool-use suite.\n"
                f"  Check capabilities: [bold]ollama show <model>[/bold]"
            )
            raise typer.Exit(code=1)


@app.command()
def run(
    suite_path: Annotated[
        Path | None,
        typer.Option("--suite", "-s", help="Suite name (e.g. 'coding-basics') or path to a YAML file. Interactive picker if omitted."),
    ] = None,
    models: Annotated[
        list[str] | None,
        typer.Option("--model", "-m", help="Model name(s). Repeat for multiple. Interactive picker if omitted."),
    ] = None,
    prompt_ids: Annotated[
        list[str] | None,
        typer.Option("--prompt-id", "-p", help="Run only these prompt IDs."),
    ] = None,
    backend_name: Annotated[
        str,
        typer.Option("--backend", envvar="PORCHBENCH_BACKEND", help="Inference backend: 'ollama' (default) or 'openai-compat'."),
    ] = "ollama",
    host: Annotated[
        str | None,
        typer.Option("--host", "-H", envvar="OLLAMA_HOST", help="Ollama server URL."),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", envvar="PORCHBENCH_BASE_URL", help="OpenAI-compat server URL."),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", envvar="PORCHBENCH_API_KEY", help="API key for OpenAI-compat servers."),
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
    do_evaluate: Annotated[
        bool,
        typer.Option("--evaluate", help="Score all results in a single post-phase batch after inference completes (judge model stays resident)."),
    ] = False,
    eval_backend: Annotated[
        str,
        typer.Option("--eval-backend", envvar="PORCHBENCH_EVAL_BACKEND", help="Evaluation backend: ollama, api, or claude-code."),
    ] = "ollama",
    eval_model: Annotated[
        str | None,
        typer.Option("--eval-model", envvar="PORCHBENCH_EVAL_MODEL", help="Judge model. Defaults per backend."),
    ] = None,
    rubric_path: Annotated[
        Path | None,
        typer.Option("--rubric", help="Rubric YAML for evaluation. Auto-resolved from suite if omitted."),
    ] = None,
    rubric_dir: Annotated[
        Path | None,
        typer.Option("--rubric-dir", help="Directory of category-specific rubrics."),
    ] = None,
    eval_timeout: Annotated[
        int,
        typer.Option("--eval-timeout", help="Timeout in seconds per prompt evaluation (claude-code backend)."),
    ] = 120,
    set_overrides: Annotated[
        list[str] | None,
        typer.Option("--set", help="Override a suite default option as KEY=VALUE (e.g. --set think=false). Repeatable. Values parsed as YAML so booleans/ints/nulls round-trip."),
    ] = None,
) -> None:
    """Run a benchmark suite against one or more models."""
    interactive = models is None or suite_path is None

    # Interactive selection when args omitted (model first, then suite)
    if models is None:
        from porchbench.interactive import select_models
        backend = construct_backend(backend_name, host=host, base_url=base_url, api_key=api_key)
        check_server_or_exit(backend, backend_name)
        models = select_models(backend)
    if suite_path is None:
        from porchbench.interactive import select_suite
        suite_path = select_suite()
    if interactive:
        from porchbench.interactive import select_run_options
        opts = select_run_options(
            default_repeats=repeats,
            defaults={"verbose": verbose, "resume": resume, "profile_vram": profile_vram},
        )
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

    overrides = parse_set_overrides(set_overrides)
    if overrides:
        from porchbench.suite import apply_option_overrides
        try:
            suite = apply_option_overrides(suite, overrides)
        except Exception as exc:
            console.print(f"[red]Invalid --set value: {exc}[/red]")
            raise typer.Exit(code=1)
        console.print(f"Overrides: {overrides}")

    suite_ref = make_suite_reference(suite_path, suite)

    console.print(f"Suite: [bold]{suite.suite.name}[/bold] v{suite.suite.version}")
    console.print(f"Prompts: {len(suite.prompts)}")
    console.print(f"Models: {', '.join(models)}")
    if repeats > 1:
        console.print(f"Repeats: {repeats}")

    if prompt_ids:
        console.print(f"Filter: {', '.join(prompt_ids)}")

    from porchbench.overnight import (
        estimate_single_suite_duration_from_history,
        format_estimate,
    )

    prompt_count = len(prompt_ids) if prompt_ids else len(suite.prompts)
    total_seconds, with_history, total_calls = estimate_single_suite_duration_from_history(
        models=models,
        suite_name=suite.suite.name,
        prompt_count=prompt_count,
        repeats=repeats,
        results_dir=output_dir,
    )
    if total_calls == 0 or with_history == 0:
        console.print(
            f"Estimated duration: [dim]no prior runs of these (model, suite) pairs "
            f"in {output_dir}/ — first run, no estimate[/dim]"
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

    # Build backend and verify connectivity
    backend = construct_backend(backend_name, host=host, base_url=base_url, api_key=api_key)

    check_server_or_exit(backend, backend_name)
    check_models_or_exit(backend, models, backend_name)
    check_tool_support_or_exit(backend, models, suite, backend_name)

    if do_evaluate:
        eval_model = resolve_eval_model_or_exit(
            eval_backend, eval_model, backend, interactive=True,
        )
        if eval_backend == "ollama":
            check_models_or_exit(backend, [eval_model], "ollama")

    eval_paths: list[Path] = []

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

                def on_complete(prompt_id: str, success: bool, duration_s: float, prompt_num: int, total: int, result=None) -> None:
                    status = "[green]ok[/green]" if success else "[red]FAIL[/red]"
                    dur = result.metrics.total_duration if result else None
                    dur_str = f"{dur / 1e9:.1f}s" if dur else ""
                    val_badge = _format_validation_badge(result)

                    if verbose and result:
                        tps = result.metrics.tokens_per_second
                        toks = result.metrics.eval_count
                        done = result.response.done_reason or "?"
                        vram = result.metrics.peak_vram_bytes
                        # Only join populated metric parts so tool-use runs
                        # (no per-token data) don't render as ', n/a, n/a'.
                        parts = []
                        if dur_str:
                            parts.append(dur_str)
                        if toks:
                            parts.append(f"{toks} tokens")
                        if tps:
                            parts.append(f"{tps:.1f} tok/s")
                        parts.append(f"done={done}")
                        if vram:
                            parts.append(f"{vram / (1024**3):.2f}GB VRAM")
                        metrics_str = ", ".join(parts)
                        sep = "  " if metrics_str else ""
                        progress.console.print(
                            f"  {prompt_id}: {status}{val_badge}{sep}"
                            f"[dim]{metrics_str}[/dim]"
                        )
                        preview = result.response.message.content[:200].replace("\n", " ")
                        progress.console.print(f"    [dim]{preview}...[/dim]")
                    else:
                        time_part = f"  [dim]{dur_str}[/dim]" if dur_str else ""
                        progress.console.print(f"  {prompt_id}: {status}{val_badge}{time_part}")
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

            written_path = result_path_for(result, output_dir)
            console.print(f"[green]Results written to {written_path}[/green]")

            if do_evaluate:
                eval_paths.append(written_path)

            # Print summary table
            _print_summary(result)
            console.print()

    if do_evaluate and eval_paths:
        console.rule("[bold]Evaluation[/bold]")
        _run_post_phase_evaluation(
            eval_paths=eval_paths,
            eval_backend_name=eval_backend,
            eval_model=eval_model,
            host=host,
            eval_timeout=eval_timeout,
            rubric_path=rubric_path,
            rubric_dir=rubric_dir,
            results=[],
        )


def _format_validation_badge(result) -> str:
    """Format the per-prompt validator outcome as a compact inline badge.

    Returns the empty string when the prompt has no validator (e.g. text-mode
    prompts) so non-tool-use suites are unaffected. For tool-use prompts the
    result is already in `result.validation_passed` from the per-prompt
    sandbox check that ran during inference — surfacing it inline lets users
    see pass/fail as it happens instead of only in the final summary.
    """
    if result is None:
        return ""
    passed = getattr(result, "validation_passed", None)
    if passed is None:
        return ""
    if passed:
        return " [bold green]\\[pass][/bold green]"
    reason = getattr(result, "validation_reason", "") or ""
    short_reason = reason.splitlines()[0][:60] if reason else ""
    suffix = f": {short_reason}" if short_reason else ""
    return f" [bold yellow]\\[val-fail][/bold yellow]{suffix}"


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
    positional_paths: Annotated[
        list[Path] | None,
        typer.Argument(help="Run result JSON files. Shell globs work on bash/zsh (e.g. `results/*.json`). Use --result/-r when shell expansion isn't available."),
    ] = None,
    result_paths: Annotated[
        list[Path] | None,
        typer.Option("--result", "-r", help="Run result JSON file(s) — explicit form, composes with positional args. Interactive picker if all omitted."),
    ] = None,
    rubric_path: Annotated[
        Path | None,
        typer.Option("--rubric", help="Path to a rubric YAML file. Auto-resolved per-result from the suite hint if omitted."),
    ] = None,
    evaluator_model: Annotated[
        str | None,
        typer.Option("--evaluator", "-e", envvar="PORCHBENCH_EVAL_MODEL", help="Judge model. Defaults: api=claude-sonnet-4-6, claude-code=sonnet. For ollama, prompts to pick from available models on first use and persists choice to .env."),
    ] = None,
    backend: Annotated[
        str,
        typer.Option("--backend", "-b", envvar="PORCHBENCH_EVAL_BACKEND", help="Evaluation backend: 'ollama' (default), 'api', or 'claude-code'."),
    ] = "ollama",
    host: Annotated[
        str | None,
        typer.Option("--host", "-H", help="Ollama server URL (for ollama backend)."),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for scorecard JSON files."),
    ] = Path("scorecards"),
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", envvar="ANTHROPIC_API_KEY", help="Anthropic API key (for api backend)."),
    ] = None,
    rubric_dir: Annotated[
        Path | None,
        typer.Option("--rubric-dir", help="Directory of category-specific rubrics (coding.yaml, reasoning.yaml, cross-domain.yaml)."),
    ] = None,
    eval_timeout: Annotated[
        int,
        typer.Option("--eval-timeout", help="Timeout in seconds per prompt evaluation (claude-code backend)."),
    ] = 120,
    skip_scored: Annotated[
        bool,
        typer.Option("--skip-scored", help="Skip result files that already have a scorecard in output-dir (matched by run_id prefix)."),
    ] = False,
) -> None:
    """Score one or more model responses for quality using an LLM-as-judge evaluator."""
    from porchbench.evaluator import (
        AnthropicEvalBackend,
        ClaudeCodeEvalBackend,
        OllamaEvalBackend,
        evaluate_run,
        load_calibration_examples,
        load_rubric,
        load_rubric_dir,
        write_scorecard,
    )

    merged_paths = (positional_paths or []) + (result_paths or [])

    # Interactive selection when args omitted
    if not merged_paths:
        from porchbench.interactive import select_results
        merged_paths = select_results()
    result_paths = merged_paths

    # ---- one-time setup (shared across all results) ----

    probe_backend = OllamaBackend(host=host) if backend == "ollama" else None
    evaluator_model = resolve_eval_model_or_exit(
        backend, evaluator_model, probe_backend, interactive=True,
    )
    if backend == "ollama":
        check_models_or_exit(probe_backend, [evaluator_model], "ollama")

    rubrics_by_category = None
    if rubric_dir:
        try:
            rubrics_by_category = load_rubric_dir(rubric_dir)
            console.print(f"Category rubrics: {', '.join(rubrics_by_category.keys())}")
        except Exception as exc:
            console.print(f"[yellow]Warning: could not load rubric dir: {exc}[/yellow]")

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

    is_batch = len(result_paths) > 1
    console.print(f"Evaluator: {backend_label}")
    if is_batch:
        console.print(f"Results to score: [bold]{len(result_paths)}[/bold]\n")

    # Cache rubric+calibration per resolved path so a shared rubric loads once
    rubric_cache: dict[Path, tuple] = {}

    def resolve_rubric_for(run_result: RunResult) -> tuple:
        rpath = rubric_path
        if rpath is None:
            hint = run_result.run.suite.rubric
            rpath = find_rubric(hint) if hint else find_rubric("default")
        if rpath in rubric_cache:
            return rubric_cache[rpath]
        loaded = load_rubric(rpath)
        cal_file = rpath.parent / "calibration-examples.yaml"
        cal = load_calibration_examples(cal_file) if cal_file.exists() else {}
        rubric_cache[rpath] = (rpath, loaded, cal)
        return rubric_cache[rpath]

    # ---- per-result loop ----

    summary: list[tuple[str, str, float | None]] = []  # (label, status, overall)

    for idx, rp in enumerate(result_paths, 1):
        prefix = f"[dim]({idx}/{len(result_paths)})[/dim] " if is_batch else ""
        console.print(f"{prefix}[bold]{rp.name}[/bold]")

        try:
            run_result = load_json_model(rp, RunResult, "run result")
        except UserError as exc:
            console.print(f"  [red]load failed: {exc}[/red]")
            summary.append((rp.name, "failed", None))
            continue

        run_label = f"{run_result.run.model.name} ({run_result.run.id[:8]})"

        # Tool-use runs are scored by sandbox validators, not LLM judges.
        # Refusing here prevents writing a misleading 0.00 scorecard from an
        # empty `scorable` list (evaluator filters out tool-call done_reasons).
        validator_results = [
            r for r in run_result.results if r.validation_passed is not None
        ]
        if validator_results and len(validator_results) == len(run_result.results):
            passed = sum(1 for r in validator_results if r.validation_passed)
            console.print(
                f"  [yellow]skipped — tool-use run, scored by validators "
                f"({passed}/{len(validator_results)} passed). LLM judging "
                f"does not apply.[/yellow]"
            )
            summary.append((run_label, "skipped", None))
            continue

        warn_if_same_family_judge(run_result.run.model.name, evaluator_model)

        if skip_scored:
            existing = list(output_dir.glob(f"*_{run_result.run.id[:8]}.json"))
            if existing:
                console.print(f"  [yellow]skipped (scorecard exists: {existing[0].name})[/yellow]")
                summary.append((run_label, "skipped", None))
                continue

        try:
            resolved_path, rubric, calibration_data = resolve_rubric_for(run_result)
        except Exception as exc:
            console.print(f"  [red]rubric resolution failed: {exc}[/red]")
            summary.append((run_label, "failed", None))
            continue

        if not is_batch:
            console.print(f"Run: [bold]{run_label}[/bold]")
            console.print(f"Rubric: {rubric.rubric.name} v{rubric.rubric.version}")
            console.print(f"Prompts to score: {len(run_result.results)}")
            if calibration_data:
                console.print(f"Calibration: {', '.join(calibration_data.keys())}")
            console.print()

        try:
            scorecard = asyncio.run(
                evaluate_run(
                    run_result, rubric, eval_backend,
                    evaluator_label=backend_label,
                    rubrics_by_category=rubrics_by_category,
                    calibration_data=calibration_data or None,
                )
            )
            written = write_scorecard(scorecard, output_dir)
        except Exception as exc:
            console.print(f"  [red]evaluation failed: {exc}[/red]")
            summary.append((run_label, "failed", None))
            continue

        overall = scorecard.aggregate.overall_weighted
        summary.append((run_label, "scored", overall))

        if is_batch:
            console.print(f"  [green]scored — {overall:.2f} → {written.name}[/green]")
        else:
            console.print(f"\n[green]Scorecard written to {written}[/green]")
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

    # ---- batch summary ----
    if is_batch:
        scored = sum(1 for _, s, _ in summary if s == "scored")
        skipped = sum(1 for _, s, _ in summary if s == "skipped")
        failed = sum(1 for _, s, _ in summary if s == "failed")

        table = Table(title="Batch Evaluation Summary", title_style="bold")
        table.add_column("Run")
        table.add_column("Status")
        table.add_column("Overall", justify="right")
        for run_label, status, overall in summary:
            overall_str = f"{overall:.2f}" if overall is not None else "—"
            table.add_row(run_label, status, overall_str)
        console.print()
        console.print(table)
        console.print(f"\n[bold]{scored} scored, {skipped} skipped, {failed} failed[/bold]")
        if failed > 0:
            raise typer.Exit(code=1)


@app.command()
def compare(
    positional_paths: Annotated[
        list[Path] | None,
        typer.Argument(help="Run result JSON files. Shell globs work on bash/zsh (e.g. `results/*.json`). Use --result/-r when shell expansion isn't available."),
    ] = None,
    result_paths: Annotated[
        list[Path] | None,
        typer.Option("--result", "-r", help="Run result JSON files to compare — explicit form, composes with positional args. Interactive picker if all omitted."),
    ] = None,
    scorecard_paths: Annotated[
        list[Path] | None,
        typer.Option("--scorecard", help="Scorecard JSON files (same order as results)."),
    ] = None,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            envvar="PORCHBENCH_SEED",
            help="RNG seed for bootstrap CIs. Fixed at 42 by default for reproducibility; override to probe sensitivity.",
        ),
    ] = 42,
) -> None:
    """Compare metrics and scores across models side-by-side."""
    from porchbench.compare import print_comparison_table

    merged_paths = (positional_paths or []) + (result_paths or [])

    # Interactive selection when args omitted
    if not merged_paths:
        from porchbench.interactive import select_results
        merged_paths = select_results()
    result_paths = merged_paths

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

    print_comparison_table(runs, scorecards, seed=seed)


@routes_app.command("discover")
def discover_routes(
    suite_path: Annotated[
        Path | None,
        typer.Option("--suite", "-s", help="Suite name (e.g. 'routing-discovery') or path to a YAML file. Interactive picker if omitted."),
    ] = None,
    models: Annotated[
        list[str] | None,
        typer.Option("--model", "-m", help="Model name(s). Repeat for each. Interactive picker if omitted."),
    ] = None,
    backend_name: Annotated[
        str,
        typer.Option("--backend", envvar="PORCHBENCH_BACKEND", help="Inference backend: 'ollama' or 'openai-compat'."),
    ] = "ollama",
    host: Annotated[
        str | None,
        typer.Option("--host", "-H", envvar="OLLAMA_HOST", help="Ollama server URL."),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", envvar="PORCHBENCH_BASE_URL", help="OpenAI-compat server URL."),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", envvar="PORCHBENCH_API_KEY", help="API key for OpenAI-compat servers."),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for result JSON files."),
    ] = Path("results"),
) -> None:
    """Run all prompt x strategy x model combinations to map capabilities."""
    from porchbench.interactive import select_models, select_suite
    from porchbench.routing import count_discovery_runs, run_discovery

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
        list[Path] | None,
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
    from porchbench.routing import analyze_routes

    # Interactive selection when args omitted
    if result_paths is None:
        from porchbench.interactive import select_results
        result_paths = select_results()

    runs = []
    for p in result_paths:
        try:
            runs.append(load_json_model(p, RunResult, "run result"))
        except UserError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1)

    non_routing = [r for r in runs if not any(pr.strategy for pr in r.prompt_results)]
    if non_routing:
        names = ", ".join(r.metadata.run_id for r in non_routing)
        console.print(
            f"[red]routes analyze requires results produced by `porchbench routes discover` "
            f"(prompts must carry a strategy tag). The following result files have no "
            f"routing strategies and cannot be analyzed: {names}[/red]"
        )
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
def leaderboard(
    positional_paths: Annotated[
        list[Path] | None,
        typer.Argument(help="Scorecard JSON files. Shell globs work on bash/zsh (e.g. `scorecards/*.json`)."),
    ] = None,
    scorecard_paths: Annotated[
        list[Path] | None,
        typer.Option("--scorecard", help="Scorecard JSON files — explicit form, composes with positional args."),
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
    from porchbench.leaderboard import (
        discover_scorecards,
        filter_comparable,
        group_scorecards,
        print_leaderboard,
    )

    merged_paths = (positional_paths or []) + (scorecard_paths or [])

    scorecards = []
    if merged_paths:
        for p in merged_paths:
            try:
                scorecards.append(load_json_model(p, Scorecard, "scorecard"))
            except UserError as exc:
                console.print(f"[red]{exc}[/red]")
                raise typer.Exit(code=1)
    else:
        if not scorecard_dir.is_dir():
            console.print(f"[red]Scorecard directory not found: {scorecard_dir}[/red]")
            console.print("  Run [bold]porchbench evaluate[/bold] on a run result first to produce a scorecard.")
            raise typer.Exit(code=1)
        scorecards = discover_scorecards(scorecard_dir, verbose=verbose)
    scorecard_paths = merged_paths if merged_paths else scorecard_paths

    if not scorecards:
        console.print("[yellow]No scorecards found.[/yellow]")
        console.print("  Run [bold]porchbench evaluate[/bold] on a run result first to produce a scorecard.")
        raise typer.Exit(code=1)

    # Interactive rubric group selection when multiple groups exist
    groups = group_scorecards(scorecards)
    if len(groups) > 1 and scorecard_paths is None:
        from porchbench.interactive import select_rubric_group
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
        Path | None,
        typer.Option("--output", "-o", help="Output path for extracted data JSON. Defaults to .claude/eval-data.json."),
    ] = None,
) -> None:
    """Pre-extract compact evaluation data from a run result file.

    Reads the full result JSON once and writes a lightweight file containing
    only prompt text, response text, expected answers, and metadata. Used by
    the /evaluate skill to avoid repeated partial reads of large result files.
    """
    from porchbench.evaluator import extract_eval_data

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
    from porchbench.evaluator import build_scorecard_from_scores

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
        list[str] | None,
        typer.Option("--model", "-m", help="Model name(s). Repeat for multiple. Interactive picker if omitted."),
    ] = None,
    suite_paths: Annotated[
        list[Path] | None,
        typer.Option("--suite", "-s", help="Suite names or YAML paths. Repeat for multiple. Omit to auto-discover."),
    ] = None,
    repeats: Annotated[
        int,
        typer.Option("--repeats", "-n", help="Repeats per standard suite (discovery suites expand by strategy)."),
    ] = 3,
    backend_name: Annotated[
        str,
        typer.Option("--backend", envvar="PORCHBENCH_BACKEND", help="Inference backend: 'ollama' or 'openai-compat'."),
    ] = "ollama",
    host: Annotated[
        str | None,
        typer.Option("--host", "-H", envvar="OLLAMA_HOST", help="Ollama server URL."),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", envvar="PORCHBENCH_BASE_URL", help="OpenAI-compat server URL."),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", envvar="PORCHBENCH_API_KEY", help="API key for OpenAI-compat servers."),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for result JSON files."),
    ] = Path("results"),
    suite_dir: Annotated[
        Path | None,
        typer.Option(
            "--suite-dir",
            help="Directory to auto-discover suites from. Defaults to ./suites if present, else the packaged suites bundled with porchbench.",
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
        typer.Option("--evaluate", help="Score all runs in a single post-phase batch after inference completes (judge model stays resident)."),
    ] = False,
    eval_backend: Annotated[
        str,
        typer.Option("--eval-backend", envvar="PORCHBENCH_EVAL_BACKEND", help="Evaluation backend: ollama, api, or claude-code."),
    ] = "ollama",
    eval_model: Annotated[
        str | None,
        typer.Option("--eval-model", envvar="PORCHBENCH_EVAL_MODEL", help="Judge model. Defaults per backend."),
    ] = None,
    rubric_path: Annotated[
        Path | None,
        typer.Option("--rubric", help="Rubric YAML for evaluation. Auto-resolved from suite if omitted."),
    ] = None,
    rubric_dir: Annotated[
        Path | None,
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
    set_overrides: Annotated[
        list[str] | None,
        typer.Option("--set", help="Override a suite default option as KEY=VALUE (e.g. --set think=false). Repeatable. Values parsed as YAML so booleans/ints/nulls round-trip."),
    ] = None,
) -> None:
    """Queue multiple suites and models for unattended batch benchmarking."""
    import time as _time

    from porchbench.overnight import (
        build_plan,
        execute_plan,
        print_plan,
        print_summary,
    )

    interactive = models is None or suite_paths is None

    # Interactive selection when args omitted
    if models is None:
        from porchbench.interactive import select_models
        backend = construct_backend(backend_name, host=host, base_url=base_url, api_key=api_key)
        check_server_or_exit(backend, backend_name)
        models = select_models(backend)

    # 1. Discover or use provided suites
    if suite_paths:
        paths = [find_suite(p) for p in suite_paths]
    else:
        from porchbench.interactive import select_suites
        paths = select_suites(resolve_suite_dir(suite_dir))

    # Interactive options screen
    if interactive:
        from porchbench.interactive import select_overnight_options
        opts = select_overnight_options(
            default_repeats=repeats,
            defaults={
                "evaluate": do_evaluate,
                "profile": do_profile,
                "profile_vram": profile_vram,
                "resume": resume,
                "verbose": verbose,
            },
        )
        repeats = opts["repeats"]
        do_evaluate = opts["evaluate"]
        do_profile = opts["profile"]
        profile_vram = opts["profile_vram"]
        resume = opts["resume"]
        verbose = opts["verbose"]

    overrides = parse_set_overrides(set_overrides)
    if overrides:
        console.print(f"Overrides: {overrides}")

    # 2. Build plan
    try:
        plan = build_plan(paths, models, repeats, option_overrides=overrides)
    except Exception as exc:
        console.print(f"[red]Failed to build plan: {exc}[/red]")
        raise typer.Exit(code=1)

    # 3. Display plan
    console.print()
    print_plan(plan, models, results_dir=output_dir)

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

    from porchbench.overnight import check_gpu_status, check_vram_cofit

    with console.status(f"  Warming up {models[0]} (loading model into VRAM)..."):
        gpu_ok, gpu_msg = asyncio.run(check_gpu_status(backend, models[0]))
    status = "[green]PASS[/green]" if gpu_ok else "[red]FAIL[/red]"
    console.print(f"  {status} GPU acceleration: {gpu_msg}")

    # Tool-calling capability check per (model, suite) — fail fast before
    # queueing hours of inference that would just produce 0/N validations.
    for task in plan:
        check_tool_support_or_exit(backend, task.models, task.suite, backend_name)

    # Resolve + validate the evaluator model before any inference work begins.
    # Without this, a missing eval model would surface only after hours of
    # inference, when post-phase scoring fails for every result.
    if do_evaluate:
        eval_model = resolve_eval_model_or_exit(
            eval_backend, eval_model, backend, interactive=not yes,
        )
        if eval_backend == "ollama":
            check_models_or_exit(backend, [eval_model], "ollama")

    # VRAM co-fit check — only meaningful when --evaluate will load a judge on the same GPU
    if do_evaluate and eval_backend == "ollama":
        cofit_ok, cofit_msg = asyncio.run(check_vram_cofit(backend, models, eval_model))
        if cofit_ok:
            console.print(f"  [green]PASS[/green] VRAM cofit: {cofit_msg}")
        else:
            console.print(f"  [yellow]WARN[/yellow] VRAM cofit: {cofit_msg}")

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
            from porchbench.profiler import print_profile_summary, profile_system, write_profile

            sys_profile = asyncio.run(profile_system(models, backend=backend))
            path = write_profile(sys_profile, output_dir)
            console.print(f"[green]Profile written to {path}[/green]\n")
            print_profile_summary(sys_profile)
            console.print()

    # 7. Announce (but don't start) post-run evaluation
    if do_evaluate:
        console.print(
            f"[bold]Post-run evaluation[/bold] will run after all inference completes "
            f"([cyan]{eval_backend}/{eval_model}[/cyan])."
        )
        console.print()

    # 8. Execute inference
    console.rule("[bold]Running benchmarks[/bold]")
    start = _time.monotonic()

    seen_models: set[str] = set()

    def on_start(task, model, repeat):
        repeat_str = f" repeat {repeat}/{task.repeats}" if repeat else ""
        console.rule(f"[bold]{task.suite.suite.name}[/bold] / {model}{repeat_str}")
        # Cold-start hint the first time we touch a model — ROCm/AMD in particular
        # spends minutes compiling compute-graph kernels on the first prompt of a
        # freshly-loaded model, and the CLI otherwise looks hung during that window.
        if model not in seen_models and model not in ("(all models)",):
            seen_models.add(model)
            console.print(
                "  [dim]First prompt can take several minutes while the model loads "
                "and kernels compile (common on ROCm/AMD).[/dim]"
            )

    def on_done(result):
        if result.success:
            path_part = f" → {result.result_path.name}" if result.result_path else ""
            console.print(f"  [green]Done[/green] ({result.duration_s:.0f}s){path_part}")
        else:
            console.print(f"  [red]Failed: {result.error}[/red]")

    def on_prompt_complete(prompt_id, success, duration_s, prompt_num, total, result=None):
        if duration_s >= 60:
            mins, secs = divmod(int(duration_s), 60)
            dur_str = f"{mins}m {secs:02d}s"
        else:
            dur_str = f"{duration_s:.1f}s"
        counter = f"{prompt_num}/{total}"
        val_badge = _format_validation_badge(result)
        if success:
            console.print(rf"  [green]\[ok][/green]{val_badge}    {counter}  {prompt_id}  ({dur_str})")
        else:
            err = ""
            if result is not None and result.response.done_reason:
                err = f" - {str(result.response.done_reason).splitlines()[0][:80]}"
            console.print(rf"  [red]\[fail][/red]  {counter}  {prompt_id}  (failed after {dur_str}{err})")

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
                on_prompt_complete=on_prompt_complete,
                profile_vram=profile_vram,
                heartbeat_s=60.0,
            )
        )
    except KeyboardInterrupt:
        elapsed = _time.monotonic() - start
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        console.print(f"Elapsed: {elapsed:.0f}s")
        raise typer.Exit(code=130)

    # 9. Post-run batch evaluation — single judge-model load for all results
    if do_evaluate:
        eval_paths = [r.result_path for r in results if r.success and r.result_path is not None]
        if eval_paths:
            console.rule("[bold]Evaluation[/bold]")
            _run_post_phase_evaluation(
                eval_paths=eval_paths,
                eval_backend_name=eval_backend,
                eval_model=eval_model,
                host=host,
                eval_timeout=eval_timeout,
                rubric_path=rubric_path,
                rubric_dir=rubric_dir,
                results=results,
            )

    # Capture elapsed after eval so the Overnight Complete duration covers
    # both inference and judge-model scoring.
    elapsed = _time.monotonic() - start

    # 10. Summary
    print_summary(results, elapsed)


def _run_post_phase_evaluation(
    eval_paths: list[Path],
    eval_backend_name: str,
    eval_model: str,
    host: str | None,
    eval_timeout: int,
    rubric_path: Path | None,
    rubric_dir: Path | None,
    results: list,
) -> None:
    """Batch-score all successful run results after inference completes.

    Holds the judge model resident once for the whole batch — no swap
    thrashing between target-model inference and judge-model scoring.
    Scores feed back into `OvernightResult.eval_score` so the final
    summary can show them. Caller is responsible for resolving `eval_model`
    via `resolve_eval_model_or_exit` before invoking this function.
    """
    from porchbench.evaluator import (
        AnthropicEvalBackend,
        ClaudeCodeEvalBackend,
        OllamaEvalBackend,
        batch_evaluate_results,
        load_rubric_dir,
    )

    if eval_backend_name == "ollama":
        eval_be = OllamaEvalBackend(model=eval_model, host=host)
    elif eval_backend_name == "api":
        eval_be = AnthropicEvalBackend(model=eval_model)
    elif eval_backend_name == "claude-code":
        eval_be = ClaudeCodeEvalBackend(model=eval_model, timeout_s=eval_timeout)
    else:
        console.print(f"[red]Unknown eval backend: {eval_backend_name}[/red]")
        return

    backend_label = f"{eval_backend_name}/{eval_model}"
    eval_rubrics_by_cat = load_rubric_dir(resolve_rubric_dir(rubric_dir))

    console.print(f"Evaluator: {backend_label}")
    console.print(f"Results to score: [bold]{len(eval_paths)}[/bold]\n")

    # Surface same-family judge bias upfront — one warn per unique target model
    # whose family root matches the judge. Prints before scoring so the user
    # sees the methodology caveat alongside the run-by-run progress.
    seen_targets: set[str] = set()
    for r in results:
        target = getattr(r, "model", None)
        if target and target not in seen_targets:
            warn_if_same_family_judge(target, eval_model)
            seen_targets.add(target)

    summary = asyncio.run(batch_evaluate_results(
        result_paths=eval_paths,
        eval_backend=eval_be,
        backend_label=backend_label,
        output_dir=Path("scorecards"),
        explicit_rubric_path=rubric_path,
        rubrics_by_category=eval_rubrics_by_cat,
    ))

    # Feed scores back into OvernightResults so print_summary can show them
    path_to_score = {}
    for (_, status, score), path in zip(summary, eval_paths):
        if status == "scored" and score is not None:
            path_to_score[path] = score
    for r in results:
        if r.result_path in path_to_score:
            r.eval_score = path_to_score[r.result_path]

    scored = sum(1 for _, s, _ in summary if s == "scored")
    failed = sum(1 for _, s, _ in summary if s == "failed")
    console.print(f"\n[bold]{scored} scored, {failed} failed[/bold]")


@app.command()
def doctor(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable JSON report instead of styled text."),
    ] = False,
    host: Annotated[
        str | None,
        typer.Option("--host", "-H", envvar="OLLAMA_HOST", help="Ollama host to probe."),
    ] = None,
) -> None:
    """Diagnose local environment: Python, Ollama, GPU, and package install state.

    Exits 0 if required checks pass (warnings allowed), 1 otherwise.
    Paste the `--json` output into bug reports for fast triage.
    """
    from porchbench.doctor import render_report, run_checks

    report = run_checks(host=host)

    if json_output:
        typer.echo(report.to_json())
    else:
        render_report(report, console)

    raise typer.Exit(code=0 if report.ok else 1)


app.add_typer(routes_app)


@app.command()
def profile(
    models: Annotated[
        list[str] | None,
        typer.Option("--model", "-m", help="Ollama model name(s) to profile. Interactive picker if omitted."),
    ] = None,
    backend_name: Annotated[
        str,
        typer.Option("--backend", envvar="PORCHBENCH_BACKEND", help="Inference backend (only 'ollama' supported for profiling)."),
    ] = "ollama",
    host: Annotated[
        str | None,
        typer.Option("--host", "-H", envvar="OLLAMA_HOST", help="Ollama server URL."),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for profile output."),
    ] = Path("results"),
) -> None:
    """Measure GPU memory, model load times, and swap costs (Ollama only)."""
    from porchbench.interactive import select_models
    from porchbench.profiler import print_profile_summary, profile_system, write_profile

    if backend_name != "ollama":
        console.print(
            f"[red]Profile requires Ollama backend (got '{backend_name}'). "
            f"VRAM and swap profiling are Ollama-specific.[/red]"
        )
        raise typer.Exit(code=1)

    backend = OllamaBackend(host=host)
    check_server_or_exit(backend, "ollama")

    if models is None:
        models = select_models(backend)

    sys_profile = asyncio.run(profile_system(models, backend=backend))

    path = write_profile(sys_profile, output_dir)
    console.print(f"\n[green]Profile written to {path}[/green]\n")

    print_profile_summary(sys_profile)


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
    except UserError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
