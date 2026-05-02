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
from porchbench.display import format_validation_badge
from porchbench.errors import UserError, load_json_model
from porchbench.runner import find_completed_prompt_ids, result_path_for, run_suite
from porchbench.schemas import PromptResult, RunResult, Scorecard
from porchbench.suite import load_suite, make_suite_reference

app = typer.Typer(
    name="porchbench",
    help="Deterministic benchmarking of local LLMs.",
    no_args_is_help=True,
)
console = Console()


# Migration breadcrumb for the removed `routes` subgroup. `routes discover`
# was consolidated into `overnight --strategies`; `routes analyze` was
# promoted to top-level `analyze-routes`. The shim costs ~20 LOC and
# saves users scripting against the old CLI from a bare "no such
# command" — pre-1.0 stance is no compatibility shim, but a migration
# message is good citizenship.
_routes_removed_app = typer.Typer(
    name="routes",
    help="Removed in v0.1 — see migration messages below.",
    hidden=True,
)


@_routes_removed_app.command("discover")
def _routes_discover_removed() -> None:
    console.print(
        "[yellow]`routes discover` was removed in v0.1.[/yellow] "
        "Use [bold]porchbench run --strategies[/bold] for the same matrix expansion."
    )
    raise typer.Exit(code=2)


@_routes_removed_app.command("analyze")
def _routes_analyze_removed() -> None:
    console.print(
        "[yellow]`routes analyze` was renamed in v0.1.[/yellow] "
        "Use [bold]porchbench analyze-routes[/bold] (now a top-level command)."
    )
    raise typer.Exit(code=2)


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


def _is_routing_discovery_result(path: Path) -> bool:
    """Cheap content check: does this result file carry strategy tags?

    Filename alone is ambiguous — a suite literally named "routing-discovery"
    run via regular `porchbench run` produces filename-matching files that
    have no per-prompt `strategy` tag. We pre-filter on the conventional
    filename substring (free) and content-check survivors via raw
    `json.loads` on the first prompt result (~milliseconds per file).
    """
    if "_routing-discovery_" not in path.name:
        return False
    try:
        import json as _json
        data = _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    results = data.get("results") or []
    return bool(results) and results[0].get("strategy") is not None


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

    Defense-in-depth for the CLI-args path (no picker involved). The
    interactive picker already badges/sorts missing-cap models, but a
    user passing `-m medgemma:4b -s tool-use` from the command line
    bypasses the picker — without this, that run would queue and waste
    minutes producing a 0/19 score. Capability is read from Ollama's
    `client.show(model).capabilities` array — only checked for ollama;
    other backends silently skip (no portable capability probe).
    """
    from porchbench.suite import required_capabilities_for_suite

    if backend_name != "ollama":
        return
    if not isinstance(backend, OllamaBackend):
        return
    needs = required_capabilities_for_suite(suite)
    if "tools" not in needs:
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
    from porchbench.suite import required_capabilities_for_suite

    interactive = models is None or suite_path is None

    # Suite-first ordering: knowing the suite lets the model picker badge
    # missing-capability models (e.g. tag a non-tools model as
    # `· missing: tools` for a tool-use suite) at selection time, instead
    # of failing later in the preflight after the user already configured
    # repeats / verbose / etc.
    if suite_path is None:
        from porchbench.interactive import select_suite
        suite_path = select_suite()

    # Resolve bare names and relative paths against cwd/packaged defaults
    try:
        suite_path = find_suite(suite_path)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    # Load and validate suite (early — gates the model picker on caps)
    try:
        suite = load_suite(suite_path)
    except Exception as exc:
        console.print(f"[red]Failed to load suite: {exc}[/red]")
        raise typer.Exit(code=1)

    if models is None:
        from porchbench.interactive import select_models
        backend = construct_backend(backend_name, host=host, base_url=base_url, api_key=api_key)
        check_server_or_exit(backend, backend_name)
        models = select_models(
            backend,
            required_capabilities=required_capabilities_for_suite(suite),
        )

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

            # Resolve the actual prompt list before opening the Progress bar.
            # Doing this here (rather than inside run_suite) keeps the resume
            # message and the bar from interleaving — Rich's live redraw was
            # repainting "0/28" both before and after the filter message —
            # and lets us short-circuit the no-op case cleanly without
            # writing an empty result JSON.
            prompts_to_run = list(suite.prompts)
            if prompt_ids:
                id_set = set(prompt_ids)
                prompts_to_run = [p for p in prompts_to_run if p.id in id_set]
                missing = id_set - {p.id for p in prompts_to_run}
                if missing:
                    console.print(
                        f"[yellow]Warning: prompt IDs not found in suite: {missing}[/yellow]"
                    )

            if resume:
                already_done = find_completed_prompt_ids(
                    suite_ref.name, model, Path(output_dir),
                )
                before = len(prompts_to_run)
                prompts_to_run = [p for p in prompts_to_run if p.id not in already_done]
                skipped = before - len(prompts_to_run)
                if skipped:
                    console.print(
                        f"[dim]Resuming: skipping {skipped} already-completed "
                        f"prompts[/dim]"
                    )

            if not prompts_to_run:
                console.print(
                    f"[dim]Nothing to run for {model} — all prompts already "
                    f"completed.[/dim]"
                )
                console.print()
                continue

            prompt_count = len(prompts_to_run)
            run_prompt_ids = [p.id for p in prompts_to_run]

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Running {model}", total=prompt_count)

                def on_complete(prompt_id: str, success: bool, duration_s: float, prompt_num: int, total: int, result: PromptResult | None = None) -> None:
                    status = "[green]ok[/green]" if success else "[red]FAIL[/red]"
                    dur = result.metrics.total_duration if result else None
                    dur_str = f"{dur / 1e9:.1f}s" if dur else ""
                    val_badge = format_validation_badge(result)

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
                        sep = " " if metrics_str else ""
                        progress.console.print(
                            f"  {prompt_id}: {status}{val_badge}{sep}"
                            f"[dim]{metrics_str}[/dim]"
                        )
                        preview = result.response.message.content[:200].replace("\n", " ")
                        progress.console.print(f"    [dim]{preview}...[/dim]")
                    else:
                        time_part = f" [dim]{dur_str}[/dim]" if dur_str else ""
                        progress.console.print(f"  {prompt_id}: {status}{val_badge}{time_part}")
                    progress.advance(task)

                result = asyncio.run(
                    run_suite(
                        suite=suite,
                        suite_ref=suite_ref,
                        model=model,
                        backend=backend,
                        prompt_ids=run_prompt_ids,
                        output_dir=output_dir,
                        on_prompt_complete=on_complete,
                        suite_dir=suite_path.parent,
                        repeat_index=repeat_i if repeats > 1 else None,
                        total_repeats=repeats if repeats > 1 else None,
                        resume=False,
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


def _print_summary(result: RunResult) -> None:
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

    # ---- one-time setup (shared across all results) ----
    # Resolve judge BEFORE the results picker so users see which model will
    # score their selection upfront (and can Ctrl+C to override before
    # picking 30 result files). Also makes the order conceptually clean:
    # set up the judge first, then choose what to judge.
    probe_backend = OllamaBackend(host=host) if backend == "ollama" else None
    evaluator_model = resolve_eval_model_or_exit(
        backend, evaluator_model, probe_backend, interactive=True,
    )
    if backend == "ollama":
        check_models_or_exit(probe_backend, [evaluator_model], "ollama")
    console.print(f"Evaluator: [cyan]{backend}/{evaluator_model}[/cyan]")
    console.print(
        "  [dim](override with --evaluator <name> or set PORCHBENCH_EVAL_MODEL)[/dim]"
    )

    # Interactive selection when args omitted
    if not merged_paths:
        from porchbench.interactive import select_results
        merged_paths = select_results()
    result_paths = merged_paths

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
        typer.Option("--scorecard", help="Scorecard JSON files (same order as results). Auto-discovered from scorecard_dir by run_id when omitted."),
    ] = None,
    scorecard_dir: Annotated[
        Path,
        typer.Option("--scorecard-dir", help="Directory to auto-discover scorecards in when --scorecard is not given."),
    ] = Path("scorecards"),
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
    elif scorecard_dir.is_dir():
        # Auto-discover scorecards by run_id prefix — saves users from
        # hand-pairing -r and --scorecard arguments. Scorecards are named
        # `{ts}_{run_id[:8]}.json` so a glob on the prefix is unambiguous.
        scorecards = []
        found_any = False
        for run in runs:
            prefix = run.run.id[:8]
            matches = sorted(scorecard_dir.glob(f"*_{prefix}.json"))
            if matches:
                try:
                    scorecards.append(load_json_model(matches[0], Scorecard, "scorecard"))
                    found_any = True
                except UserError as exc:
                    console.print(f"[yellow]Warning: {exc}[/yellow]")
                    scorecards.append(None)
            else:
                scorecards.append(None)
        if not found_any:
            scorecards = None  # nothing to add, drop the all-None list
        elif any(sc is None for sc in scorecards):
            unscored = [
                runs[i].run.model.name for i, sc in enumerate(scorecards) if sc is None
            ]
            console.print(
                f"[dim]Note: no scorecard found in {scorecard_dir}/ for: "
                f"{', '.join(unscored)}. Run [bold]porchbench evaluate[/bold] "
                f"on the corresponding result(s) to populate score columns.[/dim]"
            )

    print_comparison_table(runs, scorecards, seed=seed)


@app.command("analyze-routes")
def analyze_routes_cmd(
    result_paths: Annotated[
        list[Path] | None,
        typer.Option("--result", "-r", help="Result files from `overnight --strategies`. Interactive picker if omitted."),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for analysis output."),
    ] = Path("results"),
    summary_only: Annotated[
        bool,
        typer.Option("--summary", help="Print summary only, don't write full analysis."),
    ] = False,
    default_strategy: Annotated[
        str | None,
        typer.Option(
            "--default-strategy",
            help=(
                "Baseline strategy to compare other strategies against. "
                "Defaults to 'universal' if present, else the first strategy alphabetically."
            ),
        ),
    ] = None,
) -> None:
    """Analyze `overnight --strategies` results to find optimal routing strategies."""
    from porchbench.routing import analyze_routes

    # Interactive selection when args omitted. Prefix with a usage hint so
    # users don't pick one file, hit the "needs at least 2 distinct models"
    # refusal, and have to start the picker over. Filter the picker to
    # files that actually carry strategy tags — filename check alone
    # produces false positives when a suite is literally named
    # "routing-discovery" and got run via regular `porchbench run`.
    if result_paths is None:
        from porchbench.interactive import select_results
        console.print(
            "[bold]analyze-routes[/bold] is cross-model — "
            "pick [bold]≥2 routing-discovery results from different models[/bold] "
            "(same suite, run via [cyan]porchbench overnight --strategies[/cyan])."
        )
        result_paths = select_results(
            filter_predicate=_is_routing_discovery_result,
            filter_description="routing-discovery result",
        )

    runs: list[RunResult] = []
    paths_by_run_id: dict[str, Path] = {}
    for p in result_paths:
        try:
            run = load_json_model(p, RunResult, "run result")
        except UserError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1)
        runs.append(run)
        paths_by_run_id[run.run.id] = p

    non_routing = [r for r in runs if not any(pr.strategy for pr in r.results)]
    if non_routing:
        # Name the offending file + model — the run_id alone is opaque and
        # forces users to grep their results/ to figure out which file to
        # remove. Likely cause is `porchbench run -s <suite>` where the
        # suite is named "routing-discovery"; suggest the fix inline.
        offenders = "\n  ".join(
            f"{paths_by_run_id[r.run.id].name} ({r.run.model.name})"
            for r in non_routing
        )
        console.print(
            f"[red]analyze-routes requires results produced by `porchbench overnight --strategies` "
            f"(prompts must carry a strategy tag). These files have no strategy data:\n  "
            f"{offenders}\n"
            f"If you intended to run a strategy matrix, use `porchbench overnight --strategies "
            f"-s <suite>` (not `porchbench run`).[/red]"
        )
        raise typer.Exit(code=1)

    # Routing analysis is fundamentally cross-model: "for prompt P, is model A
    # the right pick over model B?" With a single model every routing metric
    # is degenerate (no inverse scaling, no routing-helps count, no
    # comparison cells), so the report misleads more than it helps. Refuse
    # explicitly and point at how to add another model.
    distinct_models = {r.run.model.name for r in runs}
    if len(distinct_models) < 2:
        only = next(iter(distinct_models)) if distinct_models else "<none>"
        console.print(
            f"[red]analyze-routes needs at least 2 distinct models to compare; "
            f"got 1 ({only}). Re-run `porchbench overnight --strategies` with "
            f"`-m <model> -m <other-model>` and pass both result files to "
            f"`analyze-routes`.[/red]"
        )
        raise typer.Exit(code=1)

    # Resolve baseline strategy. Without an explicit flag we prefer "universal"
    # (the conventional empty-system-message baseline used by the bundled
    # tool-use suite) and fall back to the first strategy alphabetically.
    # Validate against what's actually in the runs so a typo or removed
    # strategy fails loudly instead of silently zeroing every comparison.
    available_strategies = sorted({pr.strategy for r in runs for pr in r.results if pr.strategy})
    if default_strategy is None:
        default_strategy = "universal" if "universal" in available_strategies else available_strategies[0]
    elif default_strategy not in available_strategies:
        console.print(
            f"[red]--default-strategy '{default_strategy}' not present in result data. "
            f"Available strategies: {', '.join(available_strategies)}.[/red]"
        )
        raise typer.Exit(code=1)

    analysis = analyze_routes(runs, default_strategy=default_strategy)

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
    # The default model is the same across every BestRoute that has a
    # vs_default comparison — pluck it from the first one so the verdict
    # names which model "use the default" actually means.
    default_model = next(
        (br.vs_default.default_model for br in analysis.best_route_per_problem if br.vs_default),
        None,
    )
    fallback_label = (
        f"Use default model ({default_model})" if default_model else "Use default model"
    )
    console.print(
        f"\n[bold]Verdict[/bold]: {'Route' if v.routing_recommended else fallback_label}"
    )
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
        typer.Option("--strict", help="Require same evaluator in addition to same rubric. Auto-picks the largest evaluator group when multiple exist; use --evaluator to pin a specific judge."),
    ] = False,
    evaluator_filter: Annotated[
        str | None,
        typer.Option("--evaluator", help="Filter to scorecards from this judge (e.g. 'ollama/phi4:14b'). Implies --strict. Use when you want to compare under a specific judge instead of the default largest-group pick."),
    ] = None,
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
    comparable = filter_comparable(
        selected, strict=strict, evaluator=evaluator_filter,
    )

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
        typer.Option("--repeats", "-n", help="Repeats per suite (ignored when --strategies is set; matrix expansion replaces repeats)."),
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
    expand_strategies: Annotated[
        bool,
        typer.Option(
            "--strategies",
            help=(
                "Expand each suite across all its strategies (prompt × strategy × model matrix). "
                "Without this, overnight runs the baseline (one row per prompt). "
                "Replaces the standalone `routes discover` command."
            ),
        ),
    ] = False,
) -> None:
    """Queue multiple suites and models for unattended batch benchmarking."""
    import time as _time

    from porchbench.overnight import (
        build_plan,
        execute_plan,
        print_plan,
        print_summary,
    )

    from porchbench.suite import required_capabilities_for_suite

    interactive = models is None or suite_paths is None

    # Suite-first ordering: derive the union of required capabilities
    # across all selected suites so the model picker can mark missing-cap
    # models. Mirrors `run` flow.
    if suite_paths:
        paths = [find_suite(p) for p in suite_paths]
    else:
        from porchbench.interactive import select_suites
        paths = select_suites(resolve_suite_dir(suite_dir))

    # Validate --strategies against the selected suites BEFORE the model
    # picker fires. Single-suite + flag set + suite has no strategies block
    # = unambiguously wrong intent; hard-fail now so the user doesn't waste
    # time picking models for a doomed run. Multi-suite mixed selection is
    # legitimate (some suites expand, some baseline) — emit a per-suite
    # warning and continue.
    from porchbench.suite import suite_has_strategies
    if expand_strategies:
        without_strategies = [p for p in paths if not suite_has_strategies(p)]
        if without_strategies and len(paths) == 1:
            console.print(
                f"[red]--strategies requested but {paths[0].name} has no `strategies:` block.[/red]\n"
                f"  This suite has nothing to expand. Either drop --strategies (run baseline), "
                f"or pick a strategies-bearing suite (e.g. routing-discovery, tool-use)."
            )
            raise typer.Exit(code=1)
        for p in without_strategies:
            console.print(
                f"[yellow]Note:[/yellow] {p.name} has no strategies block — "
                f"this suite will run baseline; --strategies applies only to "
                f"strategies-bearing suites in this plan."
            )

    if models is None:
        from porchbench.interactive import select_models

        # Pre-load each suite to compute the union of required capabilities.
        # `build_plan` re-loads them later (negligible cost vs. restructuring
        # build_plan to accept already-loaded suites). Per-suite load
        # failures are swallowed here — build_plan surfaces them with its
        # own error path so we don't double-warn.
        required_caps_set: set[str] = set()
        for p in paths:
            try:
                required_caps_set.update(
                    required_capabilities_for_suite(load_suite(p))
                )
            except Exception:
                continue
        required_caps = sorted(required_caps_set)

        backend = construct_backend(backend_name, host=host, base_url=base_url, api_key=api_key)
        check_server_or_exit(backend, backend_name)
        models = select_models(backend, required_capabilities=required_caps)

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
                "strategies": expand_strategies,
            },
        )
        repeats = opts["repeats"]
        do_evaluate = opts["evaluate"]
        do_profile = opts["profile"]
        profile_vram = opts["profile_vram"]
        resume = opts["resume"]
        verbose = opts["verbose"]
        # If interactive options changed the toggle, re-validate before the
        # plan builds — covers users who flip --strategies on in the menu
        # against a non-strategies suite. Same hard-fail semantics as above.
        if opts["strategies"] and not expand_strategies:
            without_strategies = [p for p in paths if not suite_has_strategies(p)]
            if without_strategies and len(paths) == 1:
                console.print(
                    f"[red]--strategies requested via toggle but {paths[0].name} has no `strategies:` block.[/red]"
                )
                raise typer.Exit(code=1)
            for p in without_strategies:
                console.print(
                    f"[yellow]Note:[/yellow] {p.name} has no strategies block — baseline only."
                )
        expand_strategies = opts["strategies"]

    overrides = parse_set_overrides(set_overrides)
    if overrides:
        console.print(f"Overrides: {overrides}")

    # 2. Build plan
    try:
        plan = build_plan(
            paths, models, repeats,
            option_overrides=overrides,
            expand_strategies=expand_strategies,
        )
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

    # Verify every target model exists on the server before doing anything
    # expensive. Without this, a typo or comma-as-separator mistake (e.g.
    # `-m a:7b,b:3b` instead of `-m a:7b -m b:3b`) silently produces
    # `<N> error: invalid model name (status code: 400)` for every prompt
    # under --yes, wasting hours on guaranteed-to-fail inference.
    check_models_or_exit(backend, models, backend_name)

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
        # Echo the resolved evaluator before the cofit check that depends
        # on it — without this, the cofit message ("target + eval fit in
        # VRAM (5.0 + 8.9 GB ...)") references an "eval" without naming
        # which model the judge will be. Mirrors the `evaluate` command's
        # `Evaluator: ollama/<judge>` preflight line.
        console.print(
            f"  [green]PASS[/green] Evaluator: [bold]{eval_backend}/{eval_model}[/bold]"
        )

    # VRAM co-fit check — only meaningful when --evaluate will load a judge on the same GPU
    if do_evaluate and eval_backend == "ollama":
        cofit_ok, cofit_msg = asyncio.run(check_vram_cofit(backend, models, eval_model))
        if cofit_ok:
            console.print(f"  [green]PASS[/green] VRAM cofit: {cofit_msg}")
        else:
            console.print(f"  [yellow]WARN[/yellow] VRAM cofit: {cofit_msg}")

    console.print()

    # The pickers + plan table + preflight PASS lines already serve as
    # visible confirmation. Ctrl-C is the off-ramp. `--yes/-y` retains its
    # role gating *other* interactive prompts (eval-model picker fallback).

    # 5. Optional profiling (Ollama only)
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

    def on_start(task, model, repeat):
        repeat_str = f" repeat {repeat}/{task.repeats}" if repeat else ""
        console.rule(f"[bold]{task.suite.suite.name}[/bold] / {model}{repeat_str}")

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
        val_badge = format_validation_badge(result)
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


# Migration shim — see `_routes_removed_app` definition near the top.
app.add_typer(_routes_removed_app)


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
    # Windows captured-output (pipes, file redirects, CI logs) defaults
    # sys.stdout/stderr to cp1252, which can't encode the Unicode
    # box-drawing, em-dash, and sparkline characters Rich emits — leaderboard
    # mid-table renders crashed with UnicodeEncodeError on `█` (U+2588).
    # Reconfigure to UTF-8 with replacement fallback so captured runs render
    # the same as interactive terminal runs. Guarded by hasattr so pytest
    # capfd / capsys replacement streams are left alone.
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
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
