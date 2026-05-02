"""Interactive model and suite selection for CLI commands.

Uses beaupy for arrow-key driven pickers when --model or --suite
arguments are omitted from CLI invocations.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import typer
from beaupy import prompt, select, select_multiple
from rich.console import Console

from porchbench.backend import InferenceBackend
from porchbench.suite import discover_suites

console = Console()


def _prompt_models_manually() -> list[str]:
    """Fall back to text input when model discovery isn't available."""
    console.print("[yellow]Could not discover models from server.[/yellow]")
    console.print("[bold]Enter model name(s)[/bold] (empty line to finish):")
    models: list[str] = []
    while True:
        value = prompt("  Model: ", raise_type_conversion_fail=False)
        if not value or not value.strip():
            break
        models.append(value.strip())
    if not models:
        console.print("[red]No models entered.[/red]")
        raise typer.Exit(code=1)
    return models


def _format_model_label(
    name: str,
    caps: list[str],
    required: list[str] | None,
) -> tuple[str, bool]:
    """Render a picker label for one model and report whether it qualifies.

    Returns (label, qualifies). `qualifies` is True when the model has
    every required capability (or when no requirement was given). Labels:

      no caps known            : "model:tag"
      caps, no requirement     : "model:tag  [tools, vision]"
      caps, requirement met    : "model:tag  [tools, vision]"
      caps, requirement missing: "model:tag  [vision]  · missing: tools"

    The `· missing: <caps>` marker is plain text rather than ANSI red
    because beaupy renders option strings verbatim without rich markup;
    a textual marker survives the picker without garbling.
    """
    badge = f"  [{', '.join(caps)}]" if caps else ""
    if not required:
        return f"{name}{badge}", True
    cap_set = set(caps)
    missing = [c for c in required if c not in cap_set]
    if missing:
        return f"{name}{badge}  · missing: {', '.join(missing)}", False
    return f"{name}{badge}", True


def select_models(
    backend: InferenceBackend,
    required_capabilities: list[str] | None = None,
) -> list[str]:
    """Prompt user to pick one or more models from the backend's available list.

    When `required_capabilities` is set (e.g. ["tools"] for a tool-use suite),
    models lacking any required capability are sorted to the bottom of the
    picker and tagged `· missing: <caps>` so the user sees the mismatch at
    selection time rather than after configuring options. The
    `check_tool_support_or_exit` preflight still hard-fails if a missing-cap
    model is selected anyway — defense-in-depth for direct CLI-args use.

    Falls back to manual text entry when the server can't be queried for a
    model list (e.g. some openai-compat servers). When the server is reachable
    but reports zero models, exits with a pull hint rather than asking the
    user to type a name they don't have pulled.
    """
    try:
        catalog = asyncio.run(backend.list_available_models_with_capabilities())
    except Exception:
        return _prompt_models_manually()

    if not catalog:
        console.print(
            "[red]No models available on the server.[/red]\n"
            "Run [bold]ollama pull <model>[/bold] (e.g. qwen3:8b) and retry."
        )
        raise typer.Exit(code=1)

    decorated = [
        (name, *_format_model_label(name, caps, required_capabilities))
        for name, caps in catalog
    ]
    # Qualifying models first (preserve alpha order from the backend),
    # missing-cap models pushed to the bottom — keeps the relevant
    # subset directly under the cursor on first open.
    decorated.sort(key=lambda row: (not row[2], row[0]))
    labels = [row[1] for row in decorated]
    names = [row[0] for row in decorated]

    if required_capabilities:
        console.print(
            f"[bold]Select model(s)[/bold] (space to toggle, enter to confirm) "
            f"[dim]· suite needs: {', '.join(required_capabilities)}[/dim]:"
        )
    else:
        console.print("[bold]Select model(s)[/bold] (space to toggle, enter to confirm):")
    # return_indices avoids parsing model names back out of decorated
    # labels — the same fragility that bit `_discover_result_files`
    # before it switched to indices.
    selected_idx = select_multiple(
        options=labels,
        minimal_count=1,
        pagination=len(labels) > 15,
        page_size=15,
        return_indices=True,
    )
    if not selected_idx:
        console.print("[red]No models selected.[/red]")
        raise typer.Exit(code=1)

    return [names[i] for i in selected_idx]


def select_evaluator_model(backend: InferenceBackend) -> str:
    """Pick a single model from the server's available list to use as LLM-as-judge."""
    try:
        models = asyncio.run(backend.list_available_models())
    except Exception:
        models = []

    if not models:
        console.print(
            "[red]No models available to use as evaluator.[/red]\n"
            "Run [bold]ollama pull <model>[/bold] (e.g. gemma3:4b) and retry."
        )
        raise typer.Exit(code=1)

    console.print("[bold]Pick a model to use as evaluator (LLM-as-judge):[/bold]")
    chosen = select(
        options=models,
        pagination=len(models) > 15,
        page_size=15,
    )
    if chosen is None:
        console.print("[red]No evaluator selected.[/red]")
        raise typer.Exit(code=1)

    return chosen


def select_suite(
    suite_dir: Path | None = None,
    filter_predicate: Callable[[Path], bool] | None = None,
    filter_description: str = "suite",
) -> Path:
    """Prompt user to pick a suite YAML file from the suite directory.

    `filter_predicate`, when provided, narrows the picker to suites for which
    it returns True. `filter_description` names the filtered class for the
    "no suites matched" error (e.g. "suite with strategies"). Used by
    commands like `routes discover` whose semantics require a suite property
    (e.g. a non-empty `strategies:` block).
    """
    from porchbench.assets import resolve_suite_dir

    suite_dir = resolve_suite_dir(suite_dir)
    try:
        paths = discover_suites(suite_dir)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    if filter_predicate is not None:
        before = len(paths)
        paths = [p for p in paths if filter_predicate(p)]
        if not paths:
            console.print(
                f"[red]No {filter_description} found in {suite_dir}/ "
                f"(scanned {before} suite{'s' if before != 1 else ''}).[/red]"
            )
            raise typer.Exit(code=1)

    labels = [p.name for p in paths]

    console.print("[bold]Select a suite:[/bold]")
    chosen = select(
        options=labels,
        pagination=len(labels) > 15,
        page_size=15,
    )
    if chosen is None:
        console.print("[red]No suite selected.[/red]")
        raise typer.Exit(code=1)

    return suite_dir / chosen


def select_suites(suite_dir: Path | None = None) -> list[Path]:
    """Prompt user to pick one or more suite YAML files from the suite directory."""
    from porchbench.assets import resolve_suite_dir

    suite_dir = resolve_suite_dir(suite_dir)
    try:
        paths = discover_suites(suite_dir)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    labels = [p.name for p in paths]

    console.print("[bold]Select suite(s)[/bold] (space to toggle, enter to confirm):")
    selected = select_multiple(
        options=labels,
        minimal_count=1,
        pagination=len(labels) > 15,
        page_size=15,
    )
    if not selected:
        console.print("[red]No suites selected.[/red]")
        raise typer.Exit(code=1)

    return [suite_dir / name for name in selected]


def _discover_result_files(result_dir: Path) -> list[tuple[str, Path]]:
    """Scan result_dir for run-result JSONs, return (label, path) sorted newest-first.

    Skips non-RunResult JSONs that share the directory (system profiles,
    routing-analysis files) — they parse but lack `run.model.name` /
    `run.suite.name`, so picker users would otherwise see a `? — ? v ()`
    entry for each one.

    Labels include date + time of day. Date alone (YYYY-MM-DD) caused
    same-day same-model runs to share an identical label, which made the
    picker's `labels.index(s)` lookup collapse every duplicate-label
    selection to the first file's path — silently rendering N identical
    columns in `compare`. Including HH:MM keeps human-readable while
    guaranteeing within-minute uniqueness; the run-id suffix is the
    final tie-breaker for the rare same-minute case.
    """
    entries: list[tuple[str, Path]] = []
    seen_labels: dict[str, int] = {}
    for p in sorted(result_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            run = data.get("run") or {}
            model = (run.get("model") or {}).get("name")
            suite = run.get("suite") or {}
            suite_name = suite.get("name")
            if not model or not suite_name:
                continue
            suite_ver = suite.get("version", "")
            ts_iso = run.get("timestamp", "")
            # 'YYYY-MM-DDTHH:MM:SS...' → 'YYYY-MM-DD HH:MM' (16 chars, T→space)
            ts_human = ts_iso[:16].replace("T", " ") if ts_iso else ""
            run_id = run.get("id", "") or ""
            ts_part = f" ({ts_human})" if ts_human else ""
            label = f"{model} — {suite_name} v{suite_ver}{ts_part}"
            # Defensive disambiguation for the rare same-minute case: append
            # a short run-id suffix when a label would otherwise collide.
            if label in seen_labels and run_id:
                label = f"{label} [{run_id[:6]}]"
            seen_labels[label] = seen_labels.get(label, 0) + 1
            entries.append((label, p))
        except Exception:
            continue
    return entries


def select_result(result_dir: Path = Path("results")) -> Path:
    """Prompt user to pick a single run-result file."""
    if not result_dir.is_dir():
        console.print(f"[red]Results directory not found: {result_dir}[/red]")
        raise typer.Exit(code=1)

    entries = _discover_result_files(result_dir)
    if not entries:
        console.print(f"[red]No result files found in {result_dir}/[/red]")
        raise typer.Exit(code=1)

    labels = [label for label, _ in entries]
    console.print("[bold]Select a result:[/bold]")
    # return_index avoids labels.index() — which silently returns the first
    # match when duplicate labels exist, collapsing distinct selections to
    # the same path.
    chosen_idx = select(
        options=labels,
        pagination=len(labels) > 15,
        page_size=15,
        return_index=True,
    )
    if chosen_idx is None:
        console.print("[red]No result selected.[/red]")
        raise typer.Exit(code=1)

    return entries[chosen_idx][1]


def select_results(
    result_dir: Path = Path("results"),
    filter_predicate: Callable[[Path], bool] | None = None,
    filter_description: str = "result",
) -> list[Path]:
    """Prompt user to pick one or more run-result files.

    `filter_predicate`, when provided, narrows the picker to files for which
    it returns True. `filter_description` names the filtered class for the
    "no files matched" error (e.g. "routing-discovery result"). Used by
    commands like `routes analyze` that only operate on a subset of run
    JSONs and want to avoid the user picking an incompatible file.
    """
    if not result_dir.is_dir():
        console.print(f"[red]Results directory not found: {result_dir}[/red]")
        raise typer.Exit(code=1)

    entries = _discover_result_files(result_dir)
    if not entries:
        console.print(f"[red]No result files found in {result_dir}/[/red]")
        raise typer.Exit(code=1)

    if filter_predicate is not None:
        before = len(entries)
        entries = [(label, path) for label, path in entries if filter_predicate(path)]
        if not entries:
            console.print(
                f"[red]No {filter_description} files found in {result_dir}/ "
                f"(scanned {before} result file{'s' if before != 1 else ''}).[/red]"
            )
            raise typer.Exit(code=1)

    labels = [label for label, _ in entries]
    console.print("[bold]Select result(s)[/bold] (space to toggle, enter to confirm):")
    # return_indices avoids the labels.index() bug — when two result files
    # share a label (e.g. same model, same suite, same minute), looking up
    # by string equality always returns the first index, which collapsed
    # multi-selections of distinct files to N copies of the first file in
    # downstream commands like compare.
    selected_idx = select_multiple(
        options=labels,
        minimal_count=1,
        pagination=len(labels) > 15,
        page_size=15,
        return_indices=True,
    )
    if not selected_idx:
        console.print("[red]No results selected.[/red]")
        raise typer.Exit(code=1)

    return [entries[i][1] for i in selected_idx]


# ---------------------------------------------------------------------------
# Scorecard / rubric group selection
# ---------------------------------------------------------------------------


def select_rubric_group(
    groups: dict[str, list],
) -> list:
    """Prompt user to pick a rubric group when multiple exist.

    groups is a dict of normalized_rubric_key -> list[Scorecard].
    Returns the selected group's scorecards. If only one group, returns it directly.
    """
    if len(groups) == 1:
        return list(groups.values())[0]

    labels = [f"{key} ({len(scs)} scorecards)" for key, scs in groups.items()]
    keys = list(groups.keys())

    console.print("[bold]Multiple rubric groups found. Select one:[/bold]")
    chosen = select(
        options=labels,
        pagination=len(labels) > 15,
        page_size=15,
    )
    if chosen is None:
        console.print("[red]No rubric group selected.[/red]")
        raise typer.Exit(code=1)

    return groups[keys[labels.index(chosen)]]


# ---------------------------------------------------------------------------
# Options screens
# ---------------------------------------------------------------------------

_RUN_TOGGLES = [
    ("Verbose output", "verbose"),
    ("Resume (skip completed)", "resume"),
    ("Profile VRAM during inference", "profile_vram"),
]

_OVERNIGHT_TOGGLES = [
    ("Evaluate all runs in a batch after inference", "evaluate"),
    ("Profile system first", "profile"),
    ("Profile VRAM during inference", "profile_vram"),
    ("Resume (skip completed)", "resume"),
    ("Verbose output", "verbose"),
]


def _prompt_repeats(default: int) -> int:
    """Ask user for repeat count, validating as a positive integer."""
    console.print(f"[bold]Repeats per suite[/bold] (default {default}):")
    value = prompt(
        "  > ",
        target_type=int,
        initial_value=str(default),
        validator=lambda v: v >= 1,
        raise_validation_fail=False,
        raise_type_conversion_fail=False,
    )
    if value is None or value < 1:
        return default
    return value


def _prompt_toggles(
    toggles: list[tuple[str, str]],
    defaults: dict[str, bool] | None = None,
) -> dict[str, bool]:
    """Show a multi-select of toggle options, return map of key -> enabled.

    When `defaults` maps a toggle key to True, that option is pre-ticked so CLI
    flags the user already passed survive the interactive picker.
    """
    labels = [label for label, _ in toggles]
    ticked = [
        idx for idx, (_, key) in enumerate(toggles)
        if defaults and defaults.get(key)
    ]
    console.print("[bold]Options[/bold] (space to toggle, enter to confirm):")
    selected = select_multiple(options=labels, ticked_indices=ticked)
    selected_set = set(selected)
    return {key: label in selected_set for label, key in toggles}


def select_run_options(
    default_repeats: int = 1,
    defaults: dict[str, bool] | None = None,
) -> dict:
    """Interactive options screen for the run command."""
    repeats = _prompt_repeats(default_repeats)
    toggles = _prompt_toggles(_RUN_TOGGLES, defaults=defaults)
    return {"repeats": repeats, **toggles}


def select_overnight_options(
    default_repeats: int = 3,
    defaults: dict[str, bool] | None = None,
) -> dict:
    """Interactive options screen for the overnight command."""
    repeats = _prompt_repeats(default_repeats)
    toggles = _prompt_toggles(_OVERNIGHT_TOGGLES, defaults=defaults)
    return {"repeats": repeats, **toggles}
