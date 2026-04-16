"""Interactive model and suite selection for CLI commands.

Uses beaupy for arrow-key driven pickers when --model or --suite
arguments are omitted from CLI invocations.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from beaupy import select, select_multiple
from rich.console import Console

from feral.backend import InferenceBackend
from feral.suite import discover_suites

console = Console()


def select_models(backend: InferenceBackend) -> list[str]:
    """Prompt user to pick one or more models from the backend's available list."""
    try:
        models = asyncio.run(backend.list_available_models())
    except Exception as exc:
        console.print(f"[red]Failed to list models: {exc}[/red]")
        raise typer.Exit(code=1)

    if not models:
        console.print("[red]No models available. Pull a model first.[/red]")
        raise typer.Exit(code=1)

    console.print("[bold]Select model(s)[/bold] (space to toggle, enter to confirm):")
    selected = select_multiple(
        options=models,
        minimal_count=1,
        pagination=len(models) > 15,
        page_size=15,
    )
    if not selected:
        console.print("[red]No models selected.[/red]")
        raise typer.Exit(code=1)

    return selected


def select_suite(suite_dir: Path = Path("suites")) -> Path:
    """Prompt user to pick a suite YAML file from the suite directory."""
    try:
        paths = discover_suites(suite_dir)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
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


def select_suites(suite_dir: Path = Path("suites")) -> list[Path]:
    """Prompt user to pick one or more suite YAML files from the suite directory."""
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
    """Scan result_dir for run-result JSONs, return (label, path) sorted newest-first."""
    entries: list[tuple[str, Path]] = []
    for p in sorted(result_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            run = data.get("run", {})
            model = run.get("model", {}).get("name", "?")
            suite_name = run.get("suite", {}).get("name", "?")
            suite_ver = run.get("suite", {}).get("version", "")
            ts = run.get("timestamp", "")[:10]
            label = f"{model} — {suite_name} v{suite_ver} ({ts})"
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
    chosen = select(options=labels, pagination=len(labels) > 15, page_size=15)
    if chosen is None:
        console.print("[red]No result selected.[/red]")
        raise typer.Exit(code=1)

    return entries[labels.index(chosen)][1]


def select_results(result_dir: Path = Path("results")) -> list[Path]:
    """Prompt user to pick one or more run-result files."""
    if not result_dir.is_dir():
        console.print(f"[red]Results directory not found: {result_dir}[/red]")
        raise typer.Exit(code=1)

    entries = _discover_result_files(result_dir)
    if not entries:
        console.print(f"[red]No result files found in {result_dir}/[/red]")
        raise typer.Exit(code=1)

    labels = [label for label, _ in entries]
    console.print("[bold]Select result(s)[/bold] (space to toggle, enter to confirm):")
    selected = select_multiple(
        options=labels,
        minimal_count=1,
        pagination=len(labels) > 15,
        page_size=15,
    )
    if not selected:
        console.print("[red]No results selected.[/red]")
        raise typer.Exit(code=1)

    return [entries[labels.index(s)][1] for s in selected]
