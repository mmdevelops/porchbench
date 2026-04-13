"""System profiler for routing cost estimation.

Measures model load times, inference throughput, VRAM usage, swap times,
and co-residency capacity. Produces a SystemProfile that feeds into the
routing analysis cost model.
"""

from __future__ import annotations

import time
from itertools import combinations
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ollama_bench import client
from ollama_bench.schemas import (
    CoexistenceTest,
    ModelProfile,
    SwapMeasurement,
    SystemProfile,
    compute_derived_metrics,
)

console = Console()

# Standard prompt used for inference baseline measurement
BASELINE_PROMPT = "Explain in one paragraph what a hash table is and why it is useful."


async def profile_system(
    models: list[str],
    host: str | None = None,
) -> SystemProfile:
    """Run the full system profiling suite."""
    ollama_version = await client.get_server_version(host)

    console.print(f"Ollama version: {ollama_version}")
    console.print(f"Models to profile: {', '.join(models)}")
    console.print()

    # Phase 1: profile each model individually
    model_profiles: dict[str, ModelProfile] = {}
    for model_name in models:
        console.print(f"[bold]Profiling {model_name}...[/bold]")
        profile = await _profile_single_model(model_name, host)
        model_profiles[model_name] = profile

        vram_str = f"{profile.vram_gb:.1f}GB" if profile.vram_gb else "?"
        tps_str = f"{profile.tokens_per_second:.1f}" if profile.tokens_per_second else "?"
        load_str = f"{profile.load_time_s:.1f}s" if profile.load_time_s else "?"
        console.print(f"  VRAM: {vram_str}  tok/s: {tps_str}  load: {load_str}")

    # Phase 2: measure swap times between model pairs
    swap_times: list[SwapMeasurement] = []
    if len(models) >= 2:
        console.print("\n[bold]Measuring swap times...[/bold]")
        for i, model_a in enumerate(models):
            for model_b in models[i + 1:]:
                # A -> B
                swap_ab = await _measure_swap_time(model_a, model_b, host)
                swap_times.append(swap_ab)
                console.print(f"  {model_a} -> {model_b}: {swap_ab.swap_time_s:.1f}s")

                # B -> A
                swap_ba = await _measure_swap_time(model_b, model_a, host)
                swap_times.append(swap_ba)
                console.print(f"  {model_b} -> {model_a}: {swap_ba.swap_time_s:.1f}s")

    # Phase 3: test co-residency (can models coexist in VRAM?)
    coexistence: list[CoexistenceTest] = []
    if len(models) >= 2:
        console.print("\n[bold]Testing co-residency...[/bold]")
        for pair in combinations(models, 2):
            test = _estimate_coexistence(list(pair), model_profiles)
            fits_str = "[green]fits[/green]" if test.fits else "[red]does not fit[/red]"
            console.print(f"  {' + '.join(pair)}: {fits_str}")
            coexistence.append(test)

    # Determine hot/cold tiers
    hot_tier, cold_tier = _compute_tiers(models, model_profiles, coexistence)

    return SystemProfile(
        ollama_version=ollama_version,
        models=model_profiles,
        swap_times=swap_times,
        coexistence=coexistence,
        recommended_hot_tier=hot_tier,
        cold_tier=cold_tier,
    )


async def _profile_single_model(model_name: str, host: str | None) -> ModelProfile:
    """Profile a single model: load time, inference throughput, VRAM usage."""
    from ollama_bench.schemas import Message, ModelOptions

    messages = [Message(role="user", content=BASELINE_PROMPT)]
    options = ModelOptions(temperature=0, seed=42, num_predict=256, num_ctx=4096)

    # First call may load the model — captures load_duration
    response = await client.chat(messages, model_name, options, host=host)
    metrics = client.extract_metrics(response)
    computed = compute_derived_metrics(metrics)

    load_time_s = metrics.load_duration / 1e9 if metrics.load_duration else None

    # Check VRAM usage via ps()
    vram_bytes = None
    vram_gb = None
    try:
        running = await client.list_running_models(host)
        for m in running:
            if model_name in m.get("name", ""):
                vram_bytes = m.get("size_vram")
                if vram_bytes:
                    vram_gb = round(vram_bytes / (1024 ** 3), 2)
                break
    except Exception:
        pass

    return ModelProfile(
        vram_bytes=vram_bytes,
        vram_gb=vram_gb,
        load_time_s=round(load_time_s, 2) if load_time_s else None,
        tokens_per_second=round(computed.tokens_per_second, 1) if computed.tokens_per_second else None,
    )


async def _measure_swap_time(from_model: str, to_model: str, host: str | None) -> SwapMeasurement:
    """Measure time to swap from one model to another.

    Ensures from_model is loaded, then times a request to to_model
    (which forces the swap).
    """
    from ollama_bench.schemas import Message, ModelOptions

    messages = [Message(role="user", content="Hi")]
    options = ModelOptions(temperature=0, seed=42, num_predict=1, num_ctx=2048)

    # Ensure from_model is loaded
    await client.chat(messages, from_model, options, host=host)

    # Time the swap: request to to_model triggers unload + load
    start = time.monotonic()
    response = await client.chat(messages, to_model, options, host=host)
    elapsed = time.monotonic() - start

    return SwapMeasurement(
        from_model=from_model,
        to_model=to_model,
        swap_time_s=round(elapsed, 2),
    )


def _estimate_coexistence(
    model_pair: list[str],
    profiles: dict[str, ModelProfile],
) -> CoexistenceTest:
    """Estimate whether models can co-reside in VRAM based on profiled usage.

    This is a conservative estimate — actual co-residency depends on KV cache
    overhead under inference load, which the profiler measures per-model but
    doesn't test concurrently. The profiler notes this limitation.
    """
    vram_total = sum(
        profiles[m].vram_gb or 0 for m in model_pair if m in profiles
    )
    # Conservative: assume 16GB total VRAM as a baseline
    # TODO: detect actual GPU VRAM via platform-specific methods
    assumed_total = 16.0
    headroom = assumed_total - vram_total

    return CoexistenceTest(
        models=model_pair,
        fits=headroom > 1.0,  # leave 1GB headroom for system + KV cache
        combined_vram_gb=round(vram_total, 2) if vram_total > 0 else None,
        headroom_gb=round(headroom, 2),
    )


def _compute_tiers(
    models: list[str],
    profiles: dict[str, ModelProfile],
    coexistence: list[CoexistenceTest],
) -> tuple[list[str], list[str]]:
    """Determine which models should be hot-tier vs cold-tier.

    Hot tier: models that can co-reside in VRAM (zero swap cost).
    Cold tier: models too large to co-reside, routed to only when needed.
    """
    if len(models) <= 1:
        return models[:], []

    # Sort by VRAM ascending
    sorted_models = sorted(
        models,
        key=lambda m: profiles.get(m, ModelProfile()).vram_gb or float('inf'),
    )

    # Greedily add models to hot tier while they fit
    hot: list[str] = []
    total_vram = 0.0
    vram_limit = 14.0  # leave 2GB headroom from assumed 16GB

    for m in sorted_models:
        m_vram = profiles.get(m, ModelProfile()).vram_gb or 0
        if total_vram + m_vram <= vram_limit:
            hot.append(m)
            total_vram += m_vram
        else:
            break

    cold = [m for m in models if m not in hot]
    return hot, cold


def write_profile(profile: SystemProfile, output_dir: str | Path = "results") -> Path:
    """Write a system profile to a JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = profile.timestamp.strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"{ts}_system-profile.json"
    path = output_dir / filename
    path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return path


def print_profile_summary(profile: SystemProfile) -> None:
    """Print a rich summary table of the system profile."""
    # Model table
    table = Table(title="Model Profiles", title_style="bold")
    table.add_column("Model", style="bold")
    table.add_column("VRAM (GB)", justify="right")
    table.add_column("Load (s)", justify="right")
    table.add_column("tok/s", justify="right")

    for name, mp in profile.models.items():
        table.add_row(
            name,
            f"{mp.vram_gb:.1f}" if mp.vram_gb else "-",
            f"{mp.load_time_s:.1f}" if mp.load_time_s else "-",
            f"{mp.tokens_per_second:.1f}" if mp.tokens_per_second else "-",
        )
    console.print(table)

    # Swap times
    if profile.swap_times:
        swap_table = Table(title="Swap Times", title_style="bold")
        swap_table.add_column("From -> To")
        swap_table.add_column("Time (s)", justify="right")
        for s in profile.swap_times:
            swap_table.add_row(f"{s.from_model} -> {s.to_model}", f"{s.swap_time_s:.1f}")
        console.print(swap_table)

    # Co-residency
    if profile.coexistence:
        coresid_table = Table(title="Co-Residency", title_style="bold")
        coresid_table.add_column("Models")
        coresid_table.add_column("Fits?")
        coresid_table.add_column("Combined (GB)", justify="right")
        coresid_table.add_column("Headroom (GB)", justify="right")
        for c in profile.coexistence:
            fits_str = "[green]Yes[/green]" if c.fits else "[red]No[/red]"
            coresid_table.add_row(
                " + ".join(c.models), fits_str,
                f"{c.combined_vram_gb:.1f}" if c.combined_vram_gb else "-",
                f"{c.headroom_gb:.1f}" if c.headroom_gb is not None else "-",
            )
        console.print(coresid_table)

    # Tiers
    if profile.recommended_hot_tier:
        console.print(f"\n[bold]Hot tier[/bold] (co-resident): {', '.join(profile.recommended_hot_tier)}")
    if profile.cold_tier:
        console.print(f"[bold]Cold tier[/bold] (swap required): {', '.join(profile.cold_tier)}")
