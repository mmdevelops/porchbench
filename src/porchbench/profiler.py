"""System profiler for routing cost estimation.

Measures model load times, inference throughput, VRAM usage, swap times,
and co-residency capacity. Produces a SystemProfile that feeds into the
routing analysis cost model.
"""

from __future__ import annotations

import asyncio
import functools
import json
import platform
import subprocess
import time
from contextlib import asynccontextmanager
from itertools import combinations
from pathlib import Path

from rich.console import Console
from rich.table import Table

from porchbench.backend import OllamaBackend
from porchbench.schemas import (
    CoexistenceTest,
    ModelProfile,
    SwapMeasurement,
    SystemProfile,
    compute_derived_metrics,
)

console = Console()


@functools.lru_cache(maxsize=1)
def detect_gpu() -> tuple[str, float | None]:
    """Detect GPU name and total VRAM.

    Returns (gpu_name, vram_total_gb). VRAM may be None if detection fails.
    Uses platform-specific methods: nvidia-smi, WMI (Windows), lspci (Linux).
    Cached at module level — GPU doesn't change within a process, and
    dxdiag on Windows costs ~1-2s per call.
    """
    gpu_name = ""
    vram_gb = None

    # Try nvidia-smi first (works on NVIDIA GPUs, any OS)
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split(", ")
            gpu_name = parts[0]
            if len(parts) > 1:
                vram_gb = round(float(parts[1]) / 1024, 1)  # MiB -> GiB
            return gpu_name, vram_gb
    except (FileNotFoundError, Exception):
        pass

    # Try dxdiag on Windows (accurate VRAM even for >4GB GPUs)
    if platform.system() == "Windows":
        try:
            gpu_name, vram_gb = _detect_gpu_dxdiag()
            if gpu_name:
                return gpu_name, vram_gb
        except Exception:
            pass

        # Fallback: WMI for GPU name (VRAM capped at 4GB, not reliable)
        try:
            r = subprocess.run(
                ["powershell", "-Command",
                 "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0 and r.stdout.strip():
                data = json.loads(r.stdout)
                if isinstance(data, dict):
                    data = [data]
                best = max(data, key=lambda g: g.get("AdapterRAM", 0))
                gpu_name = best.get("Name", "")
        except (FileNotFoundError, json.JSONDecodeError, Exception):
            pass

    # Try lspci on Linux
    if platform.system() == "Linux" and not gpu_name:
        try:
            r = subprocess.run(
                ["lspci"], capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    if "VGA" in line or "3D" in line:
                        gpu_name = line.split(": ", 1)[-1] if ": " in line else line
                        break
        except (FileNotFoundError, Exception):
            pass

    return gpu_name, vram_gb


def _detect_gpu_dxdiag() -> tuple[str, float | None]:
    """Parse dxdiag output for GPU name and dedicated memory.

    dxdiag reports correct VRAM even for GPUs >4GB, unlike WMI.
    Runs dxdiag /t to dump diagnostics to a temp file.
    """
    import os
    import re
    import tempfile
    import time

    tmp_path = os.path.join(tempfile.gettempdir(), "porchbench_dxdiag.txt")

    # Clean up stale file
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    subprocess.run(
        ["cmd", "/c", f"dxdiag /t {tmp_path}"],
        capture_output=True, timeout=15,
    )

    # dxdiag writes asynchronously; wait for the file
    for _ in range(10):
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            break
        time.sleep(0.5)

    if not os.path.exists(tmp_path):
        return "", None

    try:
        with open(tmp_path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # Parse display device sections. dxdiag lists multiple display devices;
    # we want the discrete GPU (largest dedicated memory).
    card_names = re.findall(r"Card name:\s*(.+)", text)
    dedicated_mb = re.findall(r"Dedicated Memory:\s*(\d+)\s*MB", text)

    if not card_names:
        return "", None

    # Pair cards with their dedicated memory, pick largest
    best_idx = 0
    best_mem = 0
    for i, mem_str in enumerate(dedicated_mb):
        mem = int(mem_str)
        if mem > best_mem:
            best_mem = mem
            best_idx = i

    gpu_name = card_names[best_idx].strip() if best_idx < len(card_names) else card_names[0].strip()
    vram_gb = round(best_mem / 1024, 1) if best_mem > 0 else None

    return gpu_name, vram_gb


def _estimate_vram_total(
    profiles: dict[str, ModelProfile],
) -> float | None:
    """Estimate total VRAM from model loading behavior.

    If a model loads fully into VRAM (size_vram == size), we know VRAM >= that.
    We round up to the nearest common VRAM tier (8, 12, 16, 24, 48, 80 GB).
    """
    max_vram = max((mp.vram_gb or 0 for mp in profiles.values()), default=0)
    if max_vram <= 0:
        return None

    # Common GPU VRAM tiers
    tiers = [4, 6, 8, 10, 12, 16, 24, 32, 48, 80]
    for tier in tiers:
        if tier >= max_vram * 1.2:  # model uses at most ~80% of a tier
            return float(tier)

    return round(max_vram * 1.3, 1)  # fallback for very large GPUs


# ---------------------------------------------------------------------------
# VRAM polling during inference
# ---------------------------------------------------------------------------


class VramSample:
    """Accumulates peak VRAM observed during a polling window."""

    def __init__(self) -> None:
        self.peak_bytes: int = 0

    def update(self, size_vram: int) -> None:
        if size_vram > self.peak_bytes:
            self.peak_bytes = size_vram


@asynccontextmanager
async def measure_peak_vram(
    backend: OllamaBackend,
    model: str,
    poll_interval_s: float = 0.1,
):
    """Poll ollama.ps() in the background and track peak VRAM for a model.

    Usage::

        async with measure_peak_vram(backend, "qwen3:8b") as sample:
            await backend.chat(...)
        print(sample.peak_bytes)

    Yields a VramSample whose peak_bytes is updated continuously.
    Returns 0 if the model isn't found or polling fails.
    """
    sample = VramSample()
    stop = asyncio.Event()

    async def _poll() -> None:
        while not stop.is_set():
            try:
                running = await backend.list_running_models()
                for m in running:
                    if model in m.get("name", ""):
                        vram = m.get("size_vram")
                        if vram and isinstance(vram, int):
                            sample.update(vram)
                        break
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop.wait(), timeout=poll_interval_s)
            except TimeoutError:
                pass

    task = asyncio.create_task(_poll())
    try:
        yield sample
    finally:
        stop.set()
        await task


# Standard prompt used for inference baseline measurement
BASELINE_PROMPT = "Explain in one paragraph what a hash table is and why it is useful."


async def profile_system(
    models: list[str],
    backend: OllamaBackend,
) -> SystemProfile:
    """Run the full system profiling suite. Requires OllamaBackend for VRAM introspection."""
    ollama_version = await backend.get_server_version()
    gpu_name, vram_total_gb = detect_gpu()

    console.print(f"Ollama version: {ollama_version}")
    if gpu_name:
        console.print(f"GPU: {gpu_name}")
    if vram_total_gb:
        console.print(f"VRAM: {vram_total_gb} GB")
    else:
        console.print("[yellow]VRAM total unknown (will estimate from model loading)[/yellow]")
    console.print(f"Models to profile: {', '.join(models)}")
    console.print()

    # Phase 1: profile each model individually
    model_profiles: dict[str, ModelProfile] = {}
    for model_name in models:
        console.print(f"[bold]Profiling {model_name}...[/bold]")
        profile = await _profile_single_model(model_name, backend)
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
                swap_ab = await _measure_swap_time(model_a, model_b, backend)
                swap_times.append(swap_ab)
                console.print(f"  {model_a} -> {model_b}: {swap_ab.swap_time_s:.1f}s")

                # B -> A
                swap_ba = await _measure_swap_time(model_b, model_a, backend)
                swap_times.append(swap_ba)
                console.print(f"  {model_b} -> {model_a}: {swap_ba.swap_time_s:.1f}s")

    # Estimate total VRAM if not detected
    if vram_total_gb is None:
        vram_total_gb = _estimate_vram_total(model_profiles)
        if vram_total_gb:
            console.print(f"[yellow]Estimated VRAM total: ~{vram_total_gb} GB[/yellow]")

    # Phase 3: test co-residency (can models coexist in VRAM?)
    coexistence: list[CoexistenceTest] = []
    if len(models) >= 2:
        console.print("\n[bold]Testing co-residency...[/bold]")
        for pair in combinations(models, 2):
            test = _estimate_coexistence(list(pair), model_profiles, vram_total_gb)
            fits_str = "[green]fits[/green]" if test.fits else "[red]does not fit[/red]"
            console.print(f"  {' + '.join(pair)}: {fits_str}")
            coexistence.append(test)

    # Determine hot/cold tiers
    hot_tier, cold_tier = _compute_tiers(models, model_profiles, coexistence, vram_total_gb)

    return SystemProfile(
        gpu=gpu_name,
        vram_total_gb=vram_total_gb,
        ollama_version=ollama_version,
        models=model_profiles,
        swap_times=swap_times,
        coexistence=coexistence,
        recommended_hot_tier=hot_tier,
        cold_tier=cold_tier,
    )


async def _profile_single_model(model_name: str, backend: OllamaBackend) -> ModelProfile:
    """Profile a single model: load time, inference throughput, VRAM usage."""
    from porchbench.schemas import ModelOptions

    messages = [{"role": "user", "content": BASELINE_PROMPT}]
    options = ModelOptions(temperature=0, seed=42, num_predict=256, num_ctx=4096)

    # First call may load the model — captures load_duration
    result = await backend.chat(messages, model_name, options)
    computed = compute_derived_metrics(result.metrics)

    load_time_s = result.metrics.load_duration / 1e9 if result.metrics.load_duration else None

    # Check VRAM usage via ps()
    vram_bytes = None
    vram_gb = None
    try:
        running = await backend.list_running_models()
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


async def _measure_swap_time(from_model: str, to_model: str, backend: OllamaBackend) -> SwapMeasurement:
    """Measure time to swap from one model to another.

    Ensures from_model is loaded, then times a request to to_model
    (which forces the swap).
    """
    from porchbench.schemas import ModelOptions

    messages = [{"role": "user", "content": "Hi"}]
    options = ModelOptions(temperature=0, seed=42, num_predict=1, num_ctx=2048)

    # Ensure from_model is loaded
    await backend.chat(messages, from_model, options)

    # Time the swap: request to to_model triggers unload + load
    start = time.monotonic()
    await backend.chat(messages, to_model, options)
    elapsed = time.monotonic() - start

    return SwapMeasurement(
        from_model=from_model,
        to_model=to_model,
        swap_time_s=round(elapsed, 2),
    )


def _estimate_coexistence(
    model_pair: list[str],
    profiles: dict[str, ModelProfile],
    vram_total_gb: float | None = None,
) -> CoexistenceTest:
    """Estimate whether models can co-reside in VRAM based on profiled usage."""
    combined = sum(
        profiles[m].vram_gb or 0 for m in model_pair if m in profiles
    )
    total = vram_total_gb or 16.0  # fallback if still unknown
    headroom = total - combined

    return CoexistenceTest(
        models=model_pair,
        fits=headroom > 1.0,  # leave 1GB headroom for system + KV cache
        combined_vram_gb=round(combined, 2) if combined > 0 else None,
        headroom_gb=round(headroom, 2),
    )


def _compute_tiers(
    models: list[str],
    profiles: dict[str, ModelProfile],
    coexistence: list[CoexistenceTest],
    vram_total_gb: float | None = None,
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
    total = vram_total_gb or 16.0
    vram_limit = total - 2.0  # leave 2GB headroom

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
