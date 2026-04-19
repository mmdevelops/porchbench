"""Environment diagnostics for porchbench.

Runs a structured set of checks against Python, Ollama, GPU tooling,
and package install state. Produces a DoctorReport that can be rendered
for humans or serialized as JSON for bug reports and tooling.
"""

from __future__ import annotations

import json as jsonlib
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum

import httpx
from rich.console import Console

from porchbench import __version__
from porchbench.assets import resolve_suite_dir
from porchbench.profiler import (
    _sample_vram_via_nvidia_smi,
    _sample_vram_via_rocm_smi,
    detect_gpu,
)


class CheckStatus(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"
    INFO = "info"


@dataclass
class Check:
    name: str
    status: CheckStatus
    detail: str
    hint: str | None = None


@dataclass
class DoctorSummary:
    ok: int = 0
    warn: int = 0
    fail: int = 0
    skip: int = 0
    info: int = 0


@dataclass
class DoctorReport:
    version: str
    ok: bool
    summary: DoctorSummary
    checks: list[Check] = field(default_factory=list)

    def to_json(self) -> str:
        return jsonlib.dumps(
            {
                "version": self.version,
                "ok": self.ok,
                "summary": asdict(self.summary),
                "checks": [
                    {"name": c.name, "status": c.status.value, "detail": c.detail, "hint": c.hint}
                    for c in self.checks
                ],
            },
            indent=2,
        )


REQUIRED_CHECKS: frozenset[str] = frozenset({"python", "ollama-server", "builtin-suites"})


def _normalize_ollama_url(host: str | None) -> str:
    # Default to 127.0.0.1 rather than localhost: on Windows, localhost often
    # resolves to ::1 first, adds a ~2s IPv6 retry delay per HTTP probe.
    url = host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    return url.rstrip("/")


def check_python_version() -> Check:
    v = sys.version_info
    detail = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 11):
        return Check("python", CheckStatus.OK, detail)
    return Check(
        "python",
        CheckStatus.FAIL,
        f"{detail} (porchbench requires >= 3.11)",
        hint="Install Python 3.11 or newer.",
    )


def check_ollama_server(url: str) -> tuple[Check, bool]:
    """Probe /api/version; return the check and whether the server is reachable."""
    try:
        r = httpx.get(f"{url}/api/version", timeout=3.0)
    except httpx.ConnectError:
        return (
            Check(
                "ollama-server",
                CheckStatus.FAIL,
                f"{url} (connection refused)",
                hint="Start Ollama: `ollama serve`, or set OLLAMA_HOST if running elsewhere.",
            ),
            False,
        )
    except Exception as e:
        return Check("ollama-server", CheckStatus.FAIL, f"{url} ({type(e).__name__}: {e})"), False

    if r.status_code != 200:
        return (
            Check("ollama-server", CheckStatus.FAIL, f"{url} (HTTP {r.status_code})"),
            False,
        )
    version = "unknown"
    try:
        version = r.json().get("version", "unknown")
    except Exception:
        pass
    return Check("ollama-server", CheckStatus.OK, f"{url} (v{version})"), True


def check_models_pulled(url: str, server_ok: bool) -> Check:
    if not server_ok:
        return Check("models-pulled", CheckStatus.SKIP, "(Ollama unreachable)")
    try:
        r = httpx.get(f"{url}/api/tags", timeout=3.0)
        if r.status_code != 200:
            return Check("models-pulled", CheckStatus.WARN, f"HTTP {r.status_code}")
        models = r.json().get("models", [])
    except Exception as e:
        return Check("models-pulled", CheckStatus.WARN, f"probe failed ({type(e).__name__})")
    if models:
        return Check("models-pulled", CheckStatus.OK, f"{len(models)} model(s) available")
    return Check(
        "models-pulled",
        CheckStatus.WARN,
        "no models pulled",
        hint="Pull one: `ollama pull qwen2.5:3b`.",
    )


def check_gpu() -> Check:
    name, vram_gb = detect_gpu()
    if name and vram_gb:
        return Check("gpu", CheckStatus.OK, f"{name} ({vram_gb} GB)")
    if name:
        return Check("gpu", CheckStatus.WARN, f"{name} (VRAM unknown)")
    return Check(
        "gpu",
        CheckStatus.INFO,
        "no discrete GPU detected (CPU-only is supported but slower)",
    )


def check_vram_sampler() -> Check:
    if shutil.which("nvidia-smi") and _sample_vram_via_nvidia_smi() is not None:
        return Check("vram-sampler", CheckStatus.OK, "nvidia-smi (direct GPU polling)")
    if shutil.which("rocm-smi") and _sample_vram_via_rocm_smi() is not None:
        return Check("vram-sampler", CheckStatus.OK, "rocm-smi (direct GPU polling)")
    return Check(
        "vram-sampler",
        CheckStatus.WARN,
        "Ollama /api/ps (slower HTTP fallback)",
        hint="Add nvidia-smi or rocm-smi to PATH for faster direct VRAM sampling.",
    )


def check_gpu_acceleration(url: str, server_ok: bool) -> Check:
    if not server_ok:
        return Check("gpu-accel", CheckStatus.SKIP, "(Ollama unreachable)")
    try:
        r = httpx.get(f"{url}/api/ps", timeout=3.0)
        if r.status_code != 200:
            return Check("gpu-accel", CheckStatus.WARN, f"/api/ps returned HTTP {r.status_code}")
        processes = r.json().get("models", [])
    except Exception as e:
        return Check("gpu-accel", CheckStatus.WARN, f"probe failed ({type(e).__name__})")

    if not processes:
        return Check(
            "gpu-accel",
            CheckStatus.INFO,
            "no models currently loaded (run a benchmark to verify GPU usage)",
        )
    for proc in processes:
        size_vram = proc.get("size_vram", 0) or 0
        if size_vram > 0:
            gb = size_vram / 1e9
            return Check(
                "gpu-accel",
                CheckStatus.OK,
                f"active ({proc.get('name', '?')}: {gb:.1f} GB resident on GPU)",
            )
    return Check(
        "gpu-accel",
        CheckStatus.WARN,
        "models loaded but none on GPU (Ollama appears CPU-only)",
        hint="Verify Ollama has CUDA/ROCm support configured for your hardware.",
    )


def check_builtin_suites() -> Check:
    try:
        d = resolve_suite_dir()
        suites = list(d.glob("*.yaml")) + list(d.glob("*.yml"))
    except Exception as e:
        return Check(
            "builtin-suites",
            CheckStatus.FAIL,
            f"resolution failed ({type(e).__name__}: {e})",
            hint="Reinstall porchbench: `pip install --force-reinstall porchbench`.",
        )
    if not suites:
        return Check(
            "builtin-suites",
            CheckStatus.FAIL,
            "no suites found in package data",
            hint="Reinstall porchbench: `pip install --force-reinstall porchbench`.",
        )
    return Check("builtin-suites", CheckStatus.OK, f"{len(suites)} suite(s) loadable")


def check_api_extras() -> Check:
    try:
        import anthropic  # noqa: F401
        return Check("api-extras", CheckStatus.OK, "anthropic SDK importable")
    except ImportError:
        backend_env = os.environ.get("PORCHBENCH_EVAL_BACKEND", "").lower()
        if backend_env == "api":
            return Check(
                "api-extras",
                CheckStatus.FAIL,
                "not installed, but PORCHBENCH_EVAL_BACKEND=api is set",
                hint="Install extras: `pip install 'porchbench[api]'`.",
            )
        return Check(
            "api-extras",
            CheckStatus.INFO,
            "not installed (optional; only needed for PORCHBENCH_EVAL_BACKEND=api)",
        )


def check_env_vars() -> Check:
    relevant = [
        "OLLAMA_HOST",
        "OLLAMA_KV_CACHE_TYPE",
        "PORCHBENCH_BACKEND",
        "PORCHBENCH_EVAL_BACKEND",
        "PORCHBENCH_EVAL_MODEL",
        "PORCHBENCH_SEED",
    ]
    set_vars = [f"{k}={os.environ[k]}" for k in relevant if k in os.environ]
    if set_vars:
        return Check("env", CheckStatus.INFO, " ".join(set_vars))
    return Check("env", CheckStatus.INFO, "no porchbench-relevant env vars set")


def run_checks(host: str | None = None) -> DoctorReport:
    """Run all diagnostic checks and assemble a DoctorReport.

    Required checks (python, ollama-server, builtin-suites) gate the
    top-level `ok` flag; other checks may warn without failing the report.
    """
    url = _normalize_ollama_url(host)

    checks: list[Check] = []
    checks.append(check_python_version())
    server_check, server_ok = check_ollama_server(url)
    checks.append(server_check)
    checks.append(check_models_pulled(url, server_ok))
    checks.append(check_gpu())
    checks.append(check_vram_sampler())
    checks.append(check_gpu_acceleration(url, server_ok))
    checks.append(check_builtin_suites())
    checks.append(check_api_extras())
    checks.append(check_env_vars())

    summary = DoctorSummary()
    overall_ok = True
    for c in checks:
        if c.status is CheckStatus.OK:
            summary.ok += 1
        elif c.status is CheckStatus.WARN:
            summary.warn += 1
        elif c.status is CheckStatus.FAIL:
            summary.fail += 1
            if c.name in REQUIRED_CHECKS:
                overall_ok = False
        elif c.status is CheckStatus.SKIP:
            summary.skip += 1
            if c.name in REQUIRED_CHECKS:
                overall_ok = False
        elif c.status is CheckStatus.INFO:
            summary.info += 1

    return DoctorReport(version=__version__, ok=overall_ok, summary=summary, checks=checks)


_STATUS_MARKUP: dict[CheckStatus, str] = {
    CheckStatus.OK:   r"[green]\[ok][/green]  ",
    CheckStatus.WARN: r"[yellow]\[warn][/yellow]",
    CheckStatus.FAIL: r"[red]\[fail][/red]",
    CheckStatus.SKIP: r"[dim]\[skip][/dim]",
    CheckStatus.INFO: r"[cyan]\[info][/cyan]",
}


def render_report(report: DoctorReport, console: Console) -> None:
    """Render the report to the given Rich console.

    Rich strips markup automatically when stdout is not a TTY, so no
    separate plain-text path is needed.
    """
    console.print(f"\n[bold]porchbench doctor[/bold]  v{report.version}\n")
    for c in report.checks:
        marker = _STATUS_MARKUP[c.status]
        console.print(f"{marker}  [bold]{c.name:<16}[/bold] {c.detail}")
        if c.hint:
            console.print(f"        [dim]-> {c.hint}[/dim]")

    s = report.summary
    parts = [f"{s.ok} ok"]
    if s.warn:
        parts.append(f"{s.warn} warn")
    if s.fail:
        parts.append(f"{s.fail} fail")
    if s.skip:
        parts.append(f"{s.skip} skip")
    totals = ", ".join(parts)
    if report.ok:
        console.print(f"\n[green]{totals}[/green]")
    else:
        console.print(f"\n[red]{totals}[/red] (exit 1)")

    console.print(
        "\n[dim]Tip: run [bold]porchbench --install-completion[/bold] to enable "
        "shell tab completion.[/dim]\n"
    )
