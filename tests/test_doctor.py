"""Tests for porchbench.doctor.

External probes (httpx, subprocess) are monkeypatched so tests stay
hermetic. Focus: each check's status mapping, report assembly, exit
code semantics, JSON shape stability.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from porchbench.doctor import (
    REQUIRED_CHECKS,
    Check,
    CheckStatus,
    DoctorReport,
    DoctorSummary,
    _normalize_ollama_url,
    check_api_extras,
    check_builtin_suites,
    check_env_vars,
    check_gpu,
    check_gpu_acceleration,
    check_models_pulled,
    check_ollama_server,
    check_python_version,
    check_vram_sampler,
    run_checks,
)


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------


def test_normalize_ollama_url_defaults_to_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert _normalize_ollama_url(None) == "http://localhost:11434"


def test_normalize_ollama_url_adds_scheme() -> None:
    assert _normalize_ollama_url("example.com:11434") == "http://example.com:11434"


def test_normalize_ollama_url_strips_trailing_slash() -> None:
    assert _normalize_ollama_url("http://example.com:11434/") == "http://example.com:11434"


def test_normalize_ollama_url_respects_https() -> None:
    assert _normalize_ollama_url("https://ollama.internal") == "https://ollama.internal"


# ---------------------------------------------------------------------------
# Python version
# ---------------------------------------------------------------------------


def test_python_version_passes_on_current_interpreter() -> None:
    c = check_python_version()
    assert c.status is CheckStatus.OK
    assert c.name == "python"


# ---------------------------------------------------------------------------
# Ollama server
# ---------------------------------------------------------------------------


class _MockResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


def test_ollama_server_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        httpx, "get", lambda *a, **kw: _MockResponse(200, {"version": "0.6.4"})
    )
    check, ok = check_ollama_server("http://localhost:11434")
    assert ok is True
    assert check.status is CheckStatus.OK
    assert "0.6.4" in check.detail


def test_ollama_server_connection_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*a, **kw):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(httpx, "get", _raise)
    check, ok = check_ollama_server("http://localhost:11434")
    assert ok is False
    assert check.status is CheckStatus.FAIL
    assert check.hint is not None


def test_ollama_server_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _MockResponse(503))
    check, ok = check_ollama_server("http://localhost:11434")
    assert ok is False
    assert check.status is CheckStatus.FAIL
    assert "503" in check.detail


# ---------------------------------------------------------------------------
# Models pulled
# ---------------------------------------------------------------------------


def test_models_pulled_skipped_when_server_down() -> None:
    c = check_models_pulled("http://x", server_ok=False)
    assert c.status is CheckStatus.SKIP


def test_models_pulled_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"models": [{"name": "a"}, {"name": "b"}]}
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _MockResponse(200, payload))
    c = check_models_pulled("http://x", server_ok=True)
    assert c.status is CheckStatus.OK
    assert "2" in c.detail


def test_models_pulled_empty_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _MockResponse(200, {"models": []}))
    c = check_models_pulled("http://x", server_ok=True)
    assert c.status is CheckStatus.WARN
    assert c.hint is not None


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


def test_gpu_detected() -> None:
    with patch("porchbench.doctor.detect_gpu", return_value=("NVIDIA RTX 5090", 32.0)):
        c = check_gpu()
        assert c.status is CheckStatus.OK
        assert "RTX 5090" in c.detail
        assert "32.0" in c.detail


def test_gpu_name_only_warns() -> None:
    with patch("porchbench.doctor.detect_gpu", return_value=("Unknown GPU", None)):
        c = check_gpu()
        assert c.status is CheckStatus.WARN


def test_gpu_absent_info() -> None:
    with patch("porchbench.doctor.detect_gpu", return_value=("", None)):
        c = check_gpu()
        assert c.status is CheckStatus.INFO


# ---------------------------------------------------------------------------
# VRAM sampler
# ---------------------------------------------------------------------------


def test_vram_sampler_prefers_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    import porchbench.doctor as doctor_mod

    monkeypatch.setattr(doctor_mod.shutil, "which", lambda x: "/usr/bin/" + x)
    monkeypatch.setattr(doctor_mod, "_sample_vram_via_nvidia_smi", lambda: 1_000_000)
    monkeypatch.setattr(doctor_mod, "_sample_vram_via_rocm_smi", lambda: None)
    c = check_vram_sampler()
    assert c.status is CheckStatus.OK
    assert "nvidia-smi" in c.detail


def test_vram_sampler_falls_back_to_rocm(monkeypatch: pytest.MonkeyPatch) -> None:
    import porchbench.doctor as doctor_mod

    # nvidia-smi not on PATH
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda x: None if x == "nvidia-smi" else "/usr/bin/rocm-smi")
    monkeypatch.setattr(doctor_mod, "_sample_vram_via_rocm_smi", lambda: 1_000_000)
    c = check_vram_sampler()
    assert c.status is CheckStatus.OK
    assert "rocm-smi" in c.detail


def test_vram_sampler_warns_when_neither_available(monkeypatch: pytest.MonkeyPatch) -> None:
    import porchbench.doctor as doctor_mod

    monkeypatch.setattr(doctor_mod.shutil, "which", lambda x: None)
    c = check_vram_sampler()
    assert c.status is CheckStatus.WARN
    assert c.hint is not None


# ---------------------------------------------------------------------------
# GPU acceleration
# ---------------------------------------------------------------------------


def test_gpu_accel_skipped_when_server_down() -> None:
    c = check_gpu_acceleration("http://x", server_ok=False)
    assert c.status is CheckStatus.SKIP


def test_gpu_accel_active(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"models": [{"name": "qwen:3b", "size_vram": 3_500_000_000}]}
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _MockResponse(200, payload))
    c = check_gpu_acceleration("http://x", server_ok=True)
    assert c.status is CheckStatus.OK
    assert "qwen:3b" in c.detail


def test_gpu_accel_cpu_only_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"models": [{"name": "qwen:3b", "size_vram": 0}]}
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _MockResponse(200, payload))
    c = check_gpu_acceleration("http://x", server_ok=True)
    assert c.status is CheckStatus.WARN


def test_gpu_accel_no_loaded_models_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _MockResponse(200, {"models": []}))
    c = check_gpu_acceleration("http://x", server_ok=True)
    assert c.status is CheckStatus.INFO


# ---------------------------------------------------------------------------
# Built-in suites
# ---------------------------------------------------------------------------


def test_builtin_suites_loadable() -> None:
    c = check_builtin_suites()
    # Real package ships suites, so this should pass in the test environment
    assert c.status is CheckStatus.OK
    assert "suite" in c.detail


# ---------------------------------------------------------------------------
# API extras
# ---------------------------------------------------------------------------


def test_api_extras_optional_when_backend_not_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PORCHBENCH_EVAL_BACKEND", raising=False)
    c = check_api_extras()
    # Either installed (OK) or not (INFO); never FAIL without the env var
    assert c.status in (CheckStatus.OK, CheckStatus.INFO)


def test_api_extras_required_when_backend_is_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORCHBENCH_EVAL_BACKEND", "api")
    import sys
    if "anthropic" in sys.modules:
        pytest.skip("anthropic SDK is installed in this environment")
    c = check_api_extras()
    assert c.status is CheckStatus.FAIL
    assert c.hint is not None


# ---------------------------------------------------------------------------
# Env vars
# ---------------------------------------------------------------------------


def test_env_vars_reports_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://custom:11434")
    monkeypatch.setenv("PORCHBENCH_SEED", "7")
    c = check_env_vars()
    assert c.status is CheckStatus.INFO
    assert "OLLAMA_HOST" in c.detail
    assert "PORCHBENCH_SEED=7" in c.detail


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def test_run_checks_returns_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        httpx, "get",
        lambda *a, **kw: _MockResponse(200, {"version": "0.6", "models": []}),
    )
    with patch("porchbench.doctor.detect_gpu", return_value=("Fake GPU", 16.0)):
        report = run_checks(host="http://localhost:11434")
    assert isinstance(report, DoctorReport)
    assert report.version
    assert len(report.checks) == 9
    assert report.summary.ok + report.summary.warn + report.summary.fail + \
        report.summary.skip + report.summary.info == 9


def test_required_check_fail_sets_report_not_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    def _refuse(*a, **kw):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(httpx, "get", _refuse)
    report = run_checks(host="http://localhost:11434")
    assert report.ok is False
    # ollama-server is required — its failure propagates
    assert any(
        c.name == "ollama-server" and c.status is CheckStatus.FAIL for c in report.checks
    )


def test_warnings_do_not_fail_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        httpx, "get",
        lambda *a, **kw: _MockResponse(200, {"version": "0.6", "models": []}),
    )
    import porchbench.doctor as doctor_mod
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda x: None)  # no vram sampler
    with patch("porchbench.doctor.detect_gpu", return_value=("", None)):
        report = run_checks()
    # No required checks failed; warnings present but ok stays True
    assert report.ok is True
    assert report.summary.warn >= 1


def test_required_checks_set_is_stable() -> None:
    assert REQUIRED_CHECKS == frozenset({"python", "ollama-server", "builtin-suites"})


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def test_to_json_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        httpx, "get",
        lambda *a, **kw: _MockResponse(200, {"version": "0.6", "models": []}),
    )
    report = run_checks(host="http://localhost:11434")
    data = json.loads(report.to_json())
    assert set(data.keys()) == {"version", "ok", "summary", "checks"}
    assert set(data["summary"].keys()) == {"ok", "warn", "fail", "skip", "info"}
    assert isinstance(data["checks"], list)
    for c in data["checks"]:
        assert set(c.keys()) == {"name", "status", "detail", "hint"}
        assert c["status"] in {"ok", "warn", "fail", "skip", "info"}


def test_to_json_is_valid_json() -> None:
    report = DoctorReport(
        version="0.1.0",
        ok=True,
        summary=DoctorSummary(ok=1),
        checks=[Check("x", CheckStatus.OK, "detail", None)],
    )
    parsed = json.loads(report.to_json())
    assert parsed["ok"] is True
    assert parsed["checks"][0]["hint"] is None
