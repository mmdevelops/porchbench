"""Tests for porchbench.profiler helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from porchbench.profiler import (
    _GPU_CACHE_TTL_SECONDS,
    _read_gpu_cache,
    _write_gpu_cache,
    detect_gpu,
)


@pytest.fixture(autouse=True)
def _redirect_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the GPU cache at a tmp path so tests never touch ~/.porchbench."""
    monkeypatch.setattr(
        "porchbench.profiler._gpu_cache_path",
        lambda: tmp_path / "cache" / "gpu.json",
    )
    # Also clear the in-process lru_cache so each test gets a clean detect_gpu state
    detect_gpu.cache_clear()


def test_read_cache_returns_none_when_missing() -> None:
    assert _read_gpu_cache() is None


def test_write_then_read_roundtrip() -> None:
    _write_gpu_cache("NVIDIA RTX 5090", 32.0)
    result = _read_gpu_cache()
    assert result == ("NVIDIA RTX 5090", 32.0)


def test_read_cache_returns_none_for_stale_entry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Write a cache entry with a very old cached_at timestamp
    path = tmp_path / "cache" / "gpu.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    stale_ts = time.time() - _GPU_CACHE_TTL_SECONDS - 60
    path.write_text(json.dumps({
        "gpu_name": "Old GPU", "vram_gb": 8.0, "cached_at": stale_ts,
    }), encoding="utf-8")
    assert _read_gpu_cache() is None


def test_read_cache_returns_none_for_malformed(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "gpu.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")
    assert _read_gpu_cache() is None


def test_write_handles_null_vram() -> None:
    _write_gpu_cache("Unknown GPU", None)
    result = _read_gpu_cache()
    assert result == ("Unknown GPU", None)


def test_detect_gpu_uses_cache_when_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    _write_gpu_cache("Fake GPU", 24.0)

    def _should_not_run() -> tuple[str, float | None]:
        raise AssertionError("Uncached probe ran despite fresh cache")

    monkeypatch.setattr("porchbench.profiler._detect_gpu_uncached", _should_not_run)
    name, vram = detect_gpu()
    assert name == "Fake GPU"
    assert vram == 24.0


def test_detect_gpu_writes_cache_after_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "porchbench.profiler._detect_gpu_uncached",
        lambda: ("Probed GPU", 16.0),
    )
    name, vram = detect_gpu()
    assert (name, vram) == ("Probed GPU", 16.0)
    # Cache file should exist now; next read hits cache
    assert _read_gpu_cache() == ("Probed GPU", 16.0)
