"""Tests for porchbench.errors helpers."""

from __future__ import annotations

import pytest

from porchbench.errors import UserError, translate_inference_error


def test_translate_includes_model_and_phase() -> None:
    exc = RuntimeError("boom")
    ue = translate_inference_error(exc, "qwen2.5:7b", phase="profiling")
    assert isinstance(ue, UserError)
    assert "qwen2.5:7b" in str(ue)
    assert "Profiling failed" in str(ue)
    assert "boom" in str(ue)


def test_translate_adds_server_log_pointer_on_load_failure() -> None:
    exc = RuntimeError("model failed to load, check ollama server logs (status code: 500)")
    ue = translate_inference_error(exc, "llama3:8b", phase="profiling")
    assert "server.log" in str(ue)
    assert "journalctl" in str(ue)


def test_translate_flags_qwen35_solve_tri() -> None:
    exc = RuntimeError("model failed to load (status code: 500)")
    ue = translate_inference_error(exc, "qwen3.5:9b", phase="profiling")
    assert "SOLVE_TRI" in str(ue)
    assert "gfx1201" in str(ue)
    assert "qwen2.5" in str(ue)  # suggested alternative


def test_translate_does_not_flag_solve_tri_for_other_models() -> None:
    exc = RuntimeError("model failed to load (status code: 500)")
    ue = translate_inference_error(exc, "llama3:8b", phase="profiling")
    assert "SOLVE_TRI" not in str(ue)
    assert "gfx1201" not in str(ue)


def test_translate_omits_load_hints_for_non_load_errors() -> None:
    exc = RuntimeError("connection refused")
    ue = translate_inference_error(exc, "qwen3.5:9b", phase="profiling")
    assert "server.log" not in str(ue)
    assert "SOLVE_TRI" not in str(ue)
    assert "qwen3.5:9b" in str(ue)
    assert "connection refused" in str(ue)


def test_translate_uses_exception_class_when_no_message() -> None:
    class Empty(Exception):
        pass

    ue = translate_inference_error(Empty(), "foo:1b")
    assert "Empty" in str(ue)
    assert "foo:1b" in str(ue)
