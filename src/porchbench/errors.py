"""User-facing error helpers for the CLI.

Wraps filesystem and Pydantic errors in short, actionable messages so that
first-time users don't see raw `[Errno 2]` strings or Pydantic's
`errors.pydantic.dev` links.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

import pydantic

T = TypeVar("T", bound=pydantic.BaseModel)


class UserError(Exception):
    """Error with a message intended for direct display to the user."""


def load_json_model(path: str | Path, model_cls: type[T], label: str) -> T:
    """Load a JSON file and validate against a Pydantic model.

    Raises UserError with a friendly message on any failure (file missing,
    unreadable, invalid JSON, or schema mismatch). `label` is a short noun
    like "run result", "scorecard", or "rubric" used in messages.
    """
    path = Path(path)

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise UserError(f"{label.capitalize()} file not found: {path}")
    except PermissionError:
        raise UserError(f"Permission denied reading {label} at {path}")
    except OSError as exc:
        raise UserError(f"Could not read {label} at {path}: {exc.strerror or exc}")

    try:
        return model_cls.model_validate_json(text)
    except json.JSONDecodeError as exc:
        raise UserError(
            f"{label.capitalize()} file is not valid JSON: {path} "
            f"(line {exc.lineno}, column {exc.colno})"
        )
    except pydantic.ValidationError as exc:
        fields = _summarize_missing_fields(exc)
        if fields:
            raise UserError(
                f"{path} doesn't look like a porchbench {label} file "
                f"(missing or invalid: {fields})"
            )
        raise UserError(f"{path} doesn't look like a porchbench {label} file")


def _summarize_missing_fields(exc: pydantic.ValidationError) -> str:
    """Render a compact 'field1, field2' summary from a ValidationError."""
    names: list[str] = []
    for err in exc.errors():
        loc = err.get("loc", ())
        if loc:
            names.append(".".join(str(part) for part in loc))
        if len(names) >= 5:
            names.append("…")
            break
    return ", ".join(names)


def translate_inference_error(exc: Exception, model: str, phase: str = "inference") -> UserError:
    """Wrap an Ollama inference failure in a UserError with diagnostic context.

    Adds the model name, phase (e.g. "profiling", "swap measurement"), a
    pointer to the Ollama server log, and a specific hint for the known
    AMD RDNA 4 (gfx1201) SOLVE_TRI kernel gap when a Qwen 3.5 family
    model hits a load-time 500.
    """
    detail = str(exc).strip() or exc.__class__.__name__
    lines = [f"{phase.capitalize()} failed for model '{model}': {detail}"]

    looks_like_load_failure = (
        "model failed to load" in detail.lower()
        or "status code: 500" in detail.lower()
    )
    if looks_like_load_failure:
        lines.append(
            "Check the Ollama server log for the underlying cause "
            "(Windows: %LOCALAPPDATA%\\Ollama\\server.log; "
            "macOS: ~/.ollama/logs/server.log; "
            "Linux: journalctl -u ollama -n 100)."
        )
        is_qwen35 = any(tag in model.lower() for tag in ("qwen3.5", "qwen-3.5", "qwen35"))
        if is_qwen35:
            lines.append(
                "This model is in the Qwen 3.5 family. On AMD RDNA 4 (gfx1201), "
                "Qwen 3.5 commonly hits a known 'SOLVE_TRI failed' kernel gap in "
                "Ollama's rocBLAS. Swap to qwen2.5 / llama / gemma until Ollama "
                "ships an updated ROCm build."
            )

    return UserError("\n".join(lines))
