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
                f"{path} doesn't look like a feral {label} file "
                f"(missing or invalid: {fields})"
            )
        raise UserError(f"{path} doesn't look like a feral {label} file")


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
