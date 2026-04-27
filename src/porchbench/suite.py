"""Suite loading, validation, and option merging.

Loads YAML suite files, validates them against the Suite schema, computes
a SHA256 content hash for reproducibility tracking, and provides helpers
for resolving per-prompt options against suite defaults.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from porchbench.schemas import (
    Message,
    ModelOptions,
    Prompt,
    Suite,
    SuiteReference,
)


def load_suite(path: str | Path) -> Suite:
    """Load and validate a suite YAML file. Raises on parse or validation errors."""
    path = Path(path)
    raw_text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw_text)
    if not isinstance(data, dict):
        raise ValueError(f"Suite file {path} did not parse as a YAML mapping")
    return Suite.model_validate(data)


def compute_suite_hash(path: str | Path) -> str:
    """SHA256 of the suite file contents for reproducibility tracking."""
    path = Path(path)
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def make_suite_reference(path: str | Path, suite: Suite) -> SuiteReference:
    """Build a SuiteReference for embedding in run results.

    The `file` field uses a portable identifier instead of the absolute path
    so result JSONs don't leak local filesystem layout. The sha256 is the
    real reproducibility anchor.
    """
    path = Path(path)
    return SuiteReference(
        name=suite.suite.name,
        version=suite.suite.version,
        file=_portable_suite_id(path),
        sha256=compute_suite_hash(path),
        rubric=suite.suite.rubric,
    )


def _portable_suite_id(path: Path) -> str:
    """Format a suite path for result metadata without leaking absolute paths.

    Packaged defaults → `<bundled>/<name>.yaml`.
    Anything else → basename only. The sha256 in SuiteReference is the
    reproducibility anchor; this string is informational.
    """
    from porchbench.assets import PACKAGED_SUITES_DIR

    try:
        rel = path.resolve().relative_to(PACKAGED_SUITES_DIR.resolve())
        return f"<bundled>/{rel.as_posix()}"
    except ValueError:
        return path.name


def discover_suites(suite_dir: Path) -> list[Path]:
    """Find all .yaml suite files in a directory, sorted by name."""
    if not suite_dir.is_dir():
        raise FileNotFoundError(f"Suite directory not found: {suite_dir}")
    paths = sorted(suite_dir.glob("*.yaml"))
    if not paths:
        raise FileNotFoundError(f"No .yaml files found in {suite_dir}")
    return paths


def resolve_options(suite_defaults: ModelOptions, prompt: Prompt) -> ModelOptions:
    """Merge per-prompt option overrides over suite defaults.

    Suite defaults provide the base. Per-prompt options override individual fields.
    Any extra Ollama options in either layer are preserved.
    """
    base = suite_defaults.model_dump()
    if prompt.options is not None:
        overrides = prompt.options.model_dump(exclude_unset=True)
        base.update(overrides)
    return ModelOptions.model_validate(base)


def apply_option_overrides(suite: Suite, overrides: dict[str, object]) -> Suite:
    """Layer CLI-supplied overrides onto the suite's defaults.options.

    Returns a new suite with merged defaults; per-prompt option overrides in
    `prompts[].options` remain authoritative for those prompts (resolved later
    by `resolve_options`). Pass-through fields are preserved via ModelOptions'
    `extra="allow"` config.
    """
    if not overrides:
        return suite
    merged = suite.defaults.options.model_dump()
    merged.update(overrides)
    new_options = ModelOptions.model_validate(merged)
    new_defaults = suite.defaults.model_copy(update={"options": new_options})
    return suite.model_copy(update={"defaults": new_defaults})


def resolve_messages(prompt: Prompt, system_message: str | None = None) -> list[Message]:
    """Build the final message list, optionally prepending a system message.

    Used by routing discovery to inject strategy system prompts.
    """
    messages: list[Message] = []
    if system_message:
        messages.append(Message(role="system", content=system_message))
    messages.extend(prompt.messages)
    return messages
