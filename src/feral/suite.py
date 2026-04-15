"""Suite loading, validation, and option merging.

Loads YAML suite files, validates them against the Suite schema, computes
a SHA256 content hash for reproducibility tracking, and provides helpers
for resolving per-prompt options against suite defaults.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from feral.schemas import (
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
    """Build a SuiteReference for embedding in run results."""
    path = Path(path)
    return SuiteReference(
        name=suite.suite.name,
        version=suite.suite.version,
        file=str(path),
        sha256=compute_suite_hash(path),
    )


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


def resolve_messages(prompt: Prompt, system_message: str | None = None) -> list[Message]:
    """Build the final message list, optionally prepending a system message.

    Used by routing discovery to inject strategy system prompts.
    """
    messages: list[Message] = []
    if system_message:
        messages.append(Message(role="system", content=system_message))
    messages.extend(prompt.messages)
    return messages
