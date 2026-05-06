"""Schema round-trip for bundled examples/ artifacts.

Catches schema drift: if a future change to `RunResult`, `Scorecard`,
`RoutingAnalysis`, or `SystemProfile` makes a bundled example
uninstantiable, this test fails before the wheel ships.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from porchbench import (
    RoutingAnalysis,
    RunResult,
    Scorecard,
    SystemProfile,
)

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _dispatch_schema(data: dict[str, Any]) -> type:
    """Pick the right schema for a top-level JSON object by its keys."""
    if "run" in data and "results" in data:
        return RunResult
    if "evaluation" in data and "scores" in data:
        return Scorecard
    if "models_tested" in data and "strategies_tested" in data:
        return RoutingAnalysis
    if "gpu" in data and "ollama_version" in data:
        return SystemProfile
    raise ValueError(
        f"could not dispatch schema for top-level keys: {sorted(data)}"
    )


_EXAMPLE_PATHS = sorted(EXAMPLES_DIR.glob("*.json"))


@pytest.mark.parametrize("path", _EXAMPLE_PATHS, ids=lambda p: p.name)
def test_example_validates_and_roundtrips(path: Path) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    schema_cls = _dispatch_schema(raw)

    instance = schema_cls.model_validate(raw)
    re_dumped = json.loads(instance.model_dump_json())
    re_validated = schema_cls.model_validate(re_dumped)

    assert isinstance(re_validated, schema_cls)


def test_examples_directory_not_empty() -> None:
    """Guard against a regen accidentally deleting the bundled samples."""
    assert _EXAMPLE_PATHS, (
        f"no JSON examples found under {EXAMPLES_DIR}; "
        "bundled examples may have been deleted"
    )
