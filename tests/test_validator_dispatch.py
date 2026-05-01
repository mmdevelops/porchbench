"""Tests for build_validator — the spec→Validator dispatch layer.

Composite validator regression: prior to the fix, _build_single popped
"type" from each sub-validator dict in place. Routing discovery rebuilds
the same composite spec once per strategy (4× for the bundled tool-use
suite), so iteration 2 raised KeyError: 'type'. This file pins the
non-mutating contract.
"""

from __future__ import annotations

import copy

import pytest

from porchbench.sandbox.validator_dispatch import build_validator
from porchbench.sandbox.validators import (
    CompositeValidator,
    ContentContainsValidator,
    CsvRowCountValidator,
    FileExistsValidator,
)


def test_build_single_validator_returns_expected_type():
    v = build_validator({
        "validator": "file_exists",
        "validator_args": {"path": "out.txt"},
    })
    assert isinstance(v, FileExistsValidator)


def test_build_composite_returns_composite_with_subs():
    v = build_validator({
        "validator": "composite",
        "validator_args": {
            "validators": [
                {"type": "csv_row_count", "path": "dairy.csv", "expected_rows": 3},
                {"type": "content_contains", "path": "dairy.csv",
                 "required_substrings": ["milk"]},
            ],
        },
    })
    assert isinstance(v, CompositeValidator)


def test_build_composite_does_not_mutate_spec():
    """The dispatch layer must read the suite-owned dict non-destructively.
    Routing discovery rebuilds the same composite spec once per strategy;
    a destructive read corrupts iterations 2+."""
    spec = {
        "validator": "composite",
        "validator_args": {
            "validators": [
                {"type": "csv_row_count", "path": "x.csv", "expected_rows": 3},
                {"type": "content_contains", "path": "x.csv",
                 "required_substrings": ["a"]},
            ],
        },
    }
    snapshot = copy.deepcopy(spec)

    build_validator(spec)

    assert spec == snapshot, "build_validator mutated the spec"


def test_build_composite_succeeds_when_called_repeatedly_on_same_spec():
    """Direct regression for the routing-discovery KeyError: 'type' bug.
    The same spec dict (as it would come from a loaded Suite) must build
    cleanly on every call."""
    spec = {
        "validator": "composite",
        "validator_args": {
            "validators": [
                {"type": "csv_row_count", "path": "dairy.csv", "expected_rows": 3},
                {"type": "content_contains", "path": "dairy.csv",
                 "required_substrings": ["milk", "eggs"]},
            ],
        },
    }

    # Mirror the routing-discovery loop: build once per strategy.
    for _ in range(4):
        v = build_validator(spec)
        assert isinstance(v, CompositeValidator)


def test_build_composite_sub_validator_args_propagate():
    """Confirm sub-validators receive their args correctly after the
    type-stripping rewrite."""
    v = build_validator({
        "validator": "composite",
        "validator_args": {
            "validators": [
                {"type": "csv_row_count", "path": "x.csv",
                 "expected_rows": 7, "min_rows": 1},
            ],
        },
    })
    assert isinstance(v, CompositeValidator)
    sub = v.validators[0]
    assert isinstance(sub, CsvRowCountValidator)
    assert sub.path == "x.csv"
    assert sub.expected_rows == 7


def test_build_returns_none_for_empty_spec():
    assert build_validator({}) is None
    assert build_validator({"validator_args": {}}) is None


def test_build_unknown_validator_raises():
    with pytest.raises(ValueError, match="Unknown validator type"):
        build_validator({"validator": "totally-fake", "validator_args": {}})
