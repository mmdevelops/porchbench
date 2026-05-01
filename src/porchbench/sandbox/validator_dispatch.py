"""Maps validator specs from suite YAML to validator instances.

The tool-use suite defines expected_outcome with a validator type and args.
This module builds the appropriate Validator instance from that spec.
"""

from __future__ import annotations

from porchbench.sandbox.validators import (
    CodeOutputValidator,
    CompositeValidator,
    ContentContainsValidator,
    CsvRowCountValidator,
    CsvSortValidator,
    FileExistsValidator,
    JsonValidValidator,
    ResponseContainsValidator,
    Validator,
)


def build_validator(spec: dict) -> Validator | None:
    """Build a Validator from a suite YAML expected_outcome spec.

    Returns None if the spec doesn't define a validator (e.g., text-only prompts).
    """
    if not spec:
        return None

    validator_type = spec.get("validator")
    args = spec.get("validator_args", {})

    if validator_type is None:
        return None

    return _build_single(validator_type, args)


def _build_single(validator_type: str, args: dict) -> Validator:
    """Build a single validator by type name."""

    if validator_type == "file_exists":
        return FileExistsValidator(
            path=args["path"],
            min_size=args.get("min_size", 1),
        )

    if validator_type == "content_contains":
        path = args.get("path")
        if path:
            return ContentContainsValidator(
                path=path,
                required_substrings=args.get("required_substrings", []),
                case_sensitive=args.get("case_sensitive", False),
            )
        subs = args.get("response_contains", [])
        return ResponseContainsValidator(required=subs)

    if validator_type == "csv_sort":
        return CsvSortValidator(
            path=args["path"],
            sort_column=args["sort_column"],
            ascending=args.get("ascending", True),
            numeric=args.get("numeric", False),
        )

    if validator_type == "csv_row_count":
        return CsvRowCountValidator(
            path=args["path"],
            expected_rows=args.get("expected_rows"),
            min_rows=args.get("min_rows", 1),
        )

    if validator_type == "json_valid":
        return JsonValidValidator(
            path=args["path"],
            required_keys=args.get("required_keys"),
        )

    if validator_type == "code_output":
        return CodeOutputValidator(test_code=args["test_code"])

    if validator_type == "composite":
        validators = []
        for v_spec in args.get("validators", []):
            # Read non-destructively — these dicts are owned by the loaded Suite
            # and re-used across calls (e.g. routing discovery rebuilds the same
            # validators once per strategy). Popping `type` mutated suite state
            # and broke iterations 2+ with KeyError: 'type'.
            v_type = v_spec["type"]
            sub_args = {k: v for k, v in v_spec.items() if k != "type"}
            validators.append(_build_single(v_type, sub_args))
        return CompositeValidator(validators)

    raise ValueError(f"Unknown validator type: {validator_type}")
