"""Maps validator specs from suite YAML to validator instances.

The tool-use suite defines expected_outcome with a validator type and args.
This module builds the appropriate Validator instance from that spec.
"""

from __future__ import annotations

from ollama_bench.sandbox.validators import (
    CodeOutputValidator,
    CompositeValidator,
    ContentContainsValidator,
    CsvRowCountValidator,
    CsvSortValidator,
    FileExistsValidator,
    JsonValidValidator,
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
        return _ResponseContainsValidator(required=subs)

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
            v_type = v_spec.pop("type")
            validators.append(_build_single(v_type, v_spec))
        return CompositeValidator(validators)

    raise ValueError(f"Unknown validator type: {validator_type}")


class _ResponseContainsValidator:
    """Checks that the model's final text response contains required substrings.

    This isn't a sandbox file check -- it validates the conversation transcript.
    Used for tasks where the answer is in the model's reply, not in a file.
    """

    def __init__(self, required: list[str]):
        self.required = required

    async def validate(self, sandbox) -> tuple[bool, str]:
        return True, "Response validation deferred to transcript check"

    def check_response(self, response_text: str) -> tuple[bool, str]:
        text_lower = response_text.lower()
        missing = [s for s in self.required if s.lower() not in text_lower]
        if missing:
            return False, f"Response missing: {missing}"
        return True, "Response contains all required substrings"
