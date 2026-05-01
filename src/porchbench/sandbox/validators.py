"""Outcome validators for tool-use benchmarks.

Each validator checks whether the sandbox state and/or the model's final
text response satisfy the task's success criteria. Validators return
(passed, reason) rather than exact content matching, making them robust
to formatting differences in model output.

The Validator protocol's ``validate`` method takes both ``sandbox`` and
``response_text``. Sandbox-only validators ignore the response text;
response-only validators (``ResponseContainsValidator``) ignore the
sandbox. Composites forward both to every sub-validator. This avoids the
silent-pass trap that the earlier ``_ResponseContainsValidator.validate``
had — there is no longer any path that returns ``(True, "deferred")``
without actually inspecting what it claims to check.
"""

from __future__ import annotations

import csv
import json
from io import StringIO
from typing import Protocol

from porchbench.sandbox.base import Sandbox


class Validator(Protocol):
    """Protocol for outcome validators. Each task provides its own."""

    async def validate(
        self, sandbox: Sandbox, response_text: str = ""
    ) -> tuple[bool, str]:
        """Check whether the task outcome is correct.

        ``sandbox`` is the post-run sandbox state (file checks).
        ``response_text`` is the final assistant message (response checks).
        Implementations use whichever they need.

        Returns:
            (True, "reason") on success
            (False, "reason explaining failure") on failure
        """
        ...


class FileExistsValidator:
    """Validates that a specific file was created in the sandbox."""

    def __init__(self, path: str, min_size: int = 1):
        self.path = path
        self.min_size = min_size

    async def validate(
        self, sandbox: Sandbox, response_text: str = ""
    ) -> tuple[bool, str]:
        try:
            content = await sandbox.read_file(self.path)
            if len(content) < self.min_size:
                return False, f"{self.path} exists but is too small ({len(content)} bytes)"
            return True, f"{self.path} exists ({len(content)} bytes)"
        except FileNotFoundError:
            return False, f"{self.path} not found"


class ContentContainsValidator:
    """Validates that a file contains all required substrings."""

    def __init__(self, path: str, required_substrings: list[str], case_sensitive: bool = False):
        self.path = path
        self.required = required_substrings
        self.case_sensitive = case_sensitive

    async def validate(
        self, sandbox: Sandbox, response_text: str = ""
    ) -> tuple[bool, str]:
        try:
            content = await sandbox.read_file(self.path)
        except FileNotFoundError:
            return False, f"{self.path} not found"

        check_content = content if self.case_sensitive else content.lower()
        missing = []
        for sub in self.required:
            check_sub = sub if self.case_sensitive else sub.lower()
            if check_sub not in check_content:
                missing.append(sub)

        if missing:
            return False, f"{self.path} missing required content: {missing}"
        return True, f"{self.path} contains all required substrings"


class ResponseContainsValidator:
    """Validates that the model's final text response contains required substrings.

    Unlike sandbox file validators, this inspects the conversation
    transcript — used for tasks where the answer is in the model's reply,
    not in a file. ``sandbox`` is accepted for protocol uniformity and
    ignored.
    """

    def __init__(self, required: list[str]):
        self.required = required

    async def validate(
        self, sandbox: Sandbox, response_text: str = ""
    ) -> tuple[bool, str]:
        text_lower = response_text.lower()
        missing = [s for s in self.required if s.lower() not in text_lower]
        if missing:
            return False, f"Response missing: {missing}"
        return True, "Response contains all required substrings"


class CsvSortValidator:
    """Validates that a CSV file is sorted by a specific column."""

    def __init__(
        self,
        path: str,
        sort_column: str,
        ascending: bool = True,
        numeric: bool = False,
    ):
        self.path = path
        self.sort_column = sort_column
        self.ascending = ascending
        self.numeric = numeric

    async def validate(
        self, sandbox: Sandbox, response_text: str = ""
    ) -> tuple[bool, str]:
        try:
            content = await sandbox.read_file(self.path)
        except FileNotFoundError:
            return False, f"{self.path} not found"

        try:
            reader = csv.DictReader(StringIO(content))
            rows = list(reader)
        except Exception as exc:
            return False, f"{self.path} is not valid CSV: {exc}"

        if not rows:
            return False, f"{self.path} is empty"

        if self.sort_column not in rows[0]:
            available = list(rows[0].keys())
            return False, f"Column '{self.sort_column}' not found. Available: {available}"

        values = [r[self.sort_column] for r in rows]
        if self.numeric:
            try:
                parsed = [float(v) for v in values]
            except ValueError:
                return False, f"Column '{self.sort_column}' contains non-numeric values"
            sorted_vals = sorted(parsed, reverse=not self.ascending)
            if parsed == sorted_vals:
                return True, f"{self.path} is correctly sorted by {self.sort_column}"
            return False, f"{self.path} is not sorted by {self.sort_column}"
        else:
            sorted_vals = sorted(values, reverse=not self.ascending)
            if values == sorted_vals:
                return True, f"{self.path} is correctly sorted by {self.sort_column}"
            return False, f"{self.path} is not sorted by {self.sort_column}"


class CsvRowCountValidator:
    """Validates that a CSV file has the expected number of data rows."""

    def __init__(self, path: str, expected_rows: int | None = None, min_rows: int = 1):
        self.path = path
        self.expected_rows = expected_rows
        self.min_rows = min_rows

    async def validate(
        self, sandbox: Sandbox, response_text: str = ""
    ) -> tuple[bool, str]:
        try:
            content = await sandbox.read_file(self.path)
        except FileNotFoundError:
            return False, f"{self.path} not found"

        try:
            reader = csv.DictReader(StringIO(content))
            rows = list(reader)
        except Exception as exc:
            return False, f"{self.path} is not valid CSV: {exc}"

        if self.expected_rows is not None and len(rows) != self.expected_rows:
            return False, f"{self.path} has {len(rows)} rows, expected {self.expected_rows}"
        if len(rows) < self.min_rows:
            return False, f"{self.path} has {len(rows)} rows, expected at least {self.min_rows}"
        return True, f"{self.path} has {len(rows)} rows"


class JsonValidValidator:
    """Validates that a file contains valid JSON with optional key checks."""

    def __init__(self, path: str, required_keys: list[str] | None = None):
        self.path = path
        self.required_keys = required_keys or []

    async def validate(
        self, sandbox: Sandbox, response_text: str = ""
    ) -> tuple[bool, str]:
        try:
            content = await sandbox.read_file(self.path)
        except FileNotFoundError:
            return False, f"{self.path} not found"

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            return False, f"{self.path} is not valid JSON: {exc}"

        if self.required_keys:
            if isinstance(data, dict):
                missing = [k for k in self.required_keys if k not in data]
                if missing:
                    return False, f"{self.path} missing required keys: {missing}"
            elif isinstance(data, list) and data:
                missing = [k for k in self.required_keys if k not in data[0]]
                if missing:
                    return False, f"{self.path}[0] missing required keys: {missing}"

        return True, f"{self.path} is valid JSON"


class CodeOutputValidator:
    """Validates by executing a test script and checking its exit code."""

    def __init__(self, test_code: str):
        self.test_code = test_code

    async def validate(
        self, sandbox: Sandbox, response_text: str = ""
    ) -> tuple[bool, str]:
        from porchbench.sandbox.base import ExecutionRequest

        result = await sandbox.execute(
            ExecutionRequest(code=self.test_code, filename="_validate.py")
        )
        if result.exit_code == 0:
            return True, f"Validation passed: {result.stdout.strip()[:100]}"
        summary = _final_stderr_line(result.stderr)
        return False, f"Validation failed: {summary[:200]}"


def _final_stderr_line(stderr: str) -> str:
    """Pick the most informative single line out of a stderr blob.

    Python tracebacks place the actual exception (`ExceptionType: message`)
    on the final non-empty line; the leading lines are file/line context
    that's useful in the JSON for debugging but uninformative in the
    per-prompt val-fail badge. Returning the last non-empty line keeps
    both call sites readable: the badge gets a clean one-line summary
    and the stored ``validation_reason`` remains a single line that
    survives `splitlines()[0]` trimming downstream.
    """
    lines = [line for line in stderr.strip().splitlines() if line.strip()]
    return lines[-1] if lines else stderr.strip()


class CompositeValidator:
    """Combines multiple validators. All must pass."""

    def __init__(self, validators: list[Validator]):
        self.validators = validators

    async def validate(
        self, sandbox: Sandbox, response_text: str = ""
    ) -> tuple[bool, str]:
        results = []
        all_passed = True
        for v in self.validators:
            passed, reason = await v.validate(sandbox, response_text)
            results.append(reason)
            if not passed:
                all_passed = False

        # Dedupe identical reasons while preserving order — when sub-validators
        # all fail with the same root cause (e.g. multiple checks against a
        # file that doesn't exist), the user sees one 'X not found' instead
        # of N repeats of the same message.
        deduped = list(dict.fromkeys(results))
        summary = "; ".join(deduped)
        return all_passed, summary
