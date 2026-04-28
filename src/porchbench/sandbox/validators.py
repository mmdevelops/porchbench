"""Outcome validators for tool-use benchmarks.

Each validator checks whether the sandbox state after a harness run
satisfies the task's success criteria. Validators return (passed, reason)
rather than exact content matching, making them robust to formatting
differences in model output.
"""

from __future__ import annotations

import csv
import json
from io import StringIO
from typing import Protocol

from porchbench.sandbox.base import Sandbox


class Validator(Protocol):
    """Protocol for outcome validators. Each task provides its own."""

    async def validate(self, sandbox: Sandbox) -> tuple[bool, str]:
        """Check whether the task outcome is correct.

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

    async def validate(self, sandbox: Sandbox) -> tuple[bool, str]:
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

    async def validate(self, sandbox: Sandbox) -> tuple[bool, str]:
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

    async def validate(self, sandbox: Sandbox) -> tuple[bool, str]:
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

    async def validate(self, sandbox: Sandbox) -> tuple[bool, str]:
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

    async def validate(self, sandbox: Sandbox) -> tuple[bool, str]:
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

    async def validate(self, sandbox: Sandbox) -> tuple[bool, str]:
        from porchbench.sandbox.base import ExecutionRequest

        result = await sandbox.execute(
            ExecutionRequest(code=self.test_code, filename="_validate.py")
        )
        if result.exit_code == 0:
            return True, f"Validation passed: {result.stdout.strip()[:100]}"
        return False, f"Validation failed: {result.stderr.strip()[:200]}"


class CompositeValidator:
    """Combines multiple validators. All must pass."""

    def __init__(self, validators: list[Validator]):
        self.validators = validators

    async def validate(self, sandbox: Sandbox) -> tuple[bool, str]:
        results = []
        all_passed = True
        for v in self.validators:
            passed, reason = await v.validate(sandbox)
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
