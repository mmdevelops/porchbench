"""Tests for outcome validators."""

import pytest

from porchbench.sandbox import SandboxConfig, SubprocessSandbox
from porchbench.sandbox.base import FileContent
from porchbench.sandbox.validators import (
    CodeOutputValidator,
    CompositeValidator,
    ContentContainsValidator,
    CsvRowCountValidator,
    CsvSortValidator,
    FileExistsValidator,
    JsonValidValidator,
    ResponseContainsValidator,
)


@pytest.fixture
async def sandbox():
    sb = SubprocessSandbox()
    await sb.create(SandboxConfig(timeout_s=5))
    yield sb
    await sb.destroy()


class TestFileExistsValidator:
    @pytest.mark.asyncio
    async def test_exists(self, sandbox):
        await sandbox.write_files([FileContent(path="test.txt", content="hello")])
        v = FileExistsValidator("test.txt")
        passed, reason = await v.validate(sandbox)
        assert passed is True

    @pytest.mark.asyncio
    async def test_not_exists(self, sandbox):
        v = FileExistsValidator("missing.txt")
        passed, reason = await v.validate(sandbox)
        assert passed is False
        assert "not found" in reason


class TestContentContainsValidator:
    @pytest.mark.asyncio
    async def test_contains(self, sandbox):
        await sandbox.write_files([FileContent(path="out.txt", content="Hello World")])
        v = ContentContainsValidator("out.txt", ["hello", "world"])
        passed, _ = await v.validate(sandbox)
        assert passed is True

    @pytest.mark.asyncio
    async def test_missing_substring(self, sandbox):
        await sandbox.write_files([FileContent(path="out.txt", content="Hello")])
        v = ContentContainsValidator("out.txt", ["hello", "goodbye"])
        passed, reason = await v.validate(sandbox)
        assert passed is False
        assert "goodbye" in reason


class TestCsvSortValidator:
    @pytest.mark.asyncio
    async def test_sorted_numeric(self, sandbox):
        csv_data = "name,price\nbanana,0.75\napple,1.50\ncherry,3.00\n"
        await sandbox.write_files([FileContent(path="sorted.csv", content=csv_data)])
        v = CsvSortValidator("sorted.csv", "price", ascending=True, numeric=True)
        passed, _ = await v.validate(sandbox)
        assert passed is True

    @pytest.mark.asyncio
    async def test_not_sorted(self, sandbox):
        csv_data = "name,price\napple,1.50\nbanana,0.75\ncherry,3.00\n"
        await sandbox.write_files([FileContent(path="unsorted.csv", content=csv_data)])
        v = CsvSortValidator("unsorted.csv", "price", ascending=True, numeric=True)
        passed, _ = await v.validate(sandbox)
        assert passed is False

    @pytest.mark.asyncio
    async def test_sorted_alpha(self, sandbox):
        csv_data = "name,price\napple,1.50\nbanana,0.75\ncherry,3.00\n"
        await sandbox.write_files([FileContent(path="alpha.csv", content=csv_data)])
        v = CsvSortValidator("alpha.csv", "name", ascending=True)
        passed, _ = await v.validate(sandbox)
        assert passed is True


class TestCsvRowCountValidator:
    @pytest.mark.asyncio
    async def test_exact_count(self, sandbox):
        csv_data = "a,b\n1,2\n3,4\n5,6\n"
        await sandbox.write_files([FileContent(path="data.csv", content=csv_data)])
        v = CsvRowCountValidator("data.csv", expected_rows=3)
        passed, _ = await v.validate(sandbox)
        assert passed is True

    @pytest.mark.asyncio
    async def test_wrong_count(self, sandbox):
        csv_data = "a,b\n1,2\n"
        await sandbox.write_files([FileContent(path="data.csv", content=csv_data)])
        v = CsvRowCountValidator("data.csv", expected_rows=5)
        passed, _ = await v.validate(sandbox)
        assert passed is False


class TestJsonValidValidator:
    @pytest.mark.asyncio
    async def test_valid_json(self, sandbox):
        await sandbox.write_files([FileContent(path="data.json", content='{"name": "test"}')])
        v = JsonValidValidator("data.json", required_keys=["name"])
        passed, _ = await v.validate(sandbox)
        assert passed is True

    @pytest.mark.asyncio
    async def test_invalid_json(self, sandbox):
        await sandbox.write_files([FileContent(path="bad.json", content="not json")])
        v = JsonValidValidator("bad.json")
        passed, _ = await v.validate(sandbox)
        assert passed is False

    @pytest.mark.asyncio
    async def test_missing_key(self, sandbox):
        await sandbox.write_files([FileContent(path="data.json", content='{"a": 1}')])
        v = JsonValidValidator("data.json", required_keys=["b"])
        passed, reason = await v.validate(sandbox)
        assert passed is False


class TestCodeOutputValidator:
    @pytest.mark.asyncio
    async def test_passing_test(self, sandbox):
        await sandbox.write_files([FileContent(path="result.txt", content="42")])
        test_code = (
            "content = open('result.txt').read().strip()\n"
            "assert content == '42', f'Expected 42, got {content}'\n"
            "print('OK')\n"
        )
        v = CodeOutputValidator(test_code)
        passed, _ = await v.validate(sandbox)
        assert passed is True

    @pytest.mark.asyncio
    async def test_failing_test(self, sandbox):
        await sandbox.write_files([FileContent(path="result.txt", content="wrong")])
        test_code = (
            "content = open('result.txt').read().strip()\n"
            "assert content == '42', f'Expected 42, got {content}'\n"
        )
        v = CodeOutputValidator(test_code)
        passed, _ = await v.validate(sandbox)
        assert passed is False

    @pytest.mark.asyncio
    async def test_failing_reason_surfaces_exception_summary(self, sandbox):
        """Reason should be the final ExceptionType: message line, not the
        Traceback header. Without this, the per-prompt val-fail badge shows
        ``Validation failed: Traceback (most recent call last):`` which
        tells users nothing — observed in routing-discovery runs 2026-05-01.
        """
        await sandbox.write_files([FileContent(path="result.txt", content="wrong")])
        test_code = (
            "content = open('result.txt').read().strip()\n"
            "assert content == '42', f'Expected 42, got {content}'\n"
        )
        v = CodeOutputValidator(test_code)
        passed, reason = await v.validate(sandbox)
        assert passed is False
        assert "Traceback" not in reason
        assert "AssertionError" in reason
        assert "Expected 42, got wrong" in reason
        # And the reason fits on one line so splitlines()[0] in the badge
        # formatter renders the whole summary, not a partial line.
        assert "\n" not in reason


class TestCompositeValidator:
    @pytest.mark.asyncio
    async def test_all_pass(self, sandbox):
        await sandbox.write_files([FileContent(path="out.txt", content="hello world")])
        v = CompositeValidator([
            FileExistsValidator("out.txt"),
            ContentContainsValidator("out.txt", ["hello"]),
        ])
        passed, _ = await v.validate(sandbox)
        assert passed is True

    @pytest.mark.asyncio
    async def test_one_fails(self, sandbox):
        await sandbox.write_files([FileContent(path="out.txt", content="hello")])
        v = CompositeValidator([
            FileExistsValidator("out.txt"),
            ContentContainsValidator("out.txt", ["goodbye"]),
        ])
        passed, _ = await v.validate(sandbox)
        assert passed is False

    @pytest.mark.asyncio
    async def test_dedupes_identical_failure_reasons(self, sandbox):
        """Multiple sub-validators against a missing file should report it once.

        Without dedupe the user sees 'X.csv not found; X.csv not found' for
        every sub-check that hit the same FileNotFoundError, which is noise.
        """
        v = CompositeValidator([
            FileExistsValidator("missing.csv"),
            ContentContainsValidator("missing.csv", ["whatever"]),
        ])
        passed, reason = await v.validate(sandbox)
        assert passed is False
        # Both sub-validators raise the same 'missing.csv not found' message;
        # dedupe collapses them to one occurrence.
        assert reason.count("missing.csv not found") == 1

    @pytest.mark.asyncio
    async def test_preserves_distinct_reasons(self, sandbox):
        """Dedupe must not collapse genuinely different reasons."""
        await sandbox.write_files([FileContent(path="out.txt", content="hi")])
        v = CompositeValidator([
            FileExistsValidator("out.txt", min_size=100),  # too-small failure
            ContentContainsValidator("out.txt", ["bye"]),  # missing-substring failure
        ])
        passed, reason = await v.validate(sandbox)
        assert passed is False
        assert "too small" in reason
        assert "Missing required substring" in reason or "missing" in reason.lower()


class TestResponseContainsValidator:
    """Pins the protocol-level fix for the silent-pass trap.

    Earlier ``_ResponseContainsValidator.validate(sandbox)`` always returned
    ``(True, "deferred...")`` regardless of response content, so a Composite
    that wrapped one alongside file checks reported the response check as
    passed even when it should have failed. The new ``Validator.validate``
    signature accepts ``response_text`` so dispatch is uniform and the
    response check actually inspects the text it claims to inspect.
    """

    @pytest.mark.asyncio
    async def test_response_contains_pass(self, sandbox):
        v = ResponseContainsValidator(required=["paris"])
        passed, _ = await v.validate(sandbox, response_text="The answer is Paris.")
        assert passed is True

    @pytest.mark.asyncio
    async def test_response_contains_fail(self, sandbox):
        v = ResponseContainsValidator(required=["paris"])
        passed, reason = await v.validate(sandbox, response_text="No idea.")
        assert passed is False
        assert "paris" in reason.lower()

    @pytest.mark.asyncio
    async def test_response_contains_case_insensitive(self, sandbox):
        v = ResponseContainsValidator(required=["PARIS"])
        passed, _ = await v.validate(sandbox, response_text="paris")
        assert passed is True

    @pytest.mark.asyncio
    async def test_composite_does_not_silently_pass_response_check(self, sandbox):
        """Regression for the W1 silent-pass trap: a composite wrapping a
        ResponseContainsValidator alongside file checks must fail when the
        response is missing the required substring, even if file checks pass.
        """
        await sandbox.write_files([FileContent(path="out.txt", content="ok")])
        v = CompositeValidator([
            FileExistsValidator("out.txt"),
            ResponseContainsValidator(required=["paris"]),
        ])
        passed, reason = await v.validate(sandbox, response_text="no answer")
        assert passed is False
        assert "paris" in reason.lower() or "missing" in reason.lower()

    @pytest.mark.asyncio
    async def test_composite_passes_when_both_satisfied(self, sandbox):
        await sandbox.write_files([FileContent(path="out.txt", content="ok")])
        v = CompositeValidator([
            FileExistsValidator("out.txt"),
            ResponseContainsValidator(required=["paris"]),
        ])
        passed, _ = await v.validate(sandbox, response_text="Paris is the answer.")
        assert passed is True
