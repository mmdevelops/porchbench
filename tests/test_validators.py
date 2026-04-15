"""Tests for outcome validators."""

import pytest

from feral.sandbox import SubprocessSandbox, SandboxConfig
from feral.sandbox.base import FileContent
from feral.sandbox.validators import (
    CsvSortValidator,
    CsvRowCountValidator,
    ContentContainsValidator,
    CodeOutputValidator,
    CompositeValidator,
    FileExistsValidator,
    JsonValidValidator,
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
