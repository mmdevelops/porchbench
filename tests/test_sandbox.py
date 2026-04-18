"""Tests for the subprocess sandbox backend."""

import pytest

from porchbench.sandbox import SandboxConfig, SubprocessSandbox
from porchbench.sandbox.base import ExecutionRequest, FileContent


@pytest.fixture
async def sandbox():
    sb = SubprocessSandbox()
    await sb.create(SandboxConfig(timeout_s=5))
    yield sb
    await sb.destroy()


class TestSubprocessSandbox:
    @pytest.mark.asyncio
    async def test_basic_execution(self, sandbox):
        r = await sandbox.execute(ExecutionRequest(code='print("hello")'))
        assert r.stdout.strip() == "hello"
        assert r.exit_code == 0

    @pytest.mark.asyncio
    async def test_write_and_read_file(self, sandbox):
        await sandbox.write_files([FileContent(path="data.txt", content="line1\nline2")])
        content = await sandbox.read_file("data.txt")
        assert "line1" in content
        assert "line2" in content

    @pytest.mark.asyncio
    async def test_code_reads_written_file(self, sandbox):
        await sandbox.write_files([FileContent(path="input.txt", content="42")])
        r = await sandbox.execute(
            ExecutionRequest(code='print(open("input.txt").read().strip())')
        )
        assert r.stdout.strip() == "42"

    @pytest.mark.asyncio
    async def test_code_writes_file(self, sandbox):
        code = 'with open("out.txt", "w") as f:\n    f.write("result")\n'
        r = await sandbox.execute(ExecutionRequest(code=code))
        assert r.exit_code == 0
        content = await sandbox.read_file("out.txt")
        assert content == "result"

    @pytest.mark.asyncio
    async def test_state_persists_across_executions(self, sandbox):
        await sandbox.execute(
            ExecutionRequest(code='with open("state.txt", "w") as f:\n    f.write("first")\n')
        )
        r = await sandbox.execute(
            ExecutionRequest(code='print(open("state.txt").read())')
        )
        assert r.stdout.strip() == "first"

    @pytest.mark.asyncio
    async def test_timeout(self, sandbox):
        r = await sandbox.execute(
            ExecutionRequest(code="import time; time.sleep(30)")
        )
        assert r.timed_out is True
        assert r.exit_code == -1

    @pytest.mark.asyncio
    async def test_error_exit_code(self, sandbox):
        r = await sandbox.execute(
            ExecutionRequest(code='raise ValueError("oops")')
        )
        assert r.exit_code != 0
        assert "ValueError" in r.stderr

    @pytest.mark.asyncio
    async def test_file_not_found(self, sandbox):
        with pytest.raises(FileNotFoundError):
            await sandbox.read_file("nonexistent.txt")

    @pytest.mark.asyncio
    async def test_unsupported_language(self, sandbox):
        r = await sandbox.execute(
            ExecutionRequest(code="fn main() {}", language="rust")
        )
        assert r.exit_code == 1
        assert "Unsupported" in r.stderr

    @pytest.mark.asyncio
    async def test_nested_directory_write(self, sandbox):
        await sandbox.write_files([
            FileContent(path="subdir/nested.txt", content="deep")
        ])
        content = await sandbox.read_file("subdir/nested.txt")
        assert content == "deep"

    @pytest.mark.asyncio
    async def test_destroy_cleans_up(self):
        sb = SubprocessSandbox()
        await sb.create(SandboxConfig())
        workdir = sb.workdir
        assert workdir.exists()
        await sb.destroy()
        assert not workdir.exists()
