"""Tests for the subprocess sandbox backend."""

import pytest

from porchbench.sandbox import SandboxConfig, SubprocessSandbox
from porchbench.sandbox.base import ExecutionRequest, FileContent
from porchbench.sandbox.subprocess_backend import SandboxPathError


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


class TestSandboxPathContainment:
    """Containment checks: caller-supplied paths must not escape the workdir.

    Pins the security contract that the Phase 1 sandbox enforces filesystem
    isolation at the API boundary even though the underlying process has
    full host access. Without these guards a model-driven write_file call
    with ``"../../etc/passwd"`` would clobber arbitrary host paths.
    """

    @pytest.mark.asyncio
    async def test_write_files_rejects_parent_traversal(self, sandbox):
        with pytest.raises(SandboxPathError):
            await sandbox.write_files([
                FileContent(path="../escape.txt", content="x")
            ])

    @pytest.mark.asyncio
    async def test_write_files_rejects_deep_traversal(self, sandbox):
        with pytest.raises(SandboxPathError):
            await sandbox.write_files([
                FileContent(path="../../etc/passwd", content="x")
            ])

    @pytest.mark.asyncio
    async def test_write_files_rejects_absolute_path(self, sandbox, tmp_path):
        target = tmp_path / "outside.txt"
        with pytest.raises(SandboxPathError):
            await sandbox.write_files([
                FileContent(path=str(target), content="x")
            ])
        assert not target.exists()

    @pytest.mark.asyncio
    async def test_read_file_rejects_parent_traversal(self, sandbox):
        with pytest.raises(SandboxPathError):
            await sandbox.read_file("../somefile.txt")

    @pytest.mark.asyncio
    async def test_read_file_rejects_absolute_path(self, sandbox, tmp_path):
        target = tmp_path / "outside.txt"
        target.write_text("secret", encoding="utf-8")
        with pytest.raises(SandboxPathError):
            await sandbox.read_file(str(target))

    @pytest.mark.asyncio
    async def test_execute_rejects_filename_with_separators(self, sandbox):
        # Filenames with path separators are rejected outright — a
        # legitimate code filename has none.
        r = await sandbox.execute(
            ExecutionRequest(code='print("x")', filename="../escape.py")
        )
        assert r.exit_code == 1
        assert "Invalid execution filename" in r.stderr

    @pytest.mark.asyncio
    async def test_execute_rejects_absolute_filename(self, sandbox, tmp_path):
        target = tmp_path / "evil.py"
        r = await sandbox.execute(
            ExecutionRequest(code='print("x")', filename=str(target))
        )
        assert r.exit_code == 1
        assert "Invalid execution filename" in r.stderr
        assert not target.exists()

    @pytest.mark.asyncio
    async def test_inner_traversal_within_workdir_allowed(self, sandbox):
        # 'a/../b.txt' resolves to 'b.txt' under the workdir — accepted.
        await sandbox.write_files([
            FileContent(path="a/../b.txt", content="ok")
        ])
        assert (await sandbox.read_file("b.txt")) == "ok"
