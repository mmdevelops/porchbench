"""Subprocess sandbox backend.

Executes code in a subprocess with a temporary working directory.
Provides timeout enforcement and file I/O. This is the Phase 1 backend
— zero external dependencies, sufficient for benchmarking model-generated
code on a local machine.

Security model: process-level isolation + tempdir. No network isolation,
no memory limits. Filesystem isolation is enforced at the API boundary —
``write_files`` / ``read_file`` / ``execute`` reject any path that
resolves outside the per-session workdir, including paths containing
``..`` segments or absolute paths. Acceptable for benchmarking (we
control the suites); not for untrusted agent deployment.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

from porchbench.sandbox.base import (
    ExecutionRequest,
    ExecutionResult,
    FileContent,
    Sandbox,
    SandboxConfig,
)

# Map language names to interpreter commands
LANGUAGE_COMMANDS: dict[str, list[str]] = {
    "python": [sys.executable],
    "bash": ["bash"],
    "node": ["node"],
}


class SandboxPathError(ValueError):
    """Raised when a caller-supplied path would escape the sandbox workdir."""


def _resolve_within(workdir: Path, raw_path: str) -> Path:
    """Resolve ``raw_path`` under ``workdir`` and reject escapes.

    Empty paths, absolute paths, and paths whose resolved location lies
    outside ``workdir`` raise ``SandboxPathError``. ``..`` segments that
    happen to remain inside the workdir (e.g. ``a/../b``) are accepted
    after normalisation.
    """
    if not raw_path:
        raise SandboxPathError("empty sandbox path")

    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise SandboxPathError(f"absolute paths are not allowed: {raw_path!r}")

    workdir_resolved = workdir.resolve()
    target = (workdir_resolved / candidate).resolve()

    if not target.is_relative_to(workdir_resolved):
        raise SandboxPathError(
            f"path escapes sandbox workdir: {raw_path!r}"
        )
    return target


class SubprocessSandbox(Sandbox):
    """Sandbox backend using subprocess + tempdir.

    Each session gets a fresh temp directory. Files persist across
    executions within the session. The directory is cleaned up on destroy().
    """

    def __init__(self) -> None:
        self._workdir: Path | None = None
        self._config: SandboxConfig | None = None

    @property
    def workdir(self) -> Path:
        if self._workdir is None:
            raise RuntimeError("Sandbox not created. Call create() first.")
        return self._workdir

    async def create(self, config: SandboxConfig) -> None:
        self._config = config
        self._workdir = Path(tempfile.mkdtemp(prefix="porchbench_sandbox_"))

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        config = self._config or SandboxConfig()
        workdir = self.workdir

        # Determine interpreter
        lang = request.language.lower()
        cmd_prefix = LANGUAGE_COMMANDS.get(lang)
        if cmd_prefix is None:
            return ExecutionResult(
                stdout="",
                stderr=f"Unsupported language: {lang}",
                exit_code=1,
            )

        # Determine code filename. A bare basename is required — directory
        # components in `request.filename` are a sandbox-escape vector since
        # the resulting path is then handed to subprocess.run.
        ext = {"python": ".py", "bash": ".sh", "node": ".js"}.get(lang, ".txt")
        filename = request.filename or f"_exec{ext}"
        if "/" in filename or "\\" in filename or filename in ("", ".", ".."):
            return ExecutionResult(
                stdout="",
                stderr=f"Invalid execution filename: {filename!r}",
                exit_code=1,
            )

        try:
            code_path = _resolve_within(workdir, filename)
        except SandboxPathError as exc:
            return ExecutionResult(
                stdout="",
                stderr=f"Invalid execution filename: {exc}",
                exit_code=1,
            )

        code_path.write_text(request.code, encoding="utf-8")

        # Build environment
        env = os.environ.copy()
        env.update(config.env)

        # Async subprocess + asyncio.wait_for for timeout enforcement.
        # The synchronous `subprocess.run(timeout=...)` path wedges on
        # Windows: after TerminateProcess fires for the timed-out child,
        # the parent is still blocked in pipe-drain code that waits on
        # the now-dead OS handles, occasionally hanging indefinitely.
        # asyncio's transport-based pipe handling cancels cleanly.
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd_prefix, str(code_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(workdir),
                env=env,
            )
        except Exception as exc:
            return ExecutionResult(
                stdout="",
                stderr=str(exc),
                exit_code=-1,
            )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=config.timeout_s,
            )
            return ExecutionResult(
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                exit_code=proc.returncode if proc.returncode is not None else -1,
            )
        except TimeoutError:
            # Kill the child + drain (or short-window-wait if drain itself
            # hangs after the kill — rare but seen on Windows when the
            # child has spawned grandchildren that hold the pipe handles).
            proc.kill()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except TimeoutError:
                pass
            return ExecutionResult(
                stdout="",
                stderr=f"Execution timed out after {config.timeout_s}s",
                exit_code=-1,
                timed_out=True,
            )
        except Exception as exc:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return ExecutionResult(
                stdout="",
                stderr=str(exc),
                exit_code=-1,
            )

    async def write_files(self, files: list[FileContent]) -> None:
        workdir = self.workdir
        for f in files:
            target = _resolve_within(workdir, f.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f.content, encoding="utf-8")

    async def read_file(self, path: str) -> str:
        target = _resolve_within(self.workdir, path)
        if not target.exists():
            raise FileNotFoundError(f"File not found in sandbox: {path}")
        return target.read_text(encoding="utf-8")

    async def destroy(self) -> None:
        if self._workdir and self._workdir.exists():
            shutil.rmtree(self._workdir, ignore_errors=True)
        self._workdir = None
        self._config = None
