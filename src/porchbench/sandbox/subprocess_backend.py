"""Subprocess sandbox backend.

Executes code in a subprocess with a temporary working directory.
Provides timeout enforcement and file I/O. This is the Phase 1 backend
— zero external dependencies, sufficient for benchmarking model-generated
code on a local machine.

Security model: process-level isolation + tempdir. No network isolation,
no memory limits, no filesystem isolation beyond the tempdir. Acceptable
for benchmarking (we control the suites); not for untrusted agent deployment.
"""

from __future__ import annotations

import os
import shutil
import subprocess
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

        # Write code to a temp file in the workdir
        ext = {"python": ".py", "bash": ".sh", "node": ".js"}.get(lang, ".txt")
        filename = request.filename or f"_exec{ext}"
        code_path = workdir / filename
        code_path.write_text(request.code, encoding="utf-8")

        # Build environment
        env = os.environ.copy()
        env.update(config.env)

        try:
            result = subprocess.run(
                [*cmd_prefix, str(code_path)],
                capture_output=True,
                text=True,
                timeout=config.timeout_s,
                cwd=str(workdir),
                env=env,
            )
            return ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="",
                stderr=f"Execution timed out after {config.timeout_s}s",
                exit_code=-1,
                timed_out=True,
            )
        except Exception as exc:
            return ExecutionResult(
                stdout="",
                stderr=str(exc),
                exit_code=-1,
            )

    async def write_files(self, files: list[FileContent]) -> None:
        workdir = self.workdir
        for f in files:
            target = workdir / f.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f.content, encoding="utf-8")

    async def read_file(self, path: str) -> str:
        target = self.workdir / path
        if not target.exists():
            raise FileNotFoundError(f"File not found in sandbox: {path}")
        return target.read_text(encoding="utf-8")

    async def destroy(self) -> None:
        if self._workdir and self._workdir.exists():
            shutil.rmtree(self._workdir, ignore_errors=True)
        self._workdir = None
        self._config = None
