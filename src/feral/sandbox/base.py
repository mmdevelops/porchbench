"""Abstract sandbox interface.

Defines the contract that all sandbox backends must implement.
The sandbox is stateful within a session (filesystem persists across
executions) but has no opinion on what to run or why.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SandboxConfig:
    """Configuration for a sandbox session."""

    timeout_s: int = 30
    memory_limit_mb: int = 256
    network_enabled: bool = False
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class ExecutionRequest:
    """A request to execute code inside the sandbox."""

    code: str
    language: str = "python"
    filename: str | None = None  # override the auto-generated filename


@dataclass
class ExecutionResult:
    """Result of a single code execution."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


@dataclass
class FileContent:
    """A file to write into or read from the sandbox."""

    path: str
    content: str


class Sandbox(ABC):
    """Abstract sandbox interface. All backends implement these five operations."""

    @abstractmethod
    async def create(self, config: SandboxConfig) -> None:
        """Provision the isolated environment."""

    @abstractmethod
    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Run code inside the sandbox. Returns stdout/stderr/exit code."""

    @abstractmethod
    async def write_files(self, files: list[FileContent]) -> None:
        """Place files into the sandbox filesystem."""

    @abstractmethod
    async def read_file(self, path: str) -> str:
        """Read a file from the sandbox filesystem."""

    @abstractmethod
    async def destroy(self) -> None:
        """Tear down the environment, release resources."""
