"""What the tool layer needs from a sandbox."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class SandboxError(Exception):
    """The sandbox itself failed (not the command inside it)."""


@dataclass(frozen=True, slots=True)
class ExecResult:
    exit_code: int
    output: str  # stdout and stderr interleaved, as a terminal would show
    timed_out: bool = False


class Sandbox(Protocol):
    root: Path  # host path of the workspace
    workdir: str  # the same directory as seen by commands inside the sandbox

    async def exec(self, command: str, timeout_s: float) -> ExecResult: ...
