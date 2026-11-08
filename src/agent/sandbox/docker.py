"""Docker sandbox: every command the model asks for runs in here, never on the host.

One container per session. The workspace is bind-mounted at ``/workspace``;
the container has no network, no capabilities, a read-only root filesystem,
memory, CPU and process limits, and runs as the host user (so files it creates
stay owned by that user, not by root).
"""

from __future__ import annotations

import asyncio
import os
import secrets
import shlex
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

from agent.sandbox.base import ExecResult, SandboxError

DEFAULT_IMAGE = "agent-sandbox:0.1"
CONTAINER_WORKDIR = "/workspace"
DOCKERFILE_DIR = Path(__file__).parent


@dataclass(frozen=True, slots=True)
class Limits:
    memory: str = "1g"
    cpus: str = "2"
    pids: int = 256
    tmpfs_size: str = "256m"


async def _docker(*args: str, timeout_s: float = 60.0) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout_s)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise SandboxError(f"docker {args[0]} timed out after {timeout_s}s") from None
    return proc.returncode or 0, out.decode("utf-8", errors="replace")


async def build_image(image: str = DEFAULT_IMAGE) -> None:
    code, out = await _docker("build", "-t", image, str(DOCKERFILE_DIR), timeout_s=900)
    if code != 0:
        raise SandboxError(f"docker build failed:\n{out[-2000:]}")


async def image_exists(image: str = DEFAULT_IMAGE) -> bool:
    code, _ = await _docker("image", "inspect", image)
    return code == 0


class DockerSandbox:
    """Async context manager owning one container."""

    def __init__(
        self,
        root: Path,
        *,
        image: str = DEFAULT_IMAGE,
        limits: Limits | None = None,
        max_output_bytes: int = 1_000_000,
    ) -> None:
        self.root = root.resolve()
        self.image = image
        self.limits = limits or Limits()
        self.max_output_bytes = max_output_bytes
        self.workdir = CONTAINER_WORKDIR
        self.name = f"agent-sbx-{secrets.token_hex(4)}"
        self._started = False

    def run_args(self) -> list[str]:
        """The full ``docker run`` argument list; tested for its security flags."""
        lim = self.limits
        return [
            "run",
            "--detach",
            "--rm",
            "--name",
            self.name,
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--read-only",
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,size={lim.tmpfs_size}",  # noqa: S108
            "--memory",
            lim.memory,
            "--memory-swap",
            lim.memory,
            "--cpus",
            lim.cpus,
            "--pids-limit",
            str(lim.pids),
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--env",
            "HOME=/tmp",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--volume",
            f"{self.root}:{CONTAINER_WORKDIR}:rw",
            "--workdir",
            CONTAINER_WORKDIR,
            self.image,
            "sleep",
            "infinity",
        ]

    async def start(self) -> None:
        if not self.root.is_dir():
            raise SandboxError(f"workspace is not a directory: {self.root}")
        code, out = await _docker(*self.run_args())
        if code != 0:
            raise SandboxError(f"could not start sandbox: {out.strip()[-500:]}")
        self._started = True

    async def stop(self) -> None:
        if self._started:
            self._started = False
            await _docker("kill", self.name, timeout_s=30)

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.stop()

    async def exec(self, command: str, timeout_s: float) -> ExecResult:
        if not self._started:
            raise SandboxError("sandbox is not running")
        inner = f"timeout --kill-after=5 {timeout_s:g} bash -c {shlex.quote(command)}"
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "--workdir",
            CONTAINER_WORKDIR,
            self.name,
            "bash",
            "-c",
            inner,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout_s + 15)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return ExecResult(exit_code=124, output="", timed_out=True)
        code = proc.returncode if proc.returncode is not None else -1
        data = out[: self.max_output_bytes]
        return ExecResult(
            exit_code=code,
            output=data.decode("utf-8", errors="replace"),
            timed_out=code == 124,
        )
