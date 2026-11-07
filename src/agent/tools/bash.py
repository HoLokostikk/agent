"""The bash tool: the only way model output gets executed, and only in the sandbox."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from agent.sandbox.base import SandboxError
from agent.tools.base import Tool, ToolArgs, ToolContext, ToolKind, ToolResult
from agent.tools.truncate import head_tail


class BashArgs(ToolArgs):
    command: str = Field(description="Shell command, run with bash in the workspace root.")
    timeout_s: int = Field(default=60, ge=1, le=600, description="Kill after this many seconds.")


class Bash(Tool):
    name: ClassVar[str] = "bash"
    description: ClassVar[str] = (
        "Run a shell command in an isolated sandbox (no network) with the workspace "
        "mounted at /workspace. Use it to run tests. Output is truncated in the middle."
    )
    kind: ClassVar[ToolKind] = ToolKind.EXEC
    Args: ClassVar[type[ToolArgs]] = BashArgs

    def preview(self, args: BashArgs, ctx: ToolContext) -> str | None:
        return f"$ {args.command}"

    async def run(self, args: BashArgs, ctx: ToolContext) -> ToolResult:
        if ctx.sandbox is None:
            return ToolResult(False, "bash is unavailable: no sandbox in this session")
        try:
            result = await ctx.sandbox.exec(args.command, args.timeout_s)
        except SandboxError as exc:
            return ToolResult(False, f"sandbox error: {exc}")
        status = f"exit code: {result.exit_code}"
        if result.timed_out:
            status += f" (timed out after {args.timeout_s}s)"
        output = head_tail(result.output.rstrip()) if result.output.strip() else "(no output)"
        return ToolResult(result.exit_code == 0 and not result.timed_out, f"{status}\n{output}")
