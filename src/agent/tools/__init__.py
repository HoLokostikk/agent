"""Tools the model can call, with JSON schemas generated from pydantic models."""

from agent.tools.base import (
    Tool,
    ToolArgs,
    ToolContext,
    ToolInputError,
    ToolKind,
    ToolRegistry,
    ToolResult,
)
from agent.tools.bash import Bash
from agent.tools.files import EditFile, Glob, Grep, ListDir, ReadFile, WriteFile
from agent.tools.paths import PathError, PathGuard


def default_tools() -> ToolRegistry:
    return ToolRegistry([ReadFile(), ListDir(), Glob(), Grep(), EditFile(), WriteFile(), Bash()])


__all__ = [
    "Bash",
    "EditFile",
    "Glob",
    "Grep",
    "ListDir",
    "PathError",
    "PathGuard",
    "ReadFile",
    "Tool",
    "ToolArgs",
    "ToolContext",
    "ToolInputError",
    "ToolKind",
    "ToolRegistry",
    "ToolResult",
    "WriteFile",
    "default_tools",
]
