"""Tool interface, registry, and argument validation.

A tool never raises into the loop: every failure becomes a ``ToolResult``
with ``ok=False`` that the model reads as an observation and can react to.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel, ConfigDict, ValidationError

from agent.sandbox.base import Sandbox
from agent.tools.checkpoint import CheckpointStack
from agent.tools.paths import PathGuard


class ToolKind(StrEnum):
    READ = "read"
    WRITE = "write"
    EXEC = "exec"


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True, slots=True)
class ToolResult:
    ok: bool
    output: str


@dataclass(frozen=True, slots=True)
class ToolContext:
    guard: PathGuard
    sandbox: Sandbox | None
    checkpoints: CheckpointStack | None = None


class ToolInputError(Exception):
    """The model's call cannot be executed as given (bad JSON, schema, name)."""


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    kind: ClassVar[ToolKind]
    Args: ClassVar[type[ToolArgs]]

    @abstractmethod
    async def run(self, args: Any, ctx: ToolContext) -> ToolResult: ...

    def preview(self, args: Any, ctx: ToolContext) -> str | None:
        """What a human approver should see before this runs (e.g. a diff)."""
        return None

    def schema(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": _clean_schema(self.Args.model_json_schema()),
            },
        }


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop pydantic's ``title`` keys: they cost tokens and tell the model nothing."""
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "title":
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {
                name: _clean_schema(prop) if isinstance(prop, dict) else prop
                for name, prop in value.items()
            }
        elif isinstance(value, dict):
            cleaned[key] = _clean_schema(value)
        else:
            cleaned[key] = value
    return cleaned


def _format_validation_error(exc: ValidationError) -> str:
    problems = []
    for err in exc.errors():
        where = ".".join(str(part) for part in err["loc"]) or "arguments"
        problems.append(f"{where}: {err['msg']}")
    return "; ".join(problems)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name {tool.name!r}")
            self._tools[tool.name] = tool

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool:
        return self._tools[name]

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[ChatCompletionToolParam]:
        return [tool.schema() for tool in self._tools.values()]

    def parse(self, name: str, raw_arguments: str) -> tuple[Tool, ToolArgs]:
        """Resolve and validate a call, or raise ToolInputError with a message for the model."""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolInputError(
                f"unknown tool {name!r}; available tools: {', '.join(self._tools)}"
            )
        text = raw_arguments.strip() or "{}"
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ToolInputError(
                f"arguments for {name} are not valid JSON ({exc.msg} at char {exc.pos}); "
                "send a single JSON object"
            ) from None
        if not isinstance(data, dict):
            raise ToolInputError(f"arguments for {name} must be a JSON object")
        try:
            return tool, tool.Args.model_validate(data)
        except ValidationError as exc:
            raise ToolInputError(
                f"invalid arguments for {name}: {_format_validation_error(exc)}"
            ) from None
