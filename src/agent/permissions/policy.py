"""Permission policy: allow, ask a human, or deny each tool call.

Order of evaluation, first match wins:
1. deny rules (every mode),
2. read-only mode refuses anything that is not a read,
3. reads are always allowed,
4. auto mode allows the rest,
5. ask mode: allow-listed bash without shell metacharacters runs, the rest asks.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agent.tools.base import Tool, ToolArgs, ToolContext, ToolKind
from agent.tools.paths import PathError

_SHELL_META = re.compile(r"[;&|`<>\n]|\$\(")


class Mode(StrEnum):
    READ_ONLY = "read-only"
    ASK = "ask"
    AUTO = "auto"


class Verdict(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class Decision:
    verdict: Verdict
    reason: str


class PolicyError(Exception):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BashRules(_Strict):
    deny: list[str] = Field(default_factory=list)
    allow: list[str] = Field(default_factory=list)

    @field_validator("deny")
    @classmethod
    def _compiles(cls, patterns: list[str]) -> list[str]:
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc
        return patterns


class Rules(_Strict):
    bash: BashRules = BashRules()
    write_deny: list[str] = Field(default_factory=list)


def load_rules(path: Path) -> Rules:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return Rules.model_validate(raw)
    except FileNotFoundError as exc:
        raise PolicyError(f"permission rules not found: {path}") from exc
    except (yaml.YAMLError, ValidationError) as exc:
        raise PolicyError(f"invalid permission rules in {path}:\n{exc}") from exc


class PermissionPolicy:
    def __init__(self, mode: Mode, rules: Rules) -> None:
        self.mode = mode
        self.rules = rules
        self._deny = [re.compile(p) for p in rules.bash.deny]

    def check(self, tool: Tool, args: ToolArgs, ctx: ToolContext) -> Decision:
        if tool.kind is ToolKind.EXEC:
            command = str(getattr(args, "command", ""))
            for pattern in self._deny:
                if pattern.search(command):
                    return Decision(Verdict.DENY, f"command matches deny rule {pattern.pattern!r}")
        if tool.kind is ToolKind.WRITE:
            denied = self._denied_path(args, ctx)
            if denied is not None:
                return Decision(Verdict.DENY, denied)

        if tool.kind is ToolKind.READ:
            return Decision(Verdict.ALLOW, "read-only tool")
        if self.mode is Mode.READ_ONLY:
            return Decision(Verdict.DENY, "session is in read-only mode")
        if self.mode is Mode.AUTO:
            return Decision(Verdict.ALLOW, "auto mode")

        if tool.kind is ToolKind.EXEC:
            command = str(getattr(args, "command", "")).strip()
            if not _SHELL_META.search(command) and any(
                fnmatch.fnmatchcase(command, p) for p in self.rules.bash.allow
            ):
                return Decision(Verdict.ALLOW, "command is allow-listed")
        return Decision(Verdict.ASK, f"{tool.kind} needs approval in ask mode")

    def _denied_path(self, args: ToolArgs, ctx: ToolContext) -> str | None:
        raw = getattr(args, "path", None)
        if not isinstance(raw, str):
            return None
        try:
            rel = ctx.guard.display(ctx.guard.resolve(raw))
        except PathError:
            return None
        for pattern in self.rules.write_deny:
            if fnmatch.fnmatchcase(rel, pattern):
                return f"writes to {rel} are not allowed (rule {pattern!r})"
        return None


Approver = Callable[[Tool, Any, str | None], Awaitable[bool]]
