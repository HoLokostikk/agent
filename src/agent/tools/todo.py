"""A plan the model keeps for itself. No side effects outside the session."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.tools.base import Tool, ToolArgs, ToolContext, ToolKind, ToolResult


class TodoItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str = Field(min_length=1)
    status: Literal["pending", "in_progress", "done"] = "pending"


class TodoArgs(ToolArgs):
    items: list[TodoItem] = Field(description="The full, updated plan; replaces the old one.")


_MARK = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]"}


class TodoWrite(Tool):
    name: ClassVar[str] = "todo_write"
    description: ClassVar[str] = (
        "Write your plan as a checklist. Send the whole list every time, with each "
        "item's status. Use it for tasks with more than two steps."
    )
    kind: ClassVar[ToolKind] = ToolKind.READ
    Args: ClassVar[type[ToolArgs]] = TodoArgs

    def __init__(self) -> None:
        self.items: list[TodoItem] = []

    async def run(self, args: TodoArgs, ctx: ToolContext) -> ToolResult:
        in_progress = [i for i in args.items if i.status == "in_progress"]
        if len(in_progress) > 1:
            return ToolResult(False, "only one item may be in_progress at a time")
        self.items = list(args.items)
        lines = [f"{_MARK[i.status]} {i.content}" for i in self.items]
        done = sum(i.status == "done" for i in self.items)
        return ToolResult(True, f"plan ({done}/{len(self.items)} done):\n" + "\n".join(lines))
