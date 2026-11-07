"""The explorer: a read-only subagent with its own context.

Reading ten files to answer "where is X handled?" fills the main context with
text it no longer needs once the answer is known. The explorer does that
reading in a separate conversation and returns only its conclusion.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from agent.core.loop import Agent, Budget, ChatModel
from agent.permissions import Mode, PermissionPolicy, Rules
from agent.tools import Glob, Grep, ListDir, ReadFile, ToolRegistry
from agent.tools.base import Tool, ToolArgs, ToolContext, ToolKind, ToolResult

EXPLORER_PROMPT = """You are a read-only code explorer working for another agent. Answer its
question about the repository by reading code with list_dir, glob, grep and read_file.
You cannot change files or run commands. Reply with a short, factual answer: file
paths with line numbers, function names, and how they connect. Say what you could
not find instead of guessing."""


class ExploreArgs(ToolArgs):
    question: str = Field(description="A focused question about the codebase.")


class Explore(Tool):
    name: ClassVar[str] = "explore"
    description: ClassVar[str] = (
        "Ask a read-only subagent to investigate the codebase and report back a short "
        "answer (paths, line numbers). Use it for broad searches so your own context "
        "stays small."
    )
    kind: ClassVar[ToolKind] = ToolKind.READ
    Args: ClassVar[type[ToolArgs]] = ExploreArgs

    def __init__(self, model: ChatModel, rules: Rules, budget: Budget | None = None) -> None:
        self.model = model
        self.rules = rules
        self.budget = budget or Budget(max_turns=12, max_cost_usd=0.25)
        self.runs = 0

    async def run(self, args: ExploreArgs, ctx: ToolContext) -> ToolResult:
        self.runs += 1
        sub = Agent(
            self.model,
            ToolRegistry([ReadFile(), ListDir(), Glob(), Grep()]),
            ToolContext(ctx.guard, sandbox=None),
            PermissionPolicy(Mode.READ_ONLY, self.rules),
            system_prompt=EXPLORER_PROMPT,
            budget=self.budget,
            trace_attributes={"agent.session": "explorer"},
        )
        result = await sub.run(args.question)
        s = result.stats
        footer = f"\n[explorer: {s.turns} turns, {s.tool_calls} tool calls, {result.stop_reason}]"
        if not result.final_answer:
            return ToolResult(False, f"the explorer stopped without an answer{footer}")
        return ToolResult(True, result.final_answer + footer)
