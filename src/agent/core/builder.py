"""One place that assembles an agent, shared by the CLI and the eval runner.

Base, REPL, headless and eval runs differ only in permission mode, approver
and events; everything that affects behaviour (prompt, tools, compaction) is
built here, so an eval measures exactly what a user runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.core.context import Compactor
from agent.core.loop import Agent, AgentEvents, Budget, ChatModel
from agent.core.session import build_system_prompt
from agent.core.subagent import Explore
from agent.permissions import Approver, Mode, PermissionPolicy, Rules
from agent.rag.search import Searcher
from agent.rag.tool import SearchCode
from agent.sandbox import Sandbox
from agent.tools import PathGuard, Tool, ToolContext, ToolRegistry, default_tools
from agent.tools.checkpoint import CheckpointStack
from agent.tools.todo import TodoWrite

# Below this window the extra tool schemas cost more than they help.
EXTENDED_TOOLS_MIN_CONTEXT = 32_768


@dataclass(frozen=True, slots=True)
class AgentSpec:
    base_prompt: str
    prompt_version: str
    rules: Rules
    mode: Mode
    budget: Budget
    compaction: bool = True
    extended_tools: bool | None = None  # None: decide by the context window
    searcher: Searcher | None = None  # adds search_code when set


def build_tools(
    model: ChatModel,
    spec: AgentSpec,
    compact_model: ChatModel | None,
    extra_tools: list[Tool] | None = None,
) -> ToolRegistry:
    tools: list[Tool] = default_tools().tools()
    extended = spec.extended_tools
    if extended is None:
        extended = model.config.context_window >= EXTENDED_TOOLS_MIN_CONTEXT
    if extended:
        tools.append(TodoWrite())
        tools.append(Explore(compact_model or model, spec.rules))
    if spec.searcher is not None:
        tools.append(SearchCode(spec.searcher))
    tools.extend(extra_tools or [])
    return ToolRegistry(tools)


def build_agent(
    model: ChatModel,
    workspace: Path,
    sandbox: Sandbox | None,
    spec: AgentSpec,
    *,
    compact_model: ChatModel | None = None,
    approver: Approver | None = None,
    events: AgentEvents | None = None,
    trace_attributes: dict[str, Any] | None = None,
    extra_tools: list[Tool] | None = None,
) -> Agent:
    return Agent(
        model,
        build_tools(model, spec, compact_model, extra_tools),
        ToolContext(PathGuard(workspace), sandbox, CheckpointStack()),
        PermissionPolicy(spec.mode, spec.rules),
        system_prompt=build_system_prompt(spec.base_prompt, workspace),
        budget=spec.budget,
        approver=approver,
        events=events,
        compactor=Compactor(compact_model or model) if spec.compaction else None,
        trace_attributes={"agent.prompt_version": spec.prompt_version, **(trace_attributes or {})},
    )
