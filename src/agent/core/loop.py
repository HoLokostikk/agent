"""The agent loop: model -> tool calls -> observations -> model, until done or a budget stops it.

Invariants:
- a tool failure of any kind becomes an observation for the model; nothing a
  tool does can raise out of the loop;
- every tool call passes the permission policy before it runs;
- every stop has a reason, and budgets are checked after every turn.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from agent.llm import CallMetrics, Completion, LLMError, ModelConfig, OnDelta, ToolCall
from agent.permissions import Approver, PermissionPolicy, Verdict
from agent.tools import ToolContext, ToolInputError, ToolKind, ToolRegistry, ToolResult
from agent.tools.truncate import head_tail


class ChatModel(Protocol):
    config: ModelConfig

    async def complete(
        self,
        messages: list[ChatCompletionMessageParam],
        *,
        tools: list[ChatCompletionToolParam] | None = ...,
        on_delta: OnDelta | None = ...,
    ) -> Completion: ...


class StopReason(StrEnum):
    COMPLETED = "completed"
    MAX_TURNS = "max_turns"
    MAX_TOKENS = "max_tokens"
    MAX_COST = "max_cost"
    MAX_TIME = "max_time"
    CONTEXT_FULL = "context_full"
    LOOP_DETECTED = "loop_detected"
    EMPTY_RESPONSE = "empty_response"
    MODEL_ERROR = "model_error"


@dataclass(frozen=True, slots=True)
class Budget:
    max_turns: int = 30
    max_total_tokens: int = 1_000_000  # input + output, summed over turns
    max_cost_usd: float = 1.0
    max_wall_s: float = 1800.0
    # Stop before the server starts truncating the prompt (Ollama does it silently).
    max_context_fraction: float = 0.9
    repeat_limit: int = 3  # identical consecutive tool calls that count as a loop
    # Replies with neither text nor tool calls (e.g. a tool call the server
    # failed to parse and dropped) get a nudge instead of ending the run.
    max_empty_nudges: int = 2
    max_tool_output_chars: int = 6000


@dataclass(slots=True)
class ToolCallRecord:
    id: str
    name: str
    arguments: str
    verdict: str  # allow | approved | rejected | deny | invalid
    ok: bool
    output: str
    duration_s: float


@dataclass(slots=True)
class TurnRecord:
    turn: int
    content: str
    reasoning: str
    finish_reason: str | None
    metrics: CallMetrics
    tool_calls: list[ToolCallRecord] = field(default_factory=list)


@dataclass(slots=True)
class RunStats:
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cost_known: bool = True
    wall_s: float = 0.0
    tool_calls: int = 0
    invalid_tool_calls: int = 0
    denied_tool_calls: int = 0
    empty_responses: int = 0
    tool_errors: Counter[str] = field(default_factory=Counter)

    @property
    def invalid_tool_call_rate(self) -> float | None:
        return self.invalid_tool_calls / self.tool_calls if self.tool_calls else None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tool_errors"] = dict(self.tool_errors)
        data["invalid_tool_call_rate"] = self.invalid_tool_call_rate
        return data


@dataclass(slots=True)
class RunResult:
    stop_reason: StopReason
    final_answer: str | None
    stats: RunStats
    turns: list[TurnRecord]
    error: str | None = None

    @property
    def completed(self) -> bool:
        return self.stop_reason is StopReason.COMPLETED


class AgentEvents:
    """UI hooks. The default does nothing, which is what headless mode wants."""

    def on_delta(self, kind: str, text: str) -> None:
        pass

    def on_turn_end(self, turn: TurnRecord) -> None:
        pass

    def on_tool_start(self, call: ToolCall) -> None:
        pass

    def on_tool_end(self, record: ToolCallRecord) -> None:
        pass


def load_prompt(path: Path) -> tuple[str, str]:
    """Return (version, text); the version is the file stem, e.g. ``system.v1``."""
    return path.stem, path.read_text(encoding="utf-8").strip()


EMPTY_NUDGE = (
    "Your last reply was empty: no text and no tool call. Continue the task: call a "
    "tool, or, if the task is done, reply with a summary."
)


def _call_signature(call: ToolCall) -> tuple[str, str]:
    try:
        args = json.dumps(json.loads(call.arguments or "{}"), sort_keys=True)
    except json.JSONDecodeError:
        args = call.arguments
    return call.name, args


class Agent:
    """One conversation. ``run`` may be called repeatedly (REPL turns)."""

    def __init__(
        self,
        model: ChatModel,
        tools: ToolRegistry,
        ctx: ToolContext,
        policy: PermissionPolicy,
        *,
        system_prompt: str,
        budget: Budget | None = None,
        approver: Approver | None = None,
        events: AgentEvents | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.model = model
        self.tools = tools
        self.ctx = ctx
        self.policy = policy
        self.budget = budget or Budget()
        self.approver = approver
        self.events = events or AgentEvents()
        self._clock = clock
        self.messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt}
        ]

    async def run(self, task: str) -> RunResult:
        self.messages.append({"role": "user", "content": task})
        stats = RunStats()
        turns: list[TurnRecord] = []
        recent: list[tuple[str, str]] = []
        started = self._clock()
        schemas = self.tools.schemas()

        def finish(
            reason: StopReason, answer: str | None = None, error: str | None = None
        ) -> RunResult:
            stats.wall_s = self._clock() - started
            return RunResult(reason, answer, stats, turns, error)

        while True:
            if stats.turns >= self.budget.max_turns:
                return finish(StopReason.MAX_TURNS)
            if self._clock() - started >= self.budget.max_wall_s:
                return finish(StopReason.MAX_TIME)

            try:
                completion = await self.model.complete(
                    self.messages, tools=schemas, on_delta=self.events.on_delta
                )
            except LLMError as exc:
                return finish(StopReason.MODEL_ERROR, error=str(exc))

            stats.turns += 1
            self._account(stats, completion.metrics)
            empty = not completion.tool_calls and not completion.content.strip()
            if not empty:
                # An assistant message with neither content nor tool calls is
                # invalid for several servers, so an empty reply is not echoed.
                self.messages.append(completion.to_message())
            turn = TurnRecord(
                turn=stats.turns,
                content=completion.content,
                reasoning=completion.reasoning,
                finish_reason=completion.finish_reason,
                metrics=completion.metrics,
            )
            turns.append(turn)

            if empty:
                stats.empty_responses += 1
                self.events.on_turn_end(turn)
                if stats.empty_responses > self.budget.max_empty_nudges:
                    return finish(StopReason.EMPTY_RESPONSE)
                self.messages.append({"role": "user", "content": EMPTY_NUDGE})
                continue
            if not completion.tool_calls:
                self.events.on_turn_end(turn)
                return finish(StopReason.COMPLETED, answer=completion.content)

            for call in completion.tool_calls:
                record = await self._execute(call, stats)
                turn.tool_calls.append(record)
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": record.output}
                )
                recent.append(_call_signature(call))
            self.events.on_turn_end(turn)

            limit = self.budget.repeat_limit
            if len(recent) >= limit and len(set(recent[-limit:])) == 1:
                return finish(StopReason.LOOP_DETECTED)
            if stats.input_tokens + stats.output_tokens >= self.budget.max_total_tokens:
                return finish(StopReason.MAX_TOKENS)
            if stats.cost_known and stats.cost_usd >= self.budget.max_cost_usd:
                return finish(StopReason.MAX_COST)
            usage = completion.metrics.usage
            window = self.model.config.context_window
            if (
                usage is not None
                and usage.input_tokens >= window * self.budget.max_context_fraction
            ):
                return finish(StopReason.CONTEXT_FULL)

    @staticmethod
    def _account(stats: RunStats, metrics: CallMetrics) -> None:
        if metrics.usage is not None:
            stats.input_tokens += metrics.usage.input_tokens
            stats.output_tokens += metrics.usage.output_tokens
        if metrics.cost_usd is None:
            stats.cost_known = False
        else:
            stats.cost_usd += metrics.cost_usd

    async def _execute(self, call: ToolCall, stats: RunStats) -> ToolCallRecord:
        stats.tool_calls += 1
        self.events.on_tool_start(call)
        started = self._clock()

        def record(verdict: str, result: ToolResult) -> ToolCallRecord:
            output = head_tail(result.output, max_chars=self.budget.max_tool_output_chars)
            rec = ToolCallRecord(
                id=call.id,
                name=call.name,
                arguments=call.arguments,
                verdict=verdict,
                ok=result.ok,
                output=output,
                duration_s=self._clock() - started,
            )
            self.events.on_tool_end(rec)
            return rec

        try:
            tool, args = self.tools.parse(call.name, call.arguments)
        except ToolInputError as exc:
            stats.invalid_tool_calls += 1
            return record("invalid", ToolResult(False, f"error: {exc}"))

        decision = self.policy.check(tool, args, self.ctx)
        verdict = decision.verdict.value
        if decision.verdict is Verdict.DENY:
            stats.denied_tool_calls += 1
            return record(
                verdict,
                ToolResult(False, f"permission denied: {decision.reason}. Do not retry this call."),
            )
        if decision.verdict is Verdict.ASK:
            if self.approver is None:
                stats.denied_tool_calls += 1
                return record(
                    "deny",
                    ToolResult(
                        False,
                        "permission denied: this call needs human approval and no human is "
                        "available in this session",
                    ),
                )
            approved = await self.approver(tool, args, tool.preview(args, self.ctx))
            if not approved:
                stats.denied_tool_calls += 1
                return record(
                    "rejected",
                    ToolResult(False, "the user rejected this call; try something else or ask"),
                )
            verdict = "approved"

        try:
            result = await tool.run(args, self.ctx)
        except Exception as exc:  # a tool bug must not kill the session
            result = ToolResult(False, f"tool {call.name} crashed: {type(exc).__name__}: {exc}")
        if not result.ok:
            stats.tool_errors[call.name] += 1
            if tool.kind is not ToolKind.EXEC:
                result = ToolResult(False, f"error: {result.output}")
        return record(verdict, result)
