"""Deterministic stand-ins for the model and the sandbox."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai.types.chat import ChatCompletionMessageParam

from agent.llm import CallMetrics, Completion, LLMError, ModelConfig, OnDelta, ToolCall, Usage
from agent.llm.config import Pricing
from agent.sandbox import ExecResult


def fake_config(context_window: int = 100_000) -> ModelConfig:
    return ModelConfig(
        alias="fake",
        base_url="http://fake.test/v1",
        model="fake-model",
        revision="rev-1",
        context_window=context_window,
        pricing=Pricing(input_per_mtok=1.0, output_per_mtok=2.0),
    )


def call(name: str, call_id: str | None = None, /, **args: Any) -> ToolCall:
    return ToolCall(id=call_id or f"call_{name}", name=name, arguments=json.dumps(args))


def raw_call(name: str, arguments: str, call_id: str = "call_raw") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


@dataclass
class Step:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    input_tokens: int = 100
    output_tokens: int = 10
    error: LLMError | None = None


def say(text: str, **kw: Any) -> Step:
    return Step(content=text, **kw)


def use(*calls: ToolCall, **kw: Any) -> Step:
    return Step(tool_calls=calls, **kw)


@dataclass
class FakeLLM:
    """Plays scripted turns; records the messages it was sent."""

    steps: list[Step]
    config: ModelConfig = field(default_factory=fake_config)
    seen: list[list[ChatCompletionMessageParam]] = field(default_factory=list)

    async def complete(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        tools: Any = None,
        on_delta: OnDelta | None = None,
    ) -> Completion:
        self.seen.append(list(messages))
        if not self.steps:
            raise AssertionError("FakeLLM ran out of scripted steps")
        step = self.steps.pop(0)
        if step.error is not None:
            raise step.error
        if step.content and on_delta is not None:
            on_delta("content", step.content)
        usage = Usage(input_tokens=step.input_tokens, output_tokens=step.output_tokens)
        cost = (step.input_tokens * 1.0 + step.output_tokens * 2.0) / 1_000_000
        return Completion(
            content=step.content,
            reasoning="",
            tool_calls=step.tool_calls,
            finish_reason="tool_calls" if step.tool_calls else "stop",
            metrics=CallMetrics(
                alias="fake",
                model="fake-model",
                revision="rev-1",
                attempts=1,
                latency_s=0.01,
                ttft_s=0.001,
                finish_reason="tool_calls" if step.tool_calls else "stop",
                usage=usage,
                cost_usd=cost,
            ),
        )


@dataclass
class FakeSandbox:
    """Records commands and replies from a script; never executes anything."""

    root: Path
    replies: dict[str, ExecResult] = field(default_factory=dict)
    default: ExecResult = field(default_factory=lambda: ExecResult(0, "ok"))
    workdir: str = "/workspace"
    commands: list[str] = field(default_factory=list)

    async def exec(self, command: str, timeout_s: float) -> ExecResult:
        self.commands.append(command)
        return self.replies.get(command, self.default)
