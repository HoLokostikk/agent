"""Results of a model call and the accounting attached to each one."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageFunctionToolCallParam,
)

from agent.llm.config import Pricing


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    # Raw JSON text exactly as the model produced it. Parsing and validation
    # belong to the tool layer, which reports bad JSON back to the model.
    arguments: str


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int  # includes reasoning tokens, as in the OpenAI API
    cached_input_tokens: int = 0
    # ``None`` when the server does not report it (Ollama never does).
    reasoning_tokens: int | None = None


def compute_cost(usage: Usage, pricing: Pricing) -> float:
    """USD for one call. Cached input is billed at the cached rate if set."""
    cached = min(usage.cached_input_tokens, usage.input_tokens)
    uncached = usage.input_tokens - cached
    cached_rate = (
        pricing.cached_input_per_mtok
        if pricing.cached_input_per_mtok is not None
        else pricing.input_per_mtok
    )
    return (
        uncached * pricing.input_per_mtok
        + cached * cached_rate
        + usage.output_tokens * pricing.output_per_mtok
    ) / 1_000_000


@dataclass(frozen=True, slots=True)
class CallMetrics:
    """Accounting for one logical call (all attempts together).

    ``latency_s`` runs from the first attempt to the last byte, so it includes
    retry waits: it is what the caller experienced. ``ttft_s`` runs from the
    start of the successful attempt to its first generated token of any kind
    (reasoning, text or tool call): it measures the server, not our retries.
    """

    alias: str
    model: str
    revision: str
    attempts: int
    latency_s: float
    ttft_s: float | None
    finish_reason: str | None
    usage: Usage | None
    cost_usd: float | None

    @property
    def output_tokens_per_s(self) -> float | None:
        """Decode speed after the first token."""
        if self.usage is None or self.ttft_s is None:
            return None
        decode_s = self.latency_s - self.ttft_s
        if decode_s <= 0 or self.usage.output_tokens <= 1:
            return None
        return (self.usage.output_tokens - 1) / decode_s

    def as_log_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["output_tokens_per_s"] = self.output_tokens_per_s
        return record


@dataclass(frozen=True, slots=True)
class Completion:
    content: str
    reasoning: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str | None
    metrics: CallMetrics

    def to_message(self) -> ChatCompletionAssistantMessageParam:
        """The assistant turn to append to the conversation history.

        Reasoning is not sent back. Whether to preserve it across turns is a
        context-budget decision for the agent loop, not for the client.
        """
        message: ChatCompletionAssistantMessageParam = {
            "role": "assistant",
            "content": self.content or None,
        }
        if self.tool_calls:
            calls: list[ChatCompletionMessageFunctionToolCallParam] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
            message["tool_calls"] = calls
        return message
