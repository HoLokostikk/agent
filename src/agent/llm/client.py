"""Streaming chat client for any OpenAI-compatible endpoint.

Retries are owned here, not by the SDK (``max_retries=0``), because the policy
depends on stream state: a failure before the first token is retried, a
failure after it is not, since the caller has already shown partial output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx2
from openai import AsyncOpenAI, omit
from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from agent.llm.config import ModelConfig
from agent.llm.errors import ErrorKind, LLMError, classify
from agent.llm.retry import backoff_delay
from agent.llm.types import CallMetrics, Completion, ToolCall, Usage, compute_cost

logger = logging.getLogger("agent.llm")

DeltaKind = Literal["content", "reasoning"]
OnDelta = Callable[[DeltaKind, str], None]
Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], float]

# Field names servers use for streamed reasoning: vLLM and Ollama's thinking
# models use ``reasoning_content``, OpenRouter uses ``reasoning``.
_REASONING_FIELDS = ("reasoning_content", "reasoning")


@dataclass
class _PartialToolCall:
    id: str = ""
    name: str = ""
    arguments: list[str] = field(default_factory=list)


@dataclass
class _StreamState:
    content: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    tool_calls: dict[int, _PartialToolCall] = field(default_factory=dict)
    finish_reason: str | None = None
    usage: CompletionUsage | None = None
    first_token_at: float | None = None

    @property
    def started(self) -> bool:
        return self.first_token_at is not None


def _reasoning_text(delta: ChoiceDelta) -> str:
    extra = delta.model_extra or {}
    for name in _REASONING_FIELDS:
        value = extra.get(name)
        if isinstance(value, str) and value:
            return value
    return ""


def _usage_from(raw: CompletionUsage | None) -> Usage | None:
    if raw is None:
        return None
    cached = raw.prompt_tokens_details.cached_tokens if raw.prompt_tokens_details else None
    reasoning = (
        raw.completion_tokens_details.reasoning_tokens if raw.completion_tokens_details else None
    )
    return Usage(
        input_tokens=raw.prompt_tokens,
        output_tokens=raw.completion_tokens,
        cached_input_tokens=cached or 0,
        reasoning_tokens=reasoning,
    )


def _tool_calls_from(partials: dict[int, _PartialToolCall]) -> tuple[ToolCall, ...]:
    calls = []
    for index in sorted(partials):
        part = partials[index]
        calls.append(
            ToolCall(
                id=part.id or f"call_{index}",
                name=part.name,
                arguments="".join(part.arguments),
            )
        )
    return tuple(calls)


class LLMClient:
    """One model alias. Safe to share across concurrent tasks."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        http_client: httpx2.AsyncClient | None = None,
        sleep: Sleep = asyncio.sleep,
        rng: random.Random | None = None,
        clock: Clock = time.perf_counter,
    ) -> None:
        self.config = config
        self._sleep = sleep
        self._rng = rng or random.Random()  # noqa: S311  (jitter, not crypto)
        self._clock = clock
        t = config.timeouts
        self._client = AsyncOpenAI(
            api_key=config.resolve_api_key(),
            base_url=config.base_url,
            timeout=httpx2.Timeout(
                connect=t.connect_s, read=t.read_s, write=t.read_s, pool=t.connect_s
            ),
            max_retries=0,
            http_client=http_client,
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def complete(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        tools: Iterable[ChatCompletionToolParam] | None = None,
        tool_choice: ChatCompletionToolChoiceOptionParam | None = None,
        max_tokens: int | None = None,
        on_delta: OnDelta | None = None,
    ) -> Completion:
        """Stream one completion, retrying retryable failures before the first token.

        Raises :class:`LLMError`; nothing else escapes.
        """
        policy = self.config.retry
        total_s = self.config.timeouts.total_s
        started_at = self._clock()
        deadline = started_at + total_s
        tool_list = list(tools) if tools is not None else None
        attempt = 0
        try:
            async with asyncio.timeout(total_s):
                while True:
                    attempt += 1
                    state = _StreamState()
                    attempt_started_at = self._clock()
                    try:
                        await self._stream_once(
                            messages, tool_list, tool_choice, max_tokens, on_delta, state
                        )
                    except Exception as exc:
                        error = classify(exc)
                        error.attempts = attempt
                        if state.started:
                            raise LLMError(
                                ErrorKind.STREAM_INTERRUPTED,
                                f"stream failed after first token ({error.kind}): {error.message}",
                                status_code=error.status_code,
                                attempts=attempt,
                            ) from exc
                        if not error.retryable or attempt >= policy.max_attempts:
                            self._log_failure(error, started_at)
                            raise error from exc
                        delay = backoff_delay(attempt, policy, self._rng, error.retry_after_s)
                        if self._clock() + delay > deadline:
                            self._log_failure(error, started_at)
                            raise error from exc
                        logger.warning(
                            "llm_retry %s",
                            json.dumps(
                                {
                                    "alias": self.config.alias,
                                    "attempt": attempt,
                                    "kind": error.kind,
                                    "status_code": error.status_code,
                                    "delay_s": round(delay, 3),
                                }
                            ),
                        )
                        await self._sleep(delay)
                        continue
                    return self._finish(state, attempt, started_at, attempt_started_at)
        except TimeoutError as exc:
            error = LLMError(
                ErrorKind.TIMEOUT, f"total deadline of {total_s}s exceeded", attempts=attempt
            )
            self._log_failure(error, started_at)
            raise error from exc

    async def _stream_once(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam] | None,
        tool_choice: ChatCompletionToolChoiceOptionParam | None,
        max_tokens: int | None,
        on_delta: OnDelta | None,
        state: _StreamState,
    ) -> None:
        sampling = self.config.sampling
        extra_body: dict[str, Any] = dict(self.config.extra_body)
        if sampling.top_k is not None:
            extra_body["top_k"] = sampling.top_k

        stream = await self._client.chat.completions.create(
            model=self.config.model,
            messages=list(messages),
            stream=True,
            stream_options={"include_usage": True},
            tools=tools if tools else omit,
            tool_choice=tool_choice if tools and tool_choice is not None else omit,
            max_tokens=max_tokens if max_tokens is not None else omit,
            temperature=sampling.temperature if sampling.temperature is not None else omit,
            top_p=sampling.top_p if sampling.top_p is not None else omit,
            presence_penalty=(
                sampling.presence_penalty if sampling.presence_penalty is not None else omit
            ),
            extra_body=extra_body or None,
        )
        async with stream:
            async for chunk in stream:
                self._consume(chunk, on_delta, state)

    def _consume(
        self, chunk: ChatCompletionChunk, on_delta: OnDelta | None, state: _StreamState
    ) -> None:
        if chunk.usage is not None:
            state.usage = chunk.usage
        for choice in chunk.choices:
            if choice.index != 0:
                continue
            delta = choice.delta
            reasoning = _reasoning_text(delta)
            if reasoning:
                self._mark_first_token(state)
                state.reasoning.append(reasoning)
                if on_delta is not None:
                    on_delta("reasoning", reasoning)
            if delta.content:
                self._mark_first_token(state)
                state.content.append(delta.content)
                if on_delta is not None:
                    on_delta("content", delta.content)
            for call in delta.tool_calls or ():
                self._mark_first_token(state)
                part = state.tool_calls.setdefault(call.index, _PartialToolCall())
                if call.id:
                    part.id = call.id
                if call.function is not None:
                    if call.function.name:
                        part.name += call.function.name
                    if call.function.arguments:
                        part.arguments.append(call.function.arguments)
            if choice.finish_reason is not None:
                state.finish_reason = choice.finish_reason

    def _mark_first_token(self, state: _StreamState) -> None:
        if state.first_token_at is None:
            state.first_token_at = self._clock()

    def _finish(
        self, state: _StreamState, attempts: int, started_at: float, attempt_started_at: float
    ) -> Completion:
        finished_at = self._clock()
        usage = _usage_from(state.usage)
        metrics = CallMetrics(
            alias=self.config.alias,
            model=self.config.model,
            revision=self.config.revision,
            attempts=attempts,
            latency_s=finished_at - started_at,
            ttft_s=(
                state.first_token_at - attempt_started_at
                if state.first_token_at is not None
                else None
            ),
            finish_reason=state.finish_reason,
            usage=usage,
            cost_usd=compute_cost(usage, self.config.pricing) if usage is not None else None,
        )
        # Metrics only: message content never goes to logs.
        logger.info("llm_call %s", json.dumps(metrics.as_log_record()))
        if usage is None:
            logger.warning("llm_call %s: server sent no usage; cost unknown", self.config.alias)
        return Completion(
            content="".join(state.content),
            reasoning="".join(state.reasoning),
            tool_calls=_tool_calls_from(state.tool_calls),
            finish_reason=state.finish_reason,
            metrics=metrics,
        )

    def _log_failure(self, error: LLMError, started_at: float) -> None:
        logger.error(
            "llm_error %s",
            json.dumps(
                {
                    "alias": self.config.alias,
                    "model": self.config.model,
                    "revision": self.config.revision,
                    "kind": error.kind,
                    "status_code": error.status_code,
                    "attempts": error.attempts,
                    "latency_s": self._clock() - started_at,
                }
            ),
        )
