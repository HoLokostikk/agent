"""Fake OpenAI-compatible server built on httpx2.MockTransport.

The openai SDK 3.x runs on httpx2, which respx does not patch, so tests hand
the client an ``httpx2.AsyncClient`` with a mock transport instead.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx2

from agent.llm.config import ModelConfig, Pricing, RetryPolicy, Timeouts

Handler = Callable[[httpx2.Request], httpx2.Response]


def model_config(**overrides: Any) -> ModelConfig:
    fields: dict[str, Any] = {
        "alias": "fake",
        "base_url": "http://fake.test/v1",
        "model": "fake-model",
        "revision": "rev-1",
        "context_window": 8192,
        "pricing": Pricing(input_per_mtok=1.0, output_per_mtok=2.0),
        "retry": RetryPolicy(max_attempts=3, base_delay_s=0.1, max_delay_s=1.0),
        "timeouts": Timeouts(connect_s=1.0, read_s=1.0, total_s=30.0),
    }
    fields.update(overrides)
    return ModelConfig(**fields)


def chunk(
    delta: dict[str, Any] | None = None,
    *,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "fake-model",
        "choices": [],
    }
    if delta is not None or finish_reason is not None:
        body["choices"] = [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}]
    if usage is not None:
        body["usage"] = usage
    return body


def usage(
    prompt: int, completion: int, cached: int = 0, reasoning: int | None = None
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": cached},
    }
    if reasoning is not None:
        body["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    return body


def sse_body(chunks: list[dict[str, Any]]) -> bytes:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def sse_response(chunks: list[dict[str, Any]]) -> httpx2.Response:
    return httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, content=sse_body(chunks)
    )


def error_response(status: int, headers: dict[str, str] | None = None) -> httpx2.Response:
    return httpx2.Response(
        status,
        headers=headers or {},
        json={"error": {"message": f"fake {status}", "type": "fake"}},
    )


class BrokenStream(httpx2.AsyncByteStream):
    """Yields some SSE bytes, then fails like a dropped connection."""

    def __init__(self, first: bytes) -> None:
        self._first = first

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._first
        raise httpx2.ReadError("connection reset")


class Recorder:
    """Plays scripted responses in order and records every request."""

    def __init__(self, *steps: httpx2.Response | Exception | Handler) -> None:
        self._steps = list(steps)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        index = min(len(self.requests), len(self._steps)) - 1
        step = self._steps[index]
        if isinstance(step, Exception):
            raise step
        if isinstance(step, httpx2.Response):
            return step
        return step(request)

    def body(self, n: int = 0) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self.requests[n].content)
        return data

    def transport(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self))


class FakeSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
