import random
from typing import Any

import httpx2
import pytest
from openai.types.chat import ChatCompletionMessageParam

from agent.llm import ErrorKind, LLMClient, LLMError
from agent.llm.config import ModelConfig, Pricing, Sampling
from tests.llm.fakes import (
    BrokenStream,
    FakeSleep,
    Recorder,
    chunk,
    error_response,
    model_config,
    sse_body,
    sse_response,
    usage,
)

USER: list[ChatCompletionMessageParam] = [{"role": "user", "content": "hi"}]


def ok_stream(text: str = "hello") -> httpx2.Response:
    return sse_response(
        [
            chunk({"role": "assistant", "content": text}),
            chunk(finish_reason="stop"),
            chunk(usage=usage(10, 5)),
        ]
    )


def make(recorder: Recorder, config: ModelConfig | None = None) -> tuple[LLMClient, FakeSleep]:
    sleep = FakeSleep()
    client = LLMClient(
        config or model_config(),
        http_client=recorder.transport(),
        sleep=sleep,
        rng=random.Random(0),
    )
    return client, sleep


async def test_429_is_retried_then_succeeds() -> None:
    recorder = Recorder(error_response(429, {"retry-after": "0.25"}), ok_stream())
    client, sleep = make(recorder)

    result = await client.complete(USER)

    assert result.content == "hello"
    assert result.metrics.attempts == 2
    assert len(recorder.requests) == 2
    assert sleep.calls == [0.25]  # server's Retry-After wins over backoff


async def test_400_is_not_retried() -> None:
    recorder = Recorder(error_response(400), ok_stream())
    client, sleep = make(recorder)

    with pytest.raises(LLMError) as info:
        await client.complete(USER)

    assert info.value.kind is ErrorKind.BAD_REQUEST
    assert info.value.status_code == 400
    assert info.value.attempts == 1
    assert len(recorder.requests) == 1
    assert sleep.calls == []


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (401, ErrorKind.AUTH),
        (403, ErrorKind.AUTH),
        (404, ErrorKind.NOT_FOUND),
        (422, ErrorKind.BAD_REQUEST),
    ],
)
async def test_client_errors_are_not_retried(status: int, kind: ErrorKind) -> None:
    recorder = Recorder(error_response(status))
    client, _ = make(recorder)

    with pytest.raises(LLMError) as info:
        await client.complete(USER)

    assert info.value.kind is kind
    assert len(recorder.requests) == 1


async def test_timeout_is_retried_until_attempts_run_out() -> None:
    recorder = Recorder(httpx2.ReadTimeout("slow"))
    client, sleep = make(recorder)

    with pytest.raises(LLMError) as info:
        await client.complete(USER)

    assert info.value.kind is ErrorKind.TIMEOUT
    assert info.value.attempts == 3
    assert len(recorder.requests) == 3
    assert len(sleep.calls) == 2


async def test_server_errors_back_off_with_bounded_jitter() -> None:
    recorder = Recorder(error_response(503), error_response(502), ok_stream())
    client, sleep = make(recorder)

    result = await client.complete(USER)

    assert result.metrics.attempts == 3
    # Full jitter: attempt n waits in [0, min(max, base * 2**(n-1))].
    assert 0 <= sleep.calls[0] <= 0.1
    assert 0 <= sleep.calls[1] <= 0.2


async def test_connection_error_is_retried() -> None:
    recorder = Recorder(httpx2.ConnectError("refused"), ok_stream())
    client, _ = make(recorder)

    result = await client.complete(USER)

    assert result.metrics.attempts == 2


async def test_gives_up_when_retry_after_exceeds_deadline() -> None:
    from agent.llm.config import Timeouts

    config = model_config(timeouts=Timeouts(connect_s=1, read_s=1, total_s=5))
    recorder = Recorder(error_response(429, {"retry-after": "60"}), ok_stream())
    client, sleep = make(recorder, config)

    with pytest.raises(LLMError) as info:
        await client.complete(USER)

    assert info.value.kind is ErrorKind.RATE_LIMIT
    assert sleep.calls == []  # waiting would blow the total deadline
    assert len(recorder.requests) == 1


async def test_failure_after_first_token_is_not_retried() -> None:
    first = sse_body([chunk({"role": "assistant", "content": "partial"})]).removesuffix(
        b"data: [DONE]\n\n"
    )
    broken = httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, stream=BrokenStream(first)
    )
    recorder = Recorder(broken, ok_stream())
    client, sleep = make(recorder)
    seen: list[str] = []

    with pytest.raises(LLMError) as info:
        await client.complete(USER, on_delta=lambda _kind, text: seen.append(text))

    assert info.value.kind is ErrorKind.STREAM_INTERRUPTED
    assert seen == ["partial"]
    assert len(recorder.requests) == 1
    assert sleep.calls == []


async def test_tool_call_fragments_are_assembled() -> None:
    recorder = Recorder(
        sse_response(
            [
                chunk(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"pa'},
                            }
                        ],
                    }
                ),
                chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'th": "a.py"}'}}]}),
                chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 1,
                                "id": "call_b",
                                "type": "function",
                                "function": {"name": "grep", "arguments": "{}"},
                            }
                        ]
                    }
                ),
                chunk(finish_reason="tool_calls"),
                chunk(usage=usage(20, 8)),
            ]
        )
    )
    client, _ = make(recorder)

    result = await client.complete(USER)

    assert result.finish_reason == "tool_calls"
    assert [(c.id, c.name, c.arguments) for c in result.tool_calls] == [
        ("call_a", "read_file", '{"path": "a.py"}'),
        ("call_b", "grep", "{}"),
    ]
    message = result.to_message()
    assert message["content"] is None
    assert [c["id"] for c in message["tool_calls"]] == ["call_a", "call_b"]


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
async def test_reasoning_is_kept_out_of_content(field: str) -> None:
    recorder = Recorder(
        sse_response(
            [
                chunk({"role": "assistant", field: "let me think"}),
                chunk({"content": "answer"}),
                chunk(finish_reason="stop"),
                chunk(usage=usage(10, 30, reasoning=25)),
            ]
        )
    )
    client, _ = make(recorder)
    deltas: list[tuple[str, str]] = []

    result = await client.complete(USER, on_delta=lambda kind, text: deltas.append((kind, text)))

    assert result.content == "answer"
    assert result.reasoning == "let me think"
    assert deltas == [("reasoning", "let me think"), ("content", "answer")]
    assert result.metrics.usage is not None
    assert result.metrics.usage.reasoning_tokens == 25
    assert "reasoning" not in str(result.to_message())


async def test_cost_uses_cached_rate() -> None:
    config = model_config(
        pricing=Pricing(input_per_mtok=1.0, output_per_mtok=4.0, cached_input_per_mtok=0.1)
    )
    recorder = Recorder(
        sse_response(
            [
                chunk({"content": "x"}),
                chunk(finish_reason="stop"),
                chunk(usage=usage(1_000_000, 500_000, cached=400_000)),
            ]
        )
    )
    client, _ = make(recorder, config)

    result = await client.complete(USER)

    # 600k uncached * $1 + 400k cached * $0.1 + 500k out * $4, per 1M
    assert result.metrics.cost_usd == pytest.approx(0.6 + 0.04 + 2.0)


async def test_missing_usage_means_unknown_cost() -> None:
    recorder = Recorder(sse_response([chunk({"content": "x"}), chunk(finish_reason="stop")]))
    client, _ = make(recorder)

    result = await client.complete(USER)

    assert result.metrics.usage is None
    assert result.metrics.cost_usd is None


async def test_request_carries_pinned_model_and_sampling() -> None:
    config = model_config(
        sampling=Sampling(temperature=0.7, top_p=0.8, top_k=20),
        extra_body={"reasoning": {"enabled": False}},
    )
    recorder = Recorder(ok_stream())
    client, _ = make(recorder, config)
    tools: list[Any] = [
        {"type": "function", "function": {"name": "noop", "parameters": {"type": "object"}}}
    ]

    await client.complete(USER, tools=tools, max_tokens=64)

    body = recorder.body()
    assert body["model"] == "fake-model"
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["temperature"] == 0.7
    assert body["top_p"] == 0.8
    assert body["top_k"] == 20
    assert body["reasoning"] == {"enabled": False}
    assert body["max_tokens"] == 64
    assert body["tools"][0]["function"]["name"] == "noop"
    assert "presence_penalty" not in body


async def test_ttft_and_latency_are_measured() -> None:
    ticks = iter([0.0, 0.0, 0.5, 2.0])  # start, attempt start, first token, finish
    recorder = Recorder(ok_stream())
    client = LLMClient(model_config(), http_client=recorder.transport(), clock=lambda: next(ticks))

    result = await client.complete(USER)

    assert result.metrics.ttft_s == 0.5
    assert result.metrics.latency_s == 2.0
    assert result.metrics.output_tokens_per_s == pytest.approx(4 / 1.5)
