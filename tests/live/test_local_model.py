"""Against a real local Ollama. Run with ``make test-live``; skipped in CI."""

from pathlib import Path

import pytest
from openai.types.chat import ChatCompletionToolParam

from agent.llm import LLMClient, get_model

pytestmark = pytest.mark.live

REGISTRY = Path(__file__).parents[2] / "configs" / "models.yaml"

WEATHER: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


async def test_local_small_streams_text_with_metrics() -> None:
    client = LLMClient(get_model(REGISTRY, "local-small"))
    try:
        result = await client.complete(
            [{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=10
        )
    finally:
        await client.aclose()

    assert result.content.strip()
    assert result.metrics.usage is not None
    assert result.metrics.ttft_s is not None
    assert result.metrics.cost_usd == 0.0


async def test_local_small_calls_a_tool() -> None:
    client = LLMClient(get_model(REGISTRY, "local-small"))
    try:
        result = await client.complete(
            [{"role": "user", "content": "What is the weather in Kyiv?"}], tools=[WEATHER]
        )
    finally:
        await client.aclose()

    assert result.finish_reason == "tool_calls"
    assert [c.name for c in result.tool_calls] == ["get_weather"]
    assert "Kyiv" in result.tool_calls[0].arguments
