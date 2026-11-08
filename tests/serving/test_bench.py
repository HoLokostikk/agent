import pytest

from agent.llm import LLMClient
from agent.serving.bench import render, run_bench
from tests.llm.fakes import Handler, Recorder, chunk, model_config, sse_response, usage


def streaming_reply(n_tokens: int = 10) -> Handler:
    return lambda request: sse_response(
        [
            chunk({"content": "x"}),
            chunk(finish_reason="stop"),
            chunk(usage=usage(50, n_tokens, cached=40)),
        ]
    )


async def test_bench_reports_each_level() -> None:
    client = LLMClient(model_config(), http_client=Recorder(streaming_reply()).transport())

    levels = await run_bench(client, [1, 4], requests_per_level=4, max_tokens=16)

    assert [lvl.concurrency for lvl in levels] == [1, 4]
    assert all(lvl.errors == 0 for lvl in levels)
    assert levels[0].output_tokens == 40
    assert levels[0].throughput_tok_s > 0
    assert levels[0].cached_input_ratio == pytest.approx(0.8)
    report = render(levels, {"alias": "fake", "model": "m", "revision": "r", "max_tokens": 16})
    assert "| 4 | 4 | 0 |" in report


async def test_errors_are_counted_not_raised() -> None:
    from tests.llm.fakes import error_response

    recorder = Recorder(streaming_reply(), error_response(400))
    client = LLMClient(model_config(), http_client=recorder.transport())

    levels = await run_bench(client, [2], requests_per_level=3, max_tokens=8)

    assert levels[0].errors == 3
