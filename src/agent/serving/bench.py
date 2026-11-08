"""Load benchmark: TTFT, inter-token latency and throughput at several concurrencies.

Inter-token latency (ITL) per request is the decode time divided by the
tokens after the first: (latency - TTFT) / (output_tokens - 1). Throughput is
total output tokens of a level divided by that level's wall time, which is
what a GPU-hour buys.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from agent.llm import LLMClient, LLMError

PROMPTS = [
    "Explain what a Python context manager is and write a small example.",
    "Write a function that merges two sorted lists, then explain its complexity.",
    "What is the difference between a process and a thread? Answer in detail.",
    "Refactor this into idiomatic Python: for i in range(len(xs)): print(i, xs[i])",
]


@dataclass(frozen=True, slots=True)
class LevelResult:
    concurrency: int
    requests: int
    errors: int
    wall_s: float
    output_tokens: int
    throughput_tok_s: float
    ttft_p50_s: float | None
    ttft_p95_s: float | None
    itl_p50_ms: float | None
    itl_p95_ms: float | None
    latency_p50_s: float | None
    latency_p95_s: float | None
    cached_input_ratio: float | None


def _pct(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[round(q * 100) - 1]


async def run_level(
    client: LLMClient, concurrency: int, requests: int, max_tokens: int, shared_prefix: str = ""
) -> LevelResult:
    semaphore = asyncio.Semaphore(concurrency)
    ttft: list[float] = []
    itl: list[float] = []
    latency: list[float] = []
    out_tokens = 0
    in_tokens = 0
    cached = 0
    errors = 0

    async def one(i: int) -> None:
        nonlocal out_tokens, in_tokens, cached, errors
        prompt = shared_prefix + PROMPTS[i % len(PROMPTS)]
        async with semaphore:
            try:
                result = await client.complete(
                    [{"role": "user", "content": prompt}], max_tokens=max_tokens
                )
            except LLMError:
                errors += 1
                return
        m = result.metrics
        latency.append(m.latency_s)
        if m.ttft_s is not None:
            ttft.append(m.ttft_s)
        if m.usage is not None:
            out_tokens += m.usage.output_tokens
            in_tokens += m.usage.input_tokens
            cached += m.usage.cached_input_tokens
            if m.ttft_s is not None and m.usage.output_tokens > 1:
                itl.append((m.latency_s - m.ttft_s) / (m.usage.output_tokens - 1))

    started = time.perf_counter()
    await asyncio.gather(*(one(i) for i in range(requests)))
    wall = time.perf_counter() - started
    return LevelResult(
        concurrency=concurrency,
        requests=requests,
        errors=errors,
        wall_s=wall,
        output_tokens=out_tokens,
        throughput_tok_s=out_tokens / wall if wall > 0 else 0.0,
        ttft_p50_s=_pct(ttft, 0.5),
        ttft_p95_s=_pct(ttft, 0.95),
        itl_p50_ms=(p * 1000 if (p := _pct(itl, 0.5)) is not None else None),
        itl_p95_ms=(p * 1000 if (p := _pct(itl, 0.95)) is not None else None),
        latency_p50_s=_pct(latency, 0.5),
        latency_p95_s=_pct(latency, 0.95),
        cached_input_ratio=cached / in_tokens if in_tokens else None,
    )


async def run_bench(
    client: LLMClient,
    concurrencies: Sequence[int],
    requests_per_level: int,
    max_tokens: int,
    shared_prefix: str = "",
) -> list[LevelResult]:
    # One warm-up call so the first level does not pay model load time.
    await client.complete([{"role": "user", "content": "warm up"}], max_tokens=1)
    return [
        await run_level(client, c, max(requests_per_level, c), max_tokens, shared_prefix)
        for c in concurrencies
    ]


def render(levels: Sequence[LevelResult], meta: dict[str, Any]) -> str:
    def f(v: float | None, fmt: str) -> str:
        return "—" if v is None else format(v, fmt)

    lines = [
        f"# serving benchmark · {meta.get('alias')} · {meta.get('label', '')}".rstrip(" ·"),
        "",
        f"- model `{meta.get('model')}` revision `{meta.get('revision')}`"
        f" · max_tokens {meta.get('max_tokens')}",
        f"- shared prefix: {meta.get('shared_prefix_chars', 0)} chars (prefix-cache test when > 0)",
        "",
        "| concurrency | req | err | throughput tok/s | TTFT p50 / p95 s "
        "| ITL p50 / p95 ms | latency p50 / p95 s | cached input |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in levels:
        lines.append(
            f"| {r.concurrency} | {r.requests} | {r.errors} | {r.throughput_tok_s:.1f} | "
            f"{f(r.ttft_p50_s, '.3f')} / {f(r.ttft_p95_s, '.3f')} | "
            f"{f(r.itl_p50_ms, '.1f')} / {f(r.itl_p95_ms, '.1f')} | "
            f"{f(r.latency_p50_s, '.2f')} / {f(r.latency_p95_s, '.2f')} | "
            f"{f(r.cached_input_ratio, '.0%')} |"
        )
    return "\n".join(lines) + "\n"


def as_dicts(levels: Sequence[LevelResult]) -> list[dict[str, Any]]:
    return [asdict(level) for level in levels]
