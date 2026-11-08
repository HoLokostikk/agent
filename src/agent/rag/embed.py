"""Embeddings over any OpenAI-compatible ``/v1/embeddings`` endpoint."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Sequence
from typing import Protocol

import httpx2
import numpy as np
import numpy.typing as npt
from openai import AsyncOpenAI

from agent.llm.config import ModelConfig
from agent.llm.errors import LLMError, classify
from agent.llm.retry import backoff_delay

Vectors = npt.NDArray[np.float32]


class Embedder(Protocol):
    name: str

    async def embed(self, texts: Sequence[str]) -> Vectors: ...


def normalize(vectors: Vectors) -> Vectors:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


class OpenAIEmbedder:
    def __init__(self, config: ModelConfig, *, batch_size: int = 32) -> None:
        if config.kind != "embedding":
            raise ValueError(f"{config.alias} is not an embedding model")
        self.config = config
        self.name = f"{config.alias}@{config.revision}"
        self.batch_size = batch_size
        t = config.timeouts
        self._client = AsyncOpenAI(
            api_key=config.resolve_api_key(),
            base_url=config.base_url,
            timeout=httpx2.Timeout(
                connect=t.connect_s, read=t.read_s, write=t.read_s, pool=t.connect_s
            ),
            max_retries=0,
        )
        self._rng = random.Random()  # noqa: S311

    async def _batch(self, texts: list[str]) -> list[list[float]]:
        policy = self.config.retry
        for attempt in range(1, policy.max_attempts + 1):
            try:
                response = await self._client.embeddings.create(
                    model=self.config.model, input=texts
                )
                return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
            except Exception as exc:
                error = classify(exc)
                error.attempts = attempt
                if not error.retryable or attempt == policy.max_attempts:
                    raise error from exc
                await asyncio.sleep(backoff_delay(attempt, policy, self._rng, error.retry_after_s))
        raise LLMError(classify(RuntimeError("unreachable")).kind, "unreachable")

    async def embed(self, texts: Sequence[str]) -> Vectors:
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t[:6000] for t in texts[start : start + self.batch_size]]
            rows.extend(await self._batch(batch))
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return normalize(np.asarray(rows, dtype=np.float32))

    async def aclose(self) -> None:
        await self._client.close()
