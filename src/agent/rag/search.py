"""Retrieval: BM25, dense vectors, Reciprocal Rank Fusion, optional reranking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np

from agent.rag.bm25 import BM25
from agent.rag.chunker import Chunk
from agent.rag.embed import Embedder, Vectors

RRF_K = 60
CANDIDATES = 50


class Method(StrEnum):
    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid+rerank"


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Needs ``sentence-transformers`` (and torch); install it on a GPU machine."""

    def __init__(self, model: str = "BAAI/bge-reranker-base") -> None:
        from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]

        self.model = CrossEncoder(model)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        scores = self.model.predict([(query, t) for t in texts])
        return [float(s) for s in scores]


def rrf(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: (-scores[d], d))


@dataclass(frozen=True, slots=True)
class Hit:
    chunk: Chunk
    score: float


class Searcher:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        vectors: Vectors | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.bm25 = BM25([f"{c.path} {c.symbol}\n{c.text}" for c in self.chunks])
        self.vectors = vectors
        self.embedder = embedder
        self.reranker = reranker

    @property
    def methods(self) -> list[Method]:
        methods = [Method.BM25]
        if self.vectors is not None and self.embedder is not None:
            methods += [Method.VECTOR, Method.HYBRID]
            if self.reranker is not None:
                methods.append(Method.HYBRID_RERANK)
        return methods

    def _bm25(self, query: str) -> list[int]:
        scores = self.bm25.scores(query)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [i for i in order[:CANDIDATES] if scores[i] > 0]

    async def _vector(self, query: str) -> list[int]:
        if self.vectors is None or self.embedder is None:
            raise ValueError("vector search needs vectors and an embedder")
        q = await self.embedder.embed([query])
        sims = self.vectors @ q[0]
        return [int(i) for i in np.argsort(-sims)[:CANDIDATES]]

    async def search(self, query: str, k: int = 5, method: Method | None = None) -> list[Hit]:
        if not self.chunks:
            return []
        method = method or self.methods[-1]
        if method not in self.methods:
            raise ValueError(f"method {method} is not available for this index")
        if method is Method.BM25:
            ranked = self._bm25(query)
        elif method is Method.VECTOR:
            ranked = await self._vector(query)
        else:
            ranked = rrf([self._bm25(query), await self._vector(query)])[:CANDIDATES]
            if method is Method.HYBRID_RERANK and self.reranker is not None:
                top = ranked[:20]
                scores = self.reranker.score(query, [self.chunks[i].text for i in top])
                ranked = [i for _, i in sorted(zip(scores, top, strict=True), key=lambda p: -p[0])]
        n = len(ranked)
        return [Hit(self.chunks[i], 1.0 - r / max(n, 1)) for r, i in enumerate(ranked[:k])]
