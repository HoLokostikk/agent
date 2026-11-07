"""BM25 with code-aware tokenisation: ``parseConfig`` and ``parse_config`` both
yield ``parse`` and ``config``, so either spelling of a query finds both."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

_WORD = re.compile(r"[A-Za-z0-9]+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_STOP = frozenset(
    {"the", "a", "an", "of", "to", "in", "and", "or", "is", "it", "for", "on", "self"}
)


def tokenize(text: str) -> list[str]:
    tokens = []
    for word in _WORD.findall(text):
        parts = _CAMEL.findall(word) or [word]
        for part in parts:
            low = part.lower()
            if len(low) > 1 and low not in _STOP:
                tokens.append(low)
        if len(parts) > 1 and word.lower() not in _STOP:
            tokens.append(word.lower())  # the full identifier too, for exact hits
    return tokens


class BM25:
    def __init__(self, documents: Sequence[str], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(doc.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for doc, length in zip(self.docs, self.lengths, strict=True):
            score = 0.0
            norm = (
                self.k1 * (1 - self.b + self.b * length / self.avg_len) if self.avg_len else self.k1
            )
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
            out.append(score)
        return out
