"""Exponential backoff with full jitter."""

from __future__ import annotations

import random

from agent.llm.config import RetryPolicy


def backoff_delay(
    failed_attempt: int,
    policy: RetryPolicy,
    rng: random.Random,
    retry_after_s: float | None = None,
) -> float:
    """Seconds to wait after the ``failed_attempt``-th attempt (1-based).

    A server-provided ``Retry-After`` wins: the server knows its own limits.
    Otherwise full jitter, ``uniform(0, min(cap, base * 2**(n-1)))``: clients
    that failed together spread out instead of retrying in lockstep.
    """
    if failed_attempt < 1:
        raise ValueError("failed_attempt is 1-based")
    if retry_after_s is not None:
        return retry_after_s
    ceiling = min(policy.max_delay_s, policy.base_delay_s * 2 ** (failed_attempt - 1))
    return rng.uniform(0.0, ceiling)
