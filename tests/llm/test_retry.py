import random

import pytest

from agent.llm.config import RetryPolicy
from agent.llm.retry import backoff_delay

POLICY = RetryPolicy(max_attempts=10, base_delay_s=0.5, max_delay_s=4.0)


@pytest.mark.parametrize(("attempt", "ceiling"), [(1, 0.5), (2, 1.0), (3, 2.0), (4, 4.0), (8, 4.0)])
def test_full_jitter_stays_under_capped_ceiling(attempt: int, ceiling: float) -> None:
    rng = random.Random(42)
    delays = [backoff_delay(attempt, POLICY, rng) for _ in range(500)]

    assert all(0 <= d <= ceiling for d in delays)
    # Jitter actually spreads: not all clients wait the same time.
    assert max(delays) - min(delays) > ceiling * 0.5


def test_retry_after_overrides_backoff() -> None:
    assert backoff_delay(1, POLICY, random.Random(0), retry_after_s=7.5) == 7.5


def test_attempt_is_one_based() -> None:
    with pytest.raises(ValueError, match="1-based"):
        backoff_delay(0, POLICY, random.Random(0))
