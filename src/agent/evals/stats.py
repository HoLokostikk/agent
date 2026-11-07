"""Pass rates and confidence intervals for small eval sets.

The unit of independence is the task: tasks differ in difficulty far more
than repeated runs of one task differ from each other, so every bootstrap
here resamples tasks, never individual runs.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import comb

# task id -> outcomes of its n runs (True = passed)
Outcomes = Mapping[str, Sequence[bool]]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased P(at least one of k samples passes), Chen et al. 2021."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError(f"need 0 <= c <= n and 1 <= k <= n, got n={n} c={c} k={k}")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Unbiased P(all k samples pass), the pass^k of tau-bench (Yao et al. 2024)."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError(f"need 0 <= c <= n and 1 <= k <= n, got n={n} c={c} k={k}")
    return comb(c, k) / comb(n, k)


def _mean_over_tasks(outcomes: Outcomes, metric: Callable[[int, int], float]) -> float:
    if not outcomes:
        raise ValueError("no tasks")
    return sum(metric(len(runs), sum(runs)) for runs in outcomes.values()) / len(outcomes)


def mean_pass_at_k(outcomes: Outcomes, k: int) -> float:
    return _mean_over_tasks(outcomes, lambda n, c: pass_at_k(n, c, k))


def mean_pass_hat_k(outcomes: Outcomes, k: int) -> float:
    return _mean_over_tasks(outcomes, lambda n, c: pass_hat_k(n, c, k))


@dataclass(frozen=True, slots=True)
class Interval:
    estimate: float
    low: float
    high: float
    confidence: float

    def format_pct(self) -> str:
        return f"{self.estimate:.1%} [{self.low:.1%}, {self.high:.1%}]"


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty")
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def bootstrap_ci(
    outcomes: Outcomes,
    metric: Callable[[Outcomes], float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap over tasks."""
    tasks = sorted(outcomes)
    rng = random.Random(seed)  # noqa: S311  (resampling, not crypto)
    stats = []
    for _ in range(resamples):
        sample = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        # Keys must stay distinct when the same task is drawn twice.
        stats.append(metric({f"{i}:{t}": outcomes[t] for i, t in enumerate(sample)}))
    stats.sort()
    alpha = (1 - confidence) / 2
    return Interval(
        estimate=metric(outcomes),
        low=_percentile(stats, alpha),
        high=_percentile(stats, 1 - alpha),
        confidence=confidence,
    )


def paired_bootstrap_diff(
    baseline: Outcomes,
    candidate: Outcomes,
    metric: Callable[[Outcomes], float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> Interval:
    """CI for metric(candidate) - metric(baseline) on the tasks both ran.

    Resampling the same tasks for both sides keeps task difficulty paired,
    which gives a much tighter interval than comparing two separate CIs.
    """
    tasks = sorted(set(baseline) & set(candidate))
    if not tasks:
        raise ValueError("baseline and candidate share no tasks")
    rng = random.Random(seed)  # noqa: S311  (resampling, not crypto)
    diffs = []
    for _ in range(resamples):
        sample = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        b = {f"{i}:{t}": baseline[t] for i, t in enumerate(sample)}
        c = {f"{i}:{t}": candidate[t] for i, t in enumerate(sample)}
        diffs.append(metric(c) - metric(b))
    diffs.sort()
    alpha = (1 - confidence) / 2
    point = metric({t: candidate[t] for t in tasks}) - metric({t: baseline[t] for t in tasks})
    return Interval(
        estimate=point,
        low=_percentile(diffs, alpha),
        high=_percentile(diffs, 1 - alpha),
        confidence=confidence,
    )
