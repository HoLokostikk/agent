from itertools import combinations

import pytest

from agent.evals.stats import (
    bootstrap_ci,
    mean_pass_at_k,
    mean_pass_hat_k,
    paired_bootstrap_diff,
    pass_at_k,
    pass_hat_k,
)


def brute_force(n: int, c: int, k: int, need_all: bool) -> float:
    runs = [True] * c + [False] * (n - c)
    subsets = list(combinations(runs, k))
    hit = [all(s) if need_all else any(s) for s in subsets]
    return sum(hit) / len(subsets)


@pytest.mark.parametrize(
    ("n", "c", "k"), [(3, 0, 1), (3, 1, 1), (3, 2, 2), (5, 2, 3), (5, 5, 3), (4, 1, 4)]
)
def test_estimators_match_brute_force(n: int, c: int, k: int) -> None:
    assert pass_at_k(n, c, k) == pytest.approx(brute_force(n, c, k, need_all=False))
    assert pass_hat_k(n, c, k) == pytest.approx(brute_force(n, c, k, need_all=True))


def test_pass_hat_k_punishes_flaky_tasks() -> None:
    flaky = {"t": [True, False, True]}

    assert mean_pass_at_k(flaky, 1) == pytest.approx(2 / 3)
    assert mean_pass_hat_k(flaky, 3) == 0.0
    assert mean_pass_at_k(flaky, 3) == 1.0


@pytest.mark.parametrize(("n", "c", "k"), [(3, 4, 1), (3, 1, 0), (3, 1, 4)])
def test_estimators_reject_impossible_inputs(n: int, c: int, k: int) -> None:
    with pytest.raises(ValueError, match="need"):
        pass_at_k(n, c, k)


def test_bootstrap_interval_contains_estimate_and_is_deterministic() -> None:
    outcomes = {f"t{i}": [i % 3 == 0] * 3 for i in range(30)}

    a = bootstrap_ci(outcomes, lambda o: mean_pass_at_k(o, 1), resamples=2000, seed=1)
    b = bootstrap_ci(outcomes, lambda o: mean_pass_at_k(o, 1), resamples=2000, seed=1)

    assert a == b
    assert a.low <= a.estimate <= a.high
    assert a.estimate == pytest.approx(10 / 30)
    assert 0.15 < a.low < a.estimate < a.high < 0.55


def test_all_pass_has_degenerate_interval() -> None:
    outcomes = {f"t{i}": [True, True] for i in range(10)}

    ci = bootstrap_ci(outcomes, lambda o: mean_pass_at_k(o, 1), resamples=500)

    assert (ci.low, ci.estimate, ci.high) == (1.0, 1.0, 1.0)


def test_paired_diff_detects_a_real_regression() -> None:
    base = {f"t{i}": [True] * 3 for i in range(20)}
    cand = {f"t{i}": [i >= 10] * 3 for i in range(20)}

    diff = paired_bootstrap_diff(base, cand, lambda o: mean_pass_at_k(o, 1), resamples=2000)

    assert diff.estimate == pytest.approx(-0.5)
    assert diff.high < 0


def test_paired_diff_of_identical_runs_is_zero() -> None:
    runs = {f"t{i}": [i % 2 == 0] for i in range(20)}

    diff = paired_bootstrap_diff(runs, runs, lambda o: mean_pass_at_k(o, 1), resamples=500)

    assert (diff.low, diff.estimate, diff.high) == (0.0, 0.0, 0.0)
