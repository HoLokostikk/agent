import pytest

from stats import mean, moving_average


def test_mean() -> None:
    assert mean([1, 2, 3]) == 2


def test_moving_average() -> None:
    assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]


def test_moving_average_full_window() -> None:
    assert moving_average([2, 4], 2) == [3]


def test_moving_average_rejects_bad_window() -> None:
    with pytest.raises(ValueError, match="positive"):
        moving_average([1], 0)
