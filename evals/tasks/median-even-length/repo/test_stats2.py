from stats2 import median


def test_odd() -> None:
    assert median([3, 1, 2]) == 2


def test_even() -> None:
    assert median([4, 1, 3, 2]) == 2.5
