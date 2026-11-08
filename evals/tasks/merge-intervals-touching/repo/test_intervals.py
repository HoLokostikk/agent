from intervals import merge


def test_overlapping() -> None:
    assert merge([(1, 3), (2, 6), (8, 10)]) == [(1, 6), (8, 10)]


def test_touching_intervals_merge() -> None:
    assert merge([(1, 2), (2, 3)]) == [(1, 3)]
