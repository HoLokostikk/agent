from intervals import merge


def test_unsorted_and_nested() -> None:
    assert merge([(5, 6), (1, 10), (2, 3)]) == [(1, 10)]


def test_empty() -> None:
    assert merge([]) == []
