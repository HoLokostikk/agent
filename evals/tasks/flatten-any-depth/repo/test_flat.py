from flat import flatten


def test_one_level() -> None:
    assert flatten([1, [2, 3], 4]) == [1, 2, 3, 4]


def test_deep() -> None:
    assert flatten([1, [2, [3, [4]]]]) == [1, 2, 3, 4]
