from flat import flatten


def test_empty_lists_vanish() -> None:
    assert flatten([[], [[]], 1]) == [1]


def test_strings_are_not_split() -> None:
    assert flatten(["ab", ["cd"]]) == ["ab", "cd"]
