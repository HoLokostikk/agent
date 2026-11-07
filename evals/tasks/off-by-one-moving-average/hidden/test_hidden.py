from stats import moving_average


def test_window_of_one_is_identity() -> None:
    assert moving_average([1, 2, 3], 1) == [1, 2, 3]


def test_window_longer_than_input_is_empty() -> None:
    assert moving_average([1, 2], 3) == []
