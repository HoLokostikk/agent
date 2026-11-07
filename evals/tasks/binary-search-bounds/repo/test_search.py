from search import binary_search


def test_finds_middle() -> None:
    assert binary_search([1, 3, 5, 7, 9], 5) == 2


def test_finds_last() -> None:
    assert binary_search([1, 3, 5, 7, 9], 9) == 4


def test_single_element() -> None:
    assert binary_search([4], 4) == 0


def test_absent() -> None:
    assert binary_search([1, 3, 5], 4) == -1
