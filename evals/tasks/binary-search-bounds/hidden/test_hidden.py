from search import binary_search


def test_every_position() -> None:
    items = list(range(0, 40, 2))
    for i, x in enumerate(items):
        assert binary_search(items, x) == i


def test_empty() -> None:
    assert binary_search([], 1) == -1
