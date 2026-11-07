from calc import add, multiply


def test_add() -> None:
    assert add(2, 3) == 5


def test_add_negative() -> None:
    assert add(-1, 1) == 0


def test_multiply() -> None:
    assert multiply(4, 5) == 20
