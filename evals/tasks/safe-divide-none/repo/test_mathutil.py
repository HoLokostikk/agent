from mathutil import safe_divide


def test_divides() -> None:
    assert safe_divide(6, 3) == 2


def test_zero_divisor_gives_none() -> None:
    assert safe_divide(1, 0) is None
