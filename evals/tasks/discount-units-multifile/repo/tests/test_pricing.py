import pytest

from shop.pricing import apply_discount


def test_half_off() -> None:
    assert apply_discount(10.0, 50) == 5.0


def test_rejects_fraction_out_of_range() -> None:
    with pytest.raises(ValueError, match="between"):
        apply_discount(10.0, 150)
