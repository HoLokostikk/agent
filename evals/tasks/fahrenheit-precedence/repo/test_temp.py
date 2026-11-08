import pytest

from temp import f_to_c


def test_freezing() -> None:
    assert f_to_c(32) == pytest.approx(0)


def test_boiling() -> None:
    assert f_to_c(212) == pytest.approx(100)
