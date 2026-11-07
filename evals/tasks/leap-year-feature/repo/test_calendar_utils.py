from calendar_utils import is_leap_year


def test_ordinary_years() -> None:
    assert is_leap_year(2024)
    assert not is_leap_year(2023)
