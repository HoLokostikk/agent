from calendar_utils import is_leap_year


def test_century_rules() -> None:
    assert not is_leap_year(1900)
    assert is_leap_year(2000)
    assert not is_leap_year(2100)
