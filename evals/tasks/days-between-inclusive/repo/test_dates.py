from datetime import date

from dates import days_inclusive


def test_same_day_counts_once() -> None:
    assert days_inclusive(date(2026, 1, 1), date(2026, 1, 1)) == 1


def test_week() -> None:
    assert days_inclusive(date(2026, 1, 1), date(2026, 1, 7)) == 7
