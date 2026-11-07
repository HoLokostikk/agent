from duration import parse_duration


def test_all_units() -> None:
    assert parse_duration("2h5m10s") == 7510


def test_minutes_only() -> None:
    assert parse_duration("90m") == 5400
