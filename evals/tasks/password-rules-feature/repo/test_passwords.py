from passwords import validate_password


def test_valid() -> None:
    assert validate_password("Secret123") == []


def test_short_without_digit() -> None:
    assert validate_password("Abc") == ["length", "digit"]
