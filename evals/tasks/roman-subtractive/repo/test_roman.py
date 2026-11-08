from roman import from_roman


def test_additive() -> None:
    assert from_roman("VIII") == 8


def test_subtractive() -> None:
    assert from_roman("IV") == 4
    assert from_roman("MCMXCIV") == 1994
