from words import count_words


def test_simple() -> None:
    assert count_words("one two three") == 3


def test_irregular_whitespace() -> None:
    assert count_words("  one   two\nthree\t") == 3


def test_empty() -> None:
    assert count_words("") == 0
