from text import is_palindrome


def test_ignores_punctuation() -> None:
    assert is_palindrome("No 'x' in Nixon")


def test_empty() -> None:
    assert is_palindrome("")
