from text import is_palindrome


def test_simple() -> None:
    assert is_palindrome("level")
    assert not is_palindrome("python")


def test_ignores_case_and_spaces() -> None:
    assert is_palindrome("Never odd or even")
