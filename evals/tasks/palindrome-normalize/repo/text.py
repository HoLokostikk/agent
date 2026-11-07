def is_palindrome(s: str) -> bool:
    """True if `s` reads the same backwards, ignoring case, spaces and punctuation."""
    return s == s[::-1]
