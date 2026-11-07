def is_palindrome(s: str) -> bool:
    """True if `s` reads the same backwards, ignoring case, spaces and punctuation."""
    cleaned = [c.lower() for c in s if c.isalnum()]
    return cleaned == cleaned[::-1]
