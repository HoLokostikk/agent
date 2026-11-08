from collections import Counter


def top_words(text: str, n: int) -> list[tuple[str, int]]:
    """Most common words, case-insensitive, punctuation ignored; ties by first appearance."""
    import re

    return Counter(re.findall(r"[a-z0-9']+", text.lower())).most_common(n)
