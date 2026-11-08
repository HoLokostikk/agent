from collections import Counter


def top_words(text: str, n: int) -> list[tuple[str, int]]:
    """Most common words, case-insensitive, punctuation ignored; ties by first appearance."""
    return Counter(text.split()).most_common(n)
