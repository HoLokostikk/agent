from freq import top_words


def test_simple() -> None:
    assert top_words("a b a", 1) == [("a", 2)]


def test_case_and_punctuation() -> None:
    assert top_words("The cat. the CAT, the dog!", 2) == [("the", 3), ("cat", 2)]
