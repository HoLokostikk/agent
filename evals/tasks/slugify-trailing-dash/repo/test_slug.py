from slug import slugify


def test_basic() -> None:
    assert slugify("Hello World") == "hello-world"


def test_punctuation() -> None:
    assert slugify("Hello, World!") == "hello-world"
