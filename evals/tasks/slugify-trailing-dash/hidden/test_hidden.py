from slug import slugify


def test_leading_punctuation() -> None:
    assert slugify("  -- Python 3.12 --") == "python-3-12"
