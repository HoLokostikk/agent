from query import parse_query


def test_plain() -> None:
    assert parse_query("a=1&b=2") == {"a": "1", "b": "2"}


def test_decoding() -> None:
    assert parse_query("q=two+words&city=Kyiv%2C%20UA") == {"q": "two words", "city": "Kyiv, UA"}
