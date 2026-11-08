from rows import parse_line


def test_plain() -> None:
    assert parse_line("a,b,c") == ["a", "b", "c"]


def test_quoted_comma() -> None:
    assert parse_line('a,"b,c",d') == ["a", "b,c", "d"]
