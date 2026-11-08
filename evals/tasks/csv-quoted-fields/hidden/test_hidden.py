from rows import parse_line


def test_escaped_quote() -> None:
    assert parse_line('x,"say ""hi""",y') == ["x", 'say "hi"', "y"]


def test_empty_fields() -> None:
    assert parse_line("a,,c") == ["a", "", "c"]
