from tags import add_tag


def test_appends_to_given_list() -> None:
    assert add_tag("b", ["a"]) == ["a", "b"]


def test_calls_without_list_are_independent() -> None:
    assert add_tag("x") == ["x"]
    assert add_tag("y") == ["y"]
