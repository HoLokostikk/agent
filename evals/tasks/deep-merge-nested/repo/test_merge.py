from merge import deep_merge


def test_flat() -> None:
    assert deep_merge({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}


def test_nested() -> None:
    base = {"db": {"host": "x", "port": 1}, "debug": False}
    assert deep_merge(base, {"db": {"port": 2}}) == {"db": {"host": "x", "port": 2}, "debug": False}
